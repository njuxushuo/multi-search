#!/usr/bin/env python3
"""Offline consistency audit for the complete frozen R3.0 implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from protocol_v3 import (
    DEFAULT_CONFIG,
    canonical_prompt,
    compile_sft_example,
    file_sha256,
    load_protocol,
    normalize_answer,
    prompt_from_row,
    render_initial_prompt,
    token_count,
)


ROOT = Path(__file__).resolve().parents[1]


def manifest_hash(path: Path) -> tuple[str, list[dict]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--tokenizer-check", action="store_true")
    args = parser.parse_args()
    config = load_protocol(args.protocol_config)
    report = {
        "protocol_id": config["protocol_id"],
        "human_version": config["human_version"],
        "protocol_config_sha256": file_sha256(args.protocol_config),
        "checks": {},
    }

    documents = [
        ROOT / "README.md",
        ROOT / "docs/EXPERIMENT_PROTOCOL.md",
        ROOT / "docs/ROADMAP.md",
        ROOT / "docs/PROJECT_STATUS.md",
    ]
    for path in documents:
        text = path.read_text(encoding="utf-8")
        if config["human_version"] not in text or config["protocol_id"] not in text:
            raise SystemExit(f"version registry missing from {path}")
    report["checks"]["version_registry_documents"] = len(documents)

    prompt_rows = 0
    prompt_errors = Counter()
    for split_name, path_text in (
        ("train", config["data"]["teacher_source"]),
        ("test", config["data"]["final_test_source"]),
    ):
        rows = pq.read_table(path_text, columns=["question", "prompt"]).to_pylist()
        for row in rows:
            try:
                prompt_from_row(row, config)
                prompt_rows += 1
            except ValueError as exc:
                prompt_errors[f"{split_name}:{exc}"] += 1
    if prompt_errors:
        raise SystemExit(f"dataset prompt drift: {dict(prompt_errors)}")
    report["checks"]["dataset_prompts_exact"] = prompt_rows

    full_path = ROOT / config["data"]["interactive_dev_manifest"]
    quick_path = ROOT / config["data"]["interactive_dev_quick_manifest"]
    full_hash, full_rows = manifest_hash(full_path)
    quick_hash, quick_rows = manifest_hash(quick_path)
    if len(full_rows) != config["data"]["interactive_dev_count"]:
        raise SystemExit("interactive dev size mismatch")
    if len(quick_rows) != config["data"]["interactive_dev_quick_count"]:
        raise SystemExit("quick interactive dev size mismatch")
    if [row["source_row"] for row in quick_rows] != [row["source_row"] for row in full_rows[:len(quick_rows)]]:
        raise SystemExit("quick dev is not the frozen prefix of full dev")
    final_hash, final_rows = manifest_hash(ROOT / config["data"]["final_test_manifest"])
    final_questions = {normalize_answer(row["question"]) for row in final_rows}
    if any(normalize_answer(row["question"]) in final_questions for row in full_rows):
        raise SystemExit("interactive dev question leakage into final manifest")
    report["checks"]["interactive_dev"] = {
        "full_count": len(full_rows),
        "full_sources": dict(Counter(row["data_source"] for row in full_rows)),
        "full_sha256": full_hash,
        "quick_count": len(quick_rows),
        "quick_sources": dict(Counter(row["data_source"] for row in quick_rows)),
        "quick_sha256": quick_hash,
        "final_manifest_sha256": final_hash,
        "question_overlap": 0,
    }

    pilot_path = ROOT / config["data"]["teacher_pilot_manifest"]
    pilot_metadata_path = pilot_path.with_suffix(".metadata.json")
    pilot_hash, pilot_rows = manifest_hash(pilot_path)
    pilot_metadata = json.loads(pilot_metadata_path.read_text(encoding="utf-8"))
    pilot_question_hashes = [str(row.get("question_sha256", "")) for row in pilot_rows]
    excluded_hashes = {
        hashlib.sha256(normalize_answer(row["question"]).encode("utf-8")).hexdigest()
        for row in final_rows + full_rows
    }
    expected_pilot = int(config["teacher_sampling"]["pilot_unique_questions"])
    source_counts = Counter(str(row["data_source"]) for row in pilot_rows)
    expected_sources = {
        "hotpotqa": round(expected_pilot * float(config["teacher_sampling"]["pilot_hotpotqa_fraction"])),
        "nq": round(expected_pilot * float(config["teacher_sampling"]["pilot_nq_fraction"])),
    }
    expected_sources["hotpotqa"] += expected_pilot - sum(expected_sources.values())
    if len(pilot_rows) != expected_pilot or len(set(pilot_question_hashes)) != expected_pilot:
        raise SystemExit("Teacher pilot manifest count/uniqueness mismatch")
    if set(pilot_question_hashes) & excluded_hashes:
        raise SystemExit("Teacher pilot manifest leaks into interactive dev or final test")
    if dict(source_counts) != expected_sources:
        raise SystemExit(f"Teacher pilot source distribution mismatch: {dict(source_counts)}")
    if pilot_metadata.get("manifest_sha256") != file_sha256(pilot_path):
        raise SystemExit("Teacher pilot manifest metadata checksum mismatch")
    if pilot_metadata.get("protocol_config_sha256") != file_sha256(args.protocol_config):
        raise SystemExit("Teacher pilot manifest protocol checksum mismatch")
    report["checks"]["teacher_pilot_manifest"] = {
        "count": len(pilot_rows),
        "source_counts": dict(source_counts),
        "manifest_sha256": pilot_hash,
        "dev_final_overlap": 0,
        "unique_normalized_questions": len(set(pilot_question_hashes)),
    }

    train_config = json.loads((ROOT / "configs/sft_qwen35_4b_full_v3.json").read_text(encoding="utf-8"))
    frozen_sft = config["sft"]
    expected = {
        "model_name_or_path": config["models"]["student_initial"],
        "learning_rate": frozen_sft["learning_rate"],
        "num_train_epochs": float(frozen_sft["scheduler_horizon_epochs"]),
        "warmup_ratio": frozen_sft["warmup_ratio"],
        "eval_steps": frozen_sft["eval_steps"],
        "seed": frozen_sft["seed"],
    }
    drift = {key: (train_config.get(key), value) for key, value in expected.items() if train_config.get(key) != value}
    if drift:
        raise SystemExit(f"training config drift: {drift}")
    report["checks"]["training_config"] = "matches_frozen_values"

    if args.tokenizer_check:
        from transformers import AutoTokenizer
        tokenizer_reports = {}
        for role in ("teacher", "student_initial"):
            model_path = config["models"][role]
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            prompt = canonical_prompt("Where was the author born?", config)
            prompt_tokens = token_count(tokenizer, render_initial_prompt(tokenizer, prompt))
            if prompt_tokens > config["token_budget"]["initial_prompt_tokens"]:
                raise SystemExit(f"{role} initial prompt exceeds budget")
            events = [
                {"kind": "generated", "text": "<think>Search.</think><search>author birthplace</search>"},
                {"kind": "observation", "text": "<information>\nDoc 1(Title: Author) Born in Paris.\n</information>"},
                {"kind": "generated", "text": "<think>Answer.</think><answer>Paris</answer>"},
            ]
            compiled = compile_sft_example(
                tokenizer,
                prompt,
                events,
                config["token_budget"]["sft_cutoff_len"],
                config=config,
            )
            tokenizer_reports[role] = {
                "class": type(tokenizer).__name__,
                "initial_prompt_tokens": prompt_tokens,
                "compiled_tokens": compiled["token_count"],
                "supervised_tokens": compiled["supervised_token_count"],
            }
        report["checks"]["tokenizers"] = tokenizer_reports

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

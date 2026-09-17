#!/usr/bin/env python3
"""Compile selected R3.0 trajectories into pretokenized masked Parquet."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from protocol_v3 import (
    DEFAULT_CONFIG,
    compile_sft_example,
    file_sha256,
    load_protocol,
    validate_strict_candidate_record,
)


def quantile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def load_manifest(
    path: str, protocol_id: str, protocol_config_sha256: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metadata = None
    selections: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "metadata":
                metadata = row
            elif row.get("type") == "selection":
                candidate_id = str(row["candidate_id"])
                if candidate_id in selections:
                    raise SystemExit(f"duplicate manifest candidate_id: {candidate_id}")
                selections[candidate_id] = row
    if metadata is None or metadata.get("protocol_id") != protocol_id:
        raise SystemExit("selection manifest protocol mismatch or missing metadata")
    if metadata.get("protocol_config_sha256") != protocol_config_sha256:
        raise SystemExit("selection manifest protocol checksum mismatch")
    return metadata, selections


def load_selected_candidates(
    paths: list[str],
    selections: dict[str, dict[str, Any]],
    config: dict[str, Any],
    protocol_config_sha256: str,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for path_text in paths:
        with Path(path_text).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("type") != "candidate":
                    continue
                candidate_id = str(row.get("candidate_id", ""))
                if candidate_id not in selections:
                    continue
                if candidate_id in selected:
                    raise SystemExit(f"duplicate selected candidate across inputs: {candidate_id}")
                if row.get("protocol_id") != config["protocol_id"] or not row.get("strict_sft_eligible"):
                    raise SystemExit(f"selected candidate is not strict R3.0 eligible: {candidate_id}")
                if row.get("protocol_config_sha256") != protocol_config_sha256:
                    raise SystemExit(f"selected candidate protocol checksum mismatch: {candidate_id}")
                try:
                    validate_strict_candidate_record(row, config)
                except ValueError as exc:
                    raise SystemExit(f"selected candidate integrity failure {candidate_id}: {exc}") from exc
                selected[candidate_id] = row
    missing = sorted(set(selections) - set(selected))
    if missing:
        raise SystemExit(f"selection manifest references missing candidates: {missing[:10]}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--protocol-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--selection-manifest", default="data/processed/searchqa_repro_v3_0_0/selection_manifest.jsonl")
    parser.add_argument("--student", default=None)
    parser.add_argument("--output-dir", default="data/processed/searchqa_repro_v3_0_0/sft_tokenized")
    args = parser.parse_args()
    config = load_protocol(args.protocol_config)
    protocol_sha = file_sha256(args.protocol_config)
    student = args.student or config["models"]["student_initial"]
    manifest_metadata, selections = load_manifest(
        args.selection_manifest, config["protocol_id"], protocol_sha
    )
    candidates = load_selected_candidates(args.inputs, selections, config, protocol_sha)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(student, trust_remote_code=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "eval": [], "reserve": []}
    for candidate_id, selection in selections.items():
        candidate = candidates[candidate_id]
        compiled = compile_sft_example(
            tokenizer,
            str(candidate["initial_prompt"]),
            candidate["events"],
            max_length=config["token_budget"]["sft_cutoff_len"],
            ignore_index=config["loss"]["ignore_index"],
            config=config,
        )
        labels = compiled["labels"]
        input_ids = compiled["input_ids"]
        if len(input_ids) != len(labels) or len(input_ids) != len(compiled["attention_mask"]):
            raise SystemExit(f"tensor length mismatch: {candidate_id}")
        if any(
            labels[index] != config["loss"]["ignore_index"]
            for span in compiled["spans"] if not span["supervised"]
            for index in range(span["start"], span["end"])
        ):
            raise SystemExit(f"masked span contains supervised labels: {candidate_id}")
        if any(
            labels[index] == config["loss"]["ignore_index"]
            for span in compiled["spans"] if span["supervised"]
            for index in range(span["start"], span["end"])
        ):
            raise SystemExit(f"generated span contains masked labels: {candidate_id}")
        split = str(selection["split"])
        information_tokens = sum(
            span["end"] - span["start"] for span in compiled["spans"] if span["kind"] == "observation"
        )
        prompt_tokens = sum(
            span["end"] - span["start"]
            for span in compiled["spans"] if span["kind"] in {"prompt", "separator"}
        )
        metadata = candidate.get("metadata") or {}
        split_rows[split].append({
            "candidate_id": candidate_id,
            "question_id": str(candidate["question_id"]),
            "question_sha256": str(candidate.get("question_sha256", "")),
            "golden_answers": [str(value) for value in candidate.get("golden_answers", [])],
            "data_source": str(candidate["data_source"]),
            "hotpot_type": str(metadata.get("type") or ""),
            "hotpot_level": str(metadata.get("level") or ""),
            "search_action_count": int(candidate["search_action_count"]),
            "bm25_execution_count": int(candidate["bm25_execution_count"]),
            "unique_document_count": int(candidate["unique_document_count"]),
            "evidence_grade": str(candidate.get("evidence_grade", "")),
            "teacher": str(candidate.get("teacher", "")),
            "protocol_config_sha256": str(candidate.get("protocol_config_sha256", "")),
            "pilot_manifest_sha256": str(candidate.get("pilot_manifest_sha256", "")),
            "trajectory_audit_json": json.dumps({
                "raw_generations": candidate.get("raw_generations"),
                "canonical_trajectory": candidate.get("canonical_trajectory"),
                "retrieval_rounds": candidate.get("retrieval_rounds"),
                "termination_reason": candidate.get("termination_reason"),
                "invalid_reasons": candidate.get("invalid_reasons"),
                "sft_rejection_reasons": candidate.get("sft_rejection_reasons"),
            }, ensure_ascii=False),
            "input_ids": input_ids,
            "attention_mask": compiled["attention_mask"],
            "labels": labels,
            "token_count": int(compiled["token_count"]),
            "supervised_token_count": int(compiled["supervised_token_count"]),
            "information_token_count": information_tokens,
            "prompt_token_count": prompt_tokens,
        })

    expected = {
        "train": int(config["selection"]["train_trajectories"]),
        "eval": int(config["selection"]["teacher_forced_eval_trajectories"]),
        "reserve": int(config["selection"]["reserve_trajectories"]),
    }
    actual = {split: len(rows) for split, rows in split_rows.items()}
    if actual != expected:
        raise SystemExit(f"compiled split sizes differ from frozen protocol: {actual} != {expected}")

    schema = pa.schema([
        ("candidate_id", pa.string()),
        ("question_id", pa.string()),
        ("question_sha256", pa.string()),
        ("golden_answers", pa.list_(pa.string())),
        ("data_source", pa.string()),
        ("hotpot_type", pa.string()),
        ("hotpot_level", pa.string()),
        ("search_action_count", pa.int16()),
        ("bm25_execution_count", pa.int16()),
        ("unique_document_count", pa.int16()),
        ("evidence_grade", pa.string()),
        ("teacher", pa.string()),
        ("protocol_config_sha256", pa.string()),
        ("pilot_manifest_sha256", pa.string()),
        ("trajectory_audit_json", pa.string()),
        ("input_ids", pa.list_(pa.int32())),
        ("attention_mask", pa.list_(pa.int8())),
        ("labels", pa.list_(pa.int32())),
        ("token_count", pa.int32()),
        ("supervised_token_count", pa.int32()),
        ("information_token_count", pa.int32()),
        ("prompt_token_count", pa.int32()),
    ])
    files = {}
    stats: dict[str, Any] = {}
    for split, rows in split_rows.items():
        path = output / f"{split}.parquet"
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, path, compression="zstd")
        files[split] = {"path": str(path), "sha256": file_sha256(path)}
        lengths = [row["token_count"] for row in rows]
        supervised = [row["supervised_token_count"] for row in rows]
        information = [row["information_token_count"] for row in rows]
        stats[split] = {
            "count": len(rows),
            "tokens": {
                "mean": statistics.mean(lengths),
                "p50": quantile(lengths, 0.5),
                "p90": quantile(lengths, 0.9),
                "p95": quantile(lengths, 0.95),
                "p99": quantile(lengths, 0.99),
                "max": max(lengths),
            },
            "mean_supervised_tokens": statistics.mean(supervised),
            "mean_supervised_fraction": statistics.mean(
                active / total for active, total in zip(supervised, lengths)
            ),
            "mean_information_tokens": statistics.mean(information),
            "mean_information_mask_fraction": statistics.mean(
                masked / total for masked, total in zip(information, lengths)
            ),
            "sources": dict(Counter(row["data_source"] for row in rows)),
            "search_actions": dict(Counter(row["search_action_count"] for row in rows)),
            "evidence_grades": dict(Counter(row["evidence_grade"] for row in rows)),
        }
    metadata = {
        "protocol_id": config["protocol_id"],
        "human_version": config["human_version"],
        "protocol_config": str(Path(args.protocol_config).resolve()),
        "protocol_config_sha256": file_sha256(args.protocol_config),
        "selection_manifest": str(Path(args.selection_manifest).resolve()),
        "selection_manifest_sha256": file_sha256(args.selection_manifest),
        "selection_metadata": manifest_metadata,
        "student_tokenizer": student,
        "tokenizer_class": type(tokenizer).__name__,
        "ignore_index": config["loss"]["ignore_index"],
        "loss_mask": config["loss"],
        "files": files,
        "stats": stats,
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metadata": str(metadata_path), "stats": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()

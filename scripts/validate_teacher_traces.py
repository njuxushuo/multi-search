#!/usr/bin/env python3
"""Validate and split teacher-generated Search-R1 traces for LLaMA-Factory.

The validator intentionally rejects oracle-answer-only records and traces that
never perform a search.  It therefore cannot silently turn a pilot dataset
into the exact 15k cold-start dataset required by the experiment.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from verl.utils.reward_score.qa_em_format import is_valid_sequence
from trace_validation import validate_tool_progress


INFO_RE = re.compile(r"(<information>.*?</information>)", re.I | re.S)


def to_masked_conversations(question: str, assistant: str) -> list[dict[str, str]]:
    """Represent retrieved evidence as user turns so SFT loss excludes it."""
    turns: list[dict[str, str]] = [{"from": "human", "value": question}]
    parts = INFO_RE.split(assistant)
    for i, part in enumerate(parts):
        if not part.strip():
            continue
        role = "human" if i % 2 else "gpt"
        turns.append({"from": role, "value": part.strip()})
    if turns[-1]["from"] != "gpt":
        raise ValueError("trajectory ends with evidence turn")
    return turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL emitted by the 27B teacher")
    ap.add_argument("--output-dir", default="data/processed/search_sft_qwen35_4b")
    ap.add_argument("--train-size", type=int, default=15000)
    ap.add_argument("--eval-size", type=int, default=1000)
    args = ap.parse_args()
    records = []
    rejected = {}
    for line_no, line in enumerate(Path(args.input).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            assistant = str(row.get("assistant", row.get("trajectory", ""))).strip()
            source = str(row.get("data_source", "unknown"))
            if not question or not answer:
                raise ValueError("empty question/answer")
            if "<search>" not in assistant or "<information>" not in assistant:
                raise ValueError("no search/information turn")
            if "<answer>" not in assistant or "</answer>" not in assistant:
                raise ValueError("missing answer tags")
            valid_progress, progress_reason = validate_tool_progress(assistant, is_valid_sequence)
            if not valid_progress:
                raise ValueError(progress_reason)
            if row.get("label_type", "").endswith("oracle_answer_pilot"):
                raise ValueError("oracle pilot record")
            records.append({
                "conversations": to_masked_conversations(question, assistant),
                "metadata": {"source": source, "teacher": "Qwen3.5-27B", "line": line_no,
                             "loss_mask": "assistant_turns_only_v2"},
            })
        except Exception as exc:
            key = str(exc)
            rejected[key] = rejected.get(key, 0) + 1
    required = args.train_size + args.eval_size
    if len(records) < required:
        raise SystemExit(f"Only {len(records)} valid traces; need {required}. Rejected: {rejected}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = [("train", records[: args.train_size]), ("eval", records[args.train_size:required])]
    for name, rows in splits:
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "dataset_info.json").write_text(json.dumps({
        "searchqa_sft_train_15k": {"file_name": "train.jsonl", "formatting": "sharegpt", "columns": {"messages": "conversations"}},
        "searchqa_sft_eval": {"file_name": "eval.jsonl", "formatting": "sharegpt", "columns": {"messages": "conversations"}},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid": len(records), "train": args.train_size, "eval": args.eval_size, "rejected": rejected}, ensure_ascii=False))


if __name__ == "__main__":
    main()

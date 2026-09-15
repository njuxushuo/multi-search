#!/usr/bin/env python3
"""Merge parallel teacher-trace shards without weakening validation.

Each shard is generated from a disjoint slice of the same shuffled candidate
pool.  The merge keeps the first occurrence of each question and rechecks the
strict Search-R1 sequence before writing the canonical JSONL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verl.utils.reward_score.qa_em_format import is_valid_sequence
from trace_validation import validate_tool_progress


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("inputs", nargs="+", help="Canonical/parallel JSONL files")
    args = ap.parse_args()

    seen: set[str] = set()
    kept = rejected = duplicates = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for input_name in args.inputs:
            with open(input_name, encoding="utf-8") as src:
                for line in src:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        rejected += 1
                        continue
                    question = str(row.get("question", "")).strip()
                    if not question or question in seen:
                        duplicates += 1
                        continue
                    assistant = str(row.get("assistant", ""))
                    valid, _ = validate_tool_progress(assistant, is_valid_sequence)
                    if not valid:
                        rejected += 1
                        continue
                    seen.add(question)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
    print(json.dumps({"kept": kept, "rejected": rejected, "duplicates": duplicates, "output": str(out_path)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a small, auditable Search-R1 SFT pilot from prepared QA parquet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/nq_hotpotqa_train/train.parquet")
    parser.add_argument("--output-dir", default="data/processed/search_sft_pilot")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=100)
    args = parser.parse_args()

    table = pq.read_table(args.input, columns=["question", "golden_answers", "data_source"])
    rows = table.to_pylist()
    if len(rows) < args.train_size + args.eval_size:
        raise ValueError(f"only {len(rows)} rows available")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def convert(row: dict, index: int) -> dict:
        answers = row.get("golden_answers") or []
        if isinstance(answers, str):
            answers = [answers]
        answer = str(answers[0]).strip() if answers else ""
        if not answer:
            raise ValueError(f"empty answer at row {index}")
        question = str(row["question"]).strip()
        user = (
            "Answer the given question. Reason inside <think> and </think>, "
            "then provide the final answer inside <answer> and </answer>.\n"
            f"Question: {question}"
        )
        assistant = (
            "<think>Identify the requested fact and verify the target answer "
            "against the question.</think>\n"
            f"<answer>{answer}</answer>"
        )
        return {
            "conversations": [
                {"from": "human", "value": user},
                {"from": "gpt", "value": assistant},
            ],
            "metadata": {
                "source": row.get("data_source", "unknown"),
                "source_row": index,
                "label_type": "oracle_answer_pilot",
            },
        }

    train = [convert(row, i) for i, row in enumerate(rows[: args.train_size])]
    evaluation = [
        convert(row, args.train_size + i)
        for i, row in enumerate(rows[args.train_size : args.train_size + args.eval_size])
    ]
    for name, records in (("train", train), ("eval", evaluation)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    info = {
        "searchqa_sft_train": {
            "file_name": "train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
        },
        "searchqa_sft_eval": {
            "file_name": "eval.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
        },
    }
    (out / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"train": len(train), "eval": len(evaluation), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

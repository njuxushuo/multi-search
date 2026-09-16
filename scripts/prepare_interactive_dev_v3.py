#!/usr/bin/env python3
"""Create the deterministic R3.0 train-holdout interactive dev manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from protocol_v3 import DEFAULT_CONFIG, load_protocol, normalize_answer


def canonical_hash(rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", default=None)
    parser.add_argument("--final-manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quick-output", default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_protocol(args.protocol_config)
    data = config["data"]
    args.input = args.input or data["teacher_source"]
    args.final_manifest = args.final_manifest or data["final_test_manifest"]
    args.output = args.output or data["interactive_dev_manifest"]
    args.quick_output = args.quick_output or data["interactive_dev_quick_manifest"]
    args.count = args.count or int(data["interactive_dev_count"])
    args.seed = args.seed or int(data["interactive_dev_seed"])
    with Path(args.final_manifest).open(encoding="utf-8") as handle:
        final_rows = [json.loads(line) for line in handle if line.strip()]
    excluded_questions = {normalize_answer(row["question"]) for row in final_rows}
    table_rows = pq.read_table(args.input, columns=["question", "golden_answers", "data_source"]).to_pylist()
    pools: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for source_row, row in enumerate(table_rows):
        if normalize_answer(row.get("question", "")) not in excluded_questions:
            pools[str(row.get("data_source", "unknown"))].append((source_row, row))
    rng = random.Random(args.seed)
    for pool in pools.values():
        rng.shuffle(pool)

    targets = {
        "hotpotqa": round(args.count * float(data["interactive_dev_hotpotqa_fraction"])),
        "nq": round(args.count * float(data["interactive_dev_nq_fraction"])),
    }
    delta = args.count - sum(targets.values())
    for source in sorted(targets, key=lambda item: (-targets[item], item)):
        if delta == 0:
            break
        targets[source] += 1 if delta > 0 else -1
        delta += -1 if delta > 0 else 1
    shortages = {
        source: {"needed": target, "available": len(pools[source])}
        for source, target in targets.items() if len(pools[source]) < target
    }
    if shortages:
        raise SystemExit(f"not enough final-test-disjoint rows for stratified dev: {shortages}")

    selected = []
    for source in sorted(targets):
        for source_row, row in pools[source][:targets[source]]:
            answers = row.get("golden_answers") or []
            if hasattr(answers, "tolist"):
                answers = answers.tolist()
            selected.append({
                "source_row": source_row,
                "question": str(row.get("question", "")).strip(),
                "golden_answers": [str(answer).strip() for answer in answers if str(answer).strip()],
                "data_source": source,
            })
    rng.shuffle(selected)
    rows = [{"eval_id": index, **row} for index, row in enumerate(selected)]
    if len(rows) != args.count or any(normalize_answer(row["question"]) in excluded_questions for row in rows):
        raise SystemExit("dev manifest count/leakage assertion failed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    quick_count = int(data["interactive_dev_quick_count"])
    quick_rows = [{**row, "eval_id": index} for index, row in enumerate(rows[:quick_count])]
    quick_output = Path(args.quick_output)
    quick_output.parent.mkdir(parents=True, exist_ok=True)
    with quick_output.open("w", encoding="utf-8") as handle:
        for row in quick_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "count": len(rows),
        "sha256": canonical_hash(rows),
        "seed": args.seed,
        "excluded_final_questions": len(excluded_questions),
        "data_sources": dict(Counter(row["data_source"] for row in rows)),
        "quick_output": str(quick_output),
        "quick_count": len(quick_rows),
        "quick_sha256": canonical_hash(quick_rows),
        "quick_data_sources": dict(Counter(row["data_source"] for row in quick_rows)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

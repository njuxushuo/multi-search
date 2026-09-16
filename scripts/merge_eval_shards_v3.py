#!/usr/bin/env python3
"""Strictly merge complete R3.0 evaluation shards."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_qwen35_search_v3 import aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    prefix = Path(args.prefix)
    output = Path(args.output) if args.output else prefix.with_suffix(".jsonl")
    metadata_rows, results = [], {}
    for shard in range(args.num_shards):
        path = Path(f"{prefix}.shard{shard}.jsonl")
        if not path.exists():
            raise SystemExit(f"missing shard: {path}")
        metadata = None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("type") == "metadata":
                    if metadata is not None:
                        raise SystemExit(f"multiple metadata rows in {path}")
                    metadata = row
                elif row.get("type") == "result":
                    eval_id = int(row["eval_id"])
                    if eval_id in results:
                        raise SystemExit(f"duplicate eval_id across shards: {eval_id}")
                    if eval_id % args.num_shards != shard:
                        raise SystemExit(f"eval_id {eval_id} belongs to another shard")
                    results[eval_id] = row
        if metadata is None:
            raise SystemExit(f"missing metadata in {path}")
        metadata_rows.append(metadata)
    reference = metadata_rows[0]
    for metadata in metadata_rows[1:]:
        for key in ("protocol_id", "human_version", "model", "selection_sha256", "count", "num_shards", "protocol_config_sha256"):
            if metadata.get(key) != reference.get(key):
                raise SystemExit(f"metadata mismatch for {key}")
    expected = set(range(args.count))
    if set(results) != expected:
        raise SystemExit(f"incomplete merge: missing={sorted(expected - set(results))[:10]} extra={sorted(set(results) - expected)[:10]}")
    merged_metadata = {**reference, "type": "metadata", "shard_index": None, "merged_shards": args.num_shards}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(merged_metadata, ensure_ascii=False) + "\n")
        for eval_id in sorted(results):
            handle.write(json.dumps(results[eval_id], ensure_ascii=False) + "\n")
    rows = [results[index] for index in sorted(results)]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["data_source"])].append(row)
    summary = {
        **merged_metadata,
        "overall": aggregate(rows),
        "by_source": {source: aggregate(source_rows) for source, source_rows in sorted(by_source.items())},
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(summary_path), **summary["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

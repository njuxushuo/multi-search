#!/usr/bin/env python3
"""Strictly merge completed evaluation shards and recompute aggregate metrics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evaluate_qwen35_search import aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    manifest_rows = [json.loads(line) for line in Path(args.manifest).open(encoding="utf-8") if line.strip()]
    expected_ids = {int(row["eval_id"]) for row in manifest_rows}
    metadata: list[dict] = []
    results: dict[int, dict] = {}
    for input_name in args.inputs:
        with open(input_name, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("type") == "metadata":
                    metadata.append(row)
                    continue
                if row.get("type") != "result":
                    raise SystemExit(f"unexpected row type in {input_name}: {row.get('type')}")
                eval_id = int(row["eval_id"])
                if eval_id in results:
                    raise SystemExit(f"duplicate eval_id {eval_id}")
                results[eval_id] = row

    if len(metadata) != len(args.inputs):
        raise SystemExit(f"expected {len(args.inputs)} metadata rows, found {len(metadata)}")
    protocol_keys = ("protocol", "model", "manifest", "selection_sha256", "count", "num_shards")
    baseline = metadata[0]
    for item in metadata[1:]:
        for key in protocol_keys:
            if item.get(key) != baseline.get(key):
                raise SystemExit(f"metadata mismatch for {key}")
    shard_indexes = {int(item["shard_index"]) for item in metadata}
    if shard_indexes != set(range(len(args.inputs))):
        raise SystemExit(f"incomplete shard indexes: {sorted(shard_indexes)}")
    actual_ids = set(results)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        extra = sorted(actual_ids - expected_ids)[:10]
        raise SystemExit(
            f"incomplete results: got={len(actual_ids)} expected={len(expected_ids)} "
            f"missing={missing} extra={extra}"
        )

    ordered = [results[eval_id] for eval_id in sorted(results)]
    merged_metadata = {
        **baseline,
        "type": "metadata",
        "shard_index": None,
        "shard_count": len(ordered),
        "merged_shards": len(args.inputs),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(merged_metadata, ensure_ascii=False) + "\n")
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)

    summary = {**merged_metadata, **aggregate(ordered)}
    sources = sorted({row["data_source"] for row in ordered})
    summary["by_data_source"] = {
        source: aggregate([row for row in ordered if row["data_source"] == source])
        for source in sources
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "examples": len(ordered), "shards": len(args.inputs)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Strictly merge complete v1 evaluation shards without touching v0 outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eval_protocol_v1 import PROTOCOL, aggregate_v1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    manifest_rows = [
        json.loads(line) for line in Path(args.manifest).open(encoding="utf-8") if line.strip()
    ]
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
    keys = (
        "protocol", "model", "manifest", "selection_sha256", "count",
        "num_shards", "v0_source_sha256", "repeat_policy", "context_policy",
        "evaluator_sha256", "state_machine_sha256",
    )
    baseline = metadata[0]
    if baseline.get("protocol") != PROTOCOL:
        raise SystemExit(f"not a v1 shard: {baseline.get('protocol')}")
    for item in metadata[1:]:
        for key in keys:
            if item.get(key) != baseline.get(key):
                raise SystemExit(f"metadata mismatch for {key}")
    shard_indexes = {int(item["shard_index"]) for item in metadata}
    if shard_indexes != set(range(len(args.inputs))):
        raise SystemExit(f"incomplete shard indexes: {sorted(shard_indexes)}")
    if set(results) != expected_ids:
        missing = sorted(expected_ids - set(results))[:10]
        extra = sorted(set(results) - expected_ids)[:10]
        raise SystemExit(
            f"incomplete results: got={len(results)} expected={len(expected_ids)} "
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
    if "/v1/" not in str(output.resolve()):
        raise SystemExit("v1 merged output must be stored in an explicit v1 directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(merged_metadata, ensure_ascii=False) + "\n")
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)

    summary = {**merged_metadata, **aggregate_v1(ordered)}
    summary["by_data_source"] = {
        source: aggregate_v1([row for row in ordered if row["data_source"] == source])
        for source in sorted({row["data_source"] for row in ordered})
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "examples": len(ordered), "shards": len(args.inputs)}))


if __name__ == "__main__":
    main()

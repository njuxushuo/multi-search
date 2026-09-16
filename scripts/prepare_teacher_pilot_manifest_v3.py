#!/usr/bin/env python3
"""Build the frozen, leakage-free, stratified R3.0 Teacher pilot manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from protocol_v3 import DEFAULT_CONFIG, file_sha256, load_protocol, normalize_answer


def question_hash(question: str) -> str:
    normalized = normalize_answer(question)
    if not normalized:
        raise ValueError("empty_normalized_question")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def allocate(total: int, counts: dict[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    available = sum(counts.values())
    if available < total:
        raise SystemExit(f"Hotpot pilot pool has only {available} rows; need {total}")
    raw = {key: total * value / available for key, value in counts.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    for key in sorted(raw, key=lambda item: (-(raw[item] - result[item]), item))[: total - sum(result.values())]:
        result[key] += 1
    return result


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", default=None)
    parser.add_argument("--dev-manifest", default=None)
    parser.add_argument("--final-manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_protocol(args.protocol_config)
    data = config["data"]
    sampling = config["teacher_sampling"]
    input_path = args.input or data["teacher_source"]
    dev_path = args.dev_manifest or data["interactive_dev_manifest"]
    final_path = args.final_manifest or data["final_test_manifest"]
    output = Path(args.output or data["teacher_pilot_manifest"])
    total = int(sampling["pilot_unique_questions"])
    source_targets = {
        "hotpotqa": round(total * float(sampling["pilot_hotpotqa_fraction"])),
        "nq": round(total * float(sampling["pilot_nq_fraction"])),
    }
    source_targets["hotpotqa"] += total - sum(source_targets.values())

    dev_rows = read_jsonl(dev_path)
    final_rows = read_jsonl(final_path)
    excluded_source_rows = {int(row["source_row"]) for row in dev_rows}
    excluded_question_hashes = {question_hash(row["question"]) for row in final_rows}
    excluded_question_hashes.update(question_hash(row["question"]) for row in dev_rows)

    table = pq.read_table(input_path, columns=["id", "question", "data_source", "metadata"])
    unique: dict[str, dict[str, Any]] = {}
    exclusion_counts = Counter()
    for source_row, row in enumerate(table.to_pylist()):
        source = str(row.get("data_source", "")).strip().lower()
        if source not in source_targets:
            continue
        qhash = question_hash(str(row.get("question", "")))
        if source_row in excluded_source_rows:
            exclusion_counts["interactive_dev_source_row"] += 1
            continue
        if qhash in excluded_question_hashes:
            exclusion_counts["dev_or_final_question_hash"] += 1
            continue
        if qhash in unique:
            exclusion_counts["duplicate_normalized_question"] += 1
            continue
        metadata = row.get("metadata") or {}
        unique[qhash] = {
            "source_row": source_row,
            "question_id": str(row.get("id") or f"{source}:{source_row}"),
            "question": str(row.get("question", "")).strip(),
            "question_sha256": qhash,
            "data_source": source,
            "hotpot_type": str(metadata.get("type") or "") if source == "hotpotqa" else None,
            "hotpot_level": str(metadata.get("level") or "") if source == "hotpotqa" else None,
        }

    pools: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unique.values():
        key = (row["data_source"], row.get("hotpot_type") or "", row.get("hotpot_level") or "")
        pools[key].append(row)
    for key, rows in pools.items():
        salt = int(hashlib.sha256("/".join(key).encode("utf-8")).hexdigest()[:8], 16)
        random.Random(args.seed + salt).shuffle(rows)

    hotpot_counts = {
        (key[1], key[2]): len(rows)
        for key, rows in pools.items()
        if key[0] == "hotpotqa"
    }
    hotpot_targets = allocate(source_targets["hotpotqa"], hotpot_counts)
    selected: list[dict[str, Any]] = []
    for (hotpot_type, level), count in sorted(hotpot_targets.items()):
        selected.extend(pools[("hotpotqa", hotpot_type, level)][:count])
    nq_pool = pools[("nq", "", "")]
    if len(nq_pool) < source_targets["nq"]:
        raise SystemExit(f"NQ pilot pool has only {len(nq_pool)} rows; need {source_targets['nq']}")
    selected.extend(nq_pool[: source_targets["nq"]])
    random.Random(args.seed).shuffle(selected)
    manifest = [{"pilot_id": index, **row} for index, row in enumerate(selected)]

    hashes = [row["question_sha256"] for row in manifest]
    if len(manifest) != total or len(set(hashes)) != total:
        raise SystemExit("pilot manifest is not exactly the requested number of unique questions")
    if set(hashes) & excluded_question_hashes:
        raise SystemExit("pilot manifest leaked into interactive dev or final test")
    actual_sources = Counter(row["data_source"] for row in manifest)
    if actual_sources != Counter(source_targets):
        raise SystemExit(f"pilot source counts drifted: {dict(actual_sources)} != {source_targets}")

    write_atomic(output, manifest)
    metadata = {
        "protocol_id": config["protocol_id"],
        "human_version": config["human_version"],
        "protocol_config_sha256": file_sha256(args.protocol_config),
        "input": str(Path(input_path).resolve()),
        "input_sha256": file_sha256(input_path),
        "interactive_dev_manifest": str(Path(dev_path).resolve()),
        "interactive_dev_manifest_sha256": file_sha256(dev_path),
        "final_test_manifest": str(Path(final_path).resolve()),
        "final_test_manifest_sha256": file_sha256(final_path),
        "manifest": str(output.resolve()),
        "manifest_sha256": file_sha256(output),
        "seed": args.seed,
        "count": len(manifest),
        "source_counts": dict(actual_sources),
        "hotpot_type_level_counts": {
            f"{key[0]}/{key[1]}": value for key, value in sorted(Counter(
                (row["hotpot_type"], row["hotpot_level"])
                for row in manifest if row["data_source"] == "hotpotqa"
            ).items())
        },
        "excluded": dict(exclusion_counts),
        "unique_normalized_questions": len(set(hashes)),
        "dev_final_overlap": 0,
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()

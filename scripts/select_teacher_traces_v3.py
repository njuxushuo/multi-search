#!/usr/bin/env python3
"""Audit R3.0 Teacher candidates and create a deterministic split manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from protocol_v3 import (
    DEFAULT_CONFIG,
    file_sha256,
    load_protocol,
    validate_strict_candidate_record,
)


BUCKETS = ("one", "two", "three_plus")


def bucket(row: dict[str, Any]) -> str:
    searches = int(row.get("search_action_count", 0))
    return "one" if searches == 1 else "two" if searches == 2 else "three_plus"


def source_name(row: dict[str, Any]) -> str:
    source = str(row.get("data_source", "")).strip().lower()
    aliases = {"hotpot_qa": "hotpotqa", "natural_questions": "nq"}
    return aliases.get(source, source)


def quality_bucket(row: dict[str, Any]) -> str:
    if source_name(row) == "hotpotqa":
        return str(row.get("evidence_grade", "C"))
    return "grounded" if row.get("answer_grounded_in_visible_evidence") else "ungrounded"


def candidate_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    source = source_name(row)
    if source == "hotpotqa":
        grade_rank = {"A": 0, "B": 1, "C": 2}.get(str(row.get("evidence_grade")), 3)
        coverage = float((row.get("support_coverage") or {}).get("recall") or 0.0)
    else:
        grade_rank = 0 if row.get("answer_grounded_in_visible_evidence") else 1
        coverage = 0.0
    prompt_counts = row.get("prompt_token_count_by_round") or [0]
    observation_truncations = sum(
        bool(event.get("truncated"))
        for event in row.get("events", []) if event.get("kind") == "observation"
    )
    return (
        grade_rank,
        -coverage,
        observation_truncations,
        int(row.get("search_action_count", 0)),
        max(int(value) for value in prompt_counts),
        str(row.get("candidate_id")),
    )


def trajectory_tokens(row: dict[str, Any]) -> int:
    return max([int(value) for value in row.get("prompt_token_count_by_round", [])] or [0])


def hotpot_type_level(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata") or {}
    return str(metadata.get("type") or "unknown"), str(metadata.get("level") or "unknown")


def length_thresholds(rows: list[dict[str, Any]]) -> tuple[int, int]:
    values = sorted(trajectory_tokens(row) for row in rows)
    if not values:
        return 0, 0
    return values[round((len(values) - 1) / 3)], values[round(2 * (len(values) - 1) / 3)]


def length_bucket(row: dict[str, Any], thresholds: tuple[int, int]) -> str:
    value = trajectory_tokens(row)
    return "short" if value <= thresholds[0] else "medium" if value <= thresholds[1] else "long"


def allocate(total: int, weights: dict[tuple[str, str], float]) -> dict[tuple[str, str], int]:
    raw = {key: total * value for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(weights, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def split_allocation(cell_counts: dict[tuple[str, str], int], split_total: int) -> dict[tuple[str, str], int]:
    overall = sum(cell_counts.values())
    return allocate(split_total, {key: value / overall for key, value in cell_counts.items()})


def load_candidates(
    paths: list[str], config: dict[str, Any], protocol_config_sha256: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, metadata = [], []
    seen_ids: set[str] = set()
    for path_text in paths:
        path = Path(path_text)
        file_metadata = None
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("type") == "metadata":
                    if file_metadata is not None:
                        raise SystemExit(f"multiple metadata rows in {path}")
                    if row.get("protocol_id") != config["protocol_id"]:
                        raise SystemExit(f"protocol mismatch in {path}: {row.get('protocol_id')}")
                    if row.get("protocol_config_sha256") != protocol_config_sha256:
                        raise SystemExit(f"protocol config checksum mismatch in {path}")
                    file_metadata = row
                    continue
                if row.get("type") != "candidate":
                    raise SystemExit(f"unexpected row type at {path}:{line_no}")
                candidate_id = str(row.get("candidate_id", ""))
                if not candidate_id or candidate_id in seen_ids:
                    raise SystemExit(f"missing/duplicate candidate_id at {path}:{line_no}: {candidate_id}")
                seen_ids.add(candidate_id)
                if row.get("protocol_config_sha256") != protocol_config_sha256:
                    raise SystemExit(f"candidate config checksum mismatch at {path}:{line_no}")
                if row.get("strict_sft_eligible"):
                    try:
                        validate_strict_candidate_record(row, config)
                    except ValueError as exc:
                        raise SystemExit(
                            f"strict candidate integrity failure at {path}:{line_no}: {exc}"
                        ) from exc
                candidates.append(row)
        if file_metadata is None:
            raise SystemExit(f"missing metadata row in {path}")
        metadata.append({"path": str(path), "sha256": file_sha256(path), "metadata": file_metadata})
    return candidates, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--protocol-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default="data/processed/searchqa_repro_v3_0_0/selection_manifest.jsonl")
    parser.add_argument("--audit", default="data/processed/searchqa_repro_v3_0_0/selection_audit.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_protocol(args.protocol_config)
    protocol_sha = file_sha256(args.protocol_config)
    candidates, source_files = load_candidates(args.inputs, config, protocol_sha)

    rejection_counts = Counter(
        reason for row in candidates for reason in row.get("sft_rejection_reasons", [])
    )
    eligible = [row for row in candidates if row.get("strict_sft_eligible")]
    token_thresholds = length_thresholds(eligible)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        question_key = str(row.get("question_sha256") or row.get("question_id") or "")
        if not question_key:
            raise SystemExit(f"eligible candidate missing question_id: {row.get('candidate_id')}")
        by_question[question_key].append(row)

    # Keep the best candidate for each question within each behavior bucket so
    # the later quota selection cannot accidentally choose the same question twice.
    best_by_question_bucket: dict[tuple[str, str], dict[str, Any]] = {}
    for question_key, rows in by_question.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[bucket(row)].append(row)
        for behavior_bucket, bucket_rows in grouped.items():
            best_by_question_bucket[(question_key, behavior_bucket)] = min(bucket_rows, key=candidate_rank)

    selection = config["selection"]
    total = int(selection["train_trajectories"] + selection["teacher_forced_eval_trajectories"] + selection["reserve_trajectories"])
    source_weights = {"hotpotqa": float(selection["hotpotqa_fraction_target"]), "nq": float(selection["nq_fraction_target"])}
    three_fraction = (
        float(selection["three_plus_search_fraction_target_min"])
        + float(selection["three_plus_search_fraction_target_max"])
    ) / 2
    bucket_weights = {
        "one": 1.0 - float(selection["multi_search_fraction_min"]),
        "two": float(selection["multi_search_fraction_min"]) - three_fraction,
        "three_plus": three_fraction,
    }
    hotpot_a = float(selection["hotpot_grade_a_fraction_min"])
    nq_grounded = float(selection["nq_grounded_fraction_min"])
    quality_weights = {
        "hotpotqa": {"A": hotpot_a, "B": 1.0 - hotpot_a},
        "nq": {"grounded": nq_grounded, "ungrounded": 1.0 - nq_grounded},
    }
    cell_weights = {
        (source, behavior_bucket, quality): source_weight * bucket_weights[behavior_bucket] * quality_weight
        for source, source_weight in source_weights.items()
        for behavior_bucket in BUCKETS
        for quality, quality_weight in quality_weights[source].items()
    }
    cell_targets = allocate(total, cell_weights)

    pools: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (question_key, behavior_bucket), row in best_by_question_bucket.items():
        source = source_name(row)
        quality = quality_bucket(row)
        if source in source_weights and quality in quality_weights[source]:
            pools[(source, behavior_bucket, quality)].append(row)
    for key in pools:
        pools[key].sort(key=candidate_rank)

    shortages = {
        "/".join(key): {"needed": target, "available": len(pools[key])}
        for key, target in cell_targets.items()
        if len(pools[key]) < target
    }
    audit = {
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": file_sha256(args.protocol_config),
        "input_files": source_files,
        "total_candidates": len(candidates),
        "strict_eligible_candidates": len(eligible),
        "unique_eligible_questions": len(by_question),
        "rejection_counts_multi_label": dict(rejection_counts),
        "eligible_by_source": dict(Counter(source_name(row) for row in eligible)),
        "eligible_by_search_bucket": dict(Counter(bucket(row) for row in eligible)),
        "eligible_by_evidence_grade": dict(Counter(str(row.get("evidence_grade")) for row in eligible)),
        "cell_targets": {"/".join(key): value for key, value in cell_targets.items()},
        "cell_available": {"/".join(key): len(pools[key]) for key in cell_targets},
        "shortages": shortages,
    }
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if shortages:
        raise SystemExit(f"candidate pool does not satisfy frozen strata; see {audit_path}: {shortages}")

    selected: list[dict[str, Any]] = []
    used_questions: set[str] = set()
    for key in sorted(cell_targets):
        target = cell_targets[key]
        if target <= 0:
            continue
        available = [
            row for row in pools[key]
            if str(row.get("question_sha256") or row["question_id"]) not in used_questions
        ]
        strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in available:
            length = length_bucket(row, token_thresholds)
            stratum = (*hotpot_type_level(row), length) if key[0] == "hotpotqa" else (length,)
            strata[stratum].append(row)
        if len(available) < target:
            raise SystemExit(f"cross-cell dedup left {len(available)} rows for {'/'.join(key)}; need {target}")
        stratum_targets = allocate(
            target, {stratum: len(rows) / len(available) for stratum, rows in strata.items()}
        )
        chosen = []
        for stratum in sorted(stratum_targets):
            chosen.extend(sorted(strata[stratum], key=candidate_rank)[:stratum_targets[stratum]])
        if len(chosen) != target:
            raise SystemExit(f"stratified selection failed for {'/'.join(key)}: {len(chosen)} != {target}")
        for row in chosen:
            selected.append(row)
            used_questions.add(str(row.get("question_sha256") or row["question_id"]))
    if len(selected) != total:
        raise SystemExit(f"cross-cell question dedup left only {len(selected)} selected rows; need {total}")

    selected_cells: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        source = source_name(row)
        type_name, level = hotpot_type_level(row) if source == "hotpotqa" else ("", "")
        selected_cells[(
            source, bucket(row), quality_bucket(row), type_name, level,
            length_bucket(row, token_thresholds),
        )].append(row)
    rng = random.Random(args.seed)
    for rows in selected_cells.values():
        rng.shuffle(rows)

    split_sizes = {
        "train": int(selection["train_trajectories"]),
        "eval": int(selection["teacher_forced_eval_trajectories"]),
        "reserve": int(selection["reserve_trajectories"]),
    }
    remaining_cells = {key: list(rows) for key, rows in selected_cells.items()}
    manifest_rows: list[dict[str, Any]] = []
    for split, split_size in split_sizes.items():
        allocation = split_allocation(
            {key: len(rows) for key, rows in remaining_cells.items()}, split_size
        ) if split != "reserve" else {key: len(rows) for key, rows in remaining_cells.items()}
        for key in sorted(remaining_cells):
            count = allocation.get(key, 0)
            chosen, remaining_cells[key] = remaining_cells[key][:count], remaining_cells[key][count:]
            for row in chosen:
                manifest_rows.append({
                    "type": "selection",
                    "protocol_id": config["protocol_id"],
                    "candidate_id": row["candidate_id"],
                    "question_id": row["question_id"],
                    "question_sha256": row.get("question_sha256"),
                    "split": split,
                    "data_source": source_name(row),
                    "search_bucket": bucket(row),
                    "evidence_grade": row.get("evidence_grade"),
                    "quality_bucket": quality_bucket(row),
                    "hotpot_type": hotpot_type_level(row)[0] if source_name(row) == "hotpotqa" else None,
                    "hotpot_level": hotpot_type_level(row)[1] if source_name(row) == "hotpotqa" else None,
                    "length_bucket": length_bucket(row, token_thresholds),
                    "trajectory_tokens": trajectory_tokens(row),
                })
    actual_split_sizes = dict(Counter(row["split"] for row in manifest_rows))
    if actual_split_sizes != split_sizes:
        raise SystemExit(f"split size mismatch: {actual_split_sizes} != {split_sizes}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "type": "metadata",
        "protocol_id": config["protocol_id"],
        "human_version": config["human_version"],
        "protocol_config_sha256": file_sha256(args.protocol_config),
        "seed": args.seed,
        "split_sizes": split_sizes,
        "cell_targets": {"/".join(key): value for key, value in cell_targets.items()},
        "length_bucket_thresholds": {"short_max": token_thresholds[0], "medium_max": token_thresholds[1]},
        "input_files": source_files,
    }
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for row in sorted(manifest_rows, key=lambda item: (item["split"], str(item["candidate_id"]))):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit["selection_manifest"] = str(output)
    audit["selection_manifest_sha256"] = file_sha256(output)
    audit["selected_by_split"] = dict(Counter(row["split"] for row in manifest_rows))
    audit["selected_by_source"] = dict(Counter(row["data_source"] for row in manifest_rows))
    audit["selected_by_search_bucket"] = dict(Counter(row["search_bucket"] for row in manifest_rows))
    audit["selected_by_quality_bucket"] = dict(Counter(row["quality_bucket"] for row in manifest_rows))
    audit["selected_by_hotpot_type_level"] = dict(Counter(
        f"{row['hotpot_type']}/{row['hotpot_level']}"
        for row in manifest_rows if row["data_source"] == "hotpotqa"
    ))
    audit["selected_by_length_bucket"] = dict(Counter(row["length_bucket"] for row in manifest_rows))
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()

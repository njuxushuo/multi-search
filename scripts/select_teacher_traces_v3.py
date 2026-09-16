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

from protocol_v3 import DEFAULT_CONFIG, file_sha256, load_protocol


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
    return (
        grade_rank,
        -coverage,
        int(row.get("search_action_count", 0)),
        max(int(value) for value in prompt_counts),
        str(row.get("candidate_id")),
    )


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


def load_candidates(paths: list[str], protocol_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, metadata = [], []
    seen_ids: set[str] = set()
    for path_text in paths:
        path = Path(path_text)
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("type") == "metadata":
                    if row.get("protocol_id") != protocol_id:
                        raise SystemExit(f"protocol mismatch in {path}: {row.get('protocol_id')}")
                    metadata.append({"path": str(path), "sha256": file_sha256(path), "metadata": row})
                    continue
                if row.get("type") != "candidate":
                    raise SystemExit(f"unexpected row type at {path}:{line_no}")
                candidate_id = str(row.get("candidate_id", ""))
                if not candidate_id or candidate_id in seen_ids:
                    raise SystemExit(f"missing/duplicate candidate_id at {path}:{line_no}: {candidate_id}")
                seen_ids.add(candidate_id)
                candidates.append(row)
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
    candidates, source_files = load_candidates(args.inputs, config["protocol_id"])

    rejection_counts = Counter(
        reason for row in candidates for reason in row.get("sft_rejection_reasons", [])
    )
    eligible = [row for row in candidates if row.get("strict_sft_eligible")]
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        question_id = str(row.get("question_id", ""))
        if not question_id:
            raise SystemExit(f"eligible candidate missing question_id: {row.get('candidate_id')}")
        by_question[question_id].append(row)

    # Keep the best candidate for each question within each behavior bucket so
    # the later quota selection cannot accidentally choose the same question twice.
    best_by_question_bucket: dict[tuple[str, str], dict[str, Any]] = {}
    for question_id, rows in by_question.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[bucket(row)].append(row)
        for behavior_bucket, bucket_rows in grouped.items():
            best_by_question_bucket[(question_id, behavior_bucket)] = min(bucket_rows, key=candidate_rank)

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
    for (question_id, behavior_bucket), row in best_by_question_bucket.items():
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
        for row in pools[key]:
            question_id = str(row["question_id"])
            if question_id in used_questions:
                continue
            selected.append(row)
            used_questions.add(question_id)
            if sum(
                source_name(item) == key[0]
                and bucket(item) == key[1]
                and quality_bucket(item) == key[2]
                for item in selected
            ) >= target:
                break
    if len(selected) != total:
        raise SystemExit(f"cross-cell question dedup left only {len(selected)} selected rows; need {total}")

    selected_cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_cells[(source_name(row), bucket(row), quality_bucket(row))].append(row)
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
                    "split": split,
                    "data_source": source_name(row),
                    "search_bucket": bucket(row),
                    "evidence_grade": row.get("evidence_grade"),
                    "quality_bucket": quality_bucket(row),
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
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()

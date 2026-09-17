#!/usr/bin/env python3
"""Strictly audit complete R3.0 Teacher candidate shards and report the funnel."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from protocol_v3 import (
    DEFAULT_CONFIG,
    file_sha256,
    load_protocol,
    validate_strict_candidate_record,
)


def quantiles(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    value = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    return {
        "mean": statistics.mean(ordered),
        "p50": value(0.50),
        "p90": value(0.90),
        "p95": value(0.95),
        "p99": value(0.99),
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--protocol-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--question-count", type=int, required=True)
    parser.add_argument("--rollouts-per-question", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--operational-gate", action="store_true")
    parser.add_argument("--quality-gate", action="store_true")
    args = parser.parse_args()

    config = load_protocol(args.protocol_config)
    protocol_sha = file_sha256(args.protocol_config)
    metadata_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for input_text in args.inputs:
        path = Path(input_text)
        metadata = None
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("type") == "metadata":
                    if metadata is not None:
                        raise SystemExit(f"multiple metadata rows in {path}")
                    metadata = row
                    continue
                if row.get("type") != "candidate":
                    raise SystemExit(f"unexpected row type at {path}:{line_no}")
                candidate_id = str(row.get("candidate_id", ""))
                if not candidate_id or candidate_id in candidate_ids:
                    raise SystemExit(f"missing/duplicate candidate_id at {path}:{line_no}")
                if row.get("protocol_config_sha256") != protocol_sha:
                    raise SystemExit(f"candidate protocol checksum mismatch at {path}:{line_no}")
                raw = row.get("raw_generations")
                if not isinstance(raw, list) or len(raw) != int(row.get("assistant_generation_count", -1)):
                    raise SystemExit(f"raw generation audit mismatch at {path}:{line_no}")
                if row.get("strict_sft_eligible"):
                    try:
                        validate_strict_candidate_record(row, config)
                    except ValueError as exc:
                        raise SystemExit(f"strict candidate integrity failure at {path}:{line_no}: {exc}") from exc
                candidate_ids.add(candidate_id)
                candidates.append(row)
        if metadata is None:
            raise SystemExit(f"missing metadata row in {path}")
        metadata_rows.append(metadata)

    if len(metadata_rows) != len(args.inputs):
        raise SystemExit("metadata/input count mismatch")
    reference = metadata_rows[0]
    keys = (
        "protocol_id", "protocol_config_sha256", "teacher", "input_sha256",
        "pilot_manifest_sha256", "question_count_requested", "rollouts_per_question",
        "seed", "num_shards", "retriever_url",
    )
    for metadata in metadata_rows:
        if metadata.get("protocol_id") != config["protocol_id"] or metadata.get("protocol_config_sha256") != protocol_sha:
            raise SystemExit("Teacher shard protocol mismatch")
        for key in keys:
            if metadata.get(key) != reference.get(key):
                raise SystemExit(f"Teacher shard metadata mismatch for {key}")
    shard_indexes = {int(row["shard_index"]) for row in metadata_rows}
    if shard_indexes != set(range(int(reference["num_shards"]))):
        raise SystemExit(f"incomplete Teacher shard indexes: {sorted(shard_indexes)}")

    expected = args.question_count * args.rollouts_per_question
    if len(candidates) != expected:
        raise SystemExit(f"candidate count mismatch: {len(candidates)} != {expected}")
    by_question = Counter(str(row.get("question_sha256", "")) for row in candidates)
    if len(by_question) != args.question_count or set(by_question.values()) != {args.rollouts_per_question}:
        raise SystemExit("candidate question/rollout multiplicity mismatch")

    strict = [row for row in candidates if row.get("strict_sft_eligible")]
    termination = Counter(str(row.get("termination_reason")) for row in candidates)
    rejections = Counter(reason for row in candidates for reason in row.get("sft_rejection_reasons", []))
    search_actions = Counter(int(row.get("search_action_count", 0)) for row in candidates)
    source_counts = Counter(str(row.get("data_source")) for row in candidates)
    prompt_lengths = [
        max([int(value) for value in row.get("prompt_token_count_by_round", [])] or [0])
        for row in candidates
    ]
    report = {
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": protocol_sha,
        "teacher": reference.get("teacher"),
        "pilot_manifest_sha256": reference.get("pilot_manifest_sha256"),
        "input_files": [{"path": str(Path(path).resolve()), "sha256": file_sha256(path)} for path in args.inputs],
        "candidate_count": len(candidates),
        "unique_question_count": len(by_question),
        "strict_eligible_count": len(strict),
        "strict_eligible_rate": len(strict) / len(candidates),
        "exact_match": sum(int(row.get("exact_match", 0)) for row in candidates) / len(candidates),
        "format_compliance_rate": sum(bool(row.get("format_compliant")) for row in candidates) / len(candidates),
        "generation_truncated_rate": termination.get("generation_truncated", 0) / len(candidates),
        "search_limit_rate": termination.get("search_limit", 0) / len(candidates),
        "source_counts": dict(source_counts),
        "search_action_distribution": {str(key): value for key, value in sorted(search_actions.items())},
        "mean_search_actions": sum(int(row.get("search_action_count", 0)) for row in candidates) / len(candidates),
        "mean_bm25_executions": sum(int(row.get("bm25_execution_count", 0)) for row in candidates) / len(candidates),
        "mean_unique_documents": sum(int(row.get("unique_document_count", 0)) for row in candidates) / len(candidates),
        "multi_search_fraction": sum(
            int(row.get("search_action_count", 0)) >= 2 for row in candidates
        ) / len(candidates),
        "termination_reasons": dict(termination),
        "rejection_reasons_multi_label": dict(rejections),
        "correct_after_repeat_or_no_progress": sum(
            bool(row.get("exact_match"))
            and (int(row.get("repeated_query_count", 0)) > 0 or int(row.get("no_progress_search_count", 0)) > 0)
            for row in candidates
        ),
        "hotpot_evidence_grades": dict(Counter(
            str(row.get("evidence_grade")) for row in candidates if str(row.get("data_source")) == "hotpotqa"
        )),
        "nq_grounded": dict(Counter(
            "grounded" if row.get("answer_grounded_in_visible_evidence") else "ungrounded"
            for row in candidates if str(row.get("data_source")) == "nq"
        )),
        "max_prompt_tokens_by_candidate": quantiles(prompt_lengths),
    }
    operational_failures = {
        "retrieval_error": termination.get("retrieval_error", 0),
        "context_overflow": termination.get("context_overflow", 0),
        "missing_strict_candidate": int(not strict),
    }
    report["operational_gate"] = {
        "passed": not any(operational_failures.values()),
        "failures": operational_failures,
    }
    quality = config.get("smoke_quality_gate") or {}
    quality_failures = {}
    if quality:
        minimums = {
            "format_compliance_rate": "format_compliance_rate_min",
            "strict_eligible_rate": "strict_eligible_rate_min",
            "exact_match": "exact_match_min",
            "mean_search_actions": "mean_search_actions_min",
            "multi_search_fraction": "multi_search_fraction_min",
        }
        for metric, threshold_key in minimums.items():
            threshold = float(quality[threshold_key])
            actual = float(report[metric])
            if actual < threshold:
                quality_failures[metric] = {"actual": actual, "required_min": threshold}
        threshold = float(quality["generation_truncated_rate_max"])
        actual = float(report["generation_truncated_rate"])
        if actual > threshold:
            quality_failures["generation_truncated_rate"] = {
                "actual": actual, "required_max": threshold,
            }
        if "search_limit_rate_max" in quality:
            threshold = float(quality["search_limit_rate_max"])
            actual = float(report["search_limit_rate"])
            if actual > threshold:
                quality_failures["search_limit_rate"] = {
                    "actual": actual, "required_max": threshold,
                }
    report["quality_gate"] = {
        "configured": bool(quality),
        "passed": bool(quality) and not quality_failures,
        "failures": quality_failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if args.operational_gate and not report["operational_gate"]["passed"]:
        raise SystemExit(f"{config['human_version']} Teacher operational gate failed")
    if args.quality_gate and not report["quality_gate"]["passed"]:
        raise SystemExit(f"{config['human_version']} Teacher quality gate failed")


if __name__ == "__main__":
    main()

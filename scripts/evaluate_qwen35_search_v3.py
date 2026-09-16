#!/usr/bin/env python3
"""Evaluate one model under the frozen R3.0 protocol with resumable shards."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from eval_protocol_v3 import apply_generation, build_result, force_unfinished_termination, initial_state
from evaluate_qwen35_search import load_or_create_manifest
from protocol_v3 import (
    DEFAULT_CONFIG,
    assert_generation_fits,
    build_model_input,
    canonical_prompt,
    file_sha256,
    load_protocol,
    prompt_from_row,
    token_count,
)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    count = len(rows)
    mean = lambda key: sum(float(row.get(key, 0)) for row in rows) / count
    rate = lambda key: sum(bool(row.get(key)) for row in rows) / count
    correct = [row for row in rows if row.get("exact_match")]
    wrong = [row for row in rows if not row.get("exact_match")]
    subset_mean = lambda subset, key: (
        sum(float(row.get(key, 0)) for row in subset) / len(subset) if subset else None
    )
    support_rows = [
        row for row in rows
        if (row.get("support_coverage") or {}).get("recall") is not None
    ]
    return {
        "examples": count,
        "em": mean("exact_match"),
        "f1": mean("f1"),
        "mean_search_actions": mean("search_action_count"),
        "mean_search_actions_correct": subset_mean(correct, "search_action_count"),
        "mean_search_actions_wrong": subset_mean(wrong, "search_action_count"),
        "mean_bm25_executions": mean("bm25_execution_count"),
        "mean_bm25_executions_correct": subset_mean(correct, "bm25_execution_count"),
        "mean_bm25_executions_wrong": subset_mean(wrong, "bm25_execution_count"),
        "mean_unique_queries": mean("unique_query_count"),
        "mean_unique_documents": mean("unique_document_count"),
        "mean_progressive_searches": mean("progressive_search_count"),
        "mean_no_progress_searches": mean("no_progress_search_count"),
        "answer_rate": rate("answer_present"),
        "format_compliance_rate": rate("format_compliant"),
        "repeated_query_rate": rate("has_repeated_query"),
        "no_progress_rate": rate("has_no_progress"),
        "search_limit_rate": sum(row.get("termination_reason") == "search_limit" for row in rows) / count,
        "context_overflow_rate": sum(row.get("termination_reason") == "context_overflow" for row in rows) / count,
        "generation_truncated_rate": sum(row.get("termination_reason") == "generation_truncated" for row in rows) / count,
        "hotpot_support_recall": (
            sum(float(row["support_coverage"]["recall"]) for row in support_rows) / len(support_rows)
            if support_rows else None
        ),
        "hotpot_full_support_rate": (
            sum(float(row["support_coverage"]["recall"]) == 1.0 for row in support_rows) / len(support_rows)
            if support_rows else None
        ),
        "termination_reasons": dict(Counter(str(row.get("termination_reason")) for row in rows)),
    }


def require_v3_path(path: Path) -> None:
    parts = path.resolve().parts
    if not any(parts[index:index + 3] == ("outputs", "v3", "searchqa_repro_v3_0_0") for index in range(len(parts) - 2)):
        raise SystemExit(f"R3.0 output must be under outputs/v3/searchqa_repro_v3_0_0: {path}")


def enrich_manifest(rows: list[dict[str, Any]], input_path: str, config: dict[str, Any]) -> None:
    table = pq.read_table(input_path)
    source_rows = table.to_pylist()
    for row in rows:
        source_index = int(row["source_row"])
        source = source_rows[source_index]
        if str(source.get("question", "")).strip() != str(row["question"]).strip():
            raise SystemExit(f"manifest/source question mismatch at source_row={source_index}")
        row["prompt"] = source.get("prompt")
        row["metadata"] = source.get("metadata")
        row["question_id"] = source.get("id", f"{row.get('data_source', 'unknown')}:{source_index}")
        try:
            row["initial_prompt"] = prompt_from_row(row, config)
        except ValueError as exc:
            # Some external evaluation sources in the shared parquet may not
            # carry the Search-R1 prompt column.  The frozen text is still
            # deterministic, but the fallback is explicitly recorded.
            if str(exc) not in {"missing_dataset_prompt", "invalid_dataset_prompt"}:
                raise
            row["initial_prompt"] = canonical_prompt(row["question"], config)
            row["prompt_source_fallback"] = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model")
    parser.add_argument("--input", default="data/processed/nq_hotpotqa_train/test.parquet")
    parser.add_argument("--manifest", default="data/processed/search_eval/searchqa_50k_seed42_hnb_first.jsonl")
    parser.add_argument("--source-order", default="")
    parser.add_argument("--output")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--retriever-url", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.model or not args.output:
        parser.error("--model and --output are required")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    output = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    require_v3_path(output)
    require_v3_path(summary_path)

    config = load_protocol(args.protocol_config)
    source_order = [item.strip() for item in args.source_order.split(",") if item.strip()]
    rows, selection_hash = load_or_create_manifest(args.input, args.manifest, args.count, args.seed, source_order)
    enrich_manifest(rows, args.input, config)
    rows = [row for row in rows if int(row["eval_id"]) % args.num_shards == args.shard_index]
    retriever_url = args.retriever_url or config["retrieval"]["url"]

    import requests
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    session = requests.Session()

    def retrieve_hits(query: str) -> list[dict[str, Any]]:
        response = session.post(
            retriever_url,
            json={"queries": [query], "topk": config["retrieval"]["top_k"], "return_scores": True},
            timeout=120,
        )
        response.raise_for_status()
        batches = response.json().get("result", [])
        return list(batches[0]) if batches else []

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=config["token_budget"]["max_model_len"],
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        gdn_prefill_backend="triton",
    )
    sampling = SamplingParams(
        temperature=config["evaluation_sampling"]["temperature"],
        top_p=config["evaluation_sampling"]["top_p"],
        max_tokens=config["token_budget"]["max_new_tokens_per_action"],
        stop=["</search>", "</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )
    metadata = {
        "type": "metadata",
        "protocol_id": config["protocol_id"],
        "human_version": config["human_version"],
        "protocol_config_sha256": file_sha256(args.protocol_config),
        "evaluator_sha256": file_sha256(__file__),
        "state_machine_sha256": file_sha256(Path(__file__).with_name("eval_protocol_v3.py")),
        "model": args.model,
        "input": args.input,
        "manifest": args.manifest,
        "selection_sha256": selection_hash,
        "count": args.count,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "shard_count": len(rows),
        "seed": args.seed,
        "retriever_url": retriever_url,
        "config": config,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[int] = set()
    results: list[dict[str, Any]] = []
    if args.resume and output.exists():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("type") == "metadata":
                    for key in ("protocol_id", "model", "selection_sha256", "shard_index", "num_shards"):
                        if item.get(key) != metadata.get(key):
                            raise SystemExit(f"resume metadata mismatch for {key}")
                elif item.get("type") == "result":
                    eval_id = int(item["eval_id"])
                    if eval_id in completed:
                        raise SystemExit(f"duplicate eval_id in resume output: {eval_id}")
                    completed.add(eval_id)
                    results.append(item)
    pending = [row for row in rows if int(row["eval_id"]) not in completed]
    mode = "a" if args.resume and output.exists() else "w"
    with output.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for start in range(0, len(pending), args.batch_size):
            batch_rows = pending[start:start + args.batch_size]
            states = [
                initial_state(row, row["initial_prompt"], f"eval:{row['eval_id']}")
                for row in batch_rows
            ]
            for _ in range(config["interaction"]["max_assistant_generations"]):
                active = [state for state in states if not state["done"]]
                if not active:
                    break
                prompts, runnable = [], []
                for state in active:
                    try:
                        prompt_tokens = assert_generation_fits(
                            tokenizer, state["initial_prompt"], state["trajectory"], config
                        )
                    except ValueError:
                        state["sequence_overflow"] = True
                        state["invalid_reasons"].append("context_overflow")
                        state["done"], state["termination_reason"] = True, "context_overflow"
                        continue
                    state["prompt_token_count_by_round"].append(prompt_tokens)
                    prompt, _ = build_model_input(tokenizer, state["initial_prompt"], state["trajectory"])
                    prompts.append(prompt)
                    runnable.append(state)
                if not runnable:
                    continue
                generated_batch = llm.generate(prompts, sampling)
                for state, request_output in zip(runnable, generated_batch):
                    item = request_output.outputs[0]
                    apply_generation(
                        state, item.text, retrieve_hits, tokenizer, config,
                        finish_reason=getattr(item, "finish_reason", None),
                    )
            batch_results = []
            for state in states:
                force_unfinished_termination(state)
                candidate = build_result(state, config)
                result = {
                    **candidate,
                    "type": "result",
                    "eval_id": int(state["row"]["eval_id"]),
                    "source_row": int(state["row"]["source_row"]),
                    "answer_present": bool(candidate["prediction"]),
                    "format_compliant": state["format_compliant"],
                    "has_repeated_query": state["repeated_query_count"] > 0,
                    "has_no_progress": state["no_progress_search_count"] > 0,
                    "prompt_source_fallback": bool(state["row"].get("prompt_source_fallback")),
                }
                batch_results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed.add(result["eval_id"])
            results.extend(batch_results)
            handle.flush()
            os.fsync(handle.fileno())
            print(json.dumps({
                "processed": len(completed),
                "shard_total": len(rows),
                "running": aggregate(results),
            }, ensure_ascii=False), flush=True)

    expected = {int(row["eval_id"]) for row in rows}
    if completed != expected:
        raise SystemExit(f"incomplete shard: missing={sorted(expected - completed)[:10]}")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_search_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_source[str(result["data_source"])].append(result)
        searches = int(result.get("search_action_count", 0))
        search_bucket = "zero" if searches == 0 else "one" if searches == 1 else "two" if searches == 2 else "three_plus"
        by_search_bucket[search_bucket].append(result)
    summary = {
        **metadata,
        "overall": aggregate(results),
        "by_source": {source: aggregate(source_rows) for source, source_rows in sorted(by_source.items())},
        "by_search_bucket": {
            bucket: aggregate(bucket_rows) for bucket, bucket_rows in sorted(by_search_bucket.items())
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(summary_path), **summary["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

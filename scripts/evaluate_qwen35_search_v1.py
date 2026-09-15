#!/usr/bin/env python3
"""Evaluate search QA with the isolated v1 interaction protocol.

The frozen v0 evaluator remains in ``evaluate_qwen35_search.py``. This entry
point writes only v1 outputs and can either run a model from scratch or reuse
unaffected v0 rows while continuing only v0 ``repeated_search`` and
``max_rounds`` cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from eval_protocol_v1 import (
    PROTOCOL,
    adapt_v0_reused,
    aggregate_v1,
    apply_generation,
    build_result,
    force_unfinished_termination,
    initial_state,
    is_v0_affected,
    state_from_v0,
)
from evaluate_qwen35_search import load_or_create_manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v0_results(path: str, expected_ids: set[int], selection_hash: str) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    metadata: dict[str, Any] | None = None
    results: dict[int, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "metadata":
                if metadata is not None:
                    raise SystemExit(f"multiple metadata rows in v0 source: {path}")
                metadata = row
            elif row.get("type") == "result":
                eval_id = int(row["eval_id"])
                if eval_id in results:
                    raise SystemExit(f"duplicate eval_id {eval_id} in v0 source")
                results[eval_id] = row
            else:
                raise SystemExit(f"unexpected row type in v0 source: {row.get('type')}")
    if metadata is None:
        raise SystemExit(f"missing metadata in v0 source: {path}")
    if metadata.get("protocol") != "qwen35_search_r1_v4_dp_doc_novelty":
        raise SystemExit(f"unexpected v0 protocol: {metadata.get('protocol')}")
    if metadata.get("selection_sha256") != selection_hash:
        raise SystemExit("v0 source uses a different manifest selection")
    if set(results) != expected_ids:
        raise SystemExit(
            f"v0 source must cover the complete manifest: got={len(results)} expected={len(expected_ids)}"
        )
    return metadata, results


def validate_resume_metadata(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    keys = (
        "protocol", "model", "manifest", "selection_sha256", "count",
        "shard_index", "num_shards", "v0_source_sha256",
    )
    for key in keys:
        if existing.get(key) != expected.get(key):
            raise SystemExit(
                f"resume metadata mismatch for {key}: {existing.get(key)!r} != {expected.get(key)!r}"
            )


def require_v1_output_path(path: Path) -> None:
    parts = path.resolve().parts
    if not any(parts[index:index + 3] == ("outputs", "eval", "v1") for index in range(len(parts) - 2)):
        raise SystemExit(f"v1 output must be under outputs/eval/v1: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Hugging Face model/checkpoint path")
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--input", default="data/processed/nq_hotpotqa_train/test.parquet")
    parser.add_argument("--manifest", default="data/processed/search_eval/searchqa_50k_seed42.jsonl")
    parser.add_argument("--source-order", default="")
    parser.add_argument("--prepare-manifest-only", action="store_true")
    parser.add_argument("--v0-results", default=None, help="Complete merged v0 JSONL for selective continuation")
    parser.add_argument("--index", default="data/index/bm25")
    parser.add_argument("--corpus", default="data/corpus/wiki-18.jsonl")
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--output")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-search-attempts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_search_attempts != 8:
        raise SystemExit("v1 is frozen to exactly 8 search attempts")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    source_order = [item.strip() for item in args.source_order.split(",") if item.strip()]
    rows, selection_hash = load_or_create_manifest(
        args.input, args.manifest, args.count, args.seed, source_order
    )
    if args.prepare_manifest_only:
        print(json.dumps({
            "manifest": args.manifest,
            "rows": len(rows),
            "sha256": selection_hash,
            "data_sources": dict(Counter(row["data_source"] for row in rows)),
        }, ensure_ascii=False))
        return
    if not args.model or not args.output:
        parser.error("--model and --output are required unless --prepare-manifest-only is used")
    require_v1_output_path(Path(args.output))
    if args.summary:
        require_v1_output_path(Path(args.summary))

    expected_ids = {int(row["eval_id"]) for row in rows}
    v0_metadata: dict[str, Any] | None = None
    v0_results: dict[int, dict[str, Any]] | None = None
    v0_hash = None
    if args.v0_results:
        v0_path = Path(args.v0_results)
        v0_metadata, v0_results = load_v0_results(args.v0_results, expected_ids, selection_hash)
        if Path(str(v0_metadata.get("model", ""))).resolve() != Path(args.model).resolve():
            raise SystemExit(
                f"v0 model does not match requested v1 model: "
                f"{v0_metadata.get('model')} != {args.model}"
            )
        v0_hash = file_sha256(v0_path)

    import requests
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    retrieval_session = requests.Session()

    def retrieve_hits(query: str) -> list[dict[str, Any]]:
        response = retrieval_session.post(
            args.retriever_url,
            json={"queries": [query], "topk": args.topk, "return_scores": True},
            timeout=120,
        )
        response.raise_for_status()
        batches = response.json().get("result", [])
        documents = batches[0] if batches else []
        hits = []
        for item in documents:
            document = item.get("document", item)
            content = document.get("contents")
            if not content:
                content = (str(document.get("title", "")) + "\n" + str(document.get("text", ""))).strip()
            if content:
                hits.append({
                    "doc_id": str(document.get("id", "")),
                    "score": float(item["score"]) if item.get("score") is not None else None,
                    "content": str(content)[:1800],
                })
        return hits

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        gdn_prefill_backend="triton",
    )
    if args.lora_adapter:
        llm_kwargs.update(enable_lora=True, max_lora_rank=16)
    llm = LLM(**llm_kwargs)
    lora_request = None
    if args.lora_adapter:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("searchqa-sft-v1", 1, args.lora_adapter)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</search>", "</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    output = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "metadata",
        "protocol": PROTOCOL,
        "model": args.model,
        "lora_adapter": args.lora_adapter,
        "input": args.input,
        "manifest": args.manifest,
        "count": args.count,
        "seed": args.seed,
        "source_order": source_order,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selection_sha256": selection_hash,
        "retriever": "lucene_bm25_wikipedia_2018_http",
        "retriever_url": args.retriever_url,
        "index": args.index,
        "topk": args.topk,
        "max_search_attempts": args.max_search_attempts,
        "max_assistant_generations": args.max_search_attempts + 1,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stop_strings": ["</search>", "</answer>"],
        "context_policy": "question_latest_success_plus_current_recovery",
        "repeat_policy": "neutral_no_progress_without_retrieval",
        "v0_source": args.v0_results,
        "v0_source_sha256": v0_hash,
        "v0_protocol": v0_metadata.get("protocol") if v0_metadata else None,
        "evaluator_sha256": file_sha256(Path(__file__)),
        "state_machine_sha256": file_sha256(Path(__file__).with_name("eval_protocol_v1.py")),
    }

    results: list[dict[str, Any]] = []
    completed: set[int] = set()
    existing_metadata = None
    if args.resume and output.exists():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("type") == "metadata":
                    existing_metadata = item
                elif item.get("type") == "result":
                    eval_id = int(item["eval_id"])
                    if eval_id in completed:
                        raise SystemExit(f"duplicate eval_id in resume output: {eval_id}")
                    completed.add(eval_id)
                    results.append(item)
        if existing_metadata is None:
            raise SystemExit("resume output is missing metadata")
        validate_resume_metadata(existing_metadata, metadata)
        mode = "a"
    else:
        mode = "w"

    assigned_rows = [
        row for row in rows
        if int(row["eval_id"]) % args.num_shards == args.shard_index
    ]
    metadata["shard_count"] = len(assigned_rows)
    row_by_id = {int(row["eval_id"]): row for row in assigned_rows}
    work: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    reused: list[dict[str, Any]] = []
    reconstruction_fallbacks = 0
    for row in assigned_rows:
        eval_id = int(row["eval_id"])
        if eval_id in completed:
            continue
        old = v0_results.get(eval_id) if v0_results is not None else None
        if old is not None and not is_v0_affected(old):
            reused.append(adapt_v0_reused(old))
        else:
            work.append((row, old))

    with output.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for result in reused:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)
            completed.add(int(result["eval_id"]))
        if reused:
            handle.flush()
            os.fsync(handle.fileno())

        for start in range(0, len(work), args.batch_size):
            states = []
            for row, old in work[start:start + args.batch_size]:
                if old is None:
                    states.append(initial_state(row))
                    continue
                try:
                    states.append(state_from_v0(row, old))
                except ValueError as exc:
                    state = initial_state(row, origin="v1_rerun")
                    state["reconstruction_fallback_reason"] = str(exc)
                    state["v0_exact_match"] = int(old.get("exact_match", 0))
                    state["v0_f1"] = float(old.get("f1", 0.0))
                    state["v0_search_count"] = int(old.get("search_count", 0))
                    state["v0_search_attempt_count"] = int(old.get("search_attempt_count", 0))
                    state["v0_termination_reason"] = old.get("termination_reason")
                    states.append(state)
                    reconstruction_fallbacks += 1

            for _ in range(args.max_search_attempts + 1):
                active = [state for state in states if not state["done"]]
                if not active:
                    break
                prompts = [
                    tokenizer.apply_chat_template(state["messages"], tokenize=False, add_generation_prompt=True)
                    for state in active
                ]
                generated_batch = llm.generate(prompts, sampling, lora_request=lora_request)
                for state, request_output in zip(active, generated_batch):
                    apply_generation(
                        state,
                        request_output.outputs[0].text,
                        retrieve_hits,
                        max_search_attempts=args.max_search_attempts,
                    )
            for state in states:
                force_unfinished_termination(state)

            batch_results = [build_result(state) for state in states]
            for state, result in zip(states, batch_results):
                if "reconstruction_fallback_reason" in state:
                    result["reconstruction_fallback_reason"] = state["reconstruction_fallback_reason"]
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)
                completed.add(int(result["eval_id"]))
            handle.flush()
            os.fsync(handle.fileno())
            print(json.dumps({
                "processed": len(completed),
                "total": len(assigned_rows),
                "pending_v1_generations": max(0, len(work) - start - len(batch_results)),
                "reconstruction_fallbacks": reconstruction_fallbacks,
                "running": aggregate_v1(results),
            }, ensure_ascii=False), flush=True)

    if set(completed) != set(row_by_id):
        missing = sorted(set(row_by_id) - completed)[:10]
        raise SystemExit(f"shard incomplete after evaluation: missing={missing}")
    summary = {**metadata, **aggregate_v1(results)}
    summary["reconstruction_fallbacks"] = sum(
        row.get("origin") == "v1_rerun" for row in results
    )
    summary["by_data_source"] = {
        source: aggregate_v1([row for row in results if row["data_source"] == source])
        for source in sorted({row["data_source"] for row in results})
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

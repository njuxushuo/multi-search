#!/usr/bin/env python3
"""Evaluate Base, LoRA, Full-SFT, and Teacher with one strict BM25 protocol.

All selected examples are scored (no rejection sampling). Results are flushed
after every batch and can be resumed safely after interruption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.I | re.S)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
STRICT_ACTION_RE = re.compile(
    r"^\s*<think>.*?</think>\s*(?:<search>\s*(.*?)\s*</search>|<answer>\s*(.*?)\s*</answer>)\s*$",
    re.I | re.S,
)


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred, gold = normalize(prediction).split(), normalize(reference).split()
    if not pred or not gold:
        return float(pred == gold)
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def normalize_segment(text: str) -> str:
    text = re.sub(
        r"`(</?(?:think|search|information|answer)>)`",
        lambda m: m.group(1).replace("<", "[").replace(">", "]"),
        text,
    )
    if "</think>" in text and "<think>" not in text:
        return "<think>" + text
    return text


def mean_ci(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [mean, mean]
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    radius = 1.96 * math.sqrt(variance / len(values))
    return [max(0.0, mean - radius), mean + radius]


def wilson_ci(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 0.0]
    z, p = 1.96, successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def canonical_hash(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_or_create_manifest(
    input_path: str,
    manifest_path: str,
    count: int,
    seed: int,
    source_order: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    manifest = Path(manifest_path)
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
        if len(rows) != count:
            raise SystemExit(f"manifest contains {len(rows)} rows, expected {count}: {manifest}")
        return rows, canonical_hash(rows)
    table = pq.read_table(input_path, columns=["question", "golden_answers", "data_source"])
    source_rows = table.to_pylist()
    if count > len(source_rows):
        raise SystemExit(f"requested {count} rows but test file has {len(source_rows)}")
    indexed = list(enumerate(source_rows))
    random.Random(seed).shuffle(indexed)
    selected = indexed[:count]
    if source_order:
        priority = {source: rank for rank, source in enumerate(source_order)}
        selected.sort(key=lambda item: priority.get(str(item[1].get("data_source", "unknown")), len(priority)))
    rows = [{
        "eval_id": eval_id,
        "source_row": source_row,
        "question": str(row.get("question", "")).strip(),
        "golden_answers": [str(x).strip() for x in (row.get("golden_answers") or []) if str(x).strip()],
        "data_source": str(row.get("data_source", "unknown")),
    } for eval_id, (source_row, row) in enumerate(selected)]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, manifest)
    return rows, canonical_hash(rows)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"examples": 0}
    rate = lambda key: sum(bool(row[key]) for row in rows) / n
    em = [int(row["exact_match"]) for row in rows]
    f1 = [float(row["f1"]) for row in rows]
    searches = [float(row["search_count"]) for row in rows]
    attempts = [float(row["search_attempt_count"]) for row in rows]
    progressive = [float(row["progressive_search_count"]) for row in rows]
    redundant = [float(row["redundant_search_count"]) for row in rows]
    retrieved = [float(row["retrieved_document_count"]) for row in rows]
    unique_docs = [float(row["unique_document_count"]) for row in rows]
    novelty = [float(row["mean_novelty_ratio"]) for row in rows]
    rounds = [float(row["round_count"]) for row in rows]
    return {
        "examples": n, "em": sum(em) / n, "f1": sum(f1) / n,
        "mean_searches": sum(searches) / n,
        "mean_search_attempts": sum(attempts) / n,
        "mean_progressive_searches": sum(progressive) / n,
        "mean_redundant_searches": sum(redundant) / n,
        "mean_retrieved_documents": sum(retrieved) / n,
        "mean_unique_documents": sum(unique_docs) / n,
        "mean_novelty_ratio": sum(novelty) / n,
        "mean_rounds": sum(rounds) / n,
        "zero_search_rate": rate("zero_search"), "answer_rate": rate("answer_present"),
        "format_compliance_rate": rate("format_compliant"),
        "valid_trajectory_rate": rate("valid_trajectory"),
        "invalid_tool_call_rate": rate("invalid_tool_call"),
        "repeated_query_rate": rate("has_repeated_query"),
        "no_progress_call_rate": rate("has_no_progress_call"),
        "strict_success_rate": rate("strict_success"),
        "recovered_success_rate": rate("recovered_success"),
        "ci95": {"em": wilson_ci(sum(em), n), "f1": mean_ci(f1), "mean_searches": mean_ci(searches)},
        "termination_reasons": dict(Counter(row["termination_reason"] for row in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Base/Teacher model or full Hugging Face checkpoint")
    ap.add_argument("--lora-adapter", default=None)
    ap.add_argument("--input", default="data/processed/nq_hotpotqa_train/test.parquet")
    ap.add_argument("--manifest", default="data/processed/search_eval/searchqa_50k_seed42.jsonl")
    ap.add_argument("--source-order", default="", help="Comma-separated source priority used only when creating a manifest")
    ap.add_argument("--prepare-manifest-only", action="store_true")
    ap.add_argument("--index", default="data/index/bm25")
    ap.add_argument("--corpus", default="data/corpus/wiki-18.jsonl")
    ap.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--output")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--count", type=int, default=50000)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--max-searches", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    source_order = [item.strip() for item in args.source_order.split(",") if item.strip()]
    rows, selection_hash = load_or_create_manifest(
        args.input, args.manifest, args.count, args.seed, source_order
    )
    if args.prepare_manifest_only:
        print(json.dumps({"manifest": args.manifest, "rows": len(rows), "sha256": selection_hash,
                          "data_sources": dict(Counter(row["data_source"] for row in rows))}, ensure_ascii=False))
        return
    if not args.model or not args.output:
        ap.error("--model and --output are required unless --prepare-manifest-only is used")

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
    llm_kwargs = dict(model=args.model, trust_remote_code=True, dtype="bfloat16",
                      tensor_parallel_size=args.tensor_parallel_size, max_model_len=4096,
                      gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
                      gdn_prefill_backend="triton")
    if args.lora_adapter:
        llm_kwargs.update(enable_lora=True, max_lora_rank=16)
    llm = LLM(**llm_kwargs)
    lora_request = None
    if args.lora_adapter:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("searchqa-sft", 1, args.lora_adapter)
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
        "type": "metadata", "protocol": "qwen35_search_r1_v4_dp_doc_novelty",
        "model": args.model, "lora_adapter": args.lora_adapter, "input": args.input,
        "manifest": args.manifest, "count": args.count, "seed": args.seed,
        "source_order": source_order,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "selection_sha256": selection_hash, "retriever": "lucene_bm25_wikipedia_2018_http",
        "retriever_url": args.retriever_url,
        "index": args.index, "topk": args.topk, "max_rounds": args.max_rounds,
        "max_searches": args.max_searches, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "top_p": args.top_p,
        "stop_strings": ["</search>", "</answer>"],
    }
    results: list[dict[str, Any]] = []
    completed: set[int] = set()
    if args.resume and output.exists():
        for line in output.open(encoding="utf-8"):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("type") == "result":
                results.append(item)
                completed.add(int(item["eval_id"]))
        mode = "a"
    else:
        mode = "w"

    assigned_rows = [
        row for row in rows
        if int(row["eval_id"]) % args.num_shards == args.shard_index
    ]
    metadata["shard_count"] = len(assigned_rows)
    pending = [row for row in assigned_rows if int(row["eval_id"]) not in completed]
    with output.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            handle.flush()
        for start in range(0, len(pending), args.batch_size):
            states = []
            for row in pending[start:start + args.batch_size]:
                states.append({
                    "row": row,
                    "messages": [{"role": "user", "content": (
                        "Answer the question using the search tool when needed. Reason inside <think>...</think>. "
                        "To search, emit exactly <search>query</search>; after each search you will receive "
                        "<information>...</information>. Finish with exactly <answer>final answer</answer>.\nQuestion: "
                        + row["question"])}],
                    "turns": [], "queries": [], "searched": 0, "attempted": 0,
                    "progressive": 0, "redundant": 0, "seen_doc_keys": set(),
                    "retrieval_rounds": [],
                    "invalid_reasons": [], "repeated": 0, "no_progress": 0,
                    "prediction": "", "done": False, "termination": "max_rounds",
                    "format_compliant": True,
                })
            for _ in range(args.max_rounds):
                active = [state for state in states if not state["done"]]
                if not active:
                    break
                for state in active:
                    if len(state["messages"]) > 3:
                        state["messages"] = [state["messages"][0]] + state["messages"][-2:]
                prompts = [tokenizer.apply_chat_template(state["messages"], tokenize=False, add_generation_prompt=True)
                           for state in active]
                generated_batch = llm.generate(prompts, sampling, lora_request=lora_request)
                for state, request_output in zip(active, generated_batch):
                    generated = normalize_segment(request_output.outputs[0].text.strip())
                    state["turns"].append(generated)
                    strict = STRICT_ACTION_RE.fullmatch(generated)
                    if not strict:
                        answers = ANSWER_RE.findall(generated)
                        if answers:
                            state["prediction"] = answers[-1].strip()
                        state["format_compliant"] = False
                        state["invalid_reasons"].append("malformed_or_ambiguous_action")
                        state["done"], state["termination"] = True, "format_error"
                        continue
                    query, answer = strict.group(1), strict.group(2)
                    if answer is not None:
                        state["prediction"] = answer.strip()
                        state["done"], state["termination"] = True, "answer"
                        continue
                    state["attempted"] += 1
                    query, normalized_query = (query or "").strip(), normalize(query or "")
                    if not normalized_query:
                        state["invalid_reasons"].append("empty_search_query")
                        state["no_progress"] += 1
                        state["done"], state["termination"] = True, "invalid_search"
                        continue
                    if normalized_query in {normalize(item) for item in state["queries"]}:
                        state["invalid_reasons"].append("repeated_search_query")
                        state["repeated"] += 1
                        state["no_progress"] += 1
                        state["done"], state["termination"] = True, "repeated_search"
                        continue
                    if state["searched"] >= args.max_searches:
                        state["invalid_reasons"].append("search_limit_exceeded")
                        state["no_progress"] += 1
                        state["done"], state["termination"] = True, "search_limit"
                        continue
                    state["queries"].append(query)
                    try:
                        hits = retrieve_hits(query)
                    except Exception as exc:
                        state["invalid_reasons"].append(f"retrieval_error:{type(exc).__name__}")
                        state["no_progress"] += 1
                        state["done"], state["termination"] = True, "retrieval_error"
                        continue
                    if not hits:
                        state["invalid_reasons"].append("empty_search_result")
                        state["no_progress"] += 1
                        state["done"], state["termination"] = True, "empty_search_result"
                        continue
                    state["searched"] += 1
                    doc_keys = [
                        ("id:" + hit["doc_id"]) if hit["doc_id"] else
                        ("content:" + hashlib.sha256(hit["content"].encode()).hexdigest())
                        for hit in hits
                    ]
                    new_positions = [i for i, key in enumerate(doc_keys) if key not in state["seen_doc_keys"]]
                    new_doc_ids = [hits[i]["doc_id"] for i in new_positions]
                    novelty_ratio = len(new_positions) / len(hits)
                    if new_positions:
                        state["progressive"] += 1
                    else:
                        state["redundant"] += 1
                        state["no_progress"] += 1
                        state["invalid_reasons"].append("no_new_documents")
                    state["seen_doc_keys"].update(doc_keys)
                    state["retrieval_rounds"].append({
                        "query": query,
                        "doc_ids": [hit["doc_id"] for hit in hits],
                        "scores": [hit["score"] for hit in hits],
                        "new_doc_ids": new_doc_ids,
                        "new_doc_count": len(new_positions),
                        "novelty_ratio": novelty_ratio,
                        "no_progress": not new_positions,
                    })
                    state["messages"].append({"role": "assistant", "content": generated})
                    information = "<information>\n" + "\n\n".join(hit["content"] for hit in hits) + "\n</information>"
                    instruction = (" This is the final allowed search; do not search again. Output the required answer tag now."
                                   if state["searched"] >= args.max_searches else
                                   " If the evidence answers the question, stop searching and output the required answer tag.")
                    state["messages"].append({"role": "user", "content": information + "\nUse the evidence above." + instruction})
                    state["turns"].append(information)

            batch_results = []
            for state in states:
                if not state["done"]:
                    state["invalid_reasons"].append("max_rounds_without_answer")
                row, prediction = state["row"], state["prediction"]
                answers = row["golden_answers"]
                answer_present = bool(prediction)
                invalid_tool = bool(state["invalid_reasons"])
                exact_match = int(any(normalize(prediction) == normalize(answer) for answer in answers))
                valid_trajectory = answer_present and state["format_compliant"] and not invalid_tool
                result = {
                    "type": "result", "eval_id": row["eval_id"], "source_row": row["source_row"],
                    "question": row["question"], "data_source": row["data_source"],
                    "golden_answers": answers, "prediction": prediction,
                    "exact_match": exact_match,
                    "f1": max((token_f1(prediction, answer) for answer in answers), default=0.0),
                    "search_count": state["searched"], "search_attempt_count": state["attempted"],
                    "progressive_search_count": state["progressive"],
                    "redundant_search_count": state["redundant"],
                    "retrieved_document_count": sum(len(item["doc_ids"]) for item in state["retrieval_rounds"]),
                    "unique_document_count": len(state["seen_doc_keys"]),
                    "mean_novelty_ratio": (
                        sum(item["novelty_ratio"] for item in state["retrieval_rounds"]) /
                        len(state["retrieval_rounds"])
                    ) if state["retrieval_rounds"] else 0.0,
                    "round_count": sum(not turn.startswith("<information>") for turn in state["turns"]),
                    "zero_search": state["searched"] == 0, "answer_present": answer_present,
                    "format_compliant": state["format_compliant"],
                    "valid_trajectory": valid_trajectory,
                    "strict_success": bool(exact_match and valid_trajectory),
                    "recovered_success": bool(exact_match and answer_present and state["no_progress"] > 0),
                    "invalid_tool_call": invalid_tool, "has_repeated_query": state["repeated"] > 0,
                    "has_no_progress_call": state["no_progress"] > 0,
                    "invalid_reasons": state["invalid_reasons"], "termination_reason": state["termination"],
                    "queries": state["queries"], "assistant": "\n".join(state["turns"]),
                    "retrieval_rounds": state["retrieval_rounds"],
                }
                results.append(result)
                batch_results.append(result)
            for result in batch_results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(json.dumps({"processed": len(completed) + min(start + len(batch_results), len(pending)),
                              "total": len(assigned_rows), "running": aggregate(results)}, ensure_ascii=False), flush=True)

    summary = {**metadata, **aggregate(results)}
    summary["by_data_source"] = {source: aggregate([row for row in results if row["data_source"] == source])
                                 for source in sorted({row["data_source"] for row in results})}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

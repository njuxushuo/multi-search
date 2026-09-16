#!/usr/bin/env python3
"""Generate fully auditable R3.0 Teacher candidates.

Unlike the historical generator, this writes every rollout, including failed
ones.  It uses the frozen dataset prompt, a cumulative flat trajectory,
token-budgeted top-3 observations, four search actions, and one answer-only
generation after the search budget.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from eval_protocol_v3 import (
    aggregate_results,
    apply_generation,
    build_result,
    force_unfinished_termination,
    initial_state,
)
from protocol_v3 import (
    DEFAULT_CONFIG,
    assert_generation_fits,
    build_model_input,
    file_sha256,
    load_protocol,
    prompt_from_row,
    token_count,
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--input", default="data/processed/nq_hotpotqa_train/train.parquet")
    parser.add_argument("--exclude-manifest", default=None)
    parser.add_argument("--output", default="data/processed/searchqa_repro_v3_0_0/teacher_candidates.jsonl")
    parser.add_argument("--question-count", type=int, default=None)
    parser.add_argument("--rollouts-per-question", type=int, default=None)
    parser.add_argument("--retriever-url", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_protocol(args.protocol_config)
    teacher = args.teacher or config["models"]["teacher"]
    question_count = args.question_count or int(config["teacher_sampling"]["pilot_unique_questions"])
    rollouts = args.rollouts_per_question or int(config["teacher_sampling"]["rollouts_per_question"])
    retriever_url = args.retriever_url or config["retrieval"]["url"]
    exclude_manifest = args.exclude_manifest or config["data"]["interactive_dev_manifest"]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    table = pq.read_table(args.input)
    rows = table.to_pylist()
    for source_row, row in enumerate(rows):
        row["source_row"] = source_row
        row.setdefault("question_id", row.get("id", f"{row.get('data_source', 'unknown')}:{source_row}"))
    excluded_source_rows: set[int] = set()
    if Path(exclude_manifest).exists():
        with Path(exclude_manifest).open(encoding="utf-8") as handle:
            excluded_source_rows = {int(json.loads(line)["source_row"]) for line in handle if line.strip()}
        rows = [row for row in rows if int(row["source_row"]) not in excluded_source_rows]
    else:
        raise SystemExit(f"R3.0 interactive dev manifest must be frozen before Teacher sampling: {exclude_manifest}")
    random.Random(args.seed).shuffle(rows)
    rows = rows[:question_count]
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]

    import requests
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(teacher, trust_remote_code=True)
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
        model=teacher,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=config["token_budget"]["max_model_len"],
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        gdn_prefill_backend="triton",
    )

    states = []
    for row in rows:
        row = json_safe(row)
        initial_prompt = prompt_from_row(row, config)
        initial_tokens = token_count(
            tokenizer,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": initial_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            ),
        )
        if initial_tokens > config["token_budget"]["initial_prompt_tokens"]:
            raise SystemExit(
                f"initial prompt exceeds R3.0 budget: question_id={row['question_id']} tokens={initial_tokens}"
            )
        for rollout_index in range(rollouts):
            candidate_id = f"{row['question_id']}::seed{args.seed + rollout_index}::r{rollout_index}"
            state = initial_state(row, initial_prompt, candidate_id)
            state["rollout_index"] = rollout_index
            state["rollout_seed"] = args.seed + rollout_index
            states.append(state)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume and output.exists():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("type") == "candidate":
                    completed.add(str(item["candidate_id"]))
        states = [state for state in states if state["candidate_id"] not in completed]

    metadata = {
        "type": "metadata",
        "protocol_id": config["protocol_id"],
        "human_version": config["human_version"],
        "protocol_config": str(Path(args.protocol_config).resolve()),
        "protocol_config_sha256": file_sha256(args.protocol_config),
        "generator_sha256": file_sha256(__file__),
        "state_machine_sha256": file_sha256(Path(__file__).with_name("eval_protocol_v3.py")),
        "teacher": teacher,
        "input": args.input,
        "exclude_manifest": exclude_manifest,
        "exclude_manifest_sha256": file_sha256(exclude_manifest),
        "excluded_dev_rows": len(excluded_source_rows),
        "question_count_requested": question_count,
        "question_count_this_shard": len(rows),
        "rollouts_per_question": rollouts,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "retriever_url": retriever_url,
        "config": config,
    }
    mode = "a" if args.resume and output.exists() else "w"
    written_results: list[dict[str, Any]] = []
    with output.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for start in range(0, len(states), args.batch_size):
            batch = states[start:start + args.batch_size]
            max_generations = int(config["interaction"]["max_assistant_generations"])
            for _ in range(max_generations):
                active = [state for state in batch if not state["done"]]
                if not active:
                    break
                prompts, sampling_params, runnable = [], [], []
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
                    sampling_params.append(SamplingParams(
                        temperature=config["teacher_sampling"]["temperature"],
                        top_p=config["teacher_sampling"]["top_p"],
                        max_tokens=config["token_budget"]["max_new_tokens_per_action"],
                        stop=["</search>", "</answer>"],
                        include_stop_str_in_output=True,
                        seed=int(state["rollout_seed"]),
                    ))
                    runnable.append(state)
                if not runnable:
                    continue
                generated_batch = llm.generate(prompts, sampling_params)
                for state, request_output in zip(runnable, generated_batch):
                    output_item = request_output.outputs[0]
                    apply_generation(
                        state,
                        output_item.text,
                        retrieve_hits,
                        tokenizer,
                        config,
                        finish_reason=getattr(output_item, "finish_reason", None),
                    )
            for state in batch:
                force_unfinished_termination(state)
                result = build_result(state, config)
                result["rollout_index"] = state["rollout_index"]
                result["rollout_seed"] = state["rollout_seed"]
                handle.write(json.dumps(json_safe(result), ensure_ascii=False) + "\n")
                written_results.append(result)
            handle.flush()
            os.fsync(handle.fileno())
            print(json.dumps({
                "written_this_run": len(written_results),
                "remaining": max(0, len(states) - start - len(batch)),
                "running": aggregate_results(written_results),
            }, ensure_ascii=False), flush=True)

    print(json.dumps({
        "output": str(output),
        "new_candidates": len(written_results),
        "resume_skipped": len(completed),
        "summary": aggregate_results(written_results),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

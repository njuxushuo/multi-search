#!/usr/bin/env python3
"""Pure interaction state machine for R3.0 Teacher sampling and evaluation."""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from protocol_v3 import (
    append_generated_event,
    append_observation_event,
    document_key,
    format_observation,
    normalize_answer,
    normalize_query,
    normalized_exact_match,
    parse_action,
    support_coverage,
    support_sentence_visibility,
    support_titles,
    token_f1,
)


Retriever = Callable[[str], list[dict[str, Any]]]


def initial_state(row: dict[str, Any], initial_prompt: str, candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "row": row,
        "initial_prompt": initial_prompt,
        "trajectory": "",
        "events": [],
        "raw_generations": [],
        "queries": [],
        "query_cache": {},
        "seen_doc_keys": set(),
        "visible_documents": [],
        "retrieval_rounds": [],
        "search_action_count": 0,
        "bm25_execution_count": 0,
        "progressive_search_count": 0,
        "no_progress_search_count": 0,
        "repeated_query_count": 0,
        "assistant_generation_count": 0,
        "prompt_token_count_by_round": [],
        "invalid_reasons": [],
        "termination_reason": "in_progress",
        "prediction": "",
        "done": False,
        "final_only": False,
        "format_compliant": True,
        "generation_truncated": False,
        "sequence_overflow": False,
    }


def _reason(state: dict[str, Any], reason: str) -> None:
    if reason not in state["invalid_reasons"]:
        state["invalid_reasons"].append(reason)


def _append_event(state: dict[str, Any], kind: str, text: str, **metadata: Any) -> None:
    event = {"kind": kind, "text": text, **metadata}
    state["events"].append(event)
    if kind == "generated":
        state["trajectory"] = append_generated_event(state["trajectory"], text)
    else:
        state["trajectory"] = append_observation_event(state["trajectory"], text)


def _set_answer(state: dict[str, Any], action: Any, raw_text: str) -> None:
    _append_event(
        state, "generated", action.canonical, action="answer",
        raw_text=raw_text, canonical_text=action.canonical,
    )
    state["prediction"] = action.content
    state["done"] = True
    state["termination_reason"] = "answer_after_search_budget" if state["final_only"] else "answer"


def apply_generation(
    state: dict[str, Any],
    generated: str,
    retrieve_hits: Retriever,
    tokenizer: Any,
    config: dict[str, Any],
    finish_reason: str | None = None,
) -> None:
    """Apply one model generation without using gold labels for control flow."""
    if state["done"]:
        raise ValueError("cannot apply generation to completed state")
    state["assistant_generation_count"] += 1
    state["raw_generations"].append({
        "generation_index": state["assistant_generation_count"] - 1,
        "text": str(generated),
        "finish_reason": finish_reason,
    })
    if finish_reason == "length":
        state["generation_truncated"] = True
        state["format_compliant"] = False
        _reason(state, "generation_truncated")
        state["done"], state["termination_reason"] = True, "generation_truncated"
        return
    try:
        action = parse_action(
            generated,
            restore_qwen_think=config["interaction"]["restore_implicit_think_opener_for_qwen"],
        )
    except ValueError as exc:
        state["format_compliant"] = False
        _reason(state, str(exc))
        state["done"], state["termination_reason"] = True, "format_error"
        return

    if action.kind == "answer":
        _set_answer(state, action, str(generated))
        return
    if state["final_only"]:
        _reason(state, "search_after_search_budget")
        state["done"], state["termination_reason"] = True, "search_limit"
        return

    max_searches = int(config["interaction"]["max_search_actions"])
    state["search_action_count"] += 1
    query = action.content
    query_key = normalize_query(query)
    if not query_key:
        _reason(state, "empty_search")
        state["done"], state["termination_reason"] = True, "empty_search"
        return
    _append_event(
        state, "generated", action.canonical, action="search", query=query,
        raw_text=str(generated), canonical_text=action.canonical,
    )
    state["queries"].append(query)

    repeated = query_key in state["query_cache"]
    if repeated:
        cached = state["query_cache"][query_key]
        state["repeated_query_count"] += 1
        state["no_progress_search_count"] += 1
        _reason(state, "repeated_search_query")
        _append_event(
            state,
            "observation",
            cached["text"],
            token_count=cached["token_count"],
            truncated=cached["truncated"],
            repeated=True,
            documents=cached["documents"],
        )
        state["retrieval_rounds"].append({
            "query": query,
            "normalized_query": query_key,
            "bm25_executed": False,
            "repeated": True,
            "no_progress": True,
            "new_doc_count": 0,
            "documents": cached["documents"],
        })
    else:
        try:
            hits = retrieve_hits(query)
            state["bm25_execution_count"] += 1
        except Exception as exc:  # noqa: BLE001 - persisted error class is intentional
            _reason(state, f"retrieval_error:{type(exc).__name__}")
            state["done"], state["termination_reason"] = True, "retrieval_error"
            return
        if not hits:
            _reason(state, "empty_search_result")
            state["done"], state["termination_reason"] = True, "empty_search_result"
            return
        try:
            observation = format_observation(
                tokenizer,
                hits,
                max_tokens=int(config["token_budget"]["max_observation_tokens"]),
                top_k=int(config["retrieval"]["top_k"]),
            )
        except ValueError as exc:
            _reason(state, str(exc))
            state["done"], state["termination_reason"] = True, "observation_error"
            return
        documents = [dict(document) for document in observation.documents]
        keys = [document_key(document) for document in documents]
        new_keys = [key for key in keys if key not in state["seen_doc_keys"]]
        no_progress = not new_keys
        if no_progress:
            state["no_progress_search_count"] += 1
            _reason(state, "no_new_documents")
        else:
            state["progressive_search_count"] += 1
        state["seen_doc_keys"].update(keys)
        state["visible_documents"].extend(documents)
        cached = {
            "text": observation.text,
            "token_count": observation.token_count,
            "truncated": observation.truncated,
            "documents": documents,
        }
        state["query_cache"][query_key] = cached
        _append_event(
            state,
            "observation",
            observation.text,
            token_count=observation.token_count,
            truncated=observation.truncated,
            repeated=False,
            documents=documents,
        )
        state["retrieval_rounds"].append({
            "query": query,
            "normalized_query": query_key,
            "bm25_executed": True,
            "repeated": False,
            "no_progress": no_progress,
            "new_doc_count": len(new_keys),
            "documents": documents,
        })

    if state["search_action_count"] >= max_searches:
        state["final_only"] = True


def force_unfinished_termination(state: dict[str, Any]) -> None:
    if state["done"]:
        return
    _reason(state, "assistant_generation_budget_exhausted")
    state["done"], state["termination_reason"] = True, "generation_budget"


def strict_sft_rejection_reasons(state: dict[str, Any]) -> list[str]:
    row = state["row"]
    answers = row.get("golden_answers") or []
    if hasattr(answers, "tolist"):
        answers = answers.tolist()
    reasons: list[str] = []
    if not state["prediction"]:
        reasons.append("missing_answer")
    elif not normalized_exact_match(state["prediction"], answers):
        reasons.append("wrong_answer")
    if not state["format_compliant"]:
        reasons.append("format_error")
    if state["search_action_count"] < 1:
        reasons.append("closed_book_answer")
    if state["repeated_query_count"]:
        reasons.append("repeated_search_query")
    if state["no_progress_search_count"]:
        reasons.append("no_progress_search")
    if state["generation_truncated"]:
        reasons.append("generation_truncated")
    if state["sequence_overflow"]:
        reasons.append("sequence_overflow")
    terminal_failures = {
        "format_error", "empty_search", "empty_search_result", "retrieval_error",
        "observation_error", "search_limit", "generation_budget", "generation_truncated",
        "context_overflow",
    }
    if state["termination_reason"] in terminal_failures:
        reasons.append(f"terminal:{state['termination_reason']}")
    return sorted(set(reasons))


def build_result(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    row = state["row"]
    answers = row.get("golden_answers") or []
    if hasattr(answers, "tolist"):
        answers = answers.tolist()
    gold_support = support_titles(row.get("metadata"))
    coverage = support_coverage(gold_support, state["visible_documents"])
    sentence_visibility = support_sentence_visibility(row.get("metadata"), state["visible_documents"])
    answer_key = normalize_answer(state["prediction"])
    evidence_key = normalize_answer(" ".join(
        str(document.get("visible_text", "")) for document in state["visible_documents"]
    ))
    answer_grounded = bool(answer_key and answer_key in evidence_key)
    rejection = strict_sft_rejection_reasons(state)
    source = str(row.get("data_source", "unknown"))
    if source.lower() == "hotpotqa":
        full_title_coverage = (
            coverage["gold_title_count"]
            and coverage["matched_title_count"] == coverage["gold_title_count"]
        )
        full_sentence_visibility = (
            sentence_visibility["recall"] in (None, 1.0)
        )
        if full_title_coverage and full_sentence_visibility:
            evidence_grade = "A"
        elif coverage["matched_title_count"]:
            evidence_grade = "B"
        else:
            evidence_grade = "C"
    else:
        evidence_grade = "grounded" if answer_grounded else "ungrounded"
    return {
        "type": "candidate",
        "protocol_id": config["protocol_id"],
        "human_version": config["human_version"],
        "candidate_id": state["candidate_id"],
        "question_id": row.get("question_id", row.get("id", row.get("source_row"))),
        "question": row.get("question"),
        "golden_answers": list(answers),
        "data_source": source,
        "metadata": row.get("metadata"),
        "initial_prompt": state["initial_prompt"],
        "raw_trajectory": "\n\n".join(item["text"] for item in state["raw_generations"]),
        "raw_generations": state["raw_generations"],
        "canonical_trajectory": state["trajectory"],
        "events": state["events"],
        "prediction": state["prediction"],
        "answer_present": bool(state["prediction"]),
        "format_compliant": bool(state["format_compliant"]),
        "generation_truncated": bool(state["generation_truncated"]),
        "sequence_overflow": bool(state["sequence_overflow"]),
        "has_repeated_query": state["repeated_query_count"] > 0,
        "has_no_progress": state["no_progress_search_count"] > 0,
        "exact_match": int(normalized_exact_match(state["prediction"], answers)),
        "f1": max((token_f1(state["prediction"], answer) for answer in answers), default=0.0),
        "answer_grounded_in_visible_evidence": answer_grounded,
        "support_coverage": coverage,
        "support_sentence_visibility": sentence_visibility,
        "evidence_grade": evidence_grade,
        "search_action_count": state["search_action_count"],
        "bm25_execution_count": state["bm25_execution_count"],
        "unique_query_count": len(set(normalize_query(query) for query in state["queries"])),
        "unique_document_count": len(state["seen_doc_keys"]),
        "retrieved_document_count": sum(
            len(round_item.get("documents") or []) for round_item in state["retrieval_rounds"]
        ),
        "progressive_search_count": state["progressive_search_count"],
        "no_progress_search_count": state["no_progress_search_count"],
        "repeated_query_count": state["repeated_query_count"],
        "assistant_generation_count": state["assistant_generation_count"],
        "prompt_token_count_by_round": state["prompt_token_count_by_round"],
        "retrieval_rounds": state["retrieval_rounds"],
        "termination_reason": state["termination_reason"],
        "invalid_reasons": state["invalid_reasons"],
        "sft_rejection_reasons": rejection,
        "strict_sft_eligible": not rejection,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"count": 0}
    count = len(results)
    return {
        "count": count,
        "exact_match": sum(int(row.get("exact_match", 0)) for row in results) / count,
        "f1": sum(float(row.get("f1", 0.0)) for row in results) / count,
        "strict_sft_eligible_rate": sum(bool(row.get("strict_sft_eligible")) for row in results) / count,
        "mean_search_actions": sum(int(row.get("search_action_count", 0)) for row in results) / count,
        "mean_bm25_executions": sum(int(row.get("bm25_execution_count", 0)) for row in results) / count,
        "termination_reasons": dict(Counter(str(row.get("termination_reason")) for row in results)),
        "sft_rejection_reasons": dict(Counter(
            reason for row in results for reason in row.get("sft_rejection_reasons", [])
        )),
    }

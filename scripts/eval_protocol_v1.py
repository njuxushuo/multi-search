#!/usr/bin/env python3
"""Pure state-machine helpers for the v1 search evaluation protocol.

v1 differs from the frozen v0 protocol only in two behavioral rules:

* a repeated normalized query receives a neutral no-progress observation
  instead of terminating; and
* after eight search attempts, the model receives one reserved answer-only
  generation.

The latest successful retrieval exchange is retained while a repeated-query
correction is generated. Older successful exchanges are still discarded, so
the all-history token-budget policy remains reserved for v2.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Callable

from evaluate_qwen35_search import ANSWER_RE, STRICT_ACTION_RE, normalize, normalize_segment, token_f1


PROTOCOL = "qwen35_search_eval_v1_neutral_repeat_final_answer"
TASK_PROMPT = (
    "Answer the question using the search tool when needed. Reason inside <think>...</think>. "
    "To search, emit exactly <search>query</search>; after each search you will receive "
    "<information>...</information>. Finish with exactly <answer>final answer</answer>.\nQuestion: "
)
NEUTRAL_REPEAT_TEXT = (
    "No new information was retrieved. Use the available evidence to answer, "
    "or issue a different search query."
)
NEUTRAL_REPEAT_FINAL_TEXT = (
    "No new information was retrieved. The search budget is exhausted. "
    "Use the available evidence and output the required answer tag now."
)
NORMAL_FOLLOWUP = (
    " If the evidence answers the question, stop searching and output the required answer tag."
)
FINAL_FOLLOWUP = (
    " This is the final allowed search; do not search again. Output the required answer tag now."
)
# Environment observations always place both tags on their own lines. Anchors
# prevent literal ``<information>...</information>`` examples inside model
# reasoning from being mistaken for real tool observations during v0 replay.
INFO_BLOCK_RE = re.compile(r"^<information>\n.*?^</information>", re.I | re.S | re.M)

RECOVERABLE_REASONS = {"repeated_search_query", "no_new_documents"}
V0_AFFECTED_TERMINATIONS = {"repeated_search", "max_rounds"}


def question_message(question: str) -> dict[str, str]:
    return {"role": "user", "content": TASK_PROMPT + question}


def information_message(hits: list[dict[str, Any]], final: bool) -> tuple[str, dict[str, str]]:
    block = "<information>\n" + "\n\n".join(hit["content"] for hit in hits) + "\n</information>"
    suffix = FINAL_FOLLOWUP if final else NORMAL_FOLLOWUP
    return block, {"role": "user", "content": block + "\nUse the evidence above." + suffix}


def neutral_repeat_message(final: bool) -> tuple[str, dict[str, str]]:
    text = NEUTRAL_REPEAT_FINAL_TEXT if final else NEUTRAL_REPEAT_TEXT
    block = "<information>\n" + text + "\n</information>"
    return block, {"role": "user", "content": block}


def classify_reasons(reasons: list[str]) -> tuple[list[str], list[str]]:
    recoverable, terminal = [], []
    for reason in reasons:
        if reason in RECOVERABLE_REASONS:
            recoverable.append(reason)
        elif reason != "max_rounds_without_answer":
            terminal.append(reason)
    return recoverable, terminal


def initial_state(row: dict[str, Any], origin: str = "v1_full") -> dict[str, Any]:
    initial = question_message(str(row["question"]))
    return {
        "row": row,
        "initial_message": initial,
        "messages": [initial],
        "last_success_exchange": [],
        "turns": [],
        "queries": [],
        "searched": 0,
        "attempted": 0,
        "progressive": 0,
        "redundant": 0,
        "seen_doc_keys": set(),
        "retrieval_rounds": [],
        "invalid_reasons": [],
        "recoverable_reasons": [],
        "terminal_reasons": [],
        "repeated": 0,
        "no_progress": 0,
        "rounds": 0,
        "prediction": "",
        "done": False,
        "termination": "in_progress",
        "format_compliant": True,
        "final_only": False,
        "final_answer_opportunity": False,
        "origin": origin,
        "v0_exact_match": None,
        "v0_f1": None,
        "v0_search_count": None,
        "v0_search_attempt_count": None,
        "v0_termination_reason": None,
    }


def _last_success_exchange(v0: dict[str, Any]) -> list[dict[str, str]]:
    transcript = str(v0.get("assistant", ""))
    information = list(INFO_BLOCK_RE.finditer(transcript))
    if not information:
        raise ValueError("v0 transcript has no information block")
    latest = information[-1]
    prior_end = information[-2].end() if len(information) > 1 else 0
    search_action = transcript[prior_end:latest.start()].strip()
    strict = STRICT_ACTION_RE.fullmatch(search_action)
    if not strict or strict.group(1) is None:
        raise ValueError("cannot recover the latest successful search action")
    # v0 changed the observation suffix according to successful searches, not
    # attempts. A repeated eighth attempt may therefore follow only seven
    # successful searches and must retain the original non-final observation.
    final = int(v0.get("search_count", 0)) >= 8
    content = latest.group(0) + "\nUse the evidence above." + (FINAL_FOLLOWUP if final else NORMAL_FOLLOWUP)
    return [
        {"role": "assistant", "content": search_action},
        {"role": "user", "content": content},
    ]


def state_from_v0(row: dict[str, Any], v0: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the exact v1 continuation boundary from a v0 result.

    Raises ValueError when a reliable boundary cannot be recovered. Callers
    should then rerun that eval_id from the original question and mark it as
    ``v1_rerun``.
    """
    termination = str(v0.get("termination_reason", ""))
    if termination not in V0_AFFECTED_TERMINATIONS:
        raise ValueError(f"v0 result is not affected by v1: {termination}")
    state = initial_state(row, origin="v1_continued")
    last_exchange = _last_success_exchange(v0)
    state["last_success_exchange"] = last_exchange
    state["messages"] = [state["initial_message"], *last_exchange]
    state["turns"] = [str(v0.get("assistant", "")).strip()]
    state["queries"] = list(v0.get("queries", []))
    state["searched"] = int(v0.get("search_count", 0))
    state["attempted"] = int(v0.get("search_attempt_count", 0))
    state["progressive"] = int(v0.get("progressive_search_count", 0))
    state["redundant"] = int(v0.get("redundant_search_count", 0))
    state["retrieval_rounds"] = list(v0.get("retrieval_rounds", []))
    state["rounds"] = int(v0.get("round_count", 0))
    state["v0_exact_match"] = int(v0.get("exact_match", 0))
    state["v0_f1"] = float(v0.get("f1", 0.0))
    state["v0_search_count"] = int(v0.get("search_count", 0))
    state["v0_search_attempt_count"] = int(v0.get("search_attempt_count", 0))
    state["v0_termination_reason"] = termination
    doc_ids = [str(doc_id) for item in state["retrieval_rounds"] for doc_id in item.get("doc_ids", [])]
    if any(not doc_id for doc_id in doc_ids):
        raise ValueError("cannot reconstruct a missing document identity")
    state["seen_doc_keys"] = {"id:" + doc_id for doc_id in doc_ids}
    reasons = [reason for reason in v0.get("invalid_reasons", []) if reason != "max_rounds_without_answer"]
    recoverable, terminal = classify_reasons(reasons)
    if terminal:
        raise ValueError(f"affected v0 result also has terminal reasons: {terminal}")
    state["invalid_reasons"] = reasons
    state["recoverable_reasons"] = recoverable
    state["terminal_reasons"] = []
    state["repeated"] = sum(reason == "repeated_search_query" for reason in reasons)
    state["no_progress"] = len(recoverable)

    if termination == "repeated_search":
        transcript = str(v0.get("assistant", ""))
        latest_info = list(INFO_BLOCK_RE.finditer(transcript))[-1]
        repeated_action = transcript[latest_info.end():].strip()
        strict = STRICT_ACTION_RE.fullmatch(repeated_action)
        if not strict or strict.group(1) is None:
            raise ValueError("cannot recover the repeated search action")
        if normalize(strict.group(1)) not in {normalize(query) for query in state["queries"]}:
            raise ValueError("v0 repeated action does not match the recorded query history")
        final = state["attempted"] >= 8
        neutral_block, neutral_message = neutral_repeat_message(final)
        state["messages"] = [
            state["initial_message"],
            *last_exchange,
            {"role": "assistant", "content": repeated_action},
            neutral_message,
        ]
        state["turns"].append(neutral_block)
        if final:
            state["final_only"] = True
            state["final_answer_opportunity"] = True
    else:
        if state["attempted"] != 8:
            raise ValueError(f"max_rounds continuation expected 8 attempts, got {state['attempted']}")
        # Rebuild the last observation with the v1 final-answer instruction.
        block = INFO_BLOCK_RE.search(last_exchange[1]["content"])
        if not block:
            raise ValueError("missing final information block")
        state["last_success_exchange"][1]["content"] = (
            block.group(0) + "\nUse the evidence above." + FINAL_FOLLOWUP
        )
        state["messages"] = [state["initial_message"], *state["last_success_exchange"]]
        state["final_only"] = True
        state["final_answer_opportunity"] = True
    return state


def is_v0_affected(result: dict[str, Any]) -> bool:
    return result.get("termination_reason") in V0_AFFECTED_TERMINATIONS


def _append_reason(state: dict[str, Any], reason: str, recoverable: bool) -> None:
    state["invalid_reasons"].append(reason)
    target = "recoverable_reasons" if recoverable else "terminal_reasons"
    state[target].append(reason)


def apply_generation(
    state: dict[str, Any],
    raw_generated: str,
    retrieve_hits: Callable[[str], list[dict[str, Any]]],
    max_search_attempts: int = 8,
) -> None:
    """Apply one assistant generation to a mutable v1 trajectory state."""
    if state["done"]:
        raise ValueError("cannot advance a completed state")
    generated = normalize_segment(raw_generated.strip())
    state["turns"].append(generated)
    state["rounds"] += 1
    strict = STRICT_ACTION_RE.fullmatch(generated)
    if not strict:
        answers = ANSWER_RE.findall(generated)
        if answers:
            state["prediction"] = answers[-1].strip()
        state["format_compliant"] = False
        _append_reason(state, "malformed_or_ambiguous_action", recoverable=False)
        state["done"], state["termination"] = True, "format_error"
        return

    query, answer = strict.group(1), strict.group(2)
    if answer is not None:
        state["prediction"] = answer.strip()
        state["done"] = True
        state["termination"] = "answer_after_search_budget" if state["final_only"] else "answer"
        return

    if state["final_only"]:
        _append_reason(state, "search_after_search_budget", recoverable=False)
        state["no_progress"] += 1
        state["done"], state["termination"] = True, "search_limit"
        return

    state["attempted"] += 1
    if state["attempted"] > max_search_attempts:
        raise AssertionError("search attempt budget exceeded")
    query = (query or "").strip()
    normalized_query = normalize(query)
    if not normalized_query:
        _append_reason(state, "empty_search_query", recoverable=False)
        state["no_progress"] += 1
        state["done"], state["termination"] = True, "invalid_search"
        return

    if normalized_query in {normalize(item) for item in state["queries"]}:
        _append_reason(state, "repeated_search_query", recoverable=True)
        state["repeated"] += 1
        state["no_progress"] += 1
        final = state["attempted"] >= max_search_attempts
        neutral_block, neutral_message = neutral_repeat_message(final)
        state["messages"] = [
            state["initial_message"],
            *state["last_success_exchange"],
            {"role": "assistant", "content": generated},
            neutral_message,
        ]
        state["turns"].append(neutral_block)
        if final:
            state["final_only"] = True
            state["final_answer_opportunity"] = True
        return

    state["queries"].append(query)
    try:
        hits = retrieve_hits(query)
    except Exception as exc:  # noqa: BLE001 - persisted type is part of the result
        _append_reason(state, f"retrieval_error:{type(exc).__name__}", recoverable=False)
        state["no_progress"] += 1
        state["done"], state["termination"] = True, "retrieval_error"
        return
    if not hits:
        _append_reason(state, "empty_search_result", recoverable=False)
        state["no_progress"] += 1
        state["done"], state["termination"] = True, "empty_search_result"
        return

    state["searched"] += 1
    doc_keys = [
        ("id:" + str(hit.get("doc_id", ""))) if hit.get("doc_id") else
        ("content:" + hashlib.sha256(str(hit["content"]).encode()).hexdigest())
        for hit in hits
    ]
    new_positions = [index for index, key in enumerate(doc_keys) if key not in state["seen_doc_keys"]]
    new_doc_ids = [str(hits[index].get("doc_id", "")) for index in new_positions]
    novelty_ratio = len(new_positions) / len(hits)
    if new_positions:
        state["progressive"] += 1
    else:
        state["redundant"] += 1
        state["no_progress"] += 1
        _append_reason(state, "no_new_documents", recoverable=True)
    state["seen_doc_keys"].update(doc_keys)
    state["retrieval_rounds"].append({
        "query": query,
        "doc_ids": [str(hit.get("doc_id", "")) for hit in hits],
        "scores": [hit.get("score") for hit in hits],
        "new_doc_ids": new_doc_ids,
        "new_doc_count": len(new_positions),
        "novelty_ratio": novelty_ratio,
        "no_progress": not new_positions,
    })
    final = state["attempted"] >= max_search_attempts
    information_block, observation = information_message(hits, final)
    assistant_message = {"role": "assistant", "content": generated}
    state["last_success_exchange"] = [assistant_message, observation]
    state["messages"] = [state["initial_message"], *state["last_success_exchange"]]
    state["turns"].append(information_block)
    if final:
        state["final_only"] = True
        state["final_answer_opportunity"] = True


def force_unfinished_termination(state: dict[str, Any]) -> None:
    if state["done"]:
        return
    _append_reason(state, "generation_budget_exhausted_without_answer", recoverable=False)
    state["done"], state["termination"] = True, "generation_budget"


def build_result(state: dict[str, Any]) -> dict[str, Any]:
    row, prediction = state["row"], state["prediction"]
    answers = row["golden_answers"]
    exact_match = int(any(normalize(prediction) == normalize(answer) for answer in answers))
    f1 = max((token_f1(prediction, answer) for answer in answers), default=0.0)
    answer_present = bool(prediction)
    recoverable = bool(state["recoverable_reasons"])
    terminal_invalid = bool(state["terminal_reasons"])
    valid_trajectory = answer_present and state["format_compliant"] and not terminal_invalid
    strict_trajectory = valid_trajectory and not recoverable
    retrieval_rounds = state["retrieval_rounds"]
    return {
        "type": "result",
        "eval_id": row["eval_id"],
        "source_row": row["source_row"],
        "question": row["question"],
        "data_source": row["data_source"],
        "golden_answers": answers,
        "prediction": prediction,
        "exact_match": exact_match,
        "f1": f1,
        "search_count": state["searched"],
        "search_attempt_count": state["attempted"],
        "progressive_search_count": state["progressive"],
        "redundant_search_count": state["redundant"],
        "retrieved_document_count": sum(len(item["doc_ids"]) for item in retrieval_rounds),
        "unique_document_count": len(state["seen_doc_keys"]),
        "mean_novelty_ratio": (
            sum(item["novelty_ratio"] for item in retrieval_rounds) / len(retrieval_rounds)
            if retrieval_rounds else 0.0
        ),
        "round_count": state["rounds"],
        "zero_search": state["searched"] == 0,
        "answer_present": answer_present,
        "format_compliant": state["format_compliant"],
        "valid_trajectory": valid_trajectory,
        "strict_trajectory": strict_trajectory,
        "strict_success": bool(exact_match and strict_trajectory),
        "recovered_success": bool(exact_match and answer_present and recoverable and not terminal_invalid),
        "invalid_tool_call": terminal_invalid,
        "terminal_invalid": terminal_invalid,
        "recoverable_violation": recoverable,
        "has_any_violation": recoverable or terminal_invalid,
        "has_repeated_query": state["repeated"] > 0,
        "has_no_progress_call": state["no_progress"] > 0,
        "invalid_reasons": state["invalid_reasons"],
        "recoverable_reasons": state["recoverable_reasons"],
        "terminal_reasons": state["terminal_reasons"],
        "termination_reason": state["termination"],
        "final_answer_opportunity": state["final_answer_opportunity"],
        "origin": state["origin"],
        "v0_exact_match": state["v0_exact_match"],
        "v0_f1": state["v0_f1"],
        "v0_search_count": state["v0_search_count"],
        "v0_search_attempt_count": state["v0_search_attempt_count"],
        "v0_termination_reason": state["v0_termination_reason"],
        "exact_match_delta_vs_v0": (
            exact_match - state["v0_exact_match"] if state["v0_exact_match"] is not None else None
        ),
        "f1_delta_vs_v0": (
            f1 - state["v0_f1"]
            if state["v0_f1"] is not None else None
        ),
        "search_count_delta_vs_v0": (
            state["searched"] - state["v0_search_count"] if state["v0_search_count"] is not None else None
        ),
        "search_attempt_count_delta_vs_v0": (
            state["attempted"] - state["v0_search_attempt_count"]
            if state["v0_search_attempt_count"] is not None else None
        ),
        "queries": state["queries"],
        "assistant": "\n".join(turn for turn in state["turns"] if turn),
        "retrieval_rounds": retrieval_rounds,
    }


def adapt_v0_reused(v0: dict[str, Any]) -> dict[str, Any]:
    result = dict(v0)
    recoverable_reasons, terminal_reasons = classify_reasons(list(result.get("invalid_reasons", [])))
    answer_present = bool(result.get("prediction"))
    recoverable = bool(recoverable_reasons)
    terminal_invalid = bool(terminal_reasons)
    valid = answer_present and bool(result.get("format_compliant")) and not terminal_invalid
    strict = valid and not recoverable
    result.update({
        "origin": "v0_reused",
        "recoverable_reasons": recoverable_reasons,
        "terminal_reasons": terminal_reasons,
        "recoverable_violation": recoverable,
        "terminal_invalid": terminal_invalid,
        "has_any_violation": recoverable or terminal_invalid,
        "invalid_tool_call": terminal_invalid,
        "valid_trajectory": valid,
        "strict_trajectory": strict,
        "strict_success": bool(result.get("exact_match") and strict),
        "recovered_success": bool(result.get("exact_match") and answer_present and recoverable and not terminal_invalid),
        "final_answer_opportunity": False,
        "v0_exact_match": int(result.get("exact_match", 0)),
        "v0_f1": float(result.get("f1", 0.0)),
        "v0_search_count": int(result.get("search_count", 0)),
        "v0_search_attempt_count": int(result.get("search_attempt_count", 0)),
        "v0_termination_reason": result.get("termination_reason"),
        "exact_match_delta_vs_v0": 0,
        "f1_delta_vs_v0": 0.0,
        "search_count_delta_vs_v0": 0,
        "search_attempt_count_delta_vs_v0": 0,
    })
    return result


def aggregate_v1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from evaluate_qwen35_search import aggregate

    output = aggregate(rows)
    if not rows:
        return output
    n = len(rows)
    rate = lambda key: sum(bool(row.get(key)) for row in rows) / n
    repeated_rows = [row for row in rows if row.get("has_repeated_query")]
    final_rows = [row for row in rows if row.get("final_answer_opportunity")]
    paired_rows = [row for row in rows if row.get("v0_exact_match") is not None]
    safe_rate = lambda subset, predicate: (
        sum(bool(predicate(row)) for row in subset) / len(subset) if subset else 0.0
    )
    output.update({
        "strict_trajectory_rate": rate("strict_trajectory"),
        "recoverable_violation_rate": rate("recoverable_violation"),
        "terminal_invalid_rate": rate("terminal_invalid"),
        "any_violation_rate": rate("has_any_violation"),
        "final_answer_opportunity_rate": rate("final_answer_opportunity"),
        "repeated_query_examples": len(repeated_rows),
        "repeated_query_answer_recovery_rate": safe_rate(
            repeated_rows, lambda row: row.get("answer_present") and not row.get("terminal_invalid")
        ),
        "repeated_query_correct_recovery_rate": safe_rate(
            repeated_rows, lambda row: row.get("exact_match") and not row.get("terminal_invalid")
        ),
        "repeated_query_recovered_correct": sum(
            bool(row.get("exact_match") and not row.get("terminal_invalid")) for row in repeated_rows
        ),
        "final_answer_opportunity_examples": len(final_rows),
        "final_answer_completion_rate": safe_rate(
            final_rows, lambda row: row.get("termination_reason") == "answer_after_search_budget"
        ),
        "final_answer_correct_rate": safe_rate(
            final_rows,
            lambda row: row.get("termination_reason") == "answer_after_search_budget" and row.get("exact_match"),
        ),
        "paired_v0_examples": len(paired_rows),
        "em_delta_vs_v0": (
            sum(float(row["exact_match_delta_vs_v0"]) for row in paired_rows) / len(paired_rows)
            if paired_rows else None
        ),
        "f1_delta_vs_v0": (
            sum(float(row["f1_delta_vs_v0"]) for row in paired_rows) / len(paired_rows)
            if paired_rows else None
        ),
        "mean_search_count_delta_vs_v0": (
            sum(float(row["search_count_delta_vs_v0"]) for row in paired_rows) / len(paired_rows)
            if paired_rows else None
        ),
        "mean_search_attempt_count_delta_vs_v0": (
            sum(float(row["search_attempt_count_delta_vs_v0"]) for row in paired_rows) / len(paired_rows)
            if paired_rows else None
        ),
        "origin_counts": dict(Counter(row.get("origin", "unknown") for row in rows)),
    })
    return output

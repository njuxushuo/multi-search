from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_protocol_v1 import (  # noqa: E402
    NEUTRAL_REPEAT_FINAL_TEXT,
    NEUTRAL_REPEAT_TEXT,
    apply_generation,
    build_result,
    initial_state,
    state_from_v0,
)


def row() -> dict:
    return {
        "eval_id": 7,
        "source_row": 19,
        "question": "Where was the author born?",
        "golden_answers": ["Paris"],
        "data_source": "hotpotqa",
    }


def action(query: str) -> str:
    return f"<think>search</think><search>{query}</search>"


class ProtocolV1Tests(unittest.TestCase):
    def test_repeated_query_is_neutral_noop_and_keeps_latest_evidence(self) -> None:
        calls: list[str] = []

        def retrieve(query: str) -> list[dict]:
            calls.append(query)
            return [{"doc_id": "d1", "score": 1.0, "content": "Paris evidence"}]

        state = initial_state(row())
        apply_generation(state, action("author birthplace"), retrieve)
        first_information = state["last_success_exchange"][1]["content"]
        apply_generation(state, action("Author, birthplace!"), retrieve)

        self.assertEqual(calls, ["author birthplace"])
        self.assertEqual(state["searched"], 1)
        self.assertEqual(state["attempted"], 2)
        self.assertFalse(state["done"])
        self.assertEqual(len(state["messages"]), 5)
        self.assertIn(first_information, state["messages"][2]["content"])
        self.assertIn(NEUTRAL_REPEAT_TEXT, state["messages"][-1]["content"])

    def test_different_query_with_same_docs_still_calls_bm25_and_continues(self) -> None:
        calls: list[str] = []

        def retrieve(query: str) -> list[dict]:
            calls.append(query)
            return [{"doc_id": "same", "score": 1.0, "content": "same evidence"}]

        state = initial_state(row())
        apply_generation(state, action("author"), retrieve)
        apply_generation(state, action("birth city"), retrieve)

        self.assertEqual(calls, ["author", "birth city"])
        self.assertEqual(state["searched"], 2)
        self.assertEqual(state["attempted"], 2)
        self.assertEqual(state["redundant"], 1)
        self.assertIn("no_new_documents", state["recoverable_reasons"])
        self.assertFalse(state["done"])

    def test_eighth_repeat_gets_one_answer_only_generation(self) -> None:
        state = initial_state(row())
        state["queries"] = ["author"]
        state["attempted"] = 7
        state["searched"] = 1
        state["last_success_exchange"] = [
            {"role": "assistant", "content": action("author")},
            {"role": "user", "content": "<information>evidence</information>"},
        ]
        calls: list[str] = []
        apply_generation(state, action("author"), lambda query: calls.append(query) or [])

        self.assertEqual(calls, [])
        self.assertEqual(state["attempted"], 8)
        self.assertTrue(state["final_only"])
        self.assertTrue(state["final_answer_opportunity"])
        self.assertIn(NEUTRAL_REPEAT_FINAL_TEXT, state["messages"][-1]["content"])

        apply_generation(state, "<think>answer</think><answer>Paris</answer>", lambda _: [])
        result = build_result(state)
        self.assertEqual(result["termination_reason"], "answer_after_search_budget")
        self.assertEqual(result["exact_match"], 1)
        self.assertEqual(result["search_attempt_count"], 8)
        self.assertTrue(result["recovered_success"])
        self.assertFalse(result["terminal_invalid"])

    def test_final_answer_generation_cannot_search(self) -> None:
        state = initial_state(row())
        state["attempted"] = 8
        state["final_only"] = True
        state["final_answer_opportunity"] = True
        calls: list[str] = []
        apply_generation(state, action("another query"), lambda query: calls.append(query) or [])

        self.assertEqual(calls, [])
        self.assertEqual(state["attempted"], 8)
        self.assertTrue(state["done"])
        self.assertEqual(state["termination"], "search_limit")
        self.assertIn("search_after_search_budget", state["terminal_reasons"])

    def test_eighth_successful_search_also_reserves_final_answer(self) -> None:
        state = initial_state(row())
        state["attempted"] = 7
        state["searched"] = 7
        calls: list[str] = []

        def retrieve(query: str) -> list[dict]:
            calls.append(query)
            return [{"doc_id": "eighth", "score": 1.0, "content": "final evidence"}]

        apply_generation(state, action("eighth query"), retrieve)
        self.assertEqual(calls, ["eighth query"])
        self.assertEqual(state["attempted"], 8)
        self.assertEqual(state["searched"], 8)
        self.assertTrue(state["final_only"])
        self.assertIn("final allowed search", state["messages"][-1]["content"])

        apply_generation(state, "<think>answer</think><answer>Paris</answer>", retrieve)
        result = build_result(state)
        self.assertEqual(result["termination_reason"], "answer_after_search_budget")
        self.assertEqual(result["search_attempt_count"], 8)
        self.assertEqual(result["search_count"], 8)

    def test_v0_repeated_result_reconstructs_without_rerunning_prior_search(self) -> None:
        first = action("author")
        information = "<information>\nParis evidence\n</information>"
        repeated = action("Author!")
        v0 = {
            "termination_reason": "repeated_search",
            "assistant": "\n".join([first, information, repeated]),
            "queries": ["author"],
            "search_count": 1,
            "search_attempt_count": 2,
            "progressive_search_count": 1,
            "redundant_search_count": 0,
            "retrieval_rounds": [{
                "query": "author",
                "doc_ids": ["d1"],
                "scores": [1.0],
                "new_doc_ids": ["d1"],
                "new_doc_count": 1,
                "novelty_ratio": 1.0,
                "no_progress": False,
            }],
            "round_count": 2,
            "invalid_reasons": ["repeated_search_query"],
        }
        state = state_from_v0(row(), v0)

        self.assertEqual(state["origin"], "v1_continued")
        self.assertEqual(state["attempted"], 2)
        self.assertEqual(state["searched"], 1)
        self.assertEqual(len(state["messages"]), 5)
        self.assertIn("Paris evidence", state["messages"][2]["content"])
        self.assertIn(NEUTRAL_REPEAT_TEXT, state["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()

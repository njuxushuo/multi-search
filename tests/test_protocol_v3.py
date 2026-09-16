from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_protocol_v3 import apply_generation, build_result, initial_state  # noqa: E402
from protocol_v3 import (  # noqa: E402
    DEFAULT_CONFIG,
    canonical_prompt,
    compile_sft_example,
    format_observation,
    load_protocol,
    parse_action,
    prompt_from_row,
    support_sentence_visibility,
)


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(value) for value in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return f"<user>{messages[0]['content']}</user><assistant><think>\n"


def qa_row() -> dict:
    return {
        "eval_id": 1,
        "question_id": "hotpot:1",
        "question": "Where was the author born?",
        "golden_answers": ["Paris"],
        "data_source": "hotpotqa",
        "metadata": {"supporting_facts": {"title": ["Author", "Paris"], "sent_id": [0, 1]}},
    }


def search(query: str) -> str:
    return f"<think>Need evidence.</think><search>{query}</search>"


class ProtocolV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_protocol(DEFAULT_CONFIG)
        self.tokenizer = CharacterTokenizer()

    def test_version_and_frozen_budgets(self) -> None:
        self.assertEqual(self.config["human_version"], "R3.0")
        self.assertEqual(self.config["protocol_id"], "searchqa_repro_v3_0_0")
        self.assertEqual(self.config["token_budget"]["max_model_len"], 8192)
        self.assertEqual(self.config["token_budget"]["max_new_tokens_per_action"], 768)
        self.assertEqual(self.config["token_budget"]["max_observation_tokens"], 768)

    def test_dataset_prompt_must_match_frozen_text(self) -> None:
        row = qa_row()
        prompt = canonical_prompt(row["question"], self.config)
        self.assertTrue(prompt.endswith("?\n"))
        row["prompt"] = [{"role": "user", "content": prompt}]
        self.assertEqual(prompt_from_row(row, self.config), prompt)
        row["prompt"][0]["content"] += " changed"
        with self.assertRaisesRegex(ValueError, "dataset_prompt_drift"):
            prompt_from_row(row, self.config)

    def test_strict_action_parser_and_implicit_qwen_opener(self) -> None:
        parsed = parse_action("reason</think><search>query</search>")
        self.assertEqual(parsed.kind, "search")
        self.assertEqual(parsed.canonical, "<think>reason</think><search>query</search>")
        with self.assertRaisesRegex(ValueError, "strict_action_format_error"):
            parse_action("prefix <think>x</think><answer>Paris</answer>")

    def test_observation_balances_docs_and_never_exceeds_budget(self) -> None:
        hits = [
            {"document": {"id": f"d{index}", "title": f"T{index}", "text": letter * 300}, "score": 1.0}
            for index, letter in enumerate(("a", "b", "c"), 1)
        ]
        observation = format_observation(self.tokenizer, hits, max_tokens=240, top_k=3)
        self.assertLessEqual(observation.token_count, 240)
        self.assertTrue(observation.truncated)
        for index in range(1, 4):
            self.assertIn(f"Doc {index}(Title: T{index})", observation.text)
            self.assertGreater(observation.documents[index - 1]["visible_body_tokens"], 0)

    def test_support_sentence_visibility_uses_post_truncation_text(self) -> None:
        metadata = {
            "supporting_facts": {"title": ["Author"], "sent_id": [1]},
            "context": {"title": ["Author"], "sentences": [["Intro.", "The author was born in Paris."]]},
        }
        visible = [{"title": "Author", "visible_text": "The author was born in Paris."}]
        hidden = [{"title": "Author", "visible_text": "Intro only."}]
        self.assertEqual(support_sentence_visibility(metadata, visible)["recall"], 1.0)
        self.assertEqual(support_sentence_visibility(metadata, hidden)["recall"], 0.0)

    def test_repeated_query_continues_but_is_rejected_for_sft(self) -> None:
        row = qa_row()
        state = initial_state(row, canonical_prompt(row["question"], self.config), "c1")
        calls = []

        def retrieve(query: str):
            calls.append(query)
            return [{"document": {"id": "d1", "title": "Author", "text": "born in Paris"}, "score": 1.0}]

        apply_generation(state, search("author birthplace"), retrieve, self.tokenizer, self.config)
        apply_generation(state, search("Author, birthplace!"), retrieve, self.tokenizer, self.config)
        self.assertEqual(calls, ["author birthplace"])
        self.assertEqual(state["search_action_count"], 2)
        self.assertEqual(state["bm25_execution_count"], 1)
        self.assertFalse(state["done"])
        apply_generation(state, "<think>Evidence gives it.</think><answer>Paris</answer>", retrieve, self.tokenizer, self.config)
        result = build_result(state, self.config)
        self.assertEqual(result["exact_match"], 1)
        self.assertFalse(result["strict_sft_eligible"])
        self.assertIn("repeated_search_query", result["sft_rejection_reasons"])

    def test_four_searches_then_one_answer_only_generation(self) -> None:
        row = qa_row()
        state = initial_state(row, canonical_prompt(row["question"], self.config), "c2")
        counter = 0

        def retrieve(query: str):
            nonlocal counter
            counter += 1
            title = "Author" if counter == 1 else "Paris" if counter == 2 else f"Extra{counter}"
            return [{"document": {"id": f"d{counter}", "title": title, "text": f"new evidence {counter} Paris"}, "score": 1.0}]

        for index in range(4):
            apply_generation(state, search(f"query {index}"), retrieve, self.tokenizer, self.config)
        self.assertTrue(state["final_only"])
        self.assertFalse(state["done"])
        apply_generation(state, "<think>Combine evidence.</think><answer>Paris</answer>", retrieve, self.tokenizer, self.config)
        result = build_result(state, self.config)
        self.assertEqual(result["termination_reason"], "answer_after_search_budget")
        self.assertEqual(result["search_action_count"], 4)
        self.assertEqual(result["bm25_execution_count"], 4)
        self.assertEqual(result["support_coverage"]["recall"], 1.0)
        self.assertTrue(result["strict_sft_eligible"])

    def test_search_during_answer_only_generation_fails(self) -> None:
        row = qa_row()
        state = initial_state(row, canonical_prompt(row["question"], self.config), "c3")
        state["search_action_count"] = 4
        state["final_only"] = True
        calls = []
        apply_generation(state, search("fifth"), lambda query: calls.append(query) or [], self.tokenizer, self.config)
        self.assertEqual(calls, [])
        self.assertEqual(state["termination_reason"], "search_limit")
        self.assertEqual(state["search_action_count"], 4)

    def test_generation_length_finish_is_not_repaired(self) -> None:
        row = qa_row()
        state = initial_state(row, canonical_prompt(row["question"], self.config), "c4")
        apply_generation(
            state, "<think>unfinished", lambda _: [], self.tokenizer, self.config, finish_reason="length"
        )
        self.assertTrue(state["generation_truncated"])
        self.assertEqual(state["termination_reason"], "generation_truncated")
        self.assertFalse(build_result(state, self.config)["strict_sft_eligible"])

    def test_sft_compiler_masks_prompt_information_and_padding_only(self) -> None:
        prompt = canonical_prompt("Where?", self.config)
        events = [
            {"kind": "generated", "text": "<think>Search.</think><search>X</search>"},
            {"kind": "observation", "text": "<information>\nDoc 1(Title: X) Paris\n</information>"},
            {"kind": "generated", "text": "<think>Answer.</think><answer>Paris</answer>"},
        ]
        compiled = compile_sft_example(self.tokenizer, prompt, events, max_length=8192)
        for span in compiled["spans"]:
            values = compiled["labels"][span["start"]:span["end"]]
            if span["supervised"]:
                self.assertTrue(all(value != -100 for value in values))
            else:
                self.assertTrue(all(value == -100 for value in values))
        runtime = "".join(chr(value) for value in compiled["input_ids"])
        self.assertNotIn("<think>\n<think>", runtime)
        self.assertIn("<think>\nSearch.</think><search>X</search>", runtime)


if __name__ == "__main__":
    unittest.main()

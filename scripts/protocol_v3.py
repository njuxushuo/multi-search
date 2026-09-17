#!/usr/bin/env python3
"""Shared, CPU-testable primitives for versioned R3 protocols.

This module is the single source of truth used by Teacher sampling, SFT data
compilation and end-to-end evaluation.  Historical v0/v1 entrypoints do not
import it and therefore remain behaviorally frozen.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/protocols/searchqa_repro_v3_0_0.json"
IGNORE_INDEX = -100

ACTION_RE = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*"
    r"(?:(?:<search>(?P<search>.*?)</search>)|(?:<answer>(?P<answer>.*?)</answer>))\s*\Z",
    re.I | re.S,
)
INFO_RE = re.compile(r"<information>.*?</information>", re.I | re.S)
RESERVED_TAG_RE = re.compile(r"</?(?:think|search|answer|information)>", re.I)


@dataclass(frozen=True)
class ParsedAction:
    kind: str
    think: str
    content: str
    canonical: str


@dataclass(frozen=True)
class FormattedObservation:
    text: str
    token_count: int
    truncated: bool
    documents: tuple[dict[str, Any], ...]


def load_protocol(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "extends" in config:
        base_path = config_path.parent / str(config["extends"])
        expected_sha = str(config.get("extends_sha256", ""))
        if not expected_sha or file_sha256(base_path) != expected_sha:
            raise ValueError("protocol base config checksum mismatch")
        base = load_protocol(base_path)

        def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
            result = dict(left)
            for key, value in right.items():
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    result[key] = merge(result[key], value)
                else:
                    result[key] = value
            return result

        overlay = dict(config.get("overrides") or {})
        overlay.update({
            key: value for key, value in config.items()
            if key not in {"extends", "extends_sha256", "overrides"}
        })
        config = merge(base, overlay)
    required = {"schema_version", "human_version", "protocol_id", "prompt", "retrieval", "interaction", "token_budget"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"protocol config missing keys: {missing}")
    expected_id = "searchqa_repro_v" + config["human_version"].lower().replace("r", "").replace(".", "_") + "_0"
    if not re.fullmatch(r"R3\.\d+", str(config["human_version"])) or config["protocol_id"] != expected_id:
        raise ValueError("this module only implements versioned R3.x protocols")
    if config["retrieval"]["top_k"] != 3:
        raise ValueError("R3 protocol is frozen to top_k=3")
    if config["interaction"]["max_search_actions"] != 4:
        raise ValueError("R3 protocol is frozen to four search actions")
    budgets = config["token_budget"]
    if (budgets["max_model_len"], budgets["max_new_tokens_per_action"], budgets["max_observation_tokens"]) != (8192, 768, 768):
        raise ValueError("R3 token budget must remain 8192/768/768")
    final_tokens = int(budgets.get("max_new_tokens_final_action", budgets["max_new_tokens_per_action"]))
    worst_case = (
        int(budgets["initial_prompt_tokens"])
        + int(config["interaction"]["max_search_actions"])
        * (int(budgets["max_new_tokens_per_action"]) + int(budgets["max_observation_tokens"]))
        + final_tokens
    )
    if final_tokens < int(budgets["max_new_tokens_per_action"]) or worst_case > int(budgets["max_model_len"]):
        raise ValueError(f"R3 token budget envelope exceeds max_model_len: {worst_case}")
    return config


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalized_exact_match(prediction: str, answers: Iterable[str]) -> bool:
    prediction_key = normalize_answer(prediction)
    if not prediction_key:
        return False
    return any(
        bool(answer_key) and prediction_key == answer_key
        for answer_key in (normalize_answer(answer) for answer in answers)
    )


def token_f1(prediction: str, answer: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not prediction_tokens or not answer_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def normalize_query(text: str) -> str:
    return normalize_answer(text)


def restore_implicit_think(text: str) -> str:
    """Restore the Qwen chat-template implicit opener without hiding errors."""
    stripped = str(text).strip()
    if "</think>" in stripped.lower() and "<think>" not in stripped.lower():
        return "<think>" + stripped
    return stripped


def parse_action(text: str, restore_qwen_think: bool = True) -> ParsedAction:
    raw = restore_implicit_think(text) if restore_qwen_think else str(text).strip()
    match = ACTION_RE.fullmatch(raw)
    if not match:
        raise ValueError("strict_action_format_error")
    think = match.group("think").strip()
    if match.group("search") is not None:
        content = match.group("search").strip()
        kind = "search"
    else:
        content = match.group("answer").strip()
        kind = "answer"
    if not think:
        raise ValueError("empty_think")
    if not content:
        raise ValueError(f"empty_{kind}")
    if RESERVED_TAG_RE.search(think) or RESERVED_TAG_RE.search(content):
        raise ValueError("forbidden_protocol_tag_in_action_content")
    canonical = f"<think>{think}</think><{kind}>{content}</{kind}>"
    return ParsedAction(kind=kind, think=think, content=content, canonical=canonical)


def canonical_prompt(question: str, config: dict[str, Any]) -> str:
    question = str(question).strip()
    if not question:
        raise ValueError("empty_question")
    return str(config["prompt"]["template"]).format(question=question)


def prompt_from_row(row: dict[str, Any], config: dict[str, Any]) -> str:
    """Read the dataset prompt and reject drift from the frozen template."""
    expected = canonical_prompt(str(row.get("question", "")), config)
    source = str((config.get("prompt") or {}).get("source", ""))
    if source == "protocol_config.template":
        return expected
    if source != "parquet.prompt[0].content":
        raise ValueError(f"unsupported_prompt_source:{source}")
    prompt = row.get("prompt")
    if prompt is None:
        raise ValueError("missing_dataset_prompt")
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("invalid_dataset_prompt")
    first = prompt[0]
    if not isinstance(first, dict) or str(first.get("role", "")) != "user":
        raise ValueError("invalid_dataset_prompt_role")
    actual = str(first.get("content", ""))
    if actual != expected:
        raise ValueError("dataset_prompt_drift")
    return actual


def render_initial_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def runtime_first_action(rendered_prompt: str, action: str) -> str:
    """Avoid duplicating Qwen3.5's implicit first ``<think>`` opener.

    The canonical event always stores a complete explicit tag pair.  The
    official Qwen3.5 generation prefix already ends in ``<think>\n``, so the
    runtime continuation removes only the first event's opening tag.
    """
    if re.search(r"<think>\s*\Z", rendered_prompt, re.I) and re.match(r"\s*<think>", action, re.I):
        return re.sub(r"\A\s*<think>", "", action, count=1, flags=re.I)
    return action


def materialize_runtime_trajectory(rendered_prompt: str, trajectory: str) -> str:
    if not trajectory:
        return ""
    return runtime_first_action(rendered_prompt, trajectory)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)


def token_count(tokenizer: Any, text: str) -> int:
    return len(_token_ids(tokenizer, text))


def max_new_tokens_for_action(config: dict[str, Any], final_only: bool = False) -> int:
    budget = config["token_budget"]
    if final_only:
        return int(budget.get("max_new_tokens_final_action", budget["max_new_tokens_per_action"]))
    return int(budget["max_new_tokens_per_action"])


def _decode_prefix(tokenizer: Any, text: str, tokens: int) -> str:
    ids = _token_ids(tokenizer, text)
    return tokenizer.decode(ids[: max(0, tokens)], skip_special_tokens=False).strip()


def _hit_fields(hit: dict[str, Any], rank: int) -> dict[str, Any]:
    document = hit.get("document", hit)
    raw_contents = str(document.get("contents", hit.get("content", "")) or "")
    title = str(document.get("title", hit.get("title", "")) or "").strip()
    body = str(document.get("text", hit.get("text", "")) or "").strip()
    if raw_contents:
        lines = raw_contents.splitlines()
        if not title and lines:
            first = lines[0].strip().strip('"')
            if first and len(first) <= 300:
                title = first
                body = "\n".join(lines[1:]).strip()
            else:
                body = raw_contents.strip()
        elif not body:
            body = raw_contents.strip()
    return {
        "rank": rank,
        "doc_id": str(document.get("id", hit.get("doc_id", "")) or ""),
        "title": title or "Untitled",
        "text": body,
        "score": hit.get("score"),
    }


def format_observation(
    tokenizer: Any,
    hits: Sequence[dict[str, Any]],
    max_tokens: int = 768,
    top_k: int = 3,
) -> FormattedObservation:
    """Format top-k hits while sharing the body budget across all documents.

    The algorithm is deterministic and gold-agnostic.  It keeps every title
    and rank, distributes remaining token capacity by water filling, and then
    tightens the longest body until the fully serialized block fits exactly.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    documents = [_hit_fields(dict(hit), rank) for rank, hit in enumerate(hits[:top_k], 1)]
    if not documents:
        raise ValueError("empty_search_result")

    def build(body_limits: Sequence[int]) -> str:
        lines = ["<information>"]
        for document, limit in zip(documents, body_limits):
            body = _decode_prefix(tokenizer, document["text"], limit) if limit else ""
            suffix = f" {body}" if body else ""
            lines.append(f"Doc {document['rank']}(Title: {document['title']}){suffix}")
        lines.append("</information>")
        return "\n".join(lines)

    empty_limits = [0] * len(documents)
    empty_text = build(empty_limits)
    if token_count(tokenizer, empty_text) > max_tokens:
        raise ValueError("observation_headers_exceed_budget")

    lengths = [token_count(tokenizer, document["text"]) for document in documents]
    limits = [0] * len(documents)
    remaining = max_tokens - token_count(tokenizer, empty_text)
    active = {index for index, length in enumerate(lengths) if length > 0}
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        changed = False
        for index in sorted(active):
            addition = min(share, lengths[index] - limits[index], remaining)
            if addition > 0:
                limits[index] += addition
                remaining -= addition
                changed = True
            if limits[index] >= lengths[index]:
                active.discard(index)
            if remaining <= 0:
                break
        if not changed:
            break

    text = build(limits)
    while token_count(tokenizer, text) > max_tokens:
        candidates = [index for index, limit in enumerate(limits) if limit > 0]
        if not candidates:
            raise ValueError("observation_cannot_fit_budget")
        index = max(candidates, key=lambda item: limits[item])
        limits[index] -= 1
        text = build(limits)

    serialized_documents = []
    for document, limit, original_length in zip(documents, limits, lengths):
        visible = _decode_prefix(tokenizer, document["text"], limit) if limit else ""
        serialized_documents.append({
            **document,
            "original_text": document["text"],
            "visible_text": visible,
            "original_body_tokens": original_length,
            "visible_body_tokens": limit,
            "body_truncated": limit < original_length,
        })
    return FormattedObservation(
        text=text,
        token_count=token_count(tokenizer, text),
        truncated=any(document["body_truncated"] for document in serialized_documents),
        documents=tuple(serialized_documents),
    )


def append_generated_event(trajectory: str, action: str) -> str:
    separator = "\n\n" if trajectory else ""
    return trajectory + separator + action.strip()


def append_observation_event(trajectory: str, observation: str) -> str:
    separator = "\n\n" if trajectory else ""
    return trajectory + separator + observation.strip()


def action_generation_prefix(
    config: dict[str, Any] | None,
    trajectory: str,
    final_only: bool = False,
) -> str:
    """Return the environment-supplied prefix for a non-initial action.

    Qwen's chat template supplies ``<think>`` for the first assistant action.
    A flat tool trajectory has no later assistant header, so protocols that
    retain flat continuation must explicitly supply the same opener after an
    observation.  It is environment input, not a Teacher-generated token.
    """
    if not trajectory or not config:
        return ""
    interaction = config.get("interaction") or {}
    if interaction.get("inject_think_opener_each_generation"):
        reminder = ""
        if final_only and interaction.get("final_answer_instruction"):
            reminder = "\n\n" + str(interaction["final_answer_instruction"]).strip()
        return reminder + "\n\n<think>\n"
    return ""


def build_model_input(
    tokenizer: Any,
    initial_prompt: str,
    trajectory: str,
    config: dict[str, Any] | None = None,
    final_only: bool = False,
) -> tuple[str, int]:
    rendered = render_initial_prompt(tokenizer, initial_prompt)
    text = (
        rendered
        + materialize_runtime_trajectory(rendered, trajectory)
        + action_generation_prefix(config, trajectory, final_only)
    )
    return text, token_count(tokenizer, text)


def assert_generation_fits(
    tokenizer: Any,
    initial_prompt: str,
    trajectory: str,
    config: dict[str, Any],
    final_only: bool = False,
) -> int:
    _, prompt_tokens = build_model_input(
        tokenizer, initial_prompt, trajectory, config, final_only
    )
    budget = config["token_budget"]
    if prompt_tokens + max_new_tokens_for_action(config, final_only) > budget["max_model_len"]:
        raise ValueError("context_overflow")
    return prompt_tokens


def _event_text(event: dict[str, Any]) -> str:
    value = str(event.get("text", "")).strip()
    if not value:
        raise ValueError("empty_event")
    kind = event.get("kind")
    if kind == "generated":
        parse_action(value)
    elif kind == "observation":
        if not (value.startswith("<information>") and value.endswith("</information>")):
            raise ValueError("invalid_observation_event")
    else:
        raise ValueError(f"unknown_event_kind:{kind}")
    return value


def compile_sft_example(
    tokenizer: Any,
    initial_prompt: str,
    events: Sequence[dict[str, Any]],
    max_length: int = 8192,
    ignore_index: int = IGNORE_INDEX,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile one flat trajectory with exact span-level information masking."""
    rendered = render_initial_prompt(tokenizer, initial_prompt)
    pieces: list[tuple[str, bool, str]] = [(rendered, False, "prompt")]
    inject_action_prefix = bool(
        config
        and (config.get("interaction") or {}).get("inject_think_opener_each_generation")
    )
    searches_seen = 0
    for index, event in enumerate(events):
        event_text = _event_text(dict(event))
        parsed_action = parse_action(event_text) if event.get("kind") == "generated" else None
        if index == 0 and event.get("kind") == "generated":
            event_text = runtime_first_action(rendered, event_text)
        separator = "" if index == 0 else "\n\n"
        if separator:
            pieces.append((separator, False, "separator"))
        if (
            inject_action_prefix
            and parsed_action is not None
            and parsed_action.kind == "answer"
            and searches_seen >= int((config or {}).get("interaction", {}).get("max_search_actions", 4))
            and (config or {}).get("interaction", {}).get("final_answer_instruction")
        ):
            pieces.append((
                str(config["interaction"]["final_answer_instruction"]).strip(),
                False,
                "final_answer_instruction",
            ))
            pieces.append(("\n\n", False, "separator"))
        if (
            inject_action_prefix
            and index > 0
            and event.get("kind") == "generated"
            and events[index - 1].get("kind") == "observation"
        ):
            if not re.match(r"\s*<think>", event_text, re.I):
                raise ValueError("generated_action_missing_canonical_think")
            pieces.append(("<think>\n", False, "action_prefix"))
            event_text = re.sub(r"\A\s*<think>", "", event_text, count=1, flags=re.I)
        pieces.append((event_text, event.get("kind") == "generated", str(event.get("kind"))))
        if parsed_action is not None and parsed_action.kind == "search":
            searches_seen += 1

    input_ids: list[int] = []
    labels: list[int] = []
    spans: list[dict[str, Any]] = []
    char_text = ""
    for text, supervised, kind in pieces:
        ids = _token_ids(tokenizer, text)
        start = len(input_ids)
        input_ids.extend(ids)
        labels.extend(ids if supervised else [ignore_index] * len(ids))
        spans.append({"kind": kind, "start": start, "end": len(input_ids), "supervised": supervised})
        char_text += text

    # Segment tokenization must exactly equal runtime tokenization.  Failing
    # loudly is safer than silently shifting a label boundary at a BPE join.
    runtime_ids = _token_ids(tokenizer, char_text)
    if runtime_ids != input_ids:
        raise ValueError("segment_tokenization_mismatch")
    if len(input_ids) > max_length:
        raise ValueError("sequence_overflow")
    if not any(label != ignore_index for label in labels):
        raise ValueError("empty_supervision")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "token_count": len(input_ids),
        "supervised_token_count": sum(label != ignore_index for label in labels),
        "spans": spans,
    }


def validate_strict_candidate_record(row: dict[str, Any], config: dict[str, Any]) -> None:
    """Revalidate a selected Teacher candidate without trusting eligibility flags."""
    if row.get("protocol_id") != config["protocol_id"]:
        raise ValueError("candidate_protocol_mismatch")
    if not row.get("strict_sft_eligible") or row.get("sft_rejection_reasons"):
        raise ValueError("candidate_not_strict_eligible")
    raw_generations = row.get("raw_generations")
    if not isinstance(raw_generations, list) or len(raw_generations) != int(row.get("assistant_generation_count", -1)):
        raise ValueError("raw_generation_audit_mismatch")
    events = row.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("missing_candidate_events")
    retrieval_rounds = row.get("retrieval_rounds")
    if not isinstance(retrieval_rounds, list):
        raise ValueError("missing_retrieval_rounds")

    trajectory = ""
    search_count = 0
    round_index = 0
    expect_observation = False
    saw_answer = False
    for index, event in enumerate(events):
        kind = event.get("kind")
        text = str(event.get("text", ""))
        if kind == "generated":
            if expect_observation or saw_answer:
                raise ValueError("candidate_event_order_error")
            if not event.get("raw_text") or event.get("canonical_text") != text:
                raise ValueError("candidate_raw_canonical_missing")
            action = parse_action(text)
            trajectory = append_generated_event(trajectory, text)
            if action.kind == "search":
                search_count += 1
                if str(event.get("query", "")).strip() != action.content:
                    raise ValueError("candidate_query_event_mismatch")
                expect_observation = True
            else:
                saw_answer = True
                if index != len(events) - 1:
                    raise ValueError("candidate_answer_not_final")
        elif kind == "observation":
            if not expect_observation or saw_answer:
                raise ValueError("candidate_observation_order_error")
            if not (text.startswith("<information>") and text.endswith("</information>")):
                raise ValueError("candidate_observation_format_error")
            if round_index >= len(retrieval_rounds):
                raise ValueError("candidate_missing_retrieval_round")
            retrieval = retrieval_rounds[round_index]
            previous = events[index - 1]
            if str(retrieval.get("query", "")).strip() != str(previous.get("query", "")).strip():
                raise ValueError("candidate_query_information_mismatch")
            if retrieval.get("documents") != event.get("documents"):
                raise ValueError("candidate_retrieval_documents_mismatch")
            trajectory = append_observation_event(trajectory, text)
            expect_observation = False
            round_index += 1
        else:
            raise ValueError("candidate_unknown_event_kind")
    if expect_observation or not saw_answer:
        raise ValueError("candidate_incomplete_trajectory")
    if search_count != int(row.get("search_action_count", -1)) or round_index != len(retrieval_rounds):
        raise ValueError("candidate_search_count_mismatch")
    if trajectory != row.get("canonical_trajectory"):
        raise ValueError("candidate_canonical_trajectory_mismatch")


def document_key(document: dict[str, Any]) -> str:
    doc_id = str(document.get("doc_id", ""))
    if doc_id:
        return "id:" + doc_id
    content = str(document.get("visible_text", document.get("text", "")))
    return "content:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def support_titles(metadata: Any) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    supporting = metadata.get("supporting_facts")
    if not isinstance(supporting, dict):
        return []
    titles = supporting.get("title") or []
    if hasattr(titles, "tolist"):
        titles = titles.tolist()
    return [str(title) for title in titles]


def normalized_title(title: str) -> str:
    return " ".join(str(title).casefold().split())


def support_coverage(gold_titles: Sequence[str], visible_documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gold = {normalized_title(title) for title in gold_titles if normalized_title(title)}
    visible = {normalized_title(document.get("title", "")) for document in visible_documents}
    matched = sorted(gold & visible)
    return {
        "gold_title_count": len(gold),
        "matched_title_count": len(matched),
        "matched_titles": matched,
        "recall": (len(matched) / len(gold)) if gold else None,
    }


def support_sentence_visibility(metadata: Any, visible_documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Audit Hotpot supporting sentences after non-oracle observation truncation."""
    if not isinstance(metadata, dict):
        return {"gold_sentence_count": 0, "visible_sentence_count": 0, "recall": None}
    supporting = metadata.get("supporting_facts")
    context = metadata.get("context")
    if not isinstance(supporting, dict) or not isinstance(context, dict):
        return {"gold_sentence_count": 0, "visible_sentence_count": 0, "recall": None}
    titles = supporting.get("title") or []
    sentence_ids = supporting.get("sent_id") or []
    context_titles = context.get("title") or []
    context_sentences = context.get("sentences") or []
    for value_name, value in (("titles", titles), ("sentence_ids", sentence_ids), ("context_titles", context_titles), ("context_sentences", context_sentences)):
        if hasattr(value, "tolist"):
            if value_name == "titles":
                titles = value.tolist()
            elif value_name == "sentence_ids":
                sentence_ids = value.tolist()
            elif value_name == "context_titles":
                context_titles = value.tolist()
            else:
                context_sentences = value.tolist()
    context_map = {
        normalized_title(title): list(sentences)
        for title, sentences in zip(context_titles, context_sentences)
    }
    visible_by_title: dict[str, str] = {}
    for document in visible_documents:
        key = normalized_title(document.get("title", ""))
        visible_by_title[key] = normalize_answer(
            visible_by_title.get(key, "") + " " + str(document.get("visible_text", ""))
        )
    gold_sentences = []
    visible_count = 0
    for title, sentence_id in zip(titles, sentence_ids):
        key = normalized_title(title)
        sentences = context_map.get(key, [])
        try:
            sentence = str(sentences[int(sentence_id)])
        except (IndexError, TypeError, ValueError):
            continue
        sentence_key = normalize_answer(sentence)
        if not sentence_key:
            continue
        gold_sentences.append({"title": str(title), "sent_id": int(sentence_id), "text": sentence})
        if sentence_key in visible_by_title.get(key, ""):
            visible_count += 1
    return {
        "gold_sentence_count": len(gold_sentences),
        "visible_sentence_count": visible_count,
        "recall": (visible_count / len(gold_sentences)) if gold_sentences else None,
        "gold_sentences": gold_sentences,
    }

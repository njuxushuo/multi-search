#!/usr/bin/env python3
"""Shared structural checks for Search-R1 teacher trajectories.

The generation loop uses these checks before writing a record, while the
merge/validation stages repeat them on serialized JSONL so that an old or
manually edited shard cannot bypass the data-quality gate.
"""
from __future__ import annotations

import re


SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.I | re.S)
INFO_RE = re.compile(r"<information>\s*(.*?)\s*</information>", re.I | re.S)


def _content_fingerprint(text: str) -> str:
    """Normalize whitespace/case for detecting repeated evidence blocks."""
    return " ".join(text.lower().split())


def validate_tool_progress(assistant: str, is_valid_sequence) -> tuple[bool, str]:
    """Reject malformed, repeated, empty, or non-progressing tool traces.

    ``is_valid_sequence`` is VERL's canonical Search-R1 parser.  The extra
    checks here are deliberately conservative: every search must have a
    non-empty query and non-empty information block, normalized queries must
    be unique, and the same evidence block may not be returned twice.
    """
    valid_format, format_reason = is_valid_sequence("<|im_start|>assistant " + assistant)
    if not valid_format:
        return False, f"invalid tool-use sequence: {format_reason}"

    searches = SEARCH_RE.findall(assistant)
    infos = INFO_RE.findall(assistant)
    if not searches:
        return False, "no search turn"
    if len(searches) != len(infos):
        return False, "search/information turn count mismatch"

    seen_queries: set[str] = set()
    seen_info: set[str] = set()
    for query, info in zip(searches, infos):
        query_key = " ".join(query.lower().split())
        if not query_key:
            return False, "empty search query"
        if query_key in seen_queries:
            return False, "repeated search query (no progress)"
        seen_queries.add(query_key)

        info_key = _content_fingerprint(info)
        if not info_key:
            return False, "empty information result (no progress)"
        if info_key in seen_info:
            return False, "repeated information result (no progress)"
        seen_info.add(info_key)
    return True, "ok"

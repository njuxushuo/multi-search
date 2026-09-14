#!/usr/bin/env python3
"""Split embedded retrieval evidence into masked human turns for SFT."""
import json
import re
from pathlib import Path

INFO_RE = re.compile(r"(<information>.*?</information>)", re.I | re.S)

def convert(row):
    conv = row["conversations"]
    question, assistant = conv[0]["value"], conv[1]["value"]
    parts = INFO_RE.split(assistant)
    turns = [{"from": "human", "value": question}]
    for i, part in enumerate(parts):
        if part.strip():
            turns.append({"from": "human" if i % 2 else "gpt", "value": part.strip()})
    if turns[-1]["from"] != "gpt":
        raise ValueError("trajectory ends with information")
    row["conversations"] = turns
    row.setdefault("metadata", {})["loss_mask"] = "assistant_turns_only_v2"
    return row

root = Path("data/processed/search_sft_qwen35_4b")
for name in ("train", "eval"):
    src = root / f"{name}.jsonl"
    rows = [convert(json.loads(line)) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(name, len(rows))

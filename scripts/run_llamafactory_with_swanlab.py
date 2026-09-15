#!/usr/bin/env python3
"""Launch LLaMA-Factory while tolerating SwanLab 0.9.x get_run semantics."""
from __future__ import annotations

import sys

import swanlab


_get_run = swanlab.get_run


def _safe_get_run():
    try:
        return _get_run()
    except RuntimeError as exc:
        if "No active Run" in str(exc):
            return None
        raise


swanlab.get_run = _safe_get_run
config = sys.argv[1] if len(sys.argv) > 1 else "configs/sft_qwen35_4b_lora.yaml"
sys.argv = ["llamafactory-cli", "train", config]
from llamafactory.cli import main  # noqa: E402

main()

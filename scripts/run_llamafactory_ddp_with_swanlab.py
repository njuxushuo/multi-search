#!/usr/bin/env python3
"""Run LLaMA-Factory training under torchrun with SwanLab 0.9.x compatibility.

LLaMA-Factory's launcher starts distributed children by importing its own
launcher module, so the regular wrapper's ``get_run`` patch is not inherited.
This entrypoint applies the patch in every child and calls ``run_exp`` directly
to avoid a second nested torchrun launch.
"""
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
sys.argv = ["llamafactory-ddp", config, *sys.argv[2:]]

from llamafactory.train.tuner import run_exp  # noqa: E402

run_exp()

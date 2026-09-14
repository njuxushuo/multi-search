#!/usr/bin/env python3
"""Full-parameter LLaMA-Factory SFT entrypoint with SwanLab compatibility."""
from __future__ import annotations

import sys
import swanlab
import torch

# The llamafactory environment uses torch 2.5, while this Transformers build
# refuses to restore trusted local DeepSpeed ``.pt`` optimizer shards unless
# torch>=2.6.  The checkpoint is produced locally by this run; allow the
# existing ``weights_only=True`` loader for this trusted resume path.  Model
# weights remain safetensors.
import transformers.trainer as _transformers_trainer

_transformers_trainer.check_torch_load_is_safe = lambda: None

# Torch 2.5 cannot restore the NumPy RNG object in this locally produced
# checkpoint with ``weights_only=True``.  Permit full deserialization only for
# checkpoint-local ``.pt`` state files; the checkpoint is trusted and model
# weights are still stored/loaded as safetensors.
_torch_load = torch.load


def _trusted_checkpoint_load(*args, **kwargs):
    path = str(args[0]) if args else str(kwargs.get("f", ""))
    if ("checkpoint-" in path or "global_step" in path) and path.endswith((".pt", ".pth")):
        kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)


torch.load = _trusted_checkpoint_load


_get_run = swanlab.get_run


def _safe_get_run():
    try:
        return _get_run()
    except RuntimeError as exc:
        if "No active Run" in str(exc):
            return None
        raise


swanlab.get_run = _safe_get_run
config = sys.argv[1] if len(sys.argv) > 1 else "configs/sft_qwen35_4b_full.yaml"
sys.argv = ["llamafactory-ddp-full", config, *sys.argv[2:]]

from llamafactory.train.tuner import run_exp  # noqa: E402

run_exp()

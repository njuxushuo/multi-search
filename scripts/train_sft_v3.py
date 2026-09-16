#!/usr/bin/env python3
"""Train the pretokenized R3.0 Full SFT dataset with exact labels."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from protocol_v3 import file_sha256, load_protocol


@dataclass
class PretokenizedCollator:
    pad_token_id: int
    ignore_index: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            length = len(feature["input_ids"])
            padding = max_length - length
            input_ids.append(list(feature["input_ids"]) + [self.pad_token_id] * padding)
            attention_mask.append([1] * length + [0] * padding)
            labels.append(list(feature["labels"]) + [self.ignore_index] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--stop-after-step", type=int, default=None)
    parser.add_argument("--approve-second-epoch", action="store_true")
    args = parser.parse_args()
    train_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    os.environ.setdefault("SWANLAB_PROJECT", str(train_config["swanlab_project"]))
    os.environ.setdefault("SWANLAB_RUN_NAME", str(train_config["swanlab_run_name"]))
    protocol = load_protocol(train_config["protocol_config"])
    dataset_metadata = json.loads(Path(train_config["dataset_metadata"]).read_text(encoding="utf-8"))
    if dataset_metadata.get("protocol_id") != protocol["protocol_id"]:
        raise SystemExit("SFT dataset/protocol ID mismatch")
    if dataset_metadata.get("protocol_config_sha256") != file_sha256(train_config["protocol_config"]):
        raise SystemExit("SFT dataset was built with a different protocol config")
    expected_model = protocol["models"]["student_initial"]
    if str(Path(train_config["model_name_or_path"]).resolve()) != str(Path(expected_model).resolve()):
        raise SystemExit("R3.0 Full SFT must start from the frozen post-trained 4B checkpoint")
    frozen_sft = protocol["sft"]
    frozen_pairs = {
        "learning_rate": frozen_sft["learning_rate"],
        "num_train_epochs": float(frozen_sft["scheduler_horizon_epochs"]),
        "lr_scheduler_type": frozen_sft["scheduler"],
        "warmup_ratio": frozen_sft["warmup_ratio"],
        "eval_steps": frozen_sft["eval_steps"],
        "seed": frozen_sft["seed"],
        "bf16": frozen_sft["precision"] == "bf16",
        "gradient_checkpointing": frozen_sft["gradient_checkpointing"],
    }
    for key, expected in frozen_pairs.items():
        if train_config.get(key) != expected:
            raise SystemExit(f"R3.0 train config drift for {key}: {train_config.get(key)!r} != {expected!r}")
    target_epochs = int(train_config["num_train_epochs"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch = (
        world_size
        * int(train_config["per_device_train_batch_size"])
        * int(train_config["gradient_accumulation_steps"])
    )
    if effective_batch != int(frozen_sft["global_batch_size"]):
        raise SystemExit(
            f"R3.0 effective global batch must be {frozen_sft['global_batch_size']}, got {effective_batch}"
        )
    epoch_one_steps = int(protocol["selection"]["train_trajectories"]) // int(frozen_sft["global_batch_size"])
    if int(protocol["selection"]["train_trajectories"]) % int(frozen_sft["global_batch_size"]):
        epoch_one_steps += 1
    crosses_epoch_gate = args.stop_after_step is None or args.stop_after_step > epoch_one_steps
    if crosses_epoch_gate and not args.approve_second_epoch:
        raise SystemExit(
            f"R3.0 must stop at epoch-1 gate step {epoch_one_steps}; "
            "continuing epoch 2 requires --approve-second-epoch"
        )
    if args.approve_second_epoch and not args.resume_from_checkpoint:
        raise SystemExit("R3.0 epoch 2 approval requires resume from the epoch-1 checkpoint")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    # The local training environment uses torch 2.5.  Resume only trusted
    # checkpoint-local optimizer/RNG files produced by this exact R3.0 run;
    # model weights remain safetensors.
    import transformers.trainer as transformers_trainer
    transformers_trainer.check_torch_load_is_safe = lambda: None
    original_torch_load = torch.load

    def trusted_checkpoint_load(*load_args, **load_kwargs):
        path = str(load_args[0]) if load_args else str(load_kwargs.get("f", ""))
        if ("checkpoint-" in path or "global_step" in path) and path.endswith((".pt", ".pth")):
            load_kwargs["weights_only"] = False
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_checkpoint_load

    class StopAfterStepCallback(TrainerCallback):
        def __init__(self, target_step: int):
            self.target_step = target_step

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= self.target_step:
                control.should_save = True
                control.should_training_stop = True
            return control

    # SwanLab 0.9.x raises when integrations probe get_run before init.  Keep
    # the historical compatibility shim local to this entrypoint.
    try:
        import swanlab
        original_get_run = swanlab.get_run

        def safe_get_run():
            try:
                return original_get_run()
            except RuntimeError as exc:
                if "No active Run" in str(exc):
                    return None
                raise

        swanlab.get_run = safe_get_run
    except ImportError:
        if train_config["report_to"] == "swanlab":
            raise

    dataset = load_dataset(
        "parquet",
        data_files={"train": train_config["train_file"], "eval": train_config["eval_file"]},
    )
    expected_counts = {
        "train": protocol["selection"]["train_trajectories"],
        "eval": protocol["selection"]["teacher_forced_eval_trajectories"],
    }
    for split, count in expected_counts.items():
        if len(dataset[split]) != count:
            raise SystemExit(f"unexpected {split} size: {len(dataset[split])} != {count}")

    tokenizer = AutoTokenizer.from_pretrained(
        train_config["model_name_or_path"], trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        train_config["model_name_or_path"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    if train_config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "protocol_id": protocol["protocol_id"],
        "human_version": protocol["human_version"],
        "train_config": str(Path(args.config).resolve()),
        "train_config_sha256": file_sha256(args.config),
        "protocol_config_sha256": file_sha256(train_config["protocol_config"]),
        "dataset_metadata_sha256": file_sha256(train_config["dataset_metadata"]),
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "target_epochs": target_epochs,
        "second_epoch_approved": args.approve_second_epoch,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        num_train_epochs=target_epochs,
        lr_scheduler_type=train_config["lr_scheduler_type"],
        warmup_ratio=train_config["warmup_ratio"],
        logging_steps=train_config["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_config["eval_steps"],
        save_strategy="steps",
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        load_best_model_at_end=False,
        bf16=train_config["bf16"],
        gradient_checkpointing=train_config["gradient_checkpointing"],
        deepspeed=train_config["deepspeed"],
        group_by_length=train_config["group_by_length"],
        length_column_name="token_count",
        remove_unused_columns=True,
        report_to=[train_config["report_to"]],
        run_name=train_config["swanlab_run_name"],
        seed=train_config["seed"],
        data_seed=train_config["seed"],
        ddp_timeout=180000000,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=PretokenizedCollator(
            pad_token_id=tokenizer.pad_token_id,
            ignore_index=protocol["loss"]["ignore_index"],
        ),
        callbacks=[StopAfterStepCallback(args.stop_after_step)] if args.stop_after_step else None,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if args.stop_after_step is None:
        final_checkpoint = output_dir / f"checkpoint-{trainer.state.global_step}"
        if not final_checkpoint.is_dir():
            trainer._save_checkpoint(trainer.model, trial=None)  # noqa: SLF001 - resumable epoch gate
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))
        metrics = trainer.evaluate()
        (output_dir / "final_eval_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

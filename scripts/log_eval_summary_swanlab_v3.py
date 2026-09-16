#!/usr/bin/env python3
"""Upload one completed R3.0 interactive evaluation summary to SwanLab."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--project", default="search-r1")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if summary.get("protocol_id") != "searchqa_repro_v3_0_0":
        raise SystemExit("refusing to upload a non-R3.0 summary")
    metrics = {}
    for key, value in summary["overall"].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"interactive_dev/overall/{key}"] = value
    for group_name, groups in (
        ("source", summary.get("by_source", {})),
        ("search_bucket", summary.get("by_search_bucket", {})),
    ):
        for name, values in groups.items():
            for key, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[f"interactive_dev/{group_name}/{name}/{key}"] = value
    import swanlab
    swanlab.init(
        project=args.project,
        experiment_name=args.run_name or f"R3.0-interactive-dev-step{args.step}",
        config={
            "protocol_id": summary["protocol_id"],
            "model": summary.get("model"),
            "summary": str(Path(args.summary).resolve()),
            "checkpoint_step": args.step,
        },
    )
    swanlab.log(metrics, step=args.step)
    swanlab.finish()
    print(json.dumps({"uploaded": len(metrics), "step": args.step, "summary": args.summary}))


if __name__ == "__main__":
    main()

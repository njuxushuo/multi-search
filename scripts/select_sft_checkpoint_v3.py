#!/usr/bin/env python3
"""Select an R3.0 SFT checkpoint by interactive-dev behavior, not eval loss."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", nargs="+", help="MODEL_PATH=SUMMARY_JSON")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for item in args.candidates:
        if "=" not in item:
            raise SystemExit(f"candidate must be MODEL_PATH=SUMMARY_JSON: {item}")
        model, summary_path = item.split("=", 1)
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        if summary.get("protocol_id") != "searchqa_repro_v3_0_0":
            raise SystemExit(f"not an R3.0 summary: {summary_path}")
        metrics = summary["overall"]
        rows.append({
            "model": model,
            "summary": summary_path,
            "em": float(metrics["em"]),
            "f1": float(metrics["f1"]),
            "format_compliance_rate": float(metrics["format_compliance_rate"]),
            "mean_no_progress_searches": float(metrics["mean_no_progress_searches"]),
            "search_limit_rate": float(metrics["search_limit_rate"]),
        })
    ranked = sorted(rows, key=lambda row: (
        -row["em"],
        -row["f1"],
        row["mean_no_progress_searches"],
        -row["format_compliance_rate"],
        row["search_limit_rate"],
        row["model"],
    ))
    result = {
        "protocol_id": "searchqa_repro_v3_0_0",
        "selection_rule": [
            "max_em", "max_f1", "min_no_progress_searches",
            "max_format_compliance", "min_search_limit_rate",
        ],
        "selected": ranked[0],
        "ranking": ranked,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

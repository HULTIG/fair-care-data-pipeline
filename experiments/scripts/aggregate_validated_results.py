"""Flatten a validated pilot manifest into a traceable results table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "dataset", "config", "run_id", "score", "readiness_status", "publication_status",
    "roc_auc", "balanced_accuracy", "statistical_parity_difference",
    "equal_opportunity_difference", "disparate_impact", "evaluation_population",
    "rows_input", "rows_retained", "rows_suppressed", "bronze_seconds",
    "silver_seconds", "gold_seconds", "total_seconds",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "validated":
        raise SystemExit("refusing to aggregate a non-validated manifest")

    rows = []
    for record in manifest["runs"]:
        summary = json.loads(Path(record["summary"]).read_text())
        fairness = summary.get("fairness", {})
        utility = summary.get("utility", {})
        privacy = summary.get("privacy", {})
        runtimes = summary.get("runtimes", {})
        anonymization = summary.get("anonymization", {})
        rows.append({
            "dataset": record["dataset"],
            "config": record["config"],
            "run_id": record["run_id"],
            "score": summary.get("score"),
            "readiness_status": summary.get("status"),
            "publication_status": summary.get("publication", {}).get("status"),
            "roc_auc": utility.get("roc_auc"),
            "balanced_accuracy": utility.get("balanced_accuracy"),
            "statistical_parity_difference": fairness.get("statistical_parity_difference"),
            "equal_opportunity_difference": fairness.get("equal_opportunity_difference"),
            "disparate_impact": fairness.get("disparate_impact"),
            "evaluation_population": utility.get("evaluation_population"),
            "rows_input": anonymization.get("rows_input"),
            "rows_retained": anonymization.get("rows_retained"),
            "rows_suppressed": anonymization.get("rows_suppressed"),
            "bronze_seconds": runtimes.get("bronze"),
            "silver_seconds": runtimes.get("silver"),
            "gold_seconds": runtimes.get("gold"),
            "total_seconds": runtimes.get("total"),
        })

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} validated rows to {output}")


if __name__ == "__main__":
    main()

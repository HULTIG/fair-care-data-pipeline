"""Aggregate validated multi-seed campaign metrics and paired differences."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


METRICS = (
    "score", "roc_auc", "balanced_accuracy",
    "statistical_parity_difference", "equal_opportunity_difference",
    "disparate_impact", "rows_retained", "rows_suppressed",
)


def t_critical_975(df: int) -> float:
    """Return the two-sided .975 t critical value without requiring scipy."""
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except (ImportError, AttributeError):
        # Good fallback for the intended n=5 campaign; scipy is available in
        # the project environment, but keeping this script portable is useful.
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
        return table.get(df, 1.96)


def mean_sd_ci(values):
    values = [float(value) for value in values if value is not None]
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "sd": None, "ci95_low": None, "ci95_high": None}
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
    half_width = t_critical_975(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd,
            "ci95_low": mean - half_width, "ci95_high": mean + half_width}


def summary_path(manifest_path: Path, record: dict) -> Path:
    root = manifest_path.parent / record["dataset"] / record["config"] / record["run_id"]
    candidates = sorted(root.glob("*_metricssummary.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected one summary for {record['run_id']}, found {len(candidates)}")
    return candidates[0]


def load_rows(manifest_paths):
    rows = []
    failures = []
    for raw_path in manifest_paths:
        manifest_path = Path(raw_path).resolve()
        payload = json.loads(manifest_path.read_text())
        for run in payload.get("runs", []):
            key = {"dataset": run.get("dataset"), "config": run.get("config"),
                   "seed": int(payload.get("seed", run.get("seed")))}
            if run.get("status") != "succeeded":
                failures.append({**key, "status": run.get("status"), "error": run.get("error")})
                continue
            summary = json.loads(summary_path(manifest_path, run).read_text())
            fairness = summary.get("fairness", {})
            utility = summary.get("utility", {})
            anonymization = summary.get("anonymization", {})
            rows.append({
                **key, "run_id": run.get("run_id"),
                "score": summary.get("score"),
                "roc_auc": utility.get("roc_auc", fairness.get("roc_auc")),
                "balanced_accuracy": utility.get("balanced_accuracy", fairness.get("balanced_accuracy")),
                "statistical_parity_difference": fairness.get("statistical_parity_difference"),
                "equal_opportunity_difference": fairness.get("equal_opportunity_difference"),
                "disparate_impact": fairness.get("disparate_impact"),
                "rows_retained": anonymization.get("rows_retained"),
                "rows_suppressed": anonymization.get("rows_suppressed"),
            })
    return rows, failures


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", required=True, help="Comma-separated per-seed manifests")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline", default="baseline")
    args = parser.parse_args()
    manifest_paths = [item.strip() for item in args.manifests.split(",") if item.strip()]
    rows, failures = load_rows(manifest_paths)
    if not rows:
        raise SystemExit("no successful campaign rows found")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_seed_fields = ["dataset", "config", "seed", "run_id", *METRICS]
    write_csv(output / "per_seed_metrics.csv", per_seed_fields, rows)

    grouped = defaultdict(list)
    for row in rows:
        for metric in METRICS:
            grouped[(row["dataset"], row["config"], metric)].append(row[metric])
    aggregate_rows = []
    for (dataset, config, metric), values in sorted(grouped.items()):
        aggregate_rows.append({"dataset": dataset, "config": config, "metric": metric,
                               **mean_sd_ci(values)})
    write_csv(output / "metric_uncertainty.csv",
              ["dataset", "config", "metric", "n", "mean", "sd", "ci95_low", "ci95_high"],
              aggregate_rows)

    by_key = {(row["dataset"], row["seed"], row["config"]): row for row in rows}
    paired = []
    for (dataset, seed, config), treatment in sorted(by_key.items()):
        if config == args.baseline:
            continue
        control = by_key.get((dataset, seed, args.baseline))
        if control is None:
            failures.append({"dataset": dataset, "config": config, "seed": seed,
                             "status": "unpaired_baseline", "error": args.baseline})
            continue
        for metric in METRICS:
            if treatment[metric] is None or control[metric] is None:
                difference = None
            else:
                difference = float(treatment[metric]) - float(control[metric])
            paired.append({"dataset": dataset, "config": config, "seed": seed,
                           "metric": metric, "difference": difference})
    write_csv(output / "paired_differences.csv",
              ["dataset", "config", "seed", "metric", "difference"], paired)

    paired_groups = defaultdict(list)
    for row in paired:
        paired_groups[(row["dataset"], row["config"], row["metric"])].append(row["difference"])
    paired_summary = []
    for (dataset, config, metric), values in sorted(paired_groups.items()):
        paired_summary.append({"dataset": dataset, "config": config, "metric": metric,
                               **mean_sd_ci(values)})
    write_csv(output / "paired_uncertainty.csv",
              ["dataset", "config", "metric", "n", "mean", "sd", "ci95_low", "ci95_high"],
              paired_summary)

    failure_payload = {
        "successful_run_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "failure_counts_by_status": dict(Counter(item.get("status") for item in failures)),
    }
    (output / "failures.json").write_text(json.dumps(failure_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"successful_runs": len(rows), "failures": len(failures), "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()

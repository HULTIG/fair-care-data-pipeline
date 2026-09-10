"""Validate a multi-seed pilot campaign without conflating seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_pilot_manifests import summary_for, validate_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", required=True, help="Comma-separated per-seed manifest paths")
    parser.add_argument("--datasets", default="compas,adult,german,nij")
    parser.add_argument("--configs", default="baseline,configa,configb,configc,default")
    parser.add_argument("--seeds", required=True, help="Comma-separated expected seeds")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected = {
        (dataset.strip(), config.strip(), int(seed.strip()))
        for dataset in args.datasets.split(",") if dataset.strip()
        for config in args.configs.split(",") if config.strip()
        for seed in args.seeds.split(",") if seed.strip()
    }
    records = {}
    failures = []
    split_hashes = {}
    for raw in args.manifests.split(","):
        manifest_path = Path(raw).resolve()
        payload = json.loads(manifest_path.read_text())
        seed = int(payload["seed"])
        for run in payload.get("runs", []):
            key = (run.get("dataset"), run.get("config"), seed)
            if key not in expected:
                continue
            if run.get("status") != "succeeded":
                failures.append({"key": key, "status": run.get("status"), "error": run.get("error")})
                continue
            if key in records:
                raise SystemExit(f"duplicate campaign run: {key}")
            summary_path = summary_for(manifest_path, run)
            summary = json.loads(summary_path.read_text())
            if summary.get("run_id") != run.get("run_id") or summary.get("evidence_valid") is not True:
                raise SystemExit(f"invalid summary for {key}")
            hashes = validate_summary(summary, manifest_path)
            previous = split_hashes.setdefault(key[0], hashes)
            if previous != hashes:
                raise SystemExit(f"split mismatch across seeds for {key[0]}")
            records[key] = {
                "dataset": key[0], "config": key[1], "seed": key[2],
                "run_id": key and run["run_id"], "summary": str(summary_path),
                "test_membership_sha256": hashes["test_hash"],
                "train_membership_sha256": hashes["train_hash"],
            }

    missing = sorted(expected.difference(records))
    if missing:
        raise SystemExit(f"campaign evidence is incomplete; missing: {missing}")
    result = {
        "status": "validated", "expected_run_count": len(expected),
        "successful_run_count": len(records), "failed_runs": failures,
        "runs": [records[key] for key in sorted(records)],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "runs": len(records), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

"""Validate and consolidate server-pilot manifests into one evidence index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def summary_for(manifest_path: Path, run: dict) -> Path:
    root = manifest_path.parent / run["dataset"] / run["config"] / run["run_id"]
    candidates = sorted(root.glob("*_metricssummary.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected one summary for {run['run_id']}, found {len(candidates)}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", required=True, help="Comma-separated manifest.json paths")
    parser.add_argument("--datasets", default="compas,adult,german,nij")
    parser.add_argument("--configs", default="baseline,configa,configb,configc,default")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected = {(dataset.strip(), config.strip())
                for dataset in args.datasets.split(",") if dataset.strip()
                for config in args.configs.split(",") if config.strip()}
    successful = {}
    source_manifests = []
    for raw_path in args.manifests.split(","):
        path = Path(raw_path).resolve()
        source_manifests.append({"path": str(path), "sha256": sha256_bytes(path.read_bytes())})
        payload = json.loads(path.read_text())
        for run in payload.get("runs", []):
            key = (run.get("dataset"), run.get("config"))
            if key not in expected or run.get("status") != "succeeded":
                continue
            if key in successful:
                raise ValueError(f"duplicate successful run for {key}")
            summary_path = summary_for(path, run)
            summary = json.loads(summary_path.read_text())
            if summary.get("run_id") != run.get("run_id"):
                raise ValueError(f"summary/run ID mismatch for {run.get('run_id')}")
            if summary.get("evidence_valid") is not True:
                raise ValueError(f"run lacks valid evidence: {run.get('run_id')}")
            successful[key] = {
                "dataset": key[0],
                "config": key[1],
                "run_id": run["run_id"],
                "manifest": str(path),
                "summary": str(summary_path.resolve()),
                "publication": summary.get("publication", {}).get("status"),
                "code_revision": payload.get("code", {}).get("revision"),
                "config_checksum": payload.get("base_config_sha256"),
                "resolved_config_sha256": run.get("resolved_config_sha256"),
                "input_data_sha256": run.get("input_data_sha256"),
            }

    missing = sorted(expected.difference(successful))
    if missing:
        raise SystemExit(f"validated evidence is incomplete; missing successful runs: {missing}")

    records = [successful[key] for key in sorted(successful)]
    result = {
        "status": "validated",
        "expected_run_count": len(expected),
        "successful_run_count": len(records),
        "source_manifests": source_manifests,
        "runs": records,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True).encode()
    result["manifest_sha256"] = sha256_bytes(encoded)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "runs": len(records), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

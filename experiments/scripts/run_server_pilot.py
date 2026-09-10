"""Run the one-seed validation gate across all datasets and configurations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pace.orchestration.pipeline import run_pipeline


def deep_merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if key == "datasets":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, default=str).encode())


def revision(root: Path) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"revision": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="compas,adult,german,nij")
    parser.add_argument("--configs", default="baseline,configa,configb,configc,default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", required=True, help="Pilot output directory")
    parser.add_argument("--artifact-root", default="data/processed/server-pilot", help="Root for processed layer artifacts")
    args = parser.parse_args()

    root = ROOT
    base_path = (root / args.config).resolve()
    base = yaml.safe_load(base_path.read_text())
    pilot_id = time.strftime("pilot-%Y%m%dT%H%M%SZ", time.gmtime()) + f"-seed{args.seed}"
    pilot_dir = Path(args.output).resolve() / pilot_id
    pilot_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "pilot_id": pilot_id,
        "seed": args.seed,
        "status": "running",
        "code": revision(root),
        "base_config": str(base_path),
        "base_config_sha256": sha256_file(base_path),
        "packages": package_versions(["pandas", "numpy", "pyspark", "delta-spark", "scikit-learn", "pyyaml"]),
        "runs": [],
    }
    (pilot_dir / "manifest.running.json").write_text(json.dumps(manifest, indent=2))

    for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        for config_name in [item.strip() for item in args.configs.split(",") if item.strip()]:
            config_path = root / "experiments" / "configs" / f"{config_name}.yaml"
            resolved = copy.deepcopy(base)
            if config_name != "default":
                if not config_path.exists():
                    manifest["runs"].append({"dataset": dataset, "config": config_name, "status": "failed", "error": f"missing config: {config_path}"})
                    continue
                resolved = deep_merge(resolved, yaml.safe_load(config_path.read_text()))
            run_id = f"{pilot_id}-{dataset}-{config_name}"
            resolved["run_id"] = run_id
            run_dir = pilot_dir / dataset / config_name
            # Every treatment owns its storage paths. This prevents Delta
            # schema/state collisions when runs are executed sequentially or
            # concurrently on the server.
            layer_root = Path(args.artifact_root) / pilot_id / dataset / config_name
            resolved["datasets"][dataset]["bronze_path"] = str(layer_root / "bronze")
            resolved["datasets"][dataset]["silver_path"] = str(layer_root / "silver")
            resolved["datasets"][dataset]["gold_path"] = str(layer_root / "gold")
            raw_path = root / resolved["datasets"][dataset]["raw_path"]
            record = {
                "dataset": dataset,
                "config": config_name,
                "run_id": run_id,
                "seed": args.seed,
                "status": "running",
                "resolved_config_sha256": sha256_json(resolved),
                "input_data_sha256": sha256_file(raw_path),
            }
            try:
                result = run_pipeline(dataset, resolved, str(run_dir), seed=args.seed)
                record.update({
                    "status": "succeeded",
                    "evidence_valid": result.get("evidence_valid"),
                    "publication": result.get("publication"),
                    "utility_status": result.get("utility", {}).get("status"),
                    "metric_status": result.get("fairness", {}).get("status"),
                })
            except Exception as error:  # the failure artifact is written by the pipeline wrapper
                record.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
            manifest["runs"].append(record)
            (pilot_dir / "manifest.running.json").write_text(json.dumps(manifest, indent=2))

    manifest["status"] = "complete" if all(item["status"] == "succeeded" for item in manifest["runs"]) else "incomplete"
    (pilot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"pilot_id": pilot_id, "status": manifest["status"], "runs": len(manifest["runs"])}, indent=2))


if __name__ == "__main__":
    main()

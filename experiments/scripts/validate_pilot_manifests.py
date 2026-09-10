"""Validate and consolidate server-pilot manifests into one evidence index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def membership_hash(values) -> str:
    return sha256_bytes("\n".join(sorted(str(value) for value in values)).encode())


def close(a, b):
    return a is None and b is None or (a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-8))


def summary_for(manifest_path: Path, run: dict) -> Path:
    root = manifest_path.parent / run["dataset"] / run["config"] / run["run_id"]
    candidates = sorted(root.glob("*_metricssummary.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected one summary for {run['run_id']}, found {len(candidates)}")
    return candidates[0]


def validate_summary(summary: dict, manifest_path: Path, *, expected_seed=None,
                     expected_config_checksum=None, expected_input_checksum=None) -> dict:
    split = summary.get("split", {})
    fit = summary.get("fit", {})
    if not split.get("test_membership_sha256") or not split.get("train_membership_sha256"):
        raise ValueError(f"run lacks split membership hashes: {summary.get('run_id')}")
    if split.get("train_count", 0) + split.get("test_count", 0) != split.get("eligible_count"):
        raise ValueError(f"split counts do not reconcile: {summary.get('run_id')}")
    if summary.get("utility", {}).get("evaluation_population") != split.get("test_count"):
        raise ValueError(f"evaluation population does not match test split: {summary.get('run_id')}")
    prediction_path = Path(summary.get("prediction_artifact", ""))
    if not prediction_path.is_absolute():
        prediction_path = manifest_path.parent / prediction_path
    if not prediction_path.exists():
        raise ValueError(f"missing held-out prediction artifact: {prediction_path}")
    ids = [json.loads(line)["_record_id"] for line in prediction_path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)) or len(ids) != split.get("test_count"):
        raise ValueError(f"held-out prediction identities do not match test split: {summary.get('run_id')}")
    if membership_hash(ids) != split.get("test_membership_sha256"):
        raise ValueError(f"held-out prediction membership does not match split: {summary.get('run_id')}")
    if split.get("test_membership_ids") and set(map(str, ids)) != set(map(str, split["test_membership_ids"])):
        raise ValueError(f"held-out prediction IDs differ from recorded test membership: {summary.get('run_id')}")
    recorded_checksum = summary.get("prediction_artifact_sha256")
    if recorded_checksum != sha256_bytes(prediction_path.read_bytes()):
        raise ValueError(f"held-out prediction checksum mismatch: {summary.get('run_id')}")
    original_train_ids = set(split.get("train_membership_ids", []))
    retained_train_ids = set(fit.get("train_record_ids", []))
    if not original_train_ids or not retained_train_ids.issubset(original_train_ids):
        raise ValueError(f"fitted identities are not a subset of original training identities: {summary.get('run_id')}")
    anonymization = summary.get("anonymization", {})
    if anonymization.get("rows_input") != split.get("train_count"):
        raise ValueError(f"training input count does not match source split: {summary.get('run_id')}")
    if anonymization.get("rows_retained") != len(retained_train_ids):
        raise ValueError(f"retained training count does not match fitted identities: {summary.get('run_id')}")
    if anonymization.get("rows_suppressed") != anonymization.get("rows_input") - anonymization.get("rows_retained"):
        raise ValueError(f"training suppression counts do not reconcile: {summary.get('run_id')}")
    if fit.get("train_record_count") != len(retained_train_ids):
        raise ValueError(f"fit population does not match retained training rows: {summary.get('run_id')}")
    mitigation = summary.get("mitigation", {})
    if mitigation.get("executed"):
        if mitigation.get("weights_consumed") is not True:
            raise ValueError(f"mitigation weights were not consumed: {summary.get('run_id')}")
        if mitigation.get("weight_count") != len(retained_train_ids):
            raise ValueError(f"mitigation weight count does not match training rows: {summary.get('run_id')}")
    provenance = summary.get("provenance", {})
    if not provenance.get("resolved_config_sha256") or not provenance.get("input_data_sha256"):
        raise ValueError(f"run lacks source/config checksums: {summary.get('run_id')}")
    if expected_seed is not None and summary.get("seed") != expected_seed:
        raise ValueError(f"summary seed mismatch: {summary.get('run_id')}")
    if expected_config_checksum is not None and provenance.get("resolved_config_sha256") != expected_config_checksum:
        raise ValueError(f"summary config checksum mismatch: {summary.get('run_id')}")
    if expected_input_checksum is not None and provenance.get("input_data_sha256") != expected_input_checksum:
        raise ValueError(f"summary input checksum mismatch: {summary.get('run_id')}")
    return {"test_hash": split["test_membership_sha256"], "train_hash": split["train_membership_sha256"]}


def reconcile_predictions(summary: dict, manifest_path: Path):
    spec = summary["evaluation"]
    path = Path(summary["prediction_artifact"])
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    outcome, protected = spec["outcome_column"], spec["protected_attribute"]
    favorable = spec["favorable_label"]
    privileged = set(spec.get("privileged_values") or [next(iter(spec["privileged_group"].values()))])
    unprivileged = set(spec.get("unprivileged_values") or [next(iter(spec["unprivileged_group"].values()))])
    y = frame[outcome] == favorable
    pred = frame["prediction"] == favorable
    expected = summary["fairness"]
    p = frame[protected].isin(privileged)
    u = frame[protected].isin(unprivileged)
    for name, mask in (("privileged", p), ("unprivileged", u)):
        group = expected["groups"][name]
        gy, gp = y[mask], pred[mask]
        values = {
            "sample_count": int(mask.sum()),
            "outcome_count": int(gy.sum()),
            "selection_rate": float(gp.mean()) if len(gp) else None,
            "true_positive_rate": float((gy & gp).sum() / gy.sum()) if gy.sum() else None,
            "false_positive_rate": float(((~gy) & gp).sum() / (~gy).sum()) if (~gy).sum() else None,
        }
        for field, value in values.items():
            if not close(value, group.get(field)):
                raise ValueError(f"prediction/group metric mismatch for {name}.{field}: {summary.get('run_id')}")
    expected_counts = expected.get("outcome_counts", {})
    actual_counts = {str(key): int(value) for key, value in frame[outcome].value_counts().items()}
    if expected_counts and actual_counts != expected_counts:
        raise ValueError(f"outcome count mismatch: {summary.get('run_id')}")
    if not close(float(pred.mean()), summary["utility"].get("predicted_favorable_rate")):
        raise ValueError(f"predicted favorable rate mismatch: {summary.get('run_id')}")
    if y.nunique() > 1:
        if not close(float(balanced_accuracy_score(y, pred)), summary["utility"].get("balanced_accuracy")):
            raise ValueError(f"balanced accuracy mismatch: {summary.get('run_id')}")
        if not close(float(roc_auc_score(y, frame["prediction_score"])), summary["utility"].get("roc_auc")):
            raise ValueError(f"ROC AUC mismatch: {summary.get('run_id')}")
    p_rate, u_rate = pred[p].mean(), pred[u].mean()
    if not close(float(u_rate - p_rate), expected.get("demographic_parity_difference")):
        raise ValueError(f"DPD mismatch: {summary.get('run_id')}")
    p_tpr = ((y & pred)[p].sum() / y[p].sum()) if y[p].sum() else None
    u_tpr = ((y & pred)[u].sum() / y[u].sum()) if y[u].sum() else None
    if not close((u_tpr - p_tpr) if u_tpr is not None and p_tpr is not None else None,
                 expected.get("equal_opportunity_difference")):
        raise ValueError(f"EOD mismatch: {summary.get('run_id')}")
    expected_di = expected.get("disparate_impact")
    actual_di = float(u_rate / p_rate) if p_rate else None
    if not close(actual_di, expected_di):
        raise ValueError(f"disparate impact mismatch: {summary.get('run_id')}")
    return True


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
    split_hashes = {}
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
            hashes = validate_summary(
                summary, path, expected_seed=payload.get("seed"),
                expected_config_checksum=run.get("resolved_config_sha256"),
                expected_input_checksum=run.get("input_data_sha256"),
            )
            reconcile_predictions(summary, path)
            previous = split_hashes.setdefault(key[0], hashes)
            if previous != hashes:
                raise ValueError(f"paired split membership mismatch for dataset {key[0]}")
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

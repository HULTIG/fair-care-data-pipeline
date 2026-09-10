import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "scripts"))
from validate_campaign_manifests import record_split_hash
from validate_pilot_manifests import validate_summary


def h(values):
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def summary_fixture(tmp_path, prediction_ids=None, checksum=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prediction_ids = prediction_ids or ["t1", "t2"]
    rows = [
        {"_record_id": prediction_ids[0], "y": 0, "prediction": 0, "prediction_score": 0.1, "group": "A"},
        {"_record_id": prediction_ids[1], "y": 1, "prediction": 1, "prediction_score": 0.9, "group": "B"},
    ]
    prediction = tmp_path / "predictions.jsonl"
    prediction.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {
        "run_id": "run-42", "evidence_valid": True, "seed": 42,
        "split": {"eligible_count": 5, "train_count": 3, "test_count": 2,
                  "train_membership_ids": ["a", "b", "c"],
                  "test_membership_ids": ["t1", "t2"],
                  "train_membership_sha256": h(["a", "b", "c"]),
                  "test_membership_sha256": h(["t1", "t2"])},
        "fit": {"train_record_count": 2, "train_record_ids": ["a", "b"]},
        "anonymization": {"rows_input": 3, "rows_retained": 2, "rows_suppressed": 1},
        "mitigation": {"executed": False},
        "utility": {"evaluation_population": 2},
        "prediction_artifact": str(prediction),
        "prediction_artifact_sha256": checksum or hashlib.sha256(prediction.read_bytes()).hexdigest(),
        "provenance": {"resolved_config_sha256": "config", "input_data_sha256": "input"},
    }
    return summary


def test_validator_accepts_training_suppression_and_rejects_bad_prediction(tmp_path):
    summary = summary_fixture(tmp_path)
    assert validate_summary(summary, tmp_path / "manifest.json")["test_hash"] == h(["t1", "t2"])
    bad = summary_fixture(tmp_path / "bad", prediction_ids=["other", "t2"])
    with pytest.raises(ValueError, match="membership"):
        validate_summary(bad, tmp_path / "bad" / "manifest.json")


def test_validator_rejects_prediction_checksum(tmp_path):
    summary = summary_fixture(tmp_path, checksum="incorrect")
    with pytest.raises(ValueError, match="checksum"):
        validate_summary(summary, tmp_path / "manifest.json")


def test_campaign_split_matching_is_per_dataset_and_seed():
    seen = {}
    first = {"train_hash": h(["a"]), "test_hash": h(["b"])}
    second_seed = {"train_hash": h(["c"]), "test_hash": h(["d"])}
    record_split_hash(seen, "compas", 42, first)
    record_split_hash(seen, "compas", 43, second_seed)
    with pytest.raises(ValueError, match="within dataset/seed"):
        record_split_hash(seen, "compas", 42, second_seed)

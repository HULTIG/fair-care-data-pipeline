import numpy as np
import pandas as pd

from pace.evaluation.splitting import split_source_records
from pace.silver.utilityassessment import UtilityAssessment
from pace.silver.anonymization import AnonymizationEngine


class Frame:
    def __init__(self, value):
        self.value = value

    def toPandas(self):
        return self.value.copy()


def test_source_split_is_reusable_and_disjoint():
    frame = pd.DataFrame({
        "_record_id": [f"r{i}" for i in range(20)],
        "y": [i % 2 for i in range(20)],
    })
    train_a, test_a, meta_a = split_source_records(frame, label_column="y", seed=42)
    train_b, test_b, meta_b = split_source_records(frame, label_column="y", seed=42)
    assert set(train_a._record_id).isdisjoint(test_a._record_id)
    assert list(train_a._record_id) == list(train_b._record_id)
    assert list(test_a._record_id) == list(test_b._record_id)
    assert meta_a["test_membership_sha256"] == meta_b["test_membership_sha256"]


def test_prediction_fit_consumes_aligned_training_weights():
    train = pd.DataFrame({
        "_record_id": ["a", "b", "c", "d"],
        "x": [0, 1, 0, 1], "group": ["A", "A", "B", "B"],
        "y": [0, 1, 0, 1], "instance_weights": [1.0, 2.0, 1.0, 3.0],
    })
    test = pd.DataFrame({
        "_record_id": ["e", "f"], "x": [0, 1], "group": ["A", "B"], "y": [0, 1],
    })
    assessment = UtilityAssessment({
        "label_column": "y", "protected_attribute": "group",
        "favorable_label": 1, "predictor_allowlist": ["x"],
    })
    result = assessment.predict_held_out(Frame(train), Frame(test), seed=42)
    assert list(result["_record_id"]) == ["e", "f"]
    assert assessment.last_fit["weights_consumed"] is True
    assert assessment.last_fit["weight_sum"] == 7.0


def test_numeric_noise_is_deterministic_for_a_seed():
    frame = pd.DataFrame({"_record_id": ["a", "b"], "x": [1.0, 2.0], "y": [0, 1]})
    config = {"technique": "numeric_noise", "epsilon": 1.0,
              "label_column": "y", "protected_attribute": "group", "seed": 17}
    # The protected column is only required by the pipeline configuration;
    # direct noise fitting can use a frame without it.
    first = AnonymizationEngine(config)
    first.fit(frame)
    a = first._apply_differential_privacy(frame.copy())
    second = AnonymizationEngine(config)
    second.fit(frame)
    b = second._apply_differential_privacy(frame.copy())
    assert a["x"].tolist() == b["x"].tolist()

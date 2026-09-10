import pandas as pd
import pytest
import pandas as pd

from pace.evaluation.contract import EvaluationContract, EvaluationError


def config():
    return {
        "protected_attribute": "group",
        "privileged_groups": [{"group": "A"}],
        "unprivileged_groups": [{"group": "B"}],
        "favorable_label": 1,
    }


def test_metrics_share_prediction_population():
    frame = pd.DataFrame({
        "_record_id": range(8),
        "y": [1, 1, 0, 0, 1, 0, 0, 1],
        "prediction": [1, 0, 0, 0, 1, 1, 0, 0],
        "score": [.9, .4, .2, .1, .8, .7, .3, .2],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
    })
    report = EvaluationContract(config()).evaluate(
        frame, outcome_column="y", prediction_column="prediction", score_column="score",
        protected_attribute="group", privileged_group={"group": "A"},
        unprivileged_group={"group": "B"}, favorable_label=1,
    )
    assert report["record_count"] == 8
    assert report["groups"]["privileged"]["sample_count"] == 4
    assert report["demographic_parity_difference"] == pytest.approx(0.25)
    assert report["prediction_source"] == "held_out_predictions"


def test_missing_group_is_unavailable():
    frame = pd.DataFrame({"_record_id": [1, 2], "y": [0, 1], "prediction": [0, 1], "group": ["A", "A"]})
    report = EvaluationContract(config()).evaluate(
        frame, outcome_column="y", prediction_column="prediction",
        protected_attribute="group", privileged_group={"group": "A"},
        unprivileged_group={"group": "B"}, favorable_label=1,
    )
    assert report["status"] == "unavailable"
    assert report["demographic_parity_difference"] is None


def test_duplicate_identity_is_rejected():
    frame = pd.DataFrame({"_record_id": [1, 1], "y": [0, 1], "prediction": [0, 1], "group": ["A", "B"]})
    with pytest.raises(EvaluationError):
        EvaluationContract(config()).evaluate(
            frame, outcome_column="y", prediction_column="prediction",
            protected_attribute="group", privileged_group={"group": "A"},
            unprivileged_group={"group": "B"}, favorable_label=1,
        )


def test_constant_predictions_keep_balanced_accuracy_defined():
    frame = pd.DataFrame({
        "_record_id": range(4), "y": [0, 1, 0, 1],
        "prediction": [0, 0, 0, 0], "group": ["A", "A", "B", "B"],
    })
    report = EvaluationContract(config()).evaluate(
        frame, outcome_column="y", prediction_column="prediction",
        protected_attribute="group", privileged_group={"group": "A"},
        unprivileged_group={"group": "B"}, favorable_label=1,
    )
    assert report["balanced_accuracy"] == pytest.approx(0.5)

"""The single contract used to calculate utility and prediction-based fairness.

The contract deliberately keeps record alignment and metric validity explicit.  A
caller supplies predictions for a common held-out population; this module never
fits a second model while calculating fairness and never turns undefined values
into favourable defaults.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


class EvaluationError(ValueError):
    """Raised when an evaluation contract cannot be satisfied."""


def _finite(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


class EvaluationContract:
    """Calculate all reported metrics over one aligned prediction frame."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)

    def evaluate(
        self,
        frame: pd.DataFrame,
        *,
        outcome_column: str,
        prediction_column: str,
        score_column: Optional[str] = None,
        protected_attribute: str,
        privileged_group: Mapping[str, Any],
        unprivileged_group: Mapping[str, Any],
        favorable_label: Any,
        record_id_column: str = "_record_id",
    ) -> Dict[str, Any]:
        """Return utility, group rates, and fairness from the supplied predictions.

        The method requires the outcome, prediction, protected attribute, and
        internal record id to be present on every evaluated row.  The id is
        retained only for validation and is never used as a predictor.
        """
        required = {outcome_column, prediction_column, protected_attribute, record_id_column}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise EvaluationError(f"evaluation frame is missing required columns: {missing}")
        if frame[record_id_column].duplicated().any():
            raise EvaluationError("evaluation frame contains duplicate internal record identities")

        work = frame[[record_id_column, outcome_column, prediction_column, protected_attribute] +
                     ([score_column] if score_column else [])].copy()
        work = work.dropna(subset=[outcome_column, prediction_column, protected_attribute])
        if work.empty:
            return self._unavailable("no complete held-out records")

        privileged_value = next(iter(privileged_group.values()), None)
        unprivileged_value = next(iter(unprivileged_group.values()), None)
        privileged_values = self.config.get("privileged_values", [privileged_value])
        unprivileged_values = self.config.get("unprivileged_values", [unprivileged_value])
        groups = {
            "privileged": work[protected_attribute].isin(privileged_values),
            "unprivileged": work[protected_attribute].isin(unprivileged_values),
        }
        if not groups["privileged"].any() or not groups["unprivileged"].any():
            return self._unavailable("required comparison group is absent", work, groups)

        y_true = work[outcome_column]
        y_pred = work[prediction_column]
        favorable = y_true == favorable_label
        predicted_favorable = y_pred == favorable_label
        report: Dict[str, Any] = {
            "status": "valid",
            "record_count": int(len(work)),
            "outcome_counts": {str(k): int(v) for k, v in y_true.value_counts(dropna=False).items()},
            "prediction_source": "held_out_predictions",
        }

        if score_column and score_column in work and y_true.nunique() > 1:
            try:
                report["roc_auc"] = float(roc_auc_score(favorable.astype(int), work[score_column]))
            except (ValueError, TypeError):
                report["roc_auc"] = None
                report["roc_auc_reason"] = "score values do not support binary ROC AUC"
        else:
            report["roc_auc"] = None
            report["roc_auc_reason"] = "no held-out score column or only one outcome class"

        if y_true.nunique() > 1 and y_pred.nunique() > 1:
            report["balanced_accuracy"] = float(balanced_accuracy_score(favorable, predicted_favorable))
        else:
            report["balanced_accuracy"] = None
            report["balanced_accuracy_reason"] = "outcome or predictions contain one class"

        report["predicted_favorable_rate"] = float(predicted_favorable.mean())
        group_report = {}
        for name, mask in groups.items():
            gy = favorable[mask]
            gp = predicted_favorable[mask]
            tp = int((gy & gp).sum())
            fp = int((~gy & gp).sum())
            group_report[name] = {
                "sample_count": int(mask.sum()),
                "outcome_count": int(gy.sum()),
                "non_outcome_count": int((~gy).sum()),
                "selection_rate": _rate(int(gp.sum()), len(gp)),
                "true_positive_rate": _rate(tp, int(gy.sum())),
                "false_positive_rate": _rate(fp, int((~gy).sum())),
            }
        report["groups"] = group_report

        p_rate = group_report["privileged"]["selection_rate"]
        u_rate = group_report["unprivileged"]["selection_rate"]
        p_tpr = group_report["privileged"]["true_positive_rate"]
        u_tpr = group_report["unprivileged"]["true_positive_rate"]
        report["demographic_parity_difference"] = u_rate - p_rate if u_rate is not None and p_rate is not None else None
        report["equal_opportunity_difference"] = u_tpr - p_tpr if u_tpr is not None and p_tpr is not None else None
        report["disparate_impact"] = u_rate / p_rate if u_rate is not None and p_rate else None
        for metric, value, reason in (
            ("demographic_parity_difference", report["demographic_parity_difference"], "selection rate is undefined"),
            ("equal_opportunity_difference", report["equal_opportunity_difference"], "a group has no favorable outcomes"),
            ("disparate_impact", report["disparate_impact"], "privileged selection rate is zero or undefined"),
        ):
            if value is None:
                report[f"{metric}_reason"] = reason
        return report

    @staticmethod
    def _unavailable(reason: str, frame: Optional[pd.DataFrame] = None, groups: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        report: Dict[str, Any] = {"status": "unavailable", "reason": reason, "prediction_source": "held_out_predictions"}
        if frame is not None:
            report["record_count"] = int(len(frame))
        if groups is not None:
            report["groups"] = {name: {"sample_count": int(mask.sum())} for name, mask in groups.items()}
        for key in ("roc_auc", "balanced_accuracy", "predicted_favorable_rate", "demographic_parity_difference",
                    "equal_opportunity_difference", "disparate_impact"):
            report[key] = None
        return report

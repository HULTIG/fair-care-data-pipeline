"""Prediction-based fairness reporting."""

from pyspark.sql import DataFrame

from pace.evaluation.contract import EvaluationContract, EvaluationError


class FairnessMetrics:
    def __init__(self, config: dict):
        self.config = config

    def calculate(self, df: DataFrame, prediction_column: str = "prediction", score_column: str = "prediction_score") -> dict:
        pdf = df.toPandas()
        label = self.config.get("label_column")
        protected = self.config.get("protected_attribute")
        if label not in pdf.columns or protected not in pdf.columns:
            return self._unavailable("required outcome or protected attribute is missing")

        # Direct legacy callers may pass an observed-label frame. Pipeline runs
        # supply a real held-out prediction column and are marked accordingly.
        if prediction_column not in pdf.columns:
            if not self.config.get("allow_observed_label_prediction", True):
                return self._unavailable(f"prediction column {prediction_column!r} is missing")
            pdf[prediction_column] = pdf[label]
            prediction_source = "observed_label_compatibility"
        else:
            prediction_source = "held_out_predictions"

        for column in (label, protected, prediction_column):
            if pdf[column].dtype == "object":
                pdf[column] = pdf[column].astype(str).str.strip()
        if "_record_id" not in pdf.columns:
            pdf["_record_id"] = range(len(pdf))
        try:
            report = EvaluationContract(self.config).evaluate(
                pdf,
                outcome_column=label,
                prediction_column=prediction_column,
                score_column=score_column if score_column in pdf.columns else None,
                protected_attribute=protected,
                privileged_group=self.config.get("privileged_groups", [{}])[0],
                unprivileged_group=self.config.get("unprivileged_groups", [{}])[0],
                favorable_label=self.config.get("favorable_label", 1),
            )
            report["prediction_source"] = prediction_source
            report["statistical_parity_difference"] = report.get("demographic_parity_difference")
            if report.get("status") == "unavailable":
                report.setdefault("error", report.get("reason", "fairness metrics unavailable"))
            return report
        except (EvaluationError, IndexError, KeyError) as exc:
            return self._unavailable(str(exc))

    @staticmethod
    def _unavailable(reason: str) -> dict:
        return {
            "status": "unavailable", "error": reason, "reason": reason,
            "statistical_parity_difference": None,
            "demographic_parity_difference": None,
            "equal_opportunity_difference": None,
            "disparate_impact": None, "roc_auc": None,
            "balanced_accuracy": None,
        }

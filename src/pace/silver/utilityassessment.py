import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from sklearn.linear_model import LogisticRegression

class UtilityAssessment:
    def __init__(self, config: dict):
        self.config = config

    def assess(self, original_df: DataFrame, anonymized_df: DataFrame) -> dict:
        """
        Assesses the utility of the anonymized data compared to the original.
        """
        print("Running Utility Assessment...")
        
        # Sample for efficiency
        orig_pdf = original_df.limit(10000).toPandas()
        anon_pdf = anonymized_df.limit(10000).toPandas()
        
        report = {}
        
        # 1. Correlation Preservation (Numeric only)
        orig_corr = orig_pdf.select_dtypes(include=[np.number]).corr()
        anon_corr = anon_pdf.select_dtypes(include=[np.number]).corr()
        
        if not orig_corr.empty and not anon_corr.empty:
            # Frobenius norm of the difference
            diff = orig_corr - anon_corr
            frobenius_norm = np.linalg.norm(diff.fillna(0))
            report["correlation_distance"] = float(frobenius_norm)
        else:
            report["correlation_distance"] = 0.0

        # Predictive utility requires explicit source train/test partitions.
        # Keep this legacy comparison clearly unavailable so it cannot create a
        # second, incomparable split.
        report["status"] = "unavailable"
        report["reason"] = "legacy assess() requires explicit train/test partitions; use predict_held_out()"
        report["original_auc"] = None
        report["anonymized_auc"] = None
        report["utility_retention"] = None
        
        print(f"Utility Assessment complete: {report}")
        return report

    def predict_held_out(self, train_df: DataFrame, test_df: DataFrame, seed: int = 42):
        """Fit on the supplied training partition and predict the supplied test partition."""
        train_pdf = train_df.toPandas().copy()
        test_pdf = test_df.toPandas().copy()
        label_col = self.config.get("label_column")
        protected = self.config.get("protected_attribute")
        if label_col not in train_pdf.columns or label_col not in test_pdf.columns:
            raise ValueError(f"missing outcome column {label_col!r}")
        for frame_name, frame in (("training", train_pdf), ("test", test_pdf)):
            if "_record_id" not in frame.columns:
                raise ValueError(f"{frame_name} frame is missing _record_id")
            if protected not in frame.columns:
                raise ValueError(f"{frame_name} frame is missing protected attribute {protected!r}")
        if train_pdf["_record_id"].duplicated().any() or test_pdf["_record_id"].duplicated().any():
            raise ValueError("training or test frame contains duplicate _record_id values")
        if set(train_pdf["_record_id"]).intersection(test_pdf["_record_id"]):
            raise ValueError("training and test frames overlap by _record_id")
        train_pdf = train_pdf.dropna(subset=[label_col])
        test_pdf = test_pdf.dropna(subset=[label_col])
        if train_pdf[label_col].nunique() < 2:
            raise ValueError("training fit requires two outcome classes")
        if test_pdf.empty:
            raise ValueError("test evaluation requires at least one complete record")
        predictors = self.config.get("predictor_allowlist")
        if predictors:
            missing = sorted(set(predictors).difference(train_pdf.columns).union(set(predictors).difference(test_pdf.columns)))
            if missing:
                raise ValueError(f"required predictors are missing: {missing}")
            predictors = [c for c in predictors if c != label_col]
        else:
            predictors = [c for c in train_pdf.columns if c != label_col and not c.startswith("_")]
        predictors = [c for c in predictors if c != "instance_weights"]
        train = train_pdf[predictors].copy()
        test = test_pdf[predictors].copy()
        train = pd.get_dummies(train, dummy_na=True)
        test = pd.get_dummies(test, dummy_na=True)
        train = train.loc[:, ~train.columns.duplicated()]
        test = test.loc[:, ~test.columns.duplicated()].reindex(columns=train.columns, fill_value=0)
        train = train.apply(pd.to_numeric, errors="coerce").fillna(0)
        test = test.apply(pd.to_numeric, errors="coerce").fillna(0)
        favorable = self.config.get("favorable_label", 1)
        y_train = self._encode_labels(train_pdf[label_col], favorable)
        weights = None
        if "instance_weights" in train_pdf.columns:
            weights = pd.to_numeric(train_pdf["instance_weights"], errors="coerce").to_numpy()
            if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("training weights must be finite, nonnegative, and have positive total mass")
        model = LogisticRegression(max_iter=500, random_state=seed)
        model.fit(train, y_train, sample_weight=weights)
        scores = model.predict_proba(test)[:, 1]
        predictions = (scores >= 0.5).astype(int)
        observed = list(train_pdf[label_col].dropna().unique())
        unfavorable = next((value for value in observed if value != favorable), None)
        if unfavorable is None:
            raise ValueError("training labels do not contain an unfavorable outcome")
        held_out = test_pdf[["_record_id", label_col, protected]].copy()
        held_out["prediction"] = [favorable if value else unfavorable for value in predictions]
        held_out["prediction_score"] = scores
        self.last_fit = {
            "train_record_count": int(len(train_pdf)),
            "test_record_count": int(len(test_pdf)),
            "predictors": predictors,
            "train_record_ids": sorted(map(str, train_pdf["_record_id"])),
            "weights_consumed": weights is not None,
            "weight_count": int(len(weights)) if weights is not None else 0,
            "weight_min": float(weights.min()) if weights is not None else None,
            "weight_max": float(weights.max()) if weights is not None else None,
            "weight_sum": float(weights.sum()) if weights is not None else None,
        }
        return held_out.reset_index(drop=True)

    @staticmethod
    def _encode_labels(values, favorable):
        return (values == favorable).astype(int).to_numpy()

    def _train_eval(self, df: pd.DataFrame, label_col: str) -> float:
        raise ValueError("_train_eval is deprecated; use predict_held_out(train_df, test_df)")

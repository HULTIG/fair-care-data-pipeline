import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

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

        # 2. Predictive Utility
        label_col = self.config.get("label_column")
        if label_col and label_col in orig_pdf.columns and label_col in anon_pdf.columns:
            try:
                score_orig = self._train_eval(orig_pdf, label_col)
                score_anon = self._train_eval(anon_pdf, label_col)
                
                report["original_auc"] = score_orig
                report["anonymized_auc"] = score_anon
                
                if score_orig <= 0:
                    raise ValueError("baseline ROC AUC is not positive; retention is undefined")
                # Retention is a ratio, never an AUC-like score.
                report["utility_retention"] = float(score_anon / score_orig)
                report["status"] = "valid"
            except Exception as e:
                print(f"Predictive utility check failed: {e}")
                report["status"] = "unavailable"
                report["error"] = str(e)
                report["original_auc"] = None
                report["anonymized_auc"] = None
                report["utility_retention"] = None
        
        print(f"Utility Assessment complete: {report}")
        return report

    def predict_held_out(self, df: DataFrame, seed: int = 42):
        """Fit once on training rows and return predictions for held-out rows.

        All downstream utility and fairness metrics must consume this returned
        frame. Categorical encodings are learned from the training partition.
        """
        pdf = df.toPandas().copy()
        label_col = self.config.get("label_column")
        if label_col not in pdf.columns:
            raise ValueError(f"missing outcome column {label_col!r}")
        if "_record_id" not in pdf.columns:
            pdf["_record_id"] = range(len(pdf))
        pdf = pdf.dropna(subset=[label_col])
        if pdf[label_col].nunique() < 2:
            raise ValueError("held-out evaluation requires two outcome classes")
        train_idx, test_idx = train_test_split(pdf.index, test_size=0.3, random_state=seed, stratify=pdf[label_col])
        predictors = self.config.get("predictor_allowlist")
        if predictors:
            predictors = [c for c in predictors if c in pdf.columns and c != label_col]
        else:
            predictors = [c for c in pdf.columns if c != label_col and not c.startswith("_")]
        train = pdf.loc[train_idx, predictors].copy()
        test = pdf.loc[test_idx, predictors].copy()
        train = pd.get_dummies(train, dummy_na=True)
        test = pd.get_dummies(test, dummy_na=True).reindex(columns=train.columns, fill_value=0)
        train = train.apply(pd.to_numeric, errors="coerce").fillna(0)
        test = test.apply(pd.to_numeric, errors="coerce").fillna(0)
        y_train = self._encode_labels(pdf.loc[train_idx, label_col], self.config.get("favorable_label", 1))
        y_test = self._encode_labels(pdf.loc[test_idx, label_col], self.config.get("favorable_label", 1))
        model = LogisticRegression(max_iter=500, random_state=seed)
        model.fit(train, y_train)
        scores = model.predict_proba(test)[:, 1]
        predictions = (scores >= 0.5).astype(int)
        held_out = pdf.loc[test_idx, ["_record_id", label_col] + [self.config["protected_attribute"]]].copy()
        favorable = self.config.get("favorable_label", 1)
        held_out["prediction"] = [favorable if value else next(v for v in pdf[label_col].unique() if v != favorable) for value in predictions]
        held_out["prediction_score"] = scores
        return held_out.reset_index(drop=True)

    @staticmethod
    def _encode_labels(values, favorable):
        return (values == favorable).astype(int).to_numpy()

    def _train_eval(self, df: pd.DataFrame, label_col: str) -> float:
        # Simple preprocessing
        df = df.copy().dropna()
        if df.empty:
            raise ValueError("cannot calculate ROC AUC from an empty evaluation frame")
        
        y = df[label_col]
        X = df.drop(columns=[label_col])
        
        # Encode categoricals
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))
            
        # Exclude datetime and internal columns
        X = X.select_dtypes(exclude=['datetime', 'timedelta'])
        cols_to_drop = [c for c in X.columns if c.startswith("_")]
        X = X.drop(columns=cols_to_drop, errors='ignore')
            
        if y.nunique() < 2:
            raise ValueError("ROC AUC requires two observed outcome classes")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.config.get("seed", 42), stratify=y
        )
        
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)[:, 1]
        if len(set(y_test)) < 2:
            raise ValueError("held-out outcome has one class; ROC AUC is undefined")
        return float(roc_auc_score(y_test, probs))

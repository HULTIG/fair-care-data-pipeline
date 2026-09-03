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
                
                # Handle edge case where original model fails
                if score_orig <= 0.5:
                    # Original model failed or is random, use anonymized score directly
                    report["utility_retention"] = score_anon if score_anon > 0.5 else 0.5
                else:
                    report["utility_retention"] = score_anon / score_orig
            except Exception as e:
                print(f"Predictive utility check failed: {e}")
                report["original_auc"] = 0.5
                report["anonymized_auc"] = 0.5
                report["utility_retention"] = 0.5  # Default to neutral
        
        print(f"Utility Assessment complete: {report}")
        return report

    def _train_eval(self, df: pd.DataFrame, label_col: str) -> float:
        # Simple preprocessing
        df = df.copy().dropna()
        if df.empty: return 0.0
        
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
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        
        try:
            probs = model.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, probs)
        except:
            return 0.5 # Fallback

import pandas as pd
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from diffprivlib.mechanisms import Laplace

class AnonymizationEngine:
    def __init__(self, config: dict):
        self.config = config

    def anonymize(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        """
        Anonymizes the input DataFrame based on configuration.
        """
        print("Running Anonymization...")
        pdf = df.toPandas()
        
        technique = self.config.get("technique", "kanonymity").lower()
        supported = {"none", "kanonymity", "ldiversity", "tcloseness", "numeric_noise", "differentialprivacy"}
        if technique not in supported:
            raise ValueError(f"unsupported anonymization technique: {technique}")
        if technique in {"ldiversity", "tcloseness"} and not self.config.get("sensitive_attributes"):
            raise ValueError(f"{technique} requires at least one sensitive attribute")
        qis = self.config.get("quasi_identifiers", [])
        missing_qis = sorted(set(qis).difference(pdf.columns))
        if technique in {"kanonymity", "ldiversity", "tcloseness"} and missing_qis:
            raise ValueError(f"configured quasi-identifiers are missing: {missing_qis}")
        if technique in {"kanonymity", "ldiversity", "tcloseness"} and self.config.get("k", 0) < 1:
            raise ValueError("k must be a positive integer")
        sensitive = self.config.get("sensitive_attributes", [])
        missing_sensitive = sorted(set(sensitive).difference(pdf.columns))
        if technique in {"ldiversity", "tcloseness"} and missing_sensitive:
            raise ValueError(f"configured sensitive attributes are missing: {missing_sensitive}")
        if technique == "ldiversity" and self.config.get("l", 0) < 1:
            raise ValueError("l-diversity threshold l must be positive")
        if technique == "tcloseness" and not 0 <= self.config.get("t", -1) <= 1:
            raise ValueError("t-closeness threshold t must be between zero and one")
        if technique in {"numeric_noise", "differentialprivacy"} and self.config.get("epsilon", 0) <= 0:
            raise ValueError("numeric-noise epsilon must be positive")
        
        if technique == "kanonymity":
            pdf = self._apply_kanonymity(pdf)
        elif technique == "ldiversity":
            pdf = self._apply_ldiversity(pdf)
        elif technique == "tcloseness":
            pdf = self._apply_tcloseness(pdf)
        elif technique in {"numeric_noise", "differentialprivacy"}:
            pdf = self._apply_differential_privacy(pdf)
            
        # Convert back to Spark DataFrame
        # Handle potential type issues
        
        # 1. Drop columns that are entirely NaN/None
        pdf = pdf.dropna(axis="columns", how="all")
        
        # 2. Force object columns to string to avoid inference errors for mixed types
        for col in pdf.columns:
            if pdf[col].dtype == "object":
                pdf[col] = pdf[col].astype(str)
                
        # Calculate privacy metrics
        metadata = {
            "technique": "numeric_noise" if technique == "differentialprivacy" else technique,
            "rows_input": int(len(df.toPandas())),
            "rows_retained": int(len(pdf)),
            "rows_suppressed": int(len(df.toPandas()) - len(pdf)),
            "privacy_guarantee": None,
            # Retained for schema compatibility; no heuristic risk value is
            # reported because this implementation does not validate one.
            "risk": None,
        }
        
        if pdf.empty:
            raise ValueError(f"{technique} produced an empty release")

        return spark.createDataFrame(pdf), metadata

    def _calculate_risk(self, df: pd.DataFrame, technique: str) -> float:
        """
        Calculates heuristic re-identification risk.
        """
        if technique in {"numeric_noise", "differentialprivacy"}:
            # Numeric perturbation is a diagnostic treatment; it is not a
            # formal differential-privacy guarantee.
            return None

        # For k-anonymity based methods
        qis = self.config.get("quasi_identifiers", [])
        if not qis or qis[0] not in df.columns:
            return 1.0 # High risk if no QIs defined/found
            
        groups = df.groupby(qis)
        min_group_size = groups.size().min()
        
        if np.isnan(min_group_size): return 1.0
        
        # Risk is probability of re-identification for worst-case record
        return 1.0 / max(1, min_group_size)

    def _apply_kanonymity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies basic k-anonymity by generalizing quasi-identifiers.
        This is a simplified implementation.
        """
        k = self.config.get("k", 5)
        qis = self.config.get("quasi_identifiers", [])
        
        print(f"Applying k-anonymity (k={k}) on {qis}...")
        
        # Simple generalization: binning numeric values, masking strings
        for col in qis:
            if col not in df.columns: continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Binning
                try:
                    df[col] = pd.cut(df[col], bins=10).astype(str)
                except:
                    pass
            else:
                # Masking last characters
                df[col] = df[col].astype(str).apply(lambda x: x[:-3] + "***" if len(x) > 3 else "*")
        
        # Suppression: Remove groups with size < k
        if qis:
            groups = df.groupby(qis)
            df = groups.filter(lambda x: len(x) >= k)
            
        print(f"Rows remaining after k-anonymity suppression: {len(df)}")
        return df

    def _apply_differential_privacy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Differential Privacy using Laplace mechanism on numeric columns.
        Excludes label column and protected attributes from noise addition.
        """
        epsilon = self.config.get("epsilon", 1.0)
        label_col = self.config.get("label_column")
        protected_attr = self.config.get("protected_attribute")
        
        print(f"Applying Differential Privacy (epsilon={epsilon})...")
        
        # Columns to exclude from DP
        exclude_cols = set()
        if label_col:
            exclude_cols.add(label_col)
        if protected_attr:
            exclude_cols.add(protected_attr)
        
        mech = Laplace(epsilon=epsilon, sensitivity=1)
        
        for col in df.columns:
            # Skip internal columns, label, and protected attribute
            if col.startswith("_"): 
                continue
            if col in exclude_cols:
                print(f"  Skipping {col} (label/protected attribute)")
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].apply(mech.randomise)
                
        return df
    
    def _apply_ldiversity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies l-diversity: ensures each equivalence class has at least l distinct values
        for sensitive attributes.
        """
        k = self.config.get("k", 5)
        l = self.config.get("l", 2)
        qis = self.config.get("quasi_identifiers", [])
        sensitive_attrs = self.config.get("sensitive_attributes", [])
        
        print(f"Applying l-diversity (k={k}, l={l}) on QIs={qis}, sensitive={sensitive_attrs}...")
        
        # First apply k-anonymity generalization
        for col in qis:
            if col not in df.columns: 
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.cut(df[col], bins=10).astype(str)
                except:
                    pass
            else:
                df[col] = df[col].astype(str).apply(lambda x: x[:-3] + "***" if len(x) > 3 else "*")
        
        # Suppression based on both k-anonymity and l-diversity
        if qis and sensitive_attrs:
            groups = df.groupby(qis)
            
            def check_ldiversity(group):
                # Check k-anonymity
                if len(group) < k:
                    return False
                # Check l-diversity for each sensitive attribute
                for attr in sensitive_attrs:
                    if attr in group.columns:
                        distinct_values = group[attr].nunique()
                        if distinct_values < l:
                            return False
                return True
            
            df = groups.filter(check_ldiversity)
        
        print(f"Rows remaining after l-diversity suppression: {len(df)}")
        return df
    
    def _apply_tcloseness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies t-closeness: ensures the distribution of sensitive attributes in each
        equivalence class is within distance t of the overall distribution.
        """
        k = self.config.get("k", 5)
        t = self.config.get("t", 0.2)
        qis = self.config.get("quasi_identifiers", [])
        sensitive_attrs = self.config.get("sensitive_attributes", [])
        
        print(f"Applying t-closeness (k={k}, t={t}) on QIs={qis}, sensitive={sensitive_attrs}...")
        
        # First apply k-anonymity generalization
        for col in qis:
            if col not in df.columns:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.cut(df[col], bins=10).astype(str)
                except:
                    pass
            else:
                df[col] = df[col].astype(str).apply(lambda x: x[:-3] + "***" if len(x) > 3 else "*")
        
        # Calculate global distributions for sensitive attributes
        global_dists = {}
        for attr in sensitive_attrs:
            if attr in df.columns:
                global_dists[attr] = df[attr].value_counts(normalize=True)
        
        # Suppression based on k-anonymity and t-closeness
        if qis and sensitive_attrs and global_dists:
            groups = df.groupby(qis)
            
            def check_tcloseness(group):
                # Check k-anonymity
                if len(group) < k:
                    return False
                
                # Check t-closeness for each sensitive attribute
                for attr in sensitive_attrs:
                    if attr not in group.columns or attr not in global_dists:
                        continue
                    
                    # Calculate local distribution
                    local_dist = group[attr].value_counts(normalize=True)
                    
                    # Calculate Earth Mover's Distance (simplified as total variation distance)
                    all_values = set(global_dists[attr].index) | set(local_dist.index)
                    distance = 0
                    for val in all_values:
                        global_prob = global_dists[attr].get(val, 0)
                        local_prob = local_dist.get(val, 0)
                        distance += abs(global_prob - local_prob)
                    
                    distance = distance / 2  # Total variation distance
                    
                    if distance > t:
                        return False
                
                return True
            
            df = groups.filter(check_tcloseness)
        
        print(f"Rows remaining after t-closeness suppression: {len(df)}")
        return df

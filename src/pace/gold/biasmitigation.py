import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

class BiasMitigator:
    def __init__(self, config: dict):
        self.config = config

    def mitigate(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        """
        Mitigates bias using Reweighing.
        """
        print("Running Bias Mitigation...")
        pdf = df.toPandas()
        
        prot_attr = self.config.get("protected_attribute")
        label_col = self.config.get("label_column")
        priv_group = self.config.get("privileged_groups", [{}])[0]
        unpriv_group = self.config.get("unprivileged_groups", [{}])[0]
        
        if not (prot_attr and label_col):
            print("Missing config for bias mitigation. Skipping.")
            return df

        # Debug: Check columns
        print(f"BiasMitigation Input Columns: {pdf.columns.tolist()}")
        if prot_attr not in pdf.columns:
            print(f"ERROR: Protected attribute {prot_attr} not found in columns.")
        if label_col not in pdf.columns:
            print(f"ERROR: Label column {label_col} not found in columns.")

        # Prepare data for AIF360
        # Need to ensure numeric encoding for AIF360
        try:
            # Handle '?' as NaN (common in Adult)
            import numpy as np
            pdf = pdf.replace('?', np.nan)
            pdf = pdf.replace(' ?', np.nan) 
            
            # Impute NAs
            for col in pdf.select_dtypes(include=[np.number]).columns:
                 pdf[col] = pdf[col].fillna(0)
            for col in pdf.select_dtypes(include=['object', 'category']).columns:
                pdf[col] = pdf[col].fillna("Unknown")
            
            # Drop any lingering NAs/Infs
            pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Drop datetime/timestamp columns to avoid casting errors (COMPAS fix)
            pdf = pdf.select_dtypes(exclude=['datetime', 'timedelta', 'datetimetz'])
            
            # Strip whitespace from ALL string columns (Adult fix for ' >50K')
            df_obj = pdf.select_dtypes(['object'])
            pdf[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

            # ENCODE CATEGORICALS (Critical for AIF360)
            from sklearn.preprocessing import LabelEncoder
            label_encoders = {}
            for col in pdf.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                pdf[col] = le.fit_transform(pdf[col].astype(str))
                label_encoders[col] = le
            
            # Map Config values to Encoded values
            favorable_label = self.config.get("favorable_label", 1)
            unfavorable_label = 0 if favorable_label == 1 else 1
            
            # If label string encoded, map favorable label
            if label_col in label_encoders:
                classes = label_encoders[label_col].classes_
                if str(favorable_label) in classes:
                    favorable_label = int(label_encoders[label_col].transform([str(favorable_label)])[0])
                    # Assuming binary, unfavorable is the other one
                    unfavorable_label = 1 - favorable_label
            
            # Handle privileged group encoding if needed
            # AIF360 group dict format: [{'sex': 1}]
            priv_group_encoded = priv_group.copy()
            for k, v in priv_group.items():
                if k in label_encoders:
                    # Try to map value, fallback to v if not found (might be float vs int issue)
                    try:
                        priv_group_encoded[k] = int(label_encoders[k].transform([str(v)])[0])
                    except:
                         pass # Keep original if mapping fails

            
            dataset = BinaryLabelDataset(
                favorable_label=favorable_label,
                unfavorable_label=unfavorable_label,
                df=pdf,
                label_names=[label_col],
                protected_attribute_names=[prot_attr]
            )
            
            RW = Reweighing(
                unprivileged_groups=[unpriv_group], # Note: this logic needs update for encoding too
                privileged_groups=[priv_group_encoded]
            )
            # Fix unprivileged group similarly
            unpriv_group_encoded = unpriv_group.copy()
            for k, v in unpriv_group.items():
                if k in label_encoders:
                    try:
                        unpriv_group_encoded[k] = int(label_encoders[k].transform([str(v)])[0])
                    except:
                        pass
            
            RW = Reweighing(
                unprivileged_groups=[unpriv_group_encoded],
                privileged_groups=[priv_group_encoded]
            )
            
            dataset_transf = RW.fit_transform(dataset)
            
            # Add weights back to dataframe
            pdf['instance_weights'] = dataset_transf.instance_weights
            
            print("Bias mitigation complete. Weights added.")
            
            # Sanitize for Spark conversion
            for col in pdf.columns:
                if pdf[col].dtype == 'object':
                    pdf[col] = pdf[col].astype(str)
            
            return spark.createDataFrame(pdf)
            
        except Exception as e:
            print(f"Bias mitigation failed: {e}")
            return df

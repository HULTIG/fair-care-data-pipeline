import pandas as pd
from pyspark.sql import DataFrame
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

class FairnessMetrics:
    def __init__(self, config: dict):
        self.config = config

    def calculate(self, df: DataFrame) -> dict:
        """
        Calculates fairness metrics.
        """
        print("Calculating Fairness Metrics...")
        pdf = df.toPandas()
        
        prot_attr = self.config.get("protected_attribute")
        label_col = self.config.get("label_column")
        priv_group = self.config.get("privileged_groups", [{}])[0]
        unpriv_group = self.config.get("unprivileged_groups", [{}])[0]
        
        report = {}
        
        if prot_attr and label_col:
            try:
                # Debug: Check columns
                # print(f"FairnessMetrics Input Columns: {pdf.columns.tolist()}") 
                
                # Handle '?' as NaN (common in Adult)
                import numpy as np
                pdf = pdf.replace('?', np.nan)
                pdf = pdf.replace(' ?', np.nan)
                
                # AIF360 requires no NAs. Impute standardly:
                for col in pdf.select_dtypes(include=[np.number]).columns:
                     pdf[col] = pdf[col].fillna(0)
                for col in pdf.select_dtypes(include=['object', 'category']).columns:
                    pdf[col] = pdf[col].fillna("Unknown")
                
                # Drop any lingering NAs/Infs
                pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna()
                
                # Drop datetime/timestamp columns to avoid casting errors (COMPAS fix)
                pdf = pdf.select_dtypes(exclude=['datetime', 'timedelta', 'datetimetz'])
                
                # Strip whitespace from ALL string columns (Adult fix)
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
                
                # If label string encoded, map favorable label
                if label_col in label_encoders:
                    classes = label_encoders[label_col].classes_
                    if str(favorable_label) in classes:
                        favorable_label = int(label_encoders[label_col].transform([str(favorable_label)])[0])
                else:
                    # Label column is already numeric (e.g. german credit_risk {1, 2}).
                    # Coerce the config value to the observed dtype so it matches.
                    observed_dtype = pdf[label_col].dtype
                    try:
                        if str(observed_dtype) == "bool":
                            favorable_label = str(favorable_label).lower() in ("true", "1")
                        else:
                            favorable_label = observed_dtype.type(favorable_label)
                    except (ValueError, TypeError):
                        pass

                # Derive the unfavorable label from OBSERVED values instead of
                # assuming {0, 1} (breaks for e.g. german labels {1, 2}).
                observed_labels = sorted(pdf[label_col].unique().tolist())
                others = [v for v in observed_labels if v != favorable_label]
                if len(observed_labels) == 2 and len(others) == 1:
                    unfavorable_label = others[0]
                else:
                    raise ValueError(
                        f"Expected binary labels with favorable={favorable_label!r}, "
                        f"observed {observed_labels}"
                    )
                
                # Handle privileged group encoding
                priv_group_encoded = priv_group.copy()
                for k, v in priv_group.items():
                    if k in label_encoders:
                        try:
                            priv_group_encoded[k] = int(label_encoders[k].transform([str(v)])[0])
                        except: pass
                
                unpriv_group_encoded = unpriv_group.copy()
                for k, v in unpriv_group.items():
                    if k in label_encoders:
                        try:
                             unpriv_group_encoded[k] = int(label_encoders[k].transform([str(v)])[0])
                        except: pass

                
                dataset = BinaryLabelDataset(
                    favorable_label=favorable_label,
                    unfavorable_label=unfavorable_label,
                    df=pdf,
                    label_names=[label_col],
                    protected_attribute_names=[prot_attr]
                )
                
                metric = BinaryLabelDatasetMetric(
                    dataset,
                    unprivileged_groups=[unpriv_group_encoded],
                    privileged_groups=[priv_group_encoded]
                )
                
                spd = metric.statistical_parity_difference()
                di = metric.disparate_impact()
                
                # Handle NaN values - provide reasonable defaults
                import math
                if not math.isfinite(spd) or not math.isfinite(di):
                    raise ValueError("Fairness is undefined: check group support and favorable-label mapping")
                report["statistical_parity_difference"] = float(spd)
                report["disparate_impact"] = float(di)
                
                # EOD calculation as a diagnostic metric 
                try:
                    from sklearn.linear_model import LogisticRegression
                    dataset_train, dataset_test = dataset.split([0.7], shuffle=True)
                    
                    X_train = dataset_train.features
                    y_train = dataset_train.labels.ravel()
                    
                    if len(set(y_train)) > 1:
                        model = LogisticRegression(solver='liblinear', max_iter=200)
                        model.fit(X_train, y_train)
                        
                        dataset_pred = dataset_test.copy()
                        dataset_pred.labels = model.predict(dataset_test.features).reshape(-1,1)
                        
                        from aif360.metrics import ClassificationMetric
                        class_metric = ClassificationMetric(
                            dataset_test, dataset_pred,
                            unprivileged_groups=[unpriv_group_encoded],
                            privileged_groups=[priv_group_encoded]
                        )
                        eod = class_metric.equal_opportunity_difference()
                        if not math.isfinite(eod):
                            raise ValueError("Equal opportunity is undefined on this test split")
                        report["equal_opportunity_difference"] = float(eod)
                    else:
                        raise ValueError("Training labels contain only one class")
                except Exception as e:
                    print(f"EOD calculation failed: {e}")
                    report["equal_opportunity_difference"] = None
                    report["error"] = str(e)
                
            except Exception as e:
                print(f"Fairness metrics calculation failed: {e}")
                report["error"] = str(e)
                # Provide default values on failure
                report["statistical_parity_difference"] = None
                report["disparate_impact"] = None
                report["equal_opportunity_difference"] = None
                
        print(f"Fairness Metrics: {report}")
        return report


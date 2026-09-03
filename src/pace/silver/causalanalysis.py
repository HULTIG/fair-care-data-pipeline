import pandas as pd
from pyspark.sql import DataFrame
import dowhy
from dowhy import CausalModel

class CausalAnalyzer:
    def __init__(self, config: dict):
        self.config = config

    def analyze(self, df: DataFrame) -> dict:
        """
        Performs causal analysis to validate assumptions.
        """
        print("Running Causal Analysis...")
        pdf = df.limit(5000).toPandas()
        
        treatment = self.config.get("protected_attribute")
        outcome = self.config.get("label_column")
        common_causes = self.config.get("quasi_identifiers", [])
        
        report = {}
        
        if treatment and outcome and treatment in pdf.columns and outcome in pdf.columns:
            try:
                # Preprocessing for DoWhy:
                # 1. Drop datetime/internal columns
                pdf = pdf.select_dtypes(exclude=['datetime', 'timedelta'])
                pdf = pdf.drop(columns=[c for c in pdf.columns if c.startswith("_")], errors='ignore')
                
                # 2. Fill NAs
                pdf = pdf.fillna(method='ffill').fillna(0)
                
                # 3. Encode object columns
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for col in pdf.select_dtypes(include='object').columns:
                    pdf[col] = le.fit_transform(pdf[col].astype(str))

                # Define Causal Model
                # Define Causal Model
                model = CausalModel(
                    data=pdf,
                    treatment=treatment,
                    outcome=outcome,
                    common_causes=[c for c in common_causes if c in pdf.columns]
                )
                
                # Identify effect
                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
                
                # Estimate effect
                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name="backdoor.linear_regression"
                )
                
                report["causal_estimate"] = str(estimate.value)
                
                # Refute
                refute = model.refute_estimate(
                    identified_estimand,
                    estimate,
                    method_name="random_common_cause"
                )
                
                report["refutation_p_value"] = refute.refutation_result['p_value']
                report["causal_validity"] = "PASS" if refute.refutation_result['p_value'] > 0.05 else "FAIL"
                
            except Exception as e:
                print(f"Causal analysis failed: {e}")
                report["error"] = str(e)
        
        print(f"Causal Analysis complete: {report}")
        return report

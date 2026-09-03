from pyspark.sql import DataFrame
import pandas as pd
import re

# Optional presidio import - falls back to regex-only detection if not installed
try:
    from presidio_analyzer import AnalyzerEngine
    HAS_PRESIDIO = True
except ImportError:
    HAS_PRESIDIO = False

class PIIDetection:
    def __init__(self, config: dict):
        self.config = config
        self.analyzer = AnalyzerEngine() if HAS_PRESIDIO else None
        self.regex_patterns = {
            "email": r"[^@]+@[^@]+\.[^@]+",
            "phone": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "ssn": r"\d{3}-\d{2}-\d{4}"
        }

    def detect(self, df: DataFrame, sample_size: int = 1000) -> dict:
        """
        Detects PII in a Spark DataFrame.
        Returns a dictionary report.
        """
        print("Running PII detection...")
        # Sample data to Pandas for analysis
        sample_pdf = df.limit(sample_size).toPandas()
        
        report = {}
        
        for column in sample_pdf.columns:
            if column.startswith("_"): continue # Skip metadata columns
            
            column_report = {"pii_types": [], "confidence": [], "recommendation": "KEEP"}
            
            # Convert to string for analysis
            values = sample_pdf[column].astype(str).tolist()
            
            # 1. Regex Check
            for pii_type, pattern in self.regex_patterns.items():
                matches = [v for v in values if re.search(pattern, v)]
                if len(values) > 0 and len(matches) / len(values) > 0.1: # Threshold
                    column_report["pii_types"].append(pii_type)
                    column_report["confidence"].append(1.0)
            
            # 2. NLP Check (Presidio) - only if installed
            if self.analyzer is not None:
                # Analyze a subset of values to save time
                for value in values[:10]: 
                    results = self.analyzer.analyze(text=value, language='en')
                    for result in results:
                        if result.score >= self.config.get("confidence_threshold", 0.5):
                            if result.entity_type not in column_report["pii_types"]:
                                column_report["pii_types"].append(result.entity_type)
                                column_report["confidence"].append(result.score)
            
            if column_report["pii_types"]:
                column_report["recommendation"] = "REVIEW"
                report[column] = column_report
                
        print(f"PII Detection complete. Found potential PII in: {list(report.keys())}")
        return report

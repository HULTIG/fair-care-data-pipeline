# API Reference

## Overview

The PACE Lakehouse provides a Python API for building ethical AI data pipelines. This document describes the main classes and their usage.

## Installation

```python
pip install pace
# or
pip install -e .  # for development
```

## Quick Start

```python
from pace.orchestration.pipeline import run_pipeline

# Run full pipeline
metrics = run_pipeline(
    dataset="compas",
    config_path="configs/default.yaml",
    output_dir="results/my_run",
    verbose=True
)

print(f"PACE Score: {metrics['score']}")
```

## Bronze Layer

### DataIngestion

Ingests raw data into Bronze Delta tables with metadata.

```python
from pace.bronze.ingestion import DataIngestion
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("demo").getOrCreate()
ingestion = DataIngestion(spark)

df = ingestion.ingest(
    source_path="data/raw/compas/compas.csv",
    output_path="data/processed/bronze/compas",
    dataset_name="compas",
    source_system="manual_upload"
)
```

**Methods:**
- `ingest(source_path, output_path, dataset_name, source_system="manual_upload")`: Ingest CSV to Delta

### PIIDetection

Detects personally identifiable information using regex and NLP.

```python
from pace.bronze.piidetection import PIIDetection

config = {
    "confidence_threshold": 0.8,
    "techniques": ["regex", "nlp"],
    "entities": ["PERSON", "email", "phone", "ssn"]
}

detector = PIIDetection(config)
pii_report = detector.detect(df, sample_size=1000)

# pii_report: dict[column_name, {"pii_types": [...], "confidence": [...], "recommendation": "REVIEW"}]
```

**Methods:**
- `detect(df, sample_size=1000)`: Returns PII detection report

### AuditTrail

Logs pipeline events for provenance tracking.

```python
from pace.bronze.audittrail import AuditTrail

audit = AuditTrail(log_dir="results/logs")
audit.log_event("PII_DETECTION", {"columns_flagged": ["ssn", "email"]})
```

**Methods:**
- `log_event(event_type, details)`: Log an event with timestamp

## Silver Layer

### AnonymizationEngine

Applies privacy-preserving transformations.

```python
from pace.silver.anonymization import AnonymizationEngine

config = {
    "technique": "kanonymity",  # or "differentialprivacy"
    "k": 5,
    "epsilon": 1.0,
    "quasi_identifiers": ["age", "zip_code", "gender"]
}

anonymizer = AnonymizationEngine(config)
anonymized_df = anonymizer.anonymize(bronze_df, spark)
```

**Methods:**
- `anonymize(df, spark)`: Returns anonymized DataFrame

**Supported Techniques:**
- `kanonymity`: Generalization + suppression
- `differentialprivacy`: Laplace noise mechanism

### UtilityAssessment

Measures data utility after anonymization.

```python
from pace.silver.utilityassessment import UtilityAssessment

config = {"label_column": "recidivism"}
assessor = UtilityAssessment(config)

utility_report = assessor.assess(original_df, anonymized_df)
# Returns: {"correlation_distance": 0.15, "original_auc": 0.85, "anonymized_auc": 0.80, "utility_retention": 0.94}
```

**Methods:**
- `assess(original_df, anonymized_df)`: Returns utility metrics

### CausalAnalyzer

Validates causal assumptions using DoWhy.

```python
from pace.silver.causalanalysis import CausalAnalyzer

config = {
    "protected_attribute": "race",
    "label_column": "recidivism",
    "quasi_identifiers": ["age", "priors_count"]
}

analyzer = CausalAnalyzer(config)
causal_report = analyzer.analyze(silver_df)
# Returns: {"causal_estimate": "0.15", "refutation_p_value": 0.08, "causal_validity": "PASS"}
```

**Methods:**
- `analyze(df)`: Returns causal validation report

## Gold Layer

### BiasMitigator

Mitigates bias using AIF360 algorithms.

```python
from pace.gold.biasmitigation import BiasMitigator

config = {
    "protected_attribute": "race",
    "label_column": "recidivism",
    "privileged_groups": [{"race": "Caucasian"}],
    "unprivileged_groups": [{"race": "African-American"}],
    "favorable_label": 0
}

mitigator = BiasMitigator(config)
mitigated_df = mitigator.mitigate(silver_df, spark)
# Adds 'instance_weights' column for reweighing
```

**Methods:**
- `mitigate(df, spark)`: Returns bias-mitigated DataFrame

### FairnessMetrics

Calculates fairness metrics using AIF360.

```python
from pace.gold.fairnessmetrics import FairnessMetrics

metrics_calculator = FairnessMetrics(config)
fairness_report = metrics_calculator.calculate(gold_df)
# Returns: {"statistical_parity_difference": -0.08, "disparate_impact": 0.85}
```

**Methods:**
- `calculate(df)`: Returns fairness metrics

**Metrics:**
- `statistical_parity_difference`: P(Y=1|D=unprivileged) - P(Y=1|D=privileged)
- `disparate_impact`: P(Y=1|D=unprivileged) / P(Y=1|D=privileged)
- `equal_opportunity_difference`: TPR_unprivileged - TPR_privileged (requires predictions)

### FeatureEngineer

Performs feature quality checks.

```python
from pace.gold.featureengineering import FeatureEngineer

engineer = FeatureEngineer(config)
processed_df, quality_report = engineer.process(gold_df)
# quality_report: {"age": {"completeness": 0.98, "cardinality": 65}, ...}
```

**Methods:**
- `process(df)`: Returns (processed_df, quality_report)

### EmbeddingsGenerator

Generates text embeddings using Sentence-BERT.

```python
from pace.gold.embeddings import EmbeddingsGenerator

config = {"text_columns": ["notes", "description"]}
embedder = EmbeddingsGenerator(config)
df_with_embeddings = embedder.generate(gold_df, spark)
# Adds columns: notes_embedding, description_embedding
```

**Methods:**
- `generate(df, spark)`: Returns DataFrame with embedding columns

## Metrics Layer

### Layer Metrics

Calculate scores for each layer.

```python
from pace.metrics.layermetrics import BronzeMetrics, SilverMetrics, GoldMetrics

# Bronze Score
bronze = BronzeMetrics()
sb = bronze.calculate({
    "provenance_complete": True,
    "pii_found": False,
    "quality_score": 0.9
})

# Silver Score
silver = SilverMetrics()
ss = silver.calculate({
    "utility_retention": 0.85,
    "causal_validity": "PASS"
})

# Gold Score
gold = GoldMetrics()
sg = gold.calculate({
    "statistical_parity_difference": -0.05,
    "utility_retention": 0.85
})
```

### PACE Score

Composite ethical readiness score.

```python
from pace.metrics.pacescore import PACEScore

config = {
    "weights": {
        "bronze": 0.3,
        "silver": 0.3,
        "gold": 0.4
    }
}

scorer = PACEScore(config)
result = scorer.calculate(sb=0.9, ss=0.85, sg=0.85)
# Returns: {"score": 0.867, "status": "EXCELLENT", "components": {"bronze": 0.9, "silver": 0.85, "gold": 0.85}}
```

**Interpretation:**
- `≥ 0.85`: EXCELLENT
- `0.70-0.85`: ACCEPTABLE
- `< 0.70`: AT RISK

### ComplianceCheck

Validates regulatory compliance.

```python
from pace.metrics.compliance import ComplianceCheck

checker = ComplianceCheck(config)
compliance_report = checker.check(metadata)
# Returns: {"compliant": True, "issues": []}
```

## Orchestration

### Pipeline

End-to-end pipeline execution.

```python
from pace.orchestration.pipeline import run_pipeline

metrics = run_pipeline(
    dataset="compas",
    config_path="experiments/configs/default.yaml",
    output_dir="results/my_run",
    verbose=True,
    seed=42
)

# Returns complete metrics dict with:
# - score, status, components (SB, SS, SG)
# - fairness (DPD, EOD, DI)
# - utility (retention, AUC)
# - privacy (risk, information_loss)
# - anonymization (k, epsilon)
```

## Configuration

### YAML Structure

```yaml
datasets:
  compas:
    raw_path: "data/raw/compas/compas.csv"
    bronze_path: "data/processed/bronze/compas"
    silver_path: "data/processed/silver/compas"
    gold_path: "data/processed/gold/compas"
    protected_attribute: "race"
    label_column: "two_year_recid"
    quasi_identifiers: ["age", "sex"]

pii_detection:
  confidence_threshold: 0.8
  techniques: ["regex", "nlp"]

anonymization:
  technique: "kanonymity"
  k: 5
  epsilon: 1.0

fairness:
  metrics: ["Statistical Parity Difference"]
  thresholds:
    statistical_parity_difference: 0.1

weights:
  bronze: 0.3
  silver: 0.3
  gold: 0.4
```

## Error Handling

All components raise descriptive exceptions:

```python
try:
    metrics = run_pipeline(dataset="invalid", ...)
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Pipeline error: {e}")
```

## Examples

See `notebooks/` for complete examples:
- `01datapreparation.ipynb`: Data loading and exploration
- `02bronzeingestion.ipynb`: Bronze layer walkthrough
- `03silveranonymization.ipynb`: Anonymization techniques
- `04causalvalidation.ipynb`: Causal analysis
- `05goldfairness.ipynb`: Fairness metrics
- `06pacescore.ipynb`: PACE Score calculation

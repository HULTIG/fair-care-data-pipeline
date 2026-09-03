# PACE Architecture

## Overview

The PACE Lakehouse implements a three-layer Medallion architecture that progressively transforms raw data into ethically-governed, analysis-ready datasets.

## Architecture Layers

### Bronze Layer: Ingestion and Discovery

**Purpose**: Ingest raw data with complete provenance and identify sensitive information.

**Components**:
- **Data Ingestion** (`ingestion.py`): Reads CSV files, applies schema validation, adds metadata
- **PII Detection** (`piidetection.py`): Uses Presidio + regex to identify 18 HIPAA identifiers
- **Audit Trail** (`audittrail.py`): Logs all operations with timestamps and metadata

**Outputs**:
- Bronze Delta tables with metadata columns (`_ingestion_timestamp`, `_source_system`, `_schema_hash`)
- PII detection report (JSON)
- Bronze Score (SB): Weighted average of provenance completeness, PII detection accuracy, data quality

**Key Algorithms**:
- Algorithm 1 (Paper): PII Detection with confidence thresholding
- Regex patterns for SSN, email, phone, dates
- NLP-based entity recognition (PERSON, GPE, ORG)

### Silver Layer: Privacy and Causal Validation

**Purpose**: Apply Privacy Enhancement Technologies and validate causal assumptions.

**Components**:
- **Anonymization** (`anonymization.py`): k-anonymity, l-diversity, t-closeness, differential privacy
- **Utility Assessment** (`utilityassessment.py`): Correlation preservation, predictive utility
- **Causal Analysis** (`causalanalysis.py`): DoWhy-based causal validation and refutation

**Outputs**:
- Silver Delta tables (anonymized)
- Anonymization report (k-achieved, information loss, privacy risk)
- Utility report (correlation distance, AUROC retention)
- Causal validation report (refutation p-values)
- Silver Score (SS): Weighted average of anonymization strength, utility preservation, causal completeness

**Key Algorithms**:
- Algorithm 2 (Paper): k-Anonymity with generalization and suppression
- Algorithm 3 (Paper): Differential Privacy with Laplace mechanism
- DoWhy causal refutation tests

### Gold Layer: Fairness and Feature Engineering

**Purpose**: Mitigate bias, ensure fairness, and prepare features for downstream ML.

**Components**:
- **Bias Mitigation** (`biasmitigation.py`): AIF360 Reweighing, Disparate Impact Remover
- **Fairness Metrics** (`fairnessmetrics.py`): DPD, EOD, Disparate Impact, Counterfactual Fairness
- **Feature Engineering** (`featureengineering.py`): Quality checks, encoding, scaling
- **Embeddings** (`embeddings.py`): Sentence-BERT embeddings for text fields

**Outputs**:
- Gold Delta tables (bias-mitigated, feature-engineered)
- Fairness report (DPD, EOD, DI per protected attribute)
- Feature quality report (completeness, cardinality)
- Embeddings (vector representations)
- Gold Score (SG): Weighted average of fairness metrics, feature quality, utility retention

**Key Algorithms**:
- Algorithm 4 (Paper): Fairness-aware Reweighing
- AIF360 metrics: Statistical Parity Difference, Equal Opportunity Difference
- Fairlearn threshold optimization

## PACE Score

**Composite Metric**: Combines layer scores into a single ethical readiness score.

```
PACE Score = w_B × SB + w_S × SS + w_G × SG
```

**Default Weights**: w_B = 0.3, w_S = 0.3, w_G = 0.4

**Interpretation**:
- **≥ 0.85**: EXCELLENT - Robust ethical governance
- **0.70-0.85**: ACCEPTABLE - Governance in place, improvements recommended
- **< 0.70**: AT RISK - Significant governance gaps

**Components**:
- **SB**: Provenance (0.33) + PII Detection (0.33) + Data Quality (0.33)
- **SS**: Anonymization (0.33) + Utility (0.33) + Causal Validity (0.33)
- **SG**: Fairness (0.33) + Feature Quality (0.33) + Utility Retention (0.33)

## Data Flow

```
Raw CSV → Bronze (Ingest + PII) → Silver (Anonymize + Causal) → Gold (Fairness + Features) → ML/Analytics
           ↓ SB                    ↓ SS                          ↓ SG
           └────────────────────────┴──────────────────────────────→ PACE Score
```

## Technology Stack

- **Storage**: Delta Lake (ACID transactions, time travel)
- **Compute**: Apache Spark (distributed processing)
- **Privacy**: diffprivlib, ARX (k-anonymity)
- **Fairness**: AIF360, Fairlearn
- **Causal**: DoWhy, CausalNex
- **Dashboard**: Streamlit

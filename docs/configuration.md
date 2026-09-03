# Configuration Reference

## Overview

The PACE pipeline is configured via YAML files. This document describes all available configuration options.

## Configuration File Structure

```yaml
datasets:          # Dataset-specific settings
pii_detection:     # PII detection parameters
anonymization:     # Privacy preservation settings
fairness:          # Fairness metrics and thresholds
causal_validation: # Causal analysis settings
bias_mitigation:   # Bias mitigation settings
weights:           # Layer score weights
compliance:        # Regulatory compliance settings
```

## Dataset Configuration

Each dataset requires the following configuration:

```yaml
datasets:
  compas:  # Dataset name
    # Data paths
    source_url: "https://..."  # Optional: URL for automated download
    raw_path: "data/raw/compas/compas.csv"
    bronze_path: "data/processed/bronze/compas"
    silver_path: "data/processed/silver/compas"
    gold_path: "data/processed/gold/compas"
    
    # Fairness configuration
    sensitive_attributes: ["race", "sex"]  # All protected attributes
    protected_attribute: "race"  # Primary attribute for fairness metrics
    privileged_groups: [{"race": "Caucasian"}]
    unprivileged_groups: [{"race": "African-American"}]
    
    # Target variable
    label_column: "two_year_recid"
    favorable_label: 0  # 0 = no recidivism (favorable outcome)
    
    # Anonymization
    quasi_identifiers: ["age", "c_charge_degree", "sex"]
```

### Dataset-Specific Examples

#### COMPAS
```yaml
compas:
  protected_attribute: "race"
  privileged_groups: [{"race": "Caucasian"}]
  unprivileged_groups: [{"race": "African-American"}]
  label_column: "two_year_recid"
  favorable_label: 0
```

#### Adult Census
```yaml
adult:
  protected_attribute: "sex"
  privileged_groups: [{"sex": "Male"}]
  unprivileged_groups: [{"sex": "Female"}]
  label_column: "income"
  favorable_label: ">50K"
```

#### German Credit
```yaml
german:
  protected_attribute: "sex"
  privileged_groups: [{"sex": "male"}]
  unprivileged_groups: [{"sex": "female"}]
  label_column: "credit_risk"
  favorable_label: 1  # 1 = Good credit
```

#### NIJ Recidivism
```yaml
nij:
  protected_attribute: "Race"
  privileged_groups: [{"Race": "White"}]
  unprivileged_groups: [{"Race": "Black"}]
  label_column: "Recidivism_Arrest_Year1"
  favorable_label: false
```

## PII Detection Configuration

```yaml
pii_detection:
  confidence_threshold: 0.8  # Minimum confidence for PII detection (0.0-1.0)
  techniques: ["regex", "nlp"]  # Detection methods
  entities: ["PERSON", "email", "phone", "ssn", "LOCATION"]  # Entity types
  sample_size: 1000  # Number of rows to sample for detection
  remove_on_detection: false  # Auto-remove detected PII columns
```

### Entity Types

- **Regex patterns**: `email`, `phone`, `ssn`, `fax`, `ip`, `url`
- **NLP entities**: `PERSON`, `GPE`, `ORG`, `LOCATION`, `DATE_TIME`

## Anonymization Configuration

```yaml
anonymization:
  technique: "kanonymity"  # Options: kanonymity, ldiversity, tcloseness, differentialprivacy, none
  
  # k-anonymity parameters
  k: 5  # Minimum group size
  
  # l-diversity parameters (optional)
  l: 2  # Minimum diversity for sensitive attributes
  
  # t-closeness parameters (optional)
  t: 0.2  # Maximum distance between distributions
  
  # Differential privacy parameters
  epsilon: 1.0  # Privacy budget (lower = more private)
  delta: 0.0001  # Failure probability (for (ε,δ)-DP)
  
  # Quasi-identifiers (set per dataset, can override here)
  quasi_identifiers: []  # Usually set in dataset config
```

### Technique Comparison

| Technique | Privacy | Utility | Complexity |
|-----------|---------|---------|------------|
| `kanonymity` | Medium | High | Low |
| `ldiversity` | High | Medium | Medium |
| `tcloseness` | Very High | Medium | High |
| `differentialprivacy` | Very High | Low-Medium | Medium |
| `none` | None | Very High | None |

### Recommended Settings

**GDPR Compliance:**
```yaml
anonymization:
  technique: "kanonymity"
  k: 10
  epsilon: 0.5
```

**HIPAA Compliance:**
```yaml
anonymization:
  technique: "kanonymity"
  k: 5
  epsilon: 1.0
```

**High Utility:**
```yaml
anonymization:
  technique: "kanonymity"
  k: 3
  epsilon: 2.0
```

## Fairness Configuration

```yaml
fairness:
  metrics: [
    "Statistical Parity Difference",
    "Equal Opportunity Difference",
    "Disparate Impact"
  ]
  
  thresholds:
    statistical_parity_difference: 0.1  # |DPD| ≤ 0.1
    equal_opportunity_difference: 0.1   # |EOD| ≤ 0.1
    disparate_impact: 0.8               # DI ≥ 0.8
```

### Fairness Metrics

- **Statistical Parity Difference (DPD)**: P(Ŷ=1|D=unprivileged) - P(Ŷ=1|D=privileged)
  - Range: [-1, 1]
  - Fair: ≈ 0
  
- **Equal Opportunity Difference (EOD)**: TPR_unprivileged - TPR_privileged
  - Range: [-1, 1]
  - Fair: ≈ 0
  
- **Disparate Impact (DI)**: P(Ŷ=1|D=unprivileged) / P(Ŷ=1|D=privileged)
  - Range: [0, ∞]
  - Fair: ≈ 1
  - 80% rule: DI ≥ 0.8

## Causal Validation Configuration

```yaml
causal_validation: true  # Enable/disable causal analysis

causal:
  method: "dowhy"  # Causal inference library
  refutation_tests: ["random_common_cause", "placebo_treatment"]
  significance_level: 0.05
```

## Bias Mitigation Configuration

```yaml
bias_mitigation: true  # Enable/disable bias mitigation

mitigation:
  technique: "reweighing"  # Options: reweighing, disparate_impact_remover
  apply_to: "training"  # When to apply: training, prediction, both
```

## Layer Weights Configuration

```yaml
weights:
  bronze: 0.3  # Provenance + PII + Quality
  silver: 0.3  # Anonymization + Utility + Causal
  gold: 0.4    # Fairness + Features + Utility
```

**Must sum to 1.0**

### Use Case Weights

**Privacy-focused:**
```yaml
weights:
  bronze: 0.25
  silver: 0.50
  gold: 0.25
```

**Fairness-focused:**
```yaml
weights:
  bronze: 0.25
  silver: 0.25
  gold: 0.50
```

**Balanced:**
```yaml
weights:
  bronze: 0.33
  silver: 0.33
  gold: 0.34
```

## Compliance Configuration

```yaml
compliance:
  regulation: "GDPR"  # Options: GDPR, HIPAA, CCPA
  
  # GDPR-specific
  right_to_erasure: true
  data_minimization: true
  purpose_limitation: true
  
  # HIPAA-specific
  safe_harbor_method: true
  audit_trail_required: true
  minimum_necessary: true
  
  # CCPA-specific
  consumer_rights: ["right_to_know", "right_to_delete", "right_to_opt_out"]
  data_sale_disclosure: true
```

## Privacy Configuration

```yaml
privacy:
  max_reidentification_risk: 0.05  # Maximum acceptable risk (0.0-1.0)
  require_pseudonymization: true
  key_separation: true  # Separate storage of pseudonymization keys
  use_synthetic_data: false  # Generate synthetic data
```

## Advanced Configuration

### Random Seed

```yaml
random_seed: 42  # For reproducibility
```

### Performance Tuning

```yaml
performance:
  max_rows: null  # Limit dataset size (null = no limit)
  sample_size: 1000  # Sample size for expensive operations
  parallel_jobs: 4  # Number of parallel jobs
```

### Logging

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file: "results/logs/pipeline.log"
  console: true
```

## Complete Example Configurations

### Baseline (Traditional ETL)

```yaml
anonymization:
  technique: "none"
causal_validation: false
bias_mitigation: false
weights:
  bronze: 0.5
  silver: 0.25
  gold: 0.25
```

### Full PACE (Default)

```yaml
anonymization:
  technique: "kanonymity"
  k: 5
causal_validation: true
bias_mitigation: true
fairness:
  metrics: ["Statistical Parity Difference", "Equal Opportunity Difference", "Disparate Impact"]
  thresholds:
    statistical_parity_difference: 0.1
    equal_opportunity_difference: 0.1
    disparate_impact: 0.8
weights:
  bronze: 0.3
  silver: 0.3
  gold: 0.4
```

### GDPR Strict

```yaml
anonymization:
  technique: "kanonymity"
  k: 10
  epsilon: 0.5
privacy:
  max_reidentification_risk: 0.05
  require_pseudonymization: true
  key_separation: true
compliance:
  regulation: "GDPR"
  right_to_erasure: true
weights:
  bronze: 0.25
  silver: 0.45
  gold: 0.30
```

## Validation

The pipeline validates configurations on startup:
- Required fields present
- Weights sum to 1.0
- Thresholds in valid ranges
- Dataset paths exist
- Technique names valid

Invalid configurations raise `ValueError` with descriptive messages.

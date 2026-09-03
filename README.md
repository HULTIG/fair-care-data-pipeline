# Governing Data Before the Model: A Layered Pipeline Architecture and Readiness Metric for High-Risk AI Systems

## Overview

The PACE Lakehouse is a reference architecture for ethical AI data governance that integrates:
- **FAIR Principles**: Findability, Accessibility, Interoperability, and Reusability
- **CARE Principles**: Causality, Anonymity, Regulatory-compliance, and Ethics

This artifact implements a three-layer Medallion architecture (Bronze–Silver–Gold) with:
- Privacy Enhancement Technologies (k-anonymity, differential privacy, synthetic data)
- Causal inference validation
- Fairness metrics and bias mitigation
- Regulatory compliance checks (GDPR, HIPAA, CCPA)
- Composite PACE Score for ethical data readiness

![PACE Architecture](docs/img/arch-pace.png)

## Artifact Scope

| Paper  | Artifact Component |
|-------------|-------------------|
| Bronze Layer (Ingestion, PII Detection, Provenance) | `src/pace/bronze/` |
| Silver Layer (Anonymization, Utility, Causal Analysis) | `src/pace/silver/` |
| Gold Layer (Bias Mitigation, Fairness Metrics) | `src/pace/gold/` |
| PACE Score Framework | `src/pace/metrics/pacescore.py` |
| Experiment 1: Ablation Study | `experiments/scripts/runexperiment1.py` |
| Experiment 2: Multi-Dataset Benchmarking | `experiments/scripts/runexperiment2.py` |
| Experiment 3: Regulatory Configurations | `experiments/scripts/runexperiment3.py` |
| GDPR/HIPAA/CCPA Compliance | `experiments/configs/{gdprstrict,hipaa,ccpa}.yaml` |

## System Requirements

### Hardware
- **Recommended**: 16 GB RAM, 4+ CPU cores
- **Minimum**: 8 GB RAM, 2 CPU cores
- **GPU**: Optional

### Software
- **OS**: Linux, macOS, or Windows with WSL2
- **Docker**: 20.10+ with Docker Compose
- **Python**: 3.9+ (if running natively)
- **Internet**: Required for dataset downloads

## Quick Installation

### Option A: Docker (Recommended)

```bash
# Extract artifact
tar -xzf pace-lakehouse.tar.gz
cd pace-lakehouse

# Build and start services
docker-compose build ml
docker-compose up -d

# Verify services are running
docker-compose ps
```

### Option B: Native Python

```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Dataset Setup

**Important**: This artifact cannot include proprietary datasets. You must download them separately.

### Automated Download (COMPAS, Adult, German Credit)

```bash
python scripts/downloaddatasets.py --datasets compas,adult,german
```

### Manual Download

See `data/raw/README.md` for detailed instructions and URLs:

- **COMPAS**: ProPublica COMPAS Recidivism Dataset
- **Adult Census**: UCI Adult Income Dataset
- **German Credit**: UCI German Credit Dataset
- **NIJ Recidivism**: NIJ Recidivism Forecasting Challenge

## One-Command Pipeline Run

### Quick Demo (COMPAS)

```bash
# Using Docker
docker-compose exec ml python -m pace.orchestration.pipeline \
  --dataset compas \
  --config experiments/configs/default.yaml \
  --output results/compas_demo \
  --verbose

# Native Python
python -m pace.orchestration.pipeline \
  --dataset compas \
  --config experiments/configs/default.yaml \
  --output results/compas_demo \
  --verbose
```

### Expected Output

```
results/compas_demo/
├── logs/
│   └── audit_log.json        # Provenance trail
└── compas_metricssummary.json # PACE scores and metrics

data/processed/
├── bronze/
│   └── {dataset_name}_raw.delta/     # Raw ingested data (Bronze Layer)
├── silver/
│   └── {dataset_name}_anonymized.delta/ # Anonymized, utility-validated data (Silver Layer)
└── gold/
    └── {dataset_name}_final.delta/   # Bias-mitigated, fairness-checked data (Gold Layer)
```

**Key Metrics in `compas_metricssummary.json`**:
- `score`: Composite PACE Score (0-1)
- `status`: EXCELLENT (≥0.85), ACCEPTABLE (0.70-0.85), or AT RISK (<0.70)
- `components.bronze`: Bronze layer score (SB)
- `components.silver`: Silver layer score (SS)
- `components.gold`: Gold layer score (SG)

## Key Results from the Paper

The PACE pipeline introduces a composite PACE Score that quantitatively evaluates the ethical readiness of datasets across Bronze, Silver, and Gold layers.

### Ablation Study: Impact of PACE Layers (COMPAS Dataset)
| Configuration | PACE Score | Silver Score (SS) | Privacy Risk | Utility (AUC) |
|---------------|-----------------|-------------------|--------------|---------------|
| Baseline (No CARE) | 0.77 | 0.33 | 100.0% | 1.00 |
| Config A ($k$-anon) | 0.98 | 0.95 | 6.7% | 1.00 |
| Config B (Diff. Priv) | 0.97 | 0.90 | 10.0% | 1.00 |
| Config C (Causal) | 0.89 | 0.67 | 100.0% | 1.00 |

### Statistical Robustness (COMPAS, 5 Random Seeds)
| Configuration | PACE Score | AUC | EOD | DPD |
|---------------|-----------------|-----|-----|-----|
| Baseline | 0.77 ± 0.01 | 0.99 ± 0.01 | 0.12 ± 0.02 | 0.15 ± 0.01 |
| Config A ($k$-anon) | 0.98 ± 0.00 | 0.98 ± 0.01 | 0.06 ± 0.01 | 0.05 ± 0.01 |
| Config B (DP) | 0.97 ± 0.01 | 0.97 ± 0.02 | 0.05 ± 0.01 | 0.04 ± 0.01 |

## Reproducing Paper Experiments

### Data Partitioning
All experiments and evaluations below utilize a **70/30 train/test split**. A fixed random seed (e.g., `random_state=42`) is used to ensure reproducibility across executions. No separate validation split is used, as hyperparameter tuning was not a primary focus of this study.

### Experiment 1: Ablation Study

Tests impact of removing key components (anonymization, causal validation, bias mitigation).

```bash
docker-compose exec ml python experiments/scripts/runexperiment1.py \
  --datasets compas,adult,german,nij \
  --configs baseline,configa,configb,configc \
  --output results/exp1.csv
```

**Output**: `results/exp1.csv` with columns: dataset, config, SB, SS, SG, pacescore, dpd, eod, utility

### Experiment 2: Multi-Dataset Benchmarking

Compares PACE performance across all four datasets.

```bash
docker-compose exec ml python experiments/scripts/runexperiment2.py \
  --datasets compas,adult,german,nij \
  --config configs/default.yaml \
  --output results/exp2.csv
```

**Output**: `results/exp2.csv` with fairness, utility, and privacy metrics per dataset

### Experiment 3: Regulatory Configurations

Tests GDPR, HIPAA, and CCPA compliance modes.

```bash
docker-compose exec ml python experiments/scripts/runexperiment3.py \
  --datasets compas,adult,german,nij \
  --regulations gdpr,hipaa,ccpa \
  --output results/exp3.csv
```

**Output**: `results/exp3.csv` with compliance flags and privacy risk scores

### Experiment 4: Statistical Robustness

Executes configurations across multiple random seeds to compute the mean and standard deviation of key metrics (AUC, EOD, DPD, PACE score).

```bash
docker-compose exec ml python experiments/scripts/runexperiment4_robustness.py \
  --dataset compas \
  --configs baseline,configa,configb \
  --seeds 42,43,44,45,46 \
  --output results/exp4_robustness.json
```

**Output**: `results/exp4_robustness.json` with aggregated means and standard deviations.

### Experiment 5: Hyperparameter Sensitivity

Varies the differential privacy budget ($\epsilon$) and $k$-anonymity threshold to map out the utility-privacy-fairness trade-off surface.

```bash
docker-compose exec ml python experiments/scripts/runexperiment5_sensitivity.py \
  --dataset compas \
  --epsilons 0.1,1.0,5.0 \
  --ks 2,5,10 \
  --output results/exp5_sensitivity.json
```

**Output**: `results/exp5_sensitivity.json` with metric variations across parameter grids.

### Aggregate Results and Generate Figures

Generates the core paper figures from Experiments 1-3.

```bash
docker-compose exec ml python experiments/scripts/aggregateresults.py \
  --inputs results/exp1.csv,results/exp2.csv,results/exp3.csv \
  --output results/figures/
```

Generates the statistical robustness and sensitivity visualizations from Experiments 4-5.

```bash
docker-compose exec ml python experiments/scripts/visualize_new_experiments.py
```

**Output**: All generated plots and charts matching the paper figures will be saved in `results/figures/`.

## Running Tests

Tests should be run inside the Docker container to ensure proper Spark environment:

```bash
# Start services if not running
docker-compose up -d

# Run all tests
docker-compose exec ml pytest tests/ -v

# Run with coverage report
docker-compose exec ml pytest tests/ --cov=pace --cov-report=term-missing

# Run a specific test file
docker-compose exec ml pytest tests/test_pacescore.py -v
```

**Expected**: 50+ tests covering Bronze, Silver, Gold layers and PACE Score calculation.

## Documentation

- **[Architecture](docs/architecture.md)**: Bronze/Silver/Gold layer design
- **[Installation](docs/installation.md)**: Detailed setup instructions
- **[Experiments](docs/experiments.md)**: Step-by-step experiment reproduction
- **[Configuration](docs/configuration.md)**: Config file reference
- **[API Reference](docs/API_REFERENCE.md)**: Python API documentation

## License

This software is licensed under the Apache License 2.0. See `LICENSE` for details.


---

**Artifact Checklist**:
- ✅ Complete source code
- ✅ Configuration files for all experiments
- ✅ Automated tests (50+ unit tests)
- ✅ Documentation (README + 5 docs)
- ✅ Dataset download scripts
- ✅ One-command reproduction
- ✅ Expected runtime: ~60 minutes for full experiments
- ✅ Results tolerance: ±5% of paper values

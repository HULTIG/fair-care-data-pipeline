# FAIR-CARE Lakehouse: Ethical AI Data Governance Pipeline

**Artifact for ICSA 2026 Submission**

## Overview

The FAIR-CARE Lakehouse is a reference architecture for ethical AI data governance that integrates:
- **FAIR Principles**: Findability, Accessibility, Interoperability, and Reusability
- **CARE Principles**: Causality, Anonymity, Regulatory-compliance, and Ethics

This artifact implements a three-layer Medallion architecture (Bronze–Silver–Gold) with:
- Privacy Enhancement Technologies (k-anonymity, differential privacy, synthetic data)
- Causal inference validation
- Fairness metrics and bias mitigation
- Regulatory compliance checks (GDPR, HIPAA, CCPA)
- Composite FAIR-CARE Score for ethical data readiness

![FAIR-CARE Architecture](docs/img/arch-fair-care.png)

## Artifact Scope

| Paper  | Artifact Component |
|-------------|-------------------|
| Bronze Layer (Ingestion, PII Detection, Provenance) | `src/faircare/bronze/` |
| Silver Layer (Anonymization, Utility, Causal Analysis) | `src/faircare/silver/` |
| Gold Layer (Bias Mitigation, Fairness Metrics) | `src/faircare/gold/` |
| FAIR-CARE Score Framework | `src/faircare/metrics/faircarescore.py` |
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
tar -xzf fair-care-lakehouse.tar.gz
cd fair-care-lakehouse

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
docker-compose exec ml python -m faircare.orchestration.pipeline \
  --dataset compas \
  --config experiments/configs/default.yaml \
  --output results/compas_demo \
  --verbose

# Native Python
python -m faircare.orchestration.pipeline \
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
└── compas_metricssummary.json # FAIR-CARE scores and metrics

data/processed/
├── bronze/
│   └── {dataset_name}_raw.delta/     # Raw ingested data (Bronze Layer)
├── silver/
│   └── {dataset_name}_anonymized.delta/ # Anonymized, utility-validated data (Silver Layer)
└── gold/
    └── {dataset_name}_final.delta/   # Bias-mitigated, fairness-checked data (Gold Layer)
```

**Key Metrics in `compas_metricssummary.json`**:
- `score`: Composite FAIR-CARE Score (0-1)
- `status`: EXCELLENT (≥0.85), ACCEPTABLE (0.70-0.85), or AT RISK (<0.70)
- `components.bronze`: Bronze layer score (SB)
- `components.silver`: Silver layer score (SS)
- `components.gold`: Gold layer score (SG)

## Reproducing Paper Experiments

### Experiment 1: Ablation Study

Tests impact of removing key components (anonymization, causal validation, bias mitigation).

```bash
docker-compose exec ml python experiments/scripts/runexperiment1.py \
  --datasets compas,adult,german,nij \
  --configs baseline,configa,configb,configc \
  --output results/exp1.csv
```

**Output**: `results/exp1.csv` with columns: dataset, config, SB, SS, SG, faircarescore, dpd, eod, utility

### Experiment 2: Multi-Dataset Benchmarking

Compares FAIR-CARE performance across all four datasets.

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

Executes configurations across multiple random seeds to compute the mean and standard deviation of key metrics (AUC, EOD, DPD, FAIR-CARE score).

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
docker-compose exec ml pytest tests/ --cov=faircare --cov-report=term-missing

# Run a specific test file
docker-compose exec ml pytest tests/test_faircarescore.py -v
```

**Expected**: 50+ tests covering Bronze, Silver, Gold layers and FAIR-CARE Score calculation.

## Documentation

- **[Architecture](docs/architecture.md)**: Bronze/Silver/Gold layer design
- **[Installation](docs/installation.md)**: Detailed setup instructions
- **[Experiments](docs/experiments.md)**: Step-by-step experiment reproduction
- **[Configuration](docs/configuration.md)**: Config file reference
- **[API Reference](docs/API_REFERENCE.md)**: Python API documentation

## License and Citation

This software is licensed under the Apache License 2.0. See `LICENSE` for details.

To cite this work:
```bibtex
@software{faircare2025,
  title = {A Reference Architecture for FAIR and Ethically Governed Data Pipelines in High-Risk AI Domains},
  author = {Anonymous for Review},
  year = {2025},
  license = {Apache-2.0},
  url = {https://anonymous.4open.science/r/fair-care-lakehouse}
}
```

See `CITATION.cff` for machine-readable citation metadata.

## Contact

For questions about this artifact: **Anonymous for Review**

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

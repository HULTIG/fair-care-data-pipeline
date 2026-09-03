# Experiments Guide

This guide provides step-by-step instructions to reproduce the three main experiments from the paper.

## Overview

| Experiment | Purpose | Runtime | Output |
|------------|---------|---------|--------|
| Exp 1: Ablation | Test impact of removing components | ~30 min | `results/exp1.csv` |
| Exp 2: Benchmarking | Compare across datasets | ~45 min | `results/exp2.csv` |
| Exp 3: Regulatory | Test compliance modes | ~40 min | `results/exp3.csv` |

## Experiment 1: Ablation Study

### Objective
Measure the contribution of each PACE component by systematically removing them.

### Configurations

| Config | Description | Components Active |
|--------|-------------|-------------------|
| `baseline` | Traditional ETL only | Ingestion only |
| `configa` | + k-anonymity | Ingestion + Anonymization |
| `configb` | + Differential Privacy | Ingestion + DP |
| `configc` | + Causal validation | Ingestion + Causal |
| `default` | Full PACE | All components |

### Running the Experiment

```bash
docker-compose exec ml python experiments/scripts/runexperiment1.py \
  --datasets compas,adult,german,nij \
  --configs baseline,configa,configb,configc,default \
  --output results/exp1.csv \
  --verbose
```

### Expected Output

`results/exp1.csv`:
```csv
dataset,config,SB,SS,SG,pacescore,dpd,eod,di,utility,privacy_risk
compas,baseline,0.85,0.30,0.40,0.52,-0.25,-0.18,0.55,0.95,0.85
compas,configa,0.85,0.75,0.45,0.68,-0.12,-0.10,0.72,0.88,0.15
compas,configb,0.85,0.80,0.45,0.70,-0.15,-0.12,0.68,0.82,0.05
compas,configc,0.85,0.35,0.70,0.63,-0.20,-0.15,0.60,0.93,0.80
compas,default,0.90,0.85,0.85,0.87,-0.08,-0.06,0.85,0.85,0.08
...
```

### Key Findings (from Paper)

- **Baseline**: High utility (0.95) but poor fairness (DPD=-0.25) and privacy (risk=0.85)
- **Config A (k-anon)**: Improved privacy (risk=0.15) with moderate utility loss (0.88)
- **Config B (DP)**: Best privacy (risk=0.05) but higher utility loss (0.82)
- **Config C (Causal)**: Better fairness (DPD=-0.20) but limited privacy
- **Full PACE**: Balanced trade-offs (PACE=0.87, utility=0.85, DPD=-0.08, risk=0.08)

## Experiment 2: Multi-Dataset Benchmarking

### Objective
Evaluate PACE performance across diverse datasets with different characteristics.

### Datasets

| Dataset | Domain | Rows | Protected Attrs | Bias Type |
|---------|--------|------|-----------------|-----------|
| COMPAS | Criminal Justice | 7,214 | Race, Gender | Racial bias in recidivism |
| Adult | Income | 48,842 | Sex, Race | Gender wage gap |
| German | Credit | 1,000 | Age, Foreign | Age discrimination |
| NIJ | Parole | 25,000+ | Race, Gender | Recidivism prediction |

`results/exp2.csv`:
```csv
dataset,technique,SB,SS,SG,pacescore,dpd,eod,di,utility,info_loss,privacy_risk
compas,kanonymity,0.90,0.82,0.85,0.86,-0.08,-0.06,0.85,0.85,0.25,0.08
compas,ldiversity,0.90,0.85,0.85,0.87,-0.07,-0.05,0.87,0.83,0.30,0.05
compas,tcloseness,0.90,0.88,0.85,0.88,-0.06,-0.04,0.88,0.80,0.35,0.03
compas,dp,0.90,0.80,0.85,0.85,-0.10,-0.08,0.82,0.78,0.20,0.02
adult,kanonymity,0.88,0.80,0.82,0.83,-0.10,-0.08,0.80,0.87,0.22,0.10
...
```

### Key Findings (from Paper)

- **COMPAS**: Highest PACE scores (0.86-0.88) due to well-structured data
- **Adult**: Good performance (0.83-0.85) with gender fairness improvements
- **German**: Lower scores (0.75-0.78) due to small dataset size
- **NIJ**: Strong privacy (risk<0.05) with acceptable utility (0.80-0.85)

## Experiment 3: Regulatory Compliance

### Objective
Test PACE configurations for GDPR, HIPAA, and CCPA compliance.

### Regulatory Modes

| Regulation | Key Requirements | Config Parameters |
|------------|------------------|-------------------|
| GDPR | k≥10, ε≤0.5, risk<0.05 | `gdprstrict.yaml` |
| HIPAA | Safe Harbor or Expert Determination | `hipaa.yaml` |
| CCPA | Functional separation, deletion rights | `ccpa.yaml` |

### Running the Experiment

```bash
docker-compose exec ml python experiments/scripts/runexperiment3.py \
  --datasets compas,adult,german,nij \
  --regulations gdpr,hipaa,ccpa \
  --output results/exp3.csv \
  --verbose
```

### Expected Output

`results/exp3.csv`:
```csv
dataset,regulation,SB,SS,SG,pacescore,k,epsilon,privacy_risk,compliant
compas,gdpr,0.90,0.88,0.80,0.86,10,0.5,0.04,True
compas,hipaa,0.90,0.85,0.82,0.86,5,1.0,0.08,True
compas,ccpa,0.90,0.82,0.85,0.86,5,1.0,0.10,True
adult,gdpr,0.88,0.85,0.78,0.84,10,0.5,0.05,True
...
```

### Key Findings (from Paper)

- **GDPR**: Strictest requirements, lowest utility (0.78-0.82) but highest privacy
- **HIPAA**: Balanced approach, moderate utility loss (0.82-0.85)
- **CCPA**: Most flexible, highest utility (0.85-0.88) with acceptable privacy
- **All datasets**: Achieved compliance with PACE scores ≥0.84

## Generating Figures

After running experiments, generate paper figures:

```bash
docker-compose exec ml python experiments/scripts/aggregateresults.py \
  --inputs results/exp1.csv,results/exp2.csv,results/exp3.csv \
  --output results/figures/ \
  --format png,pdf
```

### Generated Figures

- `fig1_ablation_pace.png`: PACE scores by config
- `fig2_ablation_fairness.png`: DPD/EOD by config
- `fig3_benchmark_datasets.png`: Performance across datasets
- `fig4_benchmark_techniques.png`: Anonymization technique comparison
- `fig5_regulatory_compliance.png`: Compliance by regulation
- `fig6_tradeoffs.png`: Utility vs Privacy vs Fairness

## Tolerance and Reproducibility

### Expected Variance

Results may vary by ±5% due to:
- Random sampling in PII detection
- Stochastic elements in DP noise
- Bootstrap sampling in causal analysis
- Train/test splits in utility assessment

### Ensuring Reproducibility

Set random seeds in config:
```yaml
random_seed: 42
```

Run multiple trials with different seeds:
```bash
for i in {1..5}; do
  python experiments/scripts/runexperiment1.py --seed $i --output results/exp1_trial${i}.csv
done
```

## Troubleshooting

### Experiment Takes Too Long

Reduce dataset sizes in config:
```yaml
max_rows: 5000  # Default: all rows
```

### Out of Memory

Run experiments sequentially:
```bash
for dataset in compas adult german nij; do
  python experiments/scripts/runexperiment1.py --datasets $dataset --output results/exp1_${dataset}.csv
done
```

### Results Don't Match Paper

Check:
1. Dataset versions (URLs in `data/raw/README.md`)
2. Dependency versions (`requirements.txt`)
3. Random seeds
4. Config parameters

## Advanced Usage

### Custom Configurations

Create your own config:
```yaml
# experiments/configs/custom.yaml
datasets:
  compas:
    k: 7
    epsilon: 0.8
    fairness_threshold: 0.05
```

Run with custom config:
```bash
python experiments/scripts/runexperiment1.py --config experiments/configs/custom.yaml
```

### Parallel Execution

Use Ray for distributed experiments:
```bash
python experiments/scripts/runexperiment1.py --parallel --num-workers 4
```

# Installation Guide

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+, macOS 11+, or Windows 10/11 with WSL2
- **RAM**: 16 GB recommended (8 GB minimum)
- **Disk**: 20 GB free space
- **CPU**: 4+ cores recommended

### Software Dependencies
- Docker 20.10+ and Docker Compose 1.29+
- Python 3.9+ (for native installation)
- Git

## Installation Methods

### Method 1: Docker (Recommended)

#### Step 1: Extract Artifact
```bash
tar -xzf pace-lakehouse.tar.gz
cd pace-lakehouse
```

#### Step 2: Build Docker Image
```bash
docker-compose build ml
```

**Build Time**: ~10-15 minutes (downloads ~2 GB)

#### Step 3: Start Services
```bash
docker-compose up -d
```

#### Step 4: Verify Installation
```bash
# Check services are running
docker-compose ps

# Expected output:
# ml       running
```

#### Step 5: Test Pipeline
```bash
docker-compose exec ml python -c "import pace; print('Installation successful!')"
```

### Method 2: Native Python

#### Step 1: Create Virtual Environment
```bash
python3.9 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 3: Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

#### Step 4: Install Package
```bash
pip install -e .
```

#### Step 5: Set Environment Variables
```bash
export SPARK_HOME=/path/to/spark  # If using local Spark
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## Dataset Setup

### Automated Download

```bash
# Download COMPAS, Adult, German Credit
python scripts/downloaddatasets.py --datasets compas,adult,german
```

### Manual Download

#### COMPAS
1. Visit: https://github.com/propublica/compas-analysis
2. Download: `compas-scores-two-years.csv`
3. Place in: `data/raw/compas/compas.csv`

#### Adult Census
1. Visit: https://archive.ics.uci.edu/ml/datasets/adult
2. Download: `adult.data`
3. Place in: `data/raw/adult/adult.csv`

#### German Credit
1. Visit: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
2. Download: `german.data`
3. Place in: `data/raw/german/german.csv`

#### NIJ Recidivism (Requires Access)
1. Request access: https://nij.ojp.gov/funding/recidivism-forecasting-challenge
2. Download dataset after approval
3. Run preprocessing:
```bash
python scripts/preprocess_nij.py --input /path/to/nij_raw.csv --output data/raw/nij/nij.csv
```

## Configuration

### Environment Variables

Create `.env` file:
```bash
# Spark
SPARK_MASTER=spark://spark-master:7077

# Postgres
POSTGRES_HOST=postgres
POSTGRES_DB=pace
POSTGRES_USER=pace
POSTGRES_PASSWORD=pace

# Optional: API Keys for external services
# OPENAI_API_KEY=your_key_here
```

### Config Files

Default configuration: `experiments/configs/default.yaml`

Customize for your use case:
```yaml
datasets:
  compas:
    protected_attribute: "race"
    k: 5
    epsilon: 1.0
```

## Verification

### Quick Test
```bash
# Docker
docker-compose exec ml python -m pace.orchestration.pipeline \
  --dataset compas \
  --config experiments/configs/default.yaml \
  --output results/test

# Native
python -m pace.orchestration.pipeline \
  --dataset compas \
  --config experiments/configs/default.yaml \
  --output results/test
```

**Expected**: Completes in ~5 minutes, creates `results/test/compas_metricssummary.json`

### Run Tests
```bash
# Native Python (recommended)
pytest tests/ -v
```

**Expected**: Tests pass successfully

## Troubleshooting

### Docker Issues

**Problem**: `docker-compose build` fails with "no space left on device"
```bash
# Clean up Docker
docker system prune -a
```

**Problem**: Container exits immediately
```bash
# Check logs
docker-compose logs ml
```

### Python Issues

**Problem**: `ModuleNotFoundError: No module named 'pace'`
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

**Problem**: `ImportError: cannot import name 'Omit' from 'openai'`
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Dataset Issues

**Problem**: Download script fails for NIJ
- **Solution**: NIJ requires manual download. See `data/raw/README.md`

**Problem**: "File not found" error
```bash
# Verify dataset paths
ls -la data/raw/*/
```

### Performance Issues

**Problem**: Pipeline runs slowly
- Reduce sample sizes in config
- Use fewer datasets
- Allocate more RAM to Docker

**Problem**: Out of memory errors
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory: 8 GB+
```

## Uninstallation

### Docker
```bash
docker-compose down -v
docker rmi pace-lakehouse_ml
```

### Native
```bash
deactivate  # Exit venv
rm -rf venv
```

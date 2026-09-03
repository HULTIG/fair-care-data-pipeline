# Dataset Download Instructions

This directory contains raw datasets for the PACE pipeline experiments.

## Automated Download

For COMPAS, Adult, and German Credit datasets:

```bash
python scripts/downloaddatasets.py --datasets compas,adult,german
```

## Manual Download Instructions

### COMPAS Recidivism Dataset

**Source**: ProPublica COMPAS Analysis  
**URL**: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv  
**License**: Creative Commons Attribution 4.0

**Steps**:
1. Download the CSV file from the URL above
2. Place it in: `data/raw/compas/compas.csv`

**Expected Columns**: `age`, `c_charge_degree`, `race`, `sex`, `priors_count`, `two_year_recid`, etc.

### Adult Census Income Dataset

**Source**: UCI Machine Learning Repository  
**URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data  
**License**: Public Domain

**Steps**:
1. Download `adult.data` from the URL above
2. Rename to `adult.csv`
3. Place it in: `data/raw/adult/adult.csv`

**Expected Columns**: `age`, `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`, `income`

### German Credit Dataset

**Source**: UCI Machine Learning Repository  
**URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data  
**License**: Public Domain

**Steps**:
1. Download `german.data` from the URL above
2. Convert to CSV format (space-separated to comma-separated)
3. Add column headers based on `german.doc`
4. Place it in: `data/raw/german/german.csv`

**Expected Columns**: `checking_account`, `duration`, `credit_history`, `purpose`, `credit_amount`, `savings`, `employment`, `installment_rate`, `personal_status_sex`, `other_debtors`, `residence`, `property`, `age`, `other_installment_plans`, `housing`, `existing_credits`, `job`, `num_dependents`, `telephone`, `foreign_worker`, `credit_risk`

### NIJ Recidivism Forecasting Challenge Dataset

**Source**: National Institute of Justice (NIJ)  
**URL**: https://nij.ojp.gov/funding/recidivism-forecasting-challenge  
**License**: Requires Data Use Agreement

**Important**: This dataset requires approval from NACJD (National Archive of Criminal Justice Data).

**Steps**:
1. Visit the NIJ website and request access
2. Complete the Data Use Agreement
3. Download the dataset after approval
4. Run the preprocessing script:
   ```bash
   python scripts/preprocess_nij.py --input /path/to/nij_raw.csv --output data/raw/nij/nij.csv
   ```

**Expected Columns**: `ID`, `Age_at_Release`, `Gender`, `Race`, `Gang_Affiliation`, `Supervision_Level_First`, `Education_Level`, `Dependents`, `Prison_Offense`, `Prison_Years`, `Prior_Arrest_Episodes_*`, `Condition_*`, `Recidivism_Arrest_Year1`, `Recidivism_Arrest_Year2`, `Recidivism_Arrest_Year3`

## Dataset Characteristics

| Dataset | Rows | Columns | Protected Attributes | Target | Domain |
|---------|------|---------|---------------------|--------|--------|
| COMPAS | 7,214 | 53 | Race, Gender | 2-year recidivism | Criminal Justice |
| Adult | 48,842 | 15 | Sex, Race | Income >50K | Census/Income |
| German | 1,000 | 21 | Age, Foreign Worker | Credit Risk | Finance |
| NIJ | 25,000+ | 50+ | Race, Gender | Recidivism (1/2/3 year) | Parole/Corrections |

## Verification

After downloading, verify the datasets by checking:
- File existence in `data/raw/{dataset}/` directories
- Expected number of rows (see table above)
- Required columns present (see Expected Columns sections)
- Files open correctly with pandas

## Data Privacy Notice

These datasets contain sensitive information about real individuals. By using these datasets, you agree to:

1. Use the data only for research purposes
2. Not attempt to re-identify individuals
3. Follow all applicable data protection regulations (GDPR, HIPAA, etc.)
4. Properly cite the original data sources

## Troubleshooting

**Problem**: Download script fails for NIJ  
**Solution**: NIJ requires manual download. Follow the steps above.

**Problem**: "File not found" error when running pipeline  
**Solution**: Verify files are in the correct directories with correct names.

**Problem**: "Column not found" error  
**Solution**: Check that CSV headers match expected column names. You may need to rename columns.

## Contact

For dataset-specific questions:
- COMPAS: ProPublica (https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis)
- Adult/German: UCI ML Repository (archive@ics.uci.edu)
- NIJ: NACJD (nacjd@icpsr.umich.edu)

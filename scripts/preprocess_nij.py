"""
NIJ Recidivism Dataset Preprocessing Script

This script preprocesses the NIJ Recidivism Forecasting Challenge dataset
to make it compatible with the PACE pipeline.
"""
import argparse
import pandas as pd
import os


def preprocess_nij(input_path, output_path):
    """
    Preprocess NIJ dataset.
    
    Args:
        input_path: Path to raw NIJ CSV file
        output_path: Path to save processed CSV
    """
    print(f"Loading NIJ dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Original shape: {df.shape}")
    
    # Rename columns to standardized format
    column_mapping = {
        'ID': 'id',
        'Age_at_Release': 'age',
        'Gender': 'gender',
        'Race': 'race',
        'Gang_Affiliation': 'gang_affiliation',
        'Supervision_Level_First': 'supervision_level',
        'Education_Level': 'education',
        'Dependents': 'dependents',
        'Prison_Offense': 'prison_offense',
        'Prison_Years': 'prison_years',
        'Recidivism_Arrest_Year1': 'recidivism_year1',
        'Recidivism_Arrest_Year2': 'recidivism_year2',
        'Recidivism_Arrest_Year3': 'recidivism_year3'
    }
    
    # Select and rename columns
    available_cols = [col for col in column_mapping.keys() if col in df.columns]
    df_processed = df[available_cols].copy()
    df_processed.rename(columns={k: v for k, v in column_mapping.items() if k in available_cols}, inplace=True)
    
    # Handle missing values
    print("\nHandling missing values...")
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col].fillna('Unknown', inplace=True)
        else:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Standardize categorical values
    if 'gender' in df_processed.columns:
        df_processed['gender'] = df_processed['gender'].str.upper()
    
    if 'race' in df_processed.columns:
        df_processed['race'] = df_processed['race'].str.title()
    
    # Create binary recidivism label (Year 1)
    if 'recidivism_year1' in df_processed.columns:
        df_processed['recidivism'] = df_processed['recidivism_year1'].astype(int)
    
    print(f"\nProcessed shape: {df_processed.shape}")
    print(f"\nColumns: {list(df_processed.columns)}")
    
    # Save processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"\nProcessed dataset saved to {output_path}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(df_processed.describe())
    
    print("\n=== Categorical Value Counts ===")
    for col in ['gender', 'race', 'education']:
        if col in df_processed.columns:
            print(f"\n{col}:")
            print(df_processed[col].value_counts())


def main():
    parser = argparse.ArgumentParser(description="Preprocess NIJ Recidivism dataset")
    parser.add_argument("--input", required=True, help="Path to raw NIJ CSV file")
    parser.add_argument("--output", required=True, help="Path to save processed CSV")
    args = parser.parse_args()
    
    preprocess_nij(args.input, args.output)


if __name__ == "__main__":
    main()

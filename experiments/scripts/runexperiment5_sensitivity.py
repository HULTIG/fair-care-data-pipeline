"""
Experiment 5: Sensitivity Analysis
Varies epsilon and k-anonymity parameters and measures trade-offs.
"""
import argparse
import csv
import os
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from faircare.orchestration.pipeline import run_pipeline

def nested_update(base_dict, update_dict):
    import collections.abc
    for k, v in update_dict.items():
        if isinstance(v, collections.abc.Mapping):
            base_dict[k] = nested_update(base_dict.get(k, {}), v)
        else:
            base_dict[k] = v
    return base_dict

def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Sensitivity")
    parser.add_argument("--dataset", default="compas", help="Dataset name")
    parser.add_argument("--epsilons", default="0.1,1.0,5.0", help="Comma-separated epsilons")
    parser.add_argument("--ks", default="2,5,10", help="Comma-separated ks")
    parser.add_argument("--output", default="results/exp5_sensitivity.json", help="Output path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset
    epsilons = [float(e) for e in args.epsilons.split(',')]
    ks = [int(k) for k in args.ks.split(',')]

    base_config_path = "configs/default.yaml"
    if not os.path.exists(base_config_path):
        base_config_path = "experiments/configs/default.yaml"
        
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    results = []

    for epsilon in epsilons:
        for k in ks:
            print(f"\n{'='*60}")
            print(f"Running Sensitivity for: {dataset} | epsilon={epsilon}, k={k}")
            print(f"{'='*60}\n")
            
            try:
                merged_config = yaml.safe_load(yaml.safe_dump(base_config))
                # Base it off configa (has DP enabled)
                exp_config_path = f"experiments/configs/configa.yaml"
                if os.path.exists(exp_config_path):
                    with open(exp_config_path, 'r') as f:
                        exp_config = yaml.safe_load(f)
                    merged_config = nested_update(merged_config, exp_config)
                
                # Override parameters
                if 'anonymization' not in merged_config:
                    merged_config['anonymization'] = {}
                merged_config['anonymization']['epsilon'] = epsilon
                merged_config['anonymization']['k'] = k
                
                ds_config = merged_config['datasets'][dataset.strip()]
                base_processed = "data/processed/exp5"
                suffix = f"eps{epsilon}_k{k}"
                ds_config['bronze_path'] = f"{base_processed}/{suffix}/bronze"
                ds_config['silver_path'] = f"{base_processed}/{suffix}/silver"
                ds_config['gold_path'] = f"{base_processed}/{suffix}/gold"
                
                output_dir = f"results/exp5/{dataset}_{suffix}"
                
                metrics = run_pipeline(
                    dataset=dataset.strip(),
                    config_or_path=merged_config,
                    output_dir=output_dir,
                    verbose=args.verbose,
                    seed=42
                )
                
                results.append({
                    'dataset': dataset,
                    'epsilon': epsilon,
                    'k': k,
                    'fc_score': metrics.get('score', 0),
                    'utility': metrics.get('utility', {}).get('utility_retention', 0),
                    'privacy_risk': metrics.get('privacy', {}).get('risk', 0.1),
                    'dpd': metrics.get('fairness', {}).get('statistical_parity_difference', 0),
                    'locked': metrics.get('locked', False)
                })
            except Exception as e:
                print(f"Error on eps={epsilon}, k={k}: {e}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
            
    print(f"\nSensitivity results saved to {args.output}")

if __name__ == "__main__":
    main()

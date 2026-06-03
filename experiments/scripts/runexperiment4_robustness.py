"""
Experiment 4: Robustness and Statistical Rigor
Runs configurations across multiple random seeds and reports mean +/- std dev.
"""
import argparse
import csv
import os
import yaml
import numpy as np
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
    parser = argparse.ArgumentParser(description="Experiment 4: Robustness")
    parser.add_argument("--dataset", default="compas", help="Dataset name")
    parser.add_argument("--configs", default="baseline,configa", help="Comma-separated configs")
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated seeds")
    parser.add_argument("--output", default="results/exp4_robustness.json", help="Output path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset
    configs = args.configs.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]

    base_config_path = "configs/default.yaml"
    if not os.path.exists(base_config_path):
        base_config_path = "experiments/configs/default.yaml"
        
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    summary_results = []

    for config_name in configs:
        print(f"\n{'='*60}")
        print(f"Running Robustness for: {dataset} with {config_name}")
        print(f"{'='*60}\n")
        
        runs = []
        for seed in seeds:
            print(f"--- SEED {seed} ---")
            try:
                merged_config = yaml.safe_load(yaml.safe_dump(base_config))
                exp_config_path = f"experiments/configs/{config_name}.yaml"
                if os.path.exists(exp_config_path):
                    with open(exp_config_path, 'r') as f:
                        exp_config = yaml.safe_load(f)
                    merged_config = nested_update(merged_config, exp_config)
                
                ds_config = merged_config['datasets'][dataset.strip()]
                base_processed = "data/processed/exp4"
                ds_config['bronze_path'] = f"{base_processed}/{config_name}_{seed}/bronze"
                ds_config['silver_path'] = f"{base_processed}/{config_name}_{seed}/silver"
                ds_config['gold_path'] = f"{base_processed}/{config_name}_{seed}/gold"
                
                output_dir = f"results/exp4/{dataset}_{config_name}_{seed}"
                
                metrics = run_pipeline(
                    dataset=dataset.strip(),
                    config_or_path=merged_config,
                    output_dir=output_dir,
                    verbose=args.verbose,
                    seed=seed
                )
                runs.append(metrics)
            except Exception as e:
                print(f"Error on seed {seed}: {e}")
                
        if runs:
            # Calculate means and std
            faircare_scores = [r.get('score', 0) for r in runs]
            utils = [r.get('utility', {}).get('utility_retention', 0) for r in runs]
            dpds = [r.get('fairness', {}).get('statistical_parity_difference', 0) or 0 for r in runs]
            
            # Runtime overhead (compare to naive ETL later)
            runtimes = [r.get('runtimes', {}).get('total', 0) for r in runs]
            
            summary_results.append({
                'dataset': dataset,
                'config': config_name,
                'fc_score_mean': np.mean(faircare_scores),
                'fc_score_std': np.std(faircare_scores),
                'utility_mean': np.mean(utils),
                'utility_std': np.std(utils),
                'dpd_mean': np.mean(dpds),
                'dpd_std': np.std(dpds),
                'runtime_mean': np.mean(runtimes),
                'runtime_std': np.std(runtimes),
                'seeds_successful': len(runs)
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    import json
    with open(args.output, 'w') as f:
        json.dump(summary_results, f, indent=4)
            
    print(f"\nRobustness results saved to {args.output}")

if __name__ == "__main__":
    main()

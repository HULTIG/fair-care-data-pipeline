"""
Experiment 1: Ablation Study
Tests the impact of removing key PACE components.
"""
import argparse
import csv
import os
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pace.orchestration.pipeline import run_pipeline

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def nested_update(base_dict, update_dict):
    """Recursively update a dictionary."""
    import collections.abc
    for k, v in update_dict.items():
        if isinstance(v, collections.abc.Mapping):
            base_dict[k] = nested_update(base_dict.get(k, {}), v)
        else:
            base_dict[k] = v
    return base_dict

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Ablation Study")
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names")
    parser.add_argument("--configs", required=True, help="Comma-separated config names")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = args.datasets.split(',')
    configs = args.configs.split(',')
    
    # Load base default config first
    base_config_path = "configs/default.yaml"
    if not os.path.exists(base_config_path):
        # Fallback to experiments default if needed
        base_config_path = "experiments/configs/default.yaml"
        
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    print(f"Loaded base config from {base_config_path}")

    results = []
    
    for dataset in datasets:
        for config_name in configs:
            print(f"\n{'='*60}")
            print(f"Running: {dataset} with {config_name}")
            print(f"{'='*60}\n")
            
            try:
                # 1. Start with fresh copy of base config
                merged_config = yaml.safe_load(yaml.safe_dump(base_config))
                
                # 2. Load experiment-specific config
                exp_config_path = f"experiments/configs/{config_name}.yaml"
                if os.path.exists(exp_config_path):
                    with open(exp_config_path, 'r') as f:
                        exp_config = yaml.safe_load(f)
                    # 3. Merge experiment config OVER base config
                    merged_config = nested_update(merged_config, exp_config)
                else:
                    if config_name != "default":
                        print(f"Warning: {exp_config_path} not found. Using defaults.")
                
                # ISOLATION FIX: Update storage paths to be unique for this run
                # This prevents schema conflicts between different configs (e.g. baseline vs validaded)
                ds_config = merged_config['datasets'][dataset.strip()]
                base_processed = "data/processed/exp1"
                ds_config['bronze_path'] = f"{base_processed}/{config_name}/bronze/{dataset}"
                ds_config['silver_path'] = f"{base_processed}/{config_name}/silver/{dataset}"
                ds_config['gold_path'] = f"{base_processed}/{config_name}/gold/{dataset}"
                
                output_dir = f"results/exp1/{dataset}_{config_name}"
                
                # Run pipeline with MERGED config object
                metrics = run_pipeline(
                    dataset=dataset.strip(),
                    config_or_path=merged_config,
                    output_dir=output_dir,
                    verbose=args.verbose,
                    seed=args.seed
                )
                
                # Collect results with correct field mappings
                result = {
                    'dataset': dataset.strip(),
                    'config': config_name.strip(),
                    'SB': metrics.get('components', {}).get('bronze', 0),
                    'SS': metrics.get('components', {}).get('silver', 0),
                    'SG': metrics.get('components', {}).get('gold', 0),
                    'pacescore': metrics.get('score', 0),
                    'dpd': metrics.get('fairness', {}).get('statistical_parity_difference', None),
                    'di': metrics.get('fairness', {}).get('disparate_impact', None),
                    'utility': metrics.get('utility', {}).get('utility_retention', 0),  # Fixed: was 'retention'
                    'privacy_risk': metrics.get('privacy', {}).get('risk', 0.1),
                    'k': metrics.get('anonymization', {}).get('k', 0),
                    'epsilon': metrics.get('anonymization', {}).get('epsilon', float('inf')),
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error running {dataset} with {config_name}: {e}")
                continue
    
    # Write results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()

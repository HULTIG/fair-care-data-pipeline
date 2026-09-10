"""
Experiment 3: Regulatory Compliance
Tests GDPR, HIPAA, and CCPA compliance modes.
"""
import argparse
import csv
import os
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pace.orchestration.pipeline import run_pipeline

def nested_update(base, update):
    """Recursively update nested dictionaries."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            nested_update(base[key], value)
        else:
            base[key] = value
    return base

def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Regulatory Compliance")
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names")
    parser.add_argument("--regulations", required=True, help="Comma-separated regulation names (gdpr,hipaa,ccpa)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-root", default="results/exp3")
    args = parser.parse_args()

    datasets = args.datasets.split(',')
    regulations = args.regulations.split(',')
    
    results = []
    
    for dataset in datasets:
        for regulation in regulations:
            print(f"\n{'='*60}")
            print(f"Running: {dataset} with {regulation.upper()}")
            print(f"{'='*60}\n")
            
            # Load base config with dataset definitions
            with open('configs/default.yaml', 'r') as f:
                base_config = yaml.safe_load(f)
            
            # Load regulatory config and merge
            reg_config_path = f"experiments/configs/{regulation.strip()}.yaml"
            with open(reg_config_path, 'r') as f:
                reg_config = yaml.safe_load(f)
            
            # Merge configs
            merged_config = nested_update(base_config.copy(), reg_config)
            
            output_dir = os.path.join(args.artifact_root, "runs", f"{dataset}_{regulation}")
            
            try:
                # Run pipeline with merged config
                metrics = run_pipeline(
                    dataset=dataset.strip(),
                    config_or_path=merged_config,  # Pass dict instead of path
                    output_dir=output_dir,
                    verbose=args.verbose,
                    seed=args.seed
                )
                
                # Check compliance
                anon_config = metrics.get('anonymization', {})
                k = anon_config.get('k', 0)
                epsilon = anon_config.get('epsilon', float('inf'))
                privacy_risk = metrics.get('privacy', {}).get('risk', 1.0)
                
                # Compliance rules
                compliant = False
                if regulation.strip().lower() == 'gdpr':
                    compliant = (privacy_risk is not None and k >= 10 and epsilon <= 0.5 and privacy_risk < 0.05)
                elif regulation.strip().lower() == 'hipaa':
                    compliant = (privacy_risk is not None and k >= 5 and privacy_risk < 0.10)
                elif regulation.strip().lower() == 'ccpa':
                    compliant = (privacy_risk is not None and k >= 5 and privacy_risk < 0.15)
                
                # Collect results
                result = {
                    'dataset': dataset.strip(),
                    'regulation': regulation.strip().upper(),
                    'SB': metrics.get('components', {}).get('bronze', 0),
                    'SS': metrics.get('components', {}).get('silver', 0),
                    'SG': metrics.get('components', {}).get('gold', 0),
                    'pacescore': metrics.get('score', 0),
                    'k': k,
                    'epsilon': epsilon,
                    'privacy_risk': privacy_risk,
                    'compliant': compliant,
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error running {dataset} with {regulation}: {e}")
                results.append({
                    'dataset': dataset.strip(), 'regulation': regulation.strip().upper(),
                    'status': 'failed', 'error': str(e), 'compliant': False,
                })
    
    # Write results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        if results:
            fieldnames = list(dict.fromkeys(key for row in results for key in row))
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()

"""
Experiment 2: Multi-Dataset Benchmarking
Compares PACE performance across different datasets and anonymization techniques.
"""
import argparse
import csv
import os
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pace.orchestration.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Multi-Dataset Benchmarking")
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names")
    parser.add_argument("--config", required=True, help="Base config file")
    parser.add_argument("--techniques", default="kanonymity,ldiversity,tcloseness,numeric_noise",
                        help="Anonymization techniques to test")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-root", default="results/exp2")
    args = parser.parse_args()

    datasets = args.datasets.split(',')
    techniques = args.techniques.split(',')
    
    results = []
    
    for dataset in datasets:
        for technique in techniques:
            print(f"\n{'='*60}")
            print(f"Running: {dataset} with {technique}")
            print(f"{'='*60}\n")
            
            # Load base config and modify technique
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            config['anonymization']['technique'] = technique.strip()
            
            # Save modified config
            temp_config_path = f"experiments/configs/temp_{dataset}_{technique}.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f)
            
            output_dir = os.path.join(args.artifact_root, "runs", f"{dataset}_{technique}")
            
            try:
                # Run pipeline
                metrics = run_pipeline(
                    dataset=dataset.strip(),
                    config_or_path=temp_config_path,
                    output_dir=output_dir,
                    verbose=args.verbose,
                    seed=args.seed
                )
                
                # Collect results with correct field mappings
                result = {
                    'dataset': dataset.strip(),
                    'technique': technique.strip(),
                    'SB': metrics.get('components', {}).get('bronze', 0),
                    'SS': metrics.get('components', {}).get('silver', 0),
                    'SG': metrics.get('components', {}).get('gold', 0),
                    'pacescore': metrics.get('score', 0),
                    'dpd': metrics.get('fairness', {}).get('statistical_parity_difference', None),
                    'di': metrics.get('fairness', {}).get('disparate_impact', None),
                    'roc_auc': metrics.get('utility', {}).get('roc_auc'),
                    'utility_retention': metrics.get('utility', {}).get('utility_retention'),
                    'info_loss': metrics.get('privacy', {}).get('information_loss'),
                    'privacy_risk': metrics.get('privacy', {}).get('risk'),
                }
                results.append(result)
                
                # Clean up temp config
                os.remove(temp_config_path)
                
            except Exception as e:
                print(f"Error running {dataset} with {technique}: {e}")
                results.append({
                    'dataset': dataset.strip(), 'technique': technique.strip(),
                    'status': 'failed', 'error': str(e), 'roc_auc': None,
                })
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
                continue
    
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

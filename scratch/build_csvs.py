import os
import json
import csv

def build_exp1():
    base_dir = "results/exp1"
    if not os.path.exists(base_dir): return
    results = []
    for d in os.listdir(base_dir):
        dp = os.path.join(base_dir, d)
        if os.path.isdir(dp):
            parts = d.split('_')
            if len(parts) >= 2:
                dataset = parts[0]
                config = "_".join(parts[1:])
                json_path = os.path.join(dp, f"{dataset}_metricssummary.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        m = json.load(f)
                        result = {
                            'dataset': dataset,
                            'config': config,
                            'SB': m.get('components', {}).get('bronze', 0),
                            'SS': m.get('components', {}).get('silver', 0),
                            'SG': m.get('components', {}).get('gold', 0),
                            'pacescore': m.get('score', 0),
                            'dpd': m.get('fairness', {}).get('statistical_parity_difference', None),
                            'di': m.get('fairness', {}).get('disparate_impact', None),
                            'utility': m.get('utility', {}).get('utility_retention', 0),
                            'privacy_risk': m.get('privacy', {}).get('risk', 0.1),
                            'k': m.get('anonymization', {}).get('k', 0),
                            'epsilon': m.get('anonymization', {}).get('epsilon', float('inf')),
                        }
                        results.append(result)
    if results:
        with open("results/exp1.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote exp1.csv with {len(results)} rows")

def build_exp2():
    base_dir = "results/exp2"
    if not os.path.exists(base_dir): return
    results = []
    for d in os.listdir(base_dir):
        dp = os.path.join(base_dir, d)
        if os.path.isdir(dp):
            parts = d.split('_')
            if len(parts) >= 2:
                dataset = parts[0]
                technique = "_".join(parts[1:])
                json_path = os.path.join(dp, f"{dataset}_metricssummary.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        m = json.load(f)
                        result = {
                            'dataset': dataset,
                            'technique': technique,
                            'SB': m.get('components', {}).get('bronze', 0),
                            'SS': m.get('components', {}).get('silver', 0),
                            'SG': m.get('components', {}).get('gold', 0),
                            'pacescore': m.get('score', 0),
                            'dpd': m.get('fairness', {}).get('statistical_parity_difference', None),
                            'di': m.get('fairness', {}).get('disparate_impact', None),
                            'utility': m.get('utility', {}).get('utility_retention', 0),
                            'privacy_risk': m.get('privacy', {}).get('risk', 0.1)
                        }
                        results.append(result)
    if results:
        with open("results/exp2.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote exp2.csv with {len(results)} rows")

def build_exp3():
    base_dir = "results/exp3"
    if not os.path.exists(base_dir): return
    results = []
    for d in os.listdir(base_dir):
        dp = os.path.join(base_dir, d)
        if os.path.isdir(dp):
            parts = d.split('_')
            if len(parts) >= 2:
                dataset = parts[0]
                regulation = "_".join(parts[1:])
                json_path = os.path.join(dp, f"{dataset}_metricssummary.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        m = json.load(f)
                        result = {
                            'dataset': dataset,
                            'regulation': regulation.upper(),
                            'SB': m.get('components', {}).get('bronze', 0),
                            'SS': m.get('components', {}).get('silver', 0),
                            'SG': m.get('components', {}).get('gold', 0),
                            'pacescore': m.get('score', 0),
                            'dpd': m.get('fairness', {}).get('statistical_parity_difference', None),
                            'privacy_risk': m.get('privacy', {}).get('risk', 0.1),
                            'compliance_status': m.get('status', 'UNKNOWN')
                        }
                        results.append(result)
    if results:
        with open("results/exp3.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote exp3.csv with {len(results)} rows")

if __name__ == "__main__":
    build_exp1()
    build_exp2()
    build_exp3()

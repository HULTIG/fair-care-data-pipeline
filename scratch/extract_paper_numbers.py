import json
import os
import pandas as pd
import numpy as np

def print_section(title):
    print("\n" + "="*50)
    print(title)
    print("="*50)

# 1. Overhead
print_section("6.1 Overhead (from exp4_robustness_compas.json)")
with open("results/exp4_robustness_compas.json", "r") as f:
    r_compas = json.load(f)
    for row in r_compas:
        print(f"Config: {row['config']} -> runtime_mean: {row['runtime_mean']:.1f} +- {row['runtime_std']:.1f}")

# 2. Profiles
print_section("6.2 Profiles (from exp1.csv baseline)")
exp1 = pd.read_csv("results/exp1.csv")
baselines = exp1[exp1['config'] == 'baseline']
for _, r in baselines.iterrows():
    print(f"{r['dataset']}: Bronze={r['SB']:.3f}, Silver={r['SS']:.3f}, Gold={r['SG']:.3f}, PACE={r['pacescore']:.3f}")

# 3. NIJ EOD
print_section("6.3 NIJ Layer Attribution (from metricssummary JSONs)")
nij_b = json.load(open("results/exp1/nij_baseline/nij_metricssummary.json"))
nij_a = json.load(open("results/exp1/nij_configa/nij_metricssummary.json"))
print(f"NIJ Baseline EOD: {nij_b['fairness'].get('equal_opportunity_difference', 0):.3f}")
print(f"NIJ ConfigA EOD: {nij_a['fairness'].get('equal_opportunity_difference', 0):.3f}")
# DP is from exp2? The text says DP at epsilon=1.0 and Gold Reweighing. 
# Wait, DP at epsilon=1.0 is configb!
nij_b_dp = json.load(open("results/exp1/nij_configb/nij_metricssummary.json"))
print(f"NIJ ConfigB (DP) EOD: {nij_b_dp['fairness'].get('equal_opportunity_difference', 0):.3f}")

# 4. Mechanisms
print_section("6.4 Mechanism comparison (from exp2.csv)")
exp2 = pd.read_csv("results/exp2.csv")
tech_paces = exp2.groupby('technique')['pacescore'].mean()
for t, v in tech_paces.items():
    print(f"Technique {t}: {v:.3f}")

# 5. Ablation
print_section("6.5 Configuration ablation (from exp1.csv COMPAS)")
compas_ab = exp1[exp1['dataset'] == 'compas']
for _, r in compas_ab.iterrows():
    print(f"{r['config']}: {r['pacescore']:.3f}")

# 6. Variance
print_section("6.6 Variance (from exp4_robustness_compas.json)")
for row in r_compas:
    print(f"Config: {row['config']}")
    print(f"  AUC: {row.get('utility_mean', 0):.2f} +- {row.get('utility_std', 0):.2f}")
    # Wait, exp4 doesn't have EOD mean/std! Only DPD mean/std!
    print(f"  DPD: {row.get('dpd_mean', 0):.2f} +- {row.get('dpd_std', 0):.2f}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_robustness(json_path="results/exp4_robustness.json", out_dir="results/figures"):
    if not os.path.exists(json_path):
        print(f"File {json_path} not found. Run exp4 first.")
        return
        
    df = pd.read_json(json_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Bar chart for FAIR-CARE Score with error bars (std dev)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df, 
        x="config", 
        y="fc_score_mean", 
        hue="dataset",
        capsize=.1,
        palette="viridis"
    )
    # Add error bars manually as seaborn barplot uses raw data for ci
    for i, config in enumerate(df['config'].unique()):
        subset = df[df['config'] == config]
        plt.errorbar(
            x=[i]*len(subset), 
            y=subset['fc_score_mean'], 
            yerr=subset['fc_score_std'], 
            fmt='none', 
            c='black', 
            capsize=5
        )
        
    plt.ylim(0, 1.1)
    plt.title("Statistical Robustness: FAIR-CARE Score Across 5 Seeds")
    plt.ylabel("FAIR-CARE Score (Mean ± Std)")
    plt.xlabel("Configuration")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp4_robustness_fc_score.png"), dpi=300)
    plt.close()
    
    # 2. Runtime Overhead Chart
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="config",
        y="runtime_mean",
        hue="dataset",
        palette="magma"
    )
    plt.title("System-Level Efficiency: Pipeline Execution Time")
    plt.ylabel("Total Runtime (seconds)")
    plt.xlabel("Configuration")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp4_runtime_overhead.png"), dpi=300)
    plt.close()
    
    print(f"Robustness plots saved to {out_dir}")

def plot_sensitivity(json_path="results/exp5_sensitivity.json", out_dir="results/figures"):
    if not os.path.exists(json_path):
        print(f"File {json_path} not found. Run exp5 first.")
        return
        
    df = pd.read_json(json_path)
    os.makedirs(out_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # Plot trade-offs as epsilon changes
    plt.figure(figsize=(10, 6))
    
    # Filter for a specific k to visualize epsilon impact cleanly
    k_val = 5
    subset = df[df['k'] == k_val]
    
    if subset.empty:
        print(f"No data for k={k_val} to plot sensitivity.")
        return
        
    sns.lineplot(data=subset, x="epsilon", y="fc_score", marker='o', label="FAIR-CARE Score")
    sns.lineplot(data=subset, x="epsilon", y="utility", marker='s', label="Utility (AUC)")
    sns.lineplot(data=subset, x="epsilon", y="privacy_risk", marker='^', label="Privacy Risk")
    
    plt.xscale('log') # Epsilon is often viewed logarithmically
    plt.title(f"Hyperparameter Sensitivity Analysis (k={k_val})")
    plt.ylabel("Metric Value")
    plt.xlabel("Privacy Budget (Epsilon) - Log Scale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp5_sensitivity_tradeoff.png"), dpi=300)
    plt.close()
    
    print(f"Sensitivity plots saved to {out_dir}")

if __name__ == "__main__":
    plot_robustness()
    plot_sensitivity()

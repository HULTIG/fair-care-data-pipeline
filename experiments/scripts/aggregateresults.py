"""
Aggregate experiment results and generate figures for the paper.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_results(csv_paths):
    """Load all experiment CSV files"""
    results = {}
    for path in csv_paths:
        name = os.path.basename(path).replace('.csv', '')
        results[name] = pd.read_csv(path)
    return results


def plot_ablation_pace(df, output_dir):
    """Figure 1: PACE scores by configuration (Ablation)"""
    plt.figure(figsize=(12, 6))
    
    # Group by config and compute mean PACE score
    grouped = df.groupby('config')['pacescore'].mean().sort_values()
    
    ax = grouped.plot(kind='bar', color='steelblue')
    plt.title('Ablation Study: PACE Score by Configuration', fontsize=14, fontweight='bold')
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('PACE Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.axhline(y=0.85, color='green', linestyle='--', label='EXCELLENT threshold')
    plt.axhline(y=0.70, color='orange', linestyle='--', label='ACCEPTABLE threshold')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'fig1_ablation_pace.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig1_ablation_pace.pdf'))
    plt.close()


def plot_ablation_fairness(df, output_dir):
    """Figure 2: Fairness metrics by configuration (Ablation)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # DPD plot
    grouped_dpd = df.groupby('config')['dpd'].mean()
    grouped_dpd.plot(kind='bar', ax=axes[0], color='coral')
    axes[0].set_title('Statistical Parity Difference', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Configuration')
    axes[0].set_ylabel('DPD')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].axhline(y=0.1, color='red', linestyle='--', label='Threshold')
    axes[0].axhline(y=-0.1, color='red', linestyle='--')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    
    # DI plot
    grouped_di = df.groupby('config')['di'].mean()
    grouped_di.plot(kind='bar', ax=axes[1], color='lightgreen')
    axes[1].set_title('Disparate Impact', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Configuration')
    axes[1].set_ylabel('DI')
    axes[1].axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
    axes[1].axhline(y=0.8, color='red', linestyle='--', label='Threshold')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_ablation_fairness.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig2_ablation_fairness.pdf'))
    plt.close()


def plot_benchmark_datasets(df, output_dir):
    """Figure 3: Performance across datasets"""
    plt.figure(figsize=(12, 6))
    
    # Group by dataset
    grouped = df.groupby('dataset')[['SB', 'SS', 'SG', 'pacescore']].mean()
    
    grouped.plot(kind='bar', ax=plt.gca())
    plt.title('Multi-Dataset Benchmarking: Layer Scores', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend(['Bronze (SB)', 'Silver (SS)', 'Gold (SG)', 'PACE'], loc='lower right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'fig3_benchmark_datasets.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig3_benchmark_datasets.pdf'))
    plt.close()


def plot_benchmark_techniques(df, output_dir):
    """Figure 4: Anonymization technique comparison"""
    if 'technique' not in df.columns:
        print("Skipping technique comparison (no technique column)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PACE by technique
    grouped_fc = df.groupby('technique')['pacescore'].mean().sort_values()
    grouped_fc.plot(kind='barh', ax=axes[0], color='skyblue')
    axes[0].set_title('PACE Score by Technique', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('PACE Score')
    axes[0].set_ylabel('Technique')
    axes[0].set_xlim(0, 1.0)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.3f', padding=3)
    
    # Utility vs Privacy
    grouped = df.groupby('technique')[['utility', 'privacy_risk']].mean()
    grouped.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Utility vs Privacy Risk', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Technique')
    axes[1].set_ylabel('Score')
    axes[1].legend(['Utility Retention', 'Privacy Risk'])
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_benchmark_techniques.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig4_benchmark_techniques.pdf'))
    plt.close()


def plot_regulatory_compliance(df, output_dir):
    """Figure 5: Compliance by regulation"""
    plt.figure(figsize=(12, 6))
    
    # Group by regulation
    grouped = df.groupby('regulation')[['pacescore', 'privacy_risk']].mean()
    
    ax = grouped.plot(kind='bar')
    plt.title('Regulatory Compliance: PACE Score and Privacy Risk', fontsize=14, fontweight='bold')
    plt.xlabel('Regulation', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(['PACE Score', 'Privacy Risk'])
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'fig5_regulatory_compliance.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig5_regulatory_compliance.pdf'))
    plt.close()


def plot_tradeoffs(df, output_dir):
    """Figure 6: Utility vs Privacy vs Fairness tradeoffs"""
    if 'utility' not in df.columns or 'privacy_risk' not in df.columns:
        print("Skipping tradeoffs plot (missing columns)")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot: Utility vs Privacy, colored by Fairness
    if 'dpd' in df.columns:
        fairness_score = 1 - df['dpd'].abs()  # Convert DPD to fairness score
    else:
        fairness_score = df['pacescore']
    
    scatter = plt.scatter(df['utility'], df['privacy_risk'], 
                         c=fairness_score, cmap='RdYlGn', 
                         s=100, alpha=0.6, edgecolors='black')
    
    plt.colorbar(scatter, label='Fairness Score')
    plt.title('Utility-Privacy-Fairness Tradeoffs', fontsize=14, fontweight='bold')
    plt.xlabel('Utility Retention', fontsize=12)
    plt.ylabel('Privacy Risk', fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    
    # Add quadrant lines
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.3, label='Low Privacy Risk')
    plt.axvline(x=0.8, color='blue', linestyle='--', alpha=0.3, label='High Utility')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_tradeoffs.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig6_tradeoffs.pdf'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results and generate figures")
    parser.add_argument("--inputs", required=True, help="Comma-separated list of CSV files")
    parser.add_argument("--output", required=True, help="Output directory for figures")
    parser.add_argument("--format", default="png,pdf", help="Output formats (comma-separated)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load results
    csv_paths = args.inputs.split(',')
    results = load_results(csv_paths)

    print(f"Loaded {len(results)} result files")
    for name, df in results.items():
        print(f"  {name}: {len(df)} rows")

    # Generate figures
    print("\nGenerating figures...")
    
    if 'exp1' in results:
        print("  - Ablation study figures...")
        plot_ablation_pace(results['exp1'], args.output)
        plot_ablation_fairness(results['exp1'], args.output)
    
    if 'exp2' in results:
        print("  - Benchmarking figures...")
        plot_benchmark_datasets(results['exp2'], args.output)
        plot_benchmark_techniques(results['exp2'], args.output)
        plot_tradeoffs(results['exp2'], args.output)
    
    if 'exp3' in results:
        print("  - Regulatory compliance figures...")
        plot_regulatory_compliance(results['exp3'], args.output)

    print(f"\nFigures saved to {args.output}")
    print("Generated files:")
    for f in sorted(os.listdir(args.output)):
        print(f"  - {f}")

if __name__ == "__main__":
    main()

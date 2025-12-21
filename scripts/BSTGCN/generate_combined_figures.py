"""
Generate combined figures for the paper:
1. Uncertainty quality combined figure (Error by Decile + Risk-Coverage)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# IEEE style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})


def load_results(results_dir, data_dir):
    """Load test results from the latest BSTGCN run"""
    predictions = np.load(os.path.join(results_dir, 'test_predictions.npy'))
    uncertainties = np.load(os.path.join(results_dir, 'test_uncertainties.npy'))
    labels = np.load(os.path.join(data_dir, 'test_labels.npy'))
    return predictions, uncertainties, labels


def generate_combined_uncertainty_figure(preds, uncertainties, labels, output_path):
    """Generate combined uncertainty quality figure"""
    epistemic = uncertainties
    preds = preds.astype(int)
    labels = labels.astype(int)

    errors = (labels != preds).astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    
    # Left: Error rate by uncertainty decile
    ax1 = axes[0]
    n_deciles = 10
    decile_edges = np.percentile(epistemic, np.linspace(0, 100, n_deciles + 1))
    decile_errors = []
    decile_centers = []
    
    for i in range(n_deciles):
        mask = (epistemic >= decile_edges[i]) & (epistemic < decile_edges[i+1])
        if i == n_deciles - 1:
            mask = (epistemic >= decile_edges[i]) & (epistemic <= decile_edges[i+1])
        if mask.sum() > 0:
            decile_errors.append(errors[mask].mean())
            decile_centers.append((i + 1) * 10)
    
    ax1.bar(decile_centers, decile_errors, width=8, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Uncertainty Percentile')
    ax1.set_ylabel('Error Rate')
    ax1.set_title('(a) Error Rate by Uncertainty Decile')
    ax1.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax1.set_xlim(0, 110)
    ax1.set_ylim(0, max(decile_errors) * 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Risk-Coverage curve
    ax2 = axes[1]
    sorted_indices = np.argsort(epistemic)
    sorted_errors = errors[sorted_indices]
    
    coverages = np.linspace(0.1, 1.0, 50)
    risks = []
    for cov in coverages:
        n_samples = int(len(sorted_errors) * cov)
        if n_samples > 0:
            risks.append(sorted_errors[:n_samples].mean())
        else:
            risks.append(0)
    
    ax2.plot(coverages * 100, np.array(risks) * 100, 'b-', linewidth=2, label='BSTGCN')
    ax2.axhline(y=errors.mean() * 100, color='r', linestyle='--', linewidth=1, label='Random baseline')
    ax2.fill_between(coverages * 100, 0, np.array(risks) * 100, alpha=0.2)
    ax2.set_xlabel('Coverage (%)')
    ax2.set_ylabel('Error Rate (%)')
    ax2.set_title('(b) Risk-Coverage Curve')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, max(risks) * 100 * 1.2)
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    results_dir = 'baseline/results/bstgcn_50packets_23feat/20251220_152732'
    data_dir = 'data/processed/raw_packet_sequences_50packets_23feat'
    output_dir = 'docs/thesis/figures'
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    predictions, uncertainties, labels = load_results(results_dir, data_dir)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Uncertainties shape: {uncertainties.shape}")
    print(f"  Labels shape: {labels.shape}")

    print("\nGenerating combined uncertainty quality figure...")
    generate_combined_uncertainty_figure(
        predictions, uncertainties, labels,
        os.path.join(output_dir, 'uncertainty_quality_combined.pdf')
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()


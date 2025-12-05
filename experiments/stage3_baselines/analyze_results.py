
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.stats_utils import holm_bonferroni_correction, verify_assumptions_t_test


def analyze_results(results_path="experiments/stage3_results/results.csv", output_path="experiments/stage3_results/analysis_summary.txt"):
    print(f"Analyzing results from {results_path}...")

    try:
        df = pd.read_csv(results_path)
    except FileNotFoundError:
        print("Results file not found.")
        return

    tasks = df['task'].unique()
    models = df['model'].unique()

    with open(output_path, "w") as f:
        f.write("=== Stage 3 Validation Analysis ===\n\n")

        for task in tasks:
            f.write(f"--- Task: {task} ---\n")
            task_df = df[df['task'] == task]

            # 1. Descriptive Statistics
            f.write("Descriptive Statistics (RMSE):\n")
            stats_df = task_df.groupby('model')['rmse'].agg(['mean', 'std', 'count'])
            f.write(stats_df.to_string() + "\n\n")

            # 2. Statistical Tests (Director vs Baselines)
            f.write("Statistical Tests (Director vs Baselines):\n")
            director_scores = task_df[task_df['model'] == 'Epistemic Director']['rmse'].values

            if len(director_scores) < 2:
                f.write("Insufficient data for statistical tests.\n\n")
                continue

            p_values = []
            comparisons = []

            for model in models:
                if model == 'Epistemic Director':
                    continue

                baseline_scores = task_df[task_df['model'] == model]['rmse'].values
                if len(baseline_scores) < 2:
                    continue

                # Use stats_utils for rigorous testing
                test_type = verify_assumptions_t_test(director_scores, baseline_scores)

            if test_type == "standard_t":
                stat, p_val = stats.ttest_ind(director_scores, baseline_scores)
            elif test_type == "welch_t":
                stat, p_val = stats.ttest_ind(director_scores, baseline_scores, equal_var=False)
            elif test_type == "mann_whitney":
                stat, p_val = stats.mannwhitneyu(director_scores, baseline_scores)
            else:
                stat, p_val = 0.0, 1.0

                comparisons.append(f"Director vs {model}")
                p_values.append(p_val)

                f.write(f"  vs {model}: {test_type}, p={p_val:.4f} (Mean Diff: {np.mean(director_scores) - np.mean(baseline_scores):.4f})\n")

            # Bonferroni Correction
            if p_values:
                corrected_p, rejected = holm_bonferroni_correction(p_values)
                f.write("\n  Holm-Bonferroni Correction:\n")
                for comp, p, corr_p, rej in zip(comparisons, p_values, corrected_p, rejected):
                    sig = "*" if rej else ""
                    f.write(f"    {comp}: p_adj={corr_p:.4f} {sig}\n")

            f.write("\n")

            # 3. Soft Wall Index Analysis (if applicable)
            # For now, we just look at OOD detection rate for Director
            if task in ['chaotic', 'adversarial', 'discontinuous']:
                 f.write("Epistemic Metrics:\n")
                 # Check if formula contains warning/suppressed tags (we need the raw formula from CSV, not the stripped one used in prediction)
                 # Wait, the CSV contains the raw formula string with tags!
                 director_formulas = task_df[task_df['model'] == 'Epistemic Director']['metadata'].values

                 discontinuity_count = sum(1 for meta in director_formulas if "Discontinuity" in str(meta))
                 suppression_count = sum(1 for meta in director_formulas if "Suppressed" in str(meta))

                 f.write(f"  Discontinuity Warnings: {discontinuity_count}/{len(director_scores)}\n")
                 f.write(f"  Suppression Active: {suppression_count}/{len(director_scores)}\n")
                 f.write("\n")

    print(f"Analysis complete. Saved to {output_path}")

def analyze_ablation():
    results_file = "experiments/stage3_results/ablation_results.csv"
    if not os.path.exists(results_file):
        print("No ablation results found.")
        return

    print(f"Analyzing ablation results from {results_file}...")
    try:
        df = pd.read_csv(results_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    with open("experiments/stage3_results/analysis_summary.txt", "a") as f:
        f.write("\n\n=== Ablation Study Analysis ===\n")

        ablations = df['ablation'].unique()
        for ablation in ablations:
            subset = df[df['ablation'] == ablation]
            task = subset['task'].iloc[0] # Assuming one task per ablation type

            f.write(f"\n--- Ablation: {ablation} (Task: {task}) ---\n")

            # Descriptive Stats
            desc = subset.groupby('ablation')['rmse'].describe()
            f.write("Descriptive Statistics (RMSE):\n")
            f.write(desc.to_string())
            f.write("\n")

    print("Ablation analysis appended to summary.")

if __name__ == "__main__":
    analyze_results()
    analyze_ablation()

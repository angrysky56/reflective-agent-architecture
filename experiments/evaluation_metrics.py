"""
Evaluation Metrics for RAA Experiments
======================================

Comprehensive metrics for assessing:
1. Task performance (accuracy, success rate)
2. Entropy dynamics (reduction, variance, trajectories)
3. Search efficiency (reframing count, convergence time)
4. Comparative analysis (RAA vs baselines)
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class EntropyMetrics:
    """
    Metrics for analyzing entropy trajectories.

    Key questions:
    - Does entropy decrease during successful problem solving?
    - Is entropy reduction correlated with solution accuracy?
    - How does reframing affect entropy dynamics?
    """

    @staticmethod
    def compute_reduction(trajectory: List[float]) -> float:
        """
        Compute total entropy reduction.

        Returns:
            Δ Entropy: initial - final entropy
        """
        if not trajectory or len(trajectory) < 2:
            return 0.0
        return trajectory[0] - trajectory[-1]

    @staticmethod
    def compute_variance(trajectory: List[float]) -> float:
        """Compute variance in entropy trajectory."""
        if not trajectory:
            return 0.0
        return float(np.var(trajectory))

    @staticmethod
    def compute_convergence_rate(trajectory: List[float]) -> float:
        """
        Compute rate of entropy convergence.

        Uses exponential fit: H(t) = H_∞ + (H_0 - H_∞) * exp(-λt)
        Returns λ (convergence rate).
        """
        if not trajectory or len(trajectory) < 3:
            return 0.0

        # Simple linear approximation of log-space
        t = np.arange(len(trajectory))
        h = np.array(trajectory)

        # Avoid log of zero
        h = np.maximum(h, 1e-10)

        try:
            # Fit linear regression to log(H - min(H))
            h_shifted = h - h.min() + 1e-10
            log_h = np.log(h_shifted)

            # Linear fit
            slope, _ = np.polyfit(t, log_h, 1)
            return -slope  # Negative slope = convergence rate

        except Exception:
            return 0.0

    @staticmethod
    def detect_reframing_events(trajectory: List[float], threshold: float = 0.1) -> List[int]:
        """
        Detect sudden entropy increases (potential reframing events).

        Args:
            trajectory: Entropy values over time
            threshold: Minimum increase to count as reframing

        Returns:
            Indices where reframing likely occurred
        """
        if not trajectory or len(trajectory) < 2:
            return []

        reframing_indices = []
        for i in range(1, len(trajectory)):
            increase = trajectory[i] - trajectory[i - 1]
            if increase > threshold:
                reframing_indices.append(i)

        return reframing_indices

    @staticmethod
    def compute_exploration_efficiency(trajectory: List[float], final_success: bool) -> float:
        """
        Compute efficiency of exploration.

        Efficiency = (Entropy reduction) / (Number of steps)

        Higher efficiency = faster convergence to solution.
        """
        if not trajectory:
            return 0.0

        reduction = EntropyMetrics.compute_reduction(trajectory)
        steps = len(trajectory)

        if not final_success:
            # Penalize unsuccessful attempts
            reduction *= 0.1

        return reduction / max(steps, 1)


class PerformanceMetrics:
    """Task performance metrics."""

    @staticmethod
    def compute_accuracy(predictions: List[str], targets: List[str]) -> float:
        """Compute accuracy (exact match)."""
        if not predictions or not targets:
            return 0.0

        correct = sum(p.lower().strip() == t.lower().strip() for p, t in zip(predictions, targets))
        return correct / len(predictions)

    @staticmethod
    def compute_accuracy_by_difficulty(results: List[Dict]) -> Dict[str, float]:
        """
        Break down accuracy by difficulty level.

        Args:
            results: List of result dicts with 'difficulty' and 'correct' keys

        Returns:
            Dict mapping difficulty → accuracy
        """
        accuracy_by_diff = {}

        for difficulty in ["easy", "medium", "hard"]:
            diff_results = [r for r in results if r.get("difficulty") == difficulty]
            if diff_results:
                correct = sum(1 for r in diff_results if r.get("correct", False))
                accuracy_by_diff[difficulty] = correct / len(diff_results)

        return accuracy_by_diff

    @staticmethod
    def compute_success_rate_vs_reframing(results: List[Dict]) -> Tuple[List[int], List[float]]:
        """
        Analyze success rate as function of reframing count.

        Returns:
            reframing_counts: Unique reframing counts
            success_rates: Success rate for each count
        """
        from collections import defaultdict

        count_to_results = defaultdict(list)
        for r in results:
            count = r.get("reframing_count", 0)
            count_to_results[count].append(r.get("correct", False))

        reframing_counts = sorted(count_to_results.keys())
        success_rates = [
            sum(count_to_results[c]) / len(count_to_results[c]) for c in reframing_counts
        ]

        return reframing_counts, success_rates


class ComparativeMetrics:
    """Metrics for comparing RAA vs baseline."""

    @staticmethod
    def compute_relative_improvement(raa_accuracy: float, baseline_accuracy: float) -> float:
        """
        Compute relative improvement.

        Returns:
            Percentage improvement over baseline
        """
        if baseline_accuracy == 0:
            return float("inf") if raa_accuracy > 0 else 0.0

        return (raa_accuracy - baseline_accuracy) / baseline_accuracy * 100

    @staticmethod
    def compute_cohens_d(raa_scores: List[float], baseline_scores: List[float]) -> float:
        """
        Compute Cohen's d effect size.

        Measures standardized difference between RAA and baseline.

        Interpretation:
        - |d| < 0.2: Small effect
        - |d| < 0.5: Medium effect
        - |d| >= 0.8: Large effect
        """
        if not raa_scores or not baseline_scores:
            return 0.0

        mean_raa = np.mean(raa_scores)
        mean_baseline = np.mean(baseline_scores)

        std_raa = np.std(raa_scores)
        std_baseline = np.std(baseline_scores)

        # Pooled standard deviation
        n_raa = len(raa_scores)
        n_baseline = len(baseline_scores)

        pooled_std = np.sqrt(
            ((n_raa - 1) * std_raa**2 + (n_baseline - 1) * std_baseline**2)
            / (n_raa + n_baseline - 2)
        )

        if pooled_std == 0:
            return 0.0

        return (mean_raa - mean_baseline) / pooled_std

    @staticmethod
    def run_significance_test(
        raa_scores: List[float], baseline_scores: List[float]
    ) -> Tuple[float, float]:
        """
        Run statistical significance test (t-test).

        Returns:
            t_statistic: Test statistic
            p_value: Probability of null hypothesis
        """
        if not raa_scores or not baseline_scores:
            return 0.0, 1.0

        return stats.ttest_ind(raa_scores, baseline_scores)


class VisualizationTools:
    """Tools for visualizing evaluation results."""

    @staticmethod
    def plot_entropy_trajectories(
        successful_trajs: List[List[float]],
        failed_trajs: List[List[float]],
        output_path: Optional[str] = None,
    ):
        """
        Plot entropy trajectories for successful vs failed attempts.

        Args:
            successful_trajs: Entropy trajectories for correct solutions
            failed_trajs: Entropy trajectories for incorrect attempts
            output_path: Path to save figure (if None, display only)
        """
        plt.figure(figsize=(10, 6))

        # Plot successful trajectories
        for traj in successful_trajs:
            plt.plot(traj, color="green", alpha=0.3, linewidth=0.5)

        # Plot failed trajectories
        for traj in failed_trajs:
            plt.plot(traj, color="red", alpha=0.3, linewidth=0.5)

        # Plot means
        if successful_trajs:
            max_len = max(len(t) for t in successful_trajs)
            padded = [t + [t[-1]] * (max_len - len(t)) for t in successful_trajs]
            mean_successful = np.mean(padded, axis=0)
            plt.plot(mean_successful, color="darkgreen", linewidth=2, label="Successful (mean)")

        if failed_trajs:
            max_len = max(len(t) for t in failed_trajs)
            padded = [t + [t[-1]] * (max_len - len(t)) for t in failed_trajs]
            mean_failed = np.mean(padded, axis=0)
            plt.plot(mean_failed, color="darkred", linewidth=2, label="Failed (mean)")

        plt.xlabel("Processing Step")
        plt.ylabel("Entropy")
        plt.title("Entropy Trajectories: Successful vs Failed Solutions")
        plt.legend()
        plt.grid(alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {output_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_accuracy_comparison(
        raa_accuracy: Dict[str, float],
        baseline_accuracy: Dict[str, float],
        output_path: Optional[str] = None,
    ):
        """
        Plot accuracy comparison across difficulty levels.

        Args:
            raa_accuracy: Dict of difficulty → accuracy for RAA
            baseline_accuracy: Dict of difficulty → accuracy for baseline
            output_path: Path to save figure
        """
        difficulties = ["easy", "medium", "hard"]
        raa_scores = [raa_accuracy.get(d, 0) for d in difficulties]
        baseline_scores = [baseline_accuracy.get(d, 0) for d in difficulties]

        x = np.arange(len(difficulties))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, raa_scores, width, label="RAA", color="blue", alpha=0.7)
        ax.bar(x + width / 2, baseline_scores, width, label="Baseline", color="gray", alpha=0.7)

        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Accuracy")
        ax.set_title("RAA vs Baseline: Accuracy by Difficulty")
        ax.set_xticks(x)
        ax.set_xticklabels(difficulties)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {output_path}")
        else:
            plt.show()

        plt.close()


def generate_comparison_report(
    raa_results: Dict, baseline_results: Dict, output_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive comparison report.

    Args:
        raa_results: Results dict from RAA evaluation
        baseline_results: Results dict from baseline evaluation
        output_path: Optional path to save report

    Returns:
        Formatted report string
    """
    # Extract accuracies
    raa_acc = raa_results.get("summary", {}).get("accuracy", 0)
    baseline_acc = baseline_results.get("summary", {}).get("accuracy", 0)

    # Compute improvement
    improvement = ComparativeMetrics.compute_relative_improvement(raa_acc, baseline_acc)

    # Compute effect size (if detailed results available)
    effect_size = 0.0
    if "detailed_results" in raa_results and "detailed_results" in baseline_results:
        raa_scores = [1.0 if r["correct"] else 0.0 for r in raa_results["detailed_results"]]
        baseline_scores = [
            1.0 if r["correct"] else 0.0 for r in baseline_results["detailed_results"]
        ]
        effect_size = ComparativeMetrics.compute_cohens_d(raa_scores, baseline_scores)

    report = f"""
{'='*70}
RAA vs BASELINE: Comparative Analysis
{'='*70}

Overall Performance:
  RAA Accuracy:      {raa_acc:.1%}
  Baseline Accuracy: {baseline_acc:.1%}
  Improvement:       {improvement:+.1f}%
  Effect Size (d):   {effect_size:.3f}

"""

    # Accuracy by difficulty
    if "summary" in raa_results:
        report += "Performance by Difficulty:\n"
        for diff in ["easy", "medium", "hard"]:
            raa_diff_acc = raa_results["summary"].get(f"accuracy_{diff}", 0)
            base_diff_acc = baseline_results["summary"].get(f"accuracy_{diff}", 0)
            diff_improvement = (raa_diff_acc - base_diff_acc) * 100

            report += f"  {diff.capitalize():8s}: RAA {raa_diff_acc:.1%} vs Baseline {base_diff_acc:.1%} ({diff_improvement:+.1f}pp)\n"

    # Entropy analysis (RAA only)
    if "summary" in raa_results:
        avg_entropy_reduction = raa_results["summary"].get("avg_entropy_reduction", 0)
        avg_reframing = raa_results["summary"].get("avg_reframing_count", 0)

        report += f"""
RAA-Specific Metrics:
  Avg Entropy Reduction:  {avg_entropy_reduction:.3f}
  Avg Reframing Count:    {avg_reframing:.1f}
"""

    report += "\n" + "=" * 70

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to {output_path}")

    return report

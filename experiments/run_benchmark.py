#!/usr/bin/env python3
"""
RAA Benchmark Runner
===================

Main script to run comprehensive benchmarking suite:
1. RAA evaluation on RAT
2. Baseline evaluation on RAT
3. Comparative analysis
4. Visualization of results

Usage:
    python experiments/run_benchmark.py --mode full
    python experiments/run_benchmark.py --mode raa-only
    python experiments/run_benchmark.py --mode baseline-only
    python experiments/run_benchmark.py --mode compare
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from datetime import datetime
from pathlib import Path

from experiments.baselines.transformer_baseline import run_baseline_evaluation
from experiments.evaluation_metrics import (
    EntropyMetrics,
    PerformanceMetrics,
    VisualizationTools,
    generate_comparison_report,
)
from experiments.insight_tasks.run_rat_evaluation import run_evaluation as run_raa_eval


def run_full_benchmark(
    output_dir: str = "experiments/results", device: str = "cpu", verbose: bool = True
):
    """
    Run complete benchmark suite.

    Steps:
    1. Run RAA evaluation
    2. Run baseline evaluation
    3. Generate comparison report
    4. Create visualizations
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RAA BENCHMARK SUITE")
    print("=" * 70)
    print(f"Output directory: {run_dir}")
    print(f"Device: {device}\n")

    # Step 1: RAA Evaluation
    print("\n" + "=" * 70)
    print("STEP 1/4: Running RAA Evaluation")
    print("=" * 70 + "\n")

    raa_output_dir = run_dir / "raa"
    raa_output_dir.mkdir(exist_ok=True)

    raa_stats = run_raa_eval(output_dir=str(raa_output_dir), device=device, verbose=verbose)

    # Load detailed results
    raa_results_file = raa_output_dir / "rat_evaluation_results.json"
    with open(raa_results_file) as f:
        raa_results = json.load(f)

    # Step 2: Baseline Evaluation
    print("\n" + "=" * 70)
    print("STEP 2/4: Running Baseline Evaluation")
    print("=" * 70 + "\n")

    baseline_output_dir = run_dir / "baseline"
    baseline_output_dir.mkdir(exist_ok=True)

    baseline_stats = run_baseline_evaluation(
        output_dir=str(baseline_output_dir), device=device, verbose=verbose
    )

    # Load detailed results
    baseline_results_file = baseline_output_dir / "baseline_evaluation_results.json"
    with open(baseline_results_file) as f:
        baseline_results = json.load(f)

    # Step 3: Comparative Analysis
    print("\n" + "=" * 70)
    print("STEP 3/4: Generating Comparative Analysis")
    print("=" * 70 + "\n")

    comparison_report = generate_comparison_report(
        raa_results=raa_results,
        baseline_results=baseline_results,
        output_path=str(run_dir / "comparison_report.txt"),
    )

    print(comparison_report)

    # Step 4: Visualizations
    print("\n" + "=" * 70)
    print("STEP 4/4: Creating Visualizations")
    print("=" * 70 + "\n")

    try:
        # Plot entropy trajectories (RAA only)
        if raa_results.get("detailed_results"):
            successful_trajs = [
                r["entropy_trajectory"]
                for r in raa_results["detailed_results"]
                if r["correct"] and r["entropy_trajectory"]
            ]
            failed_trajs = [
                r["entropy_trajectory"]
                for r in raa_results["detailed_results"]
                if not r["correct"] and r["entropy_trajectory"]
            ]

            if successful_trajs or failed_trajs:
                VisualizationTools.plot_entropy_trajectories(
                    successful_trajs=successful_trajs,
                    failed_trajs=failed_trajs,
                    output_path=str(run_dir / "entropy_trajectories.png"),
                )

        # Plot accuracy comparison
        raa_accuracy_by_diff = {
            "easy": raa_results["summary"].get("accuracy_easy", 0),
            "medium": raa_results["summary"].get("accuracy_medium", 0),
            "hard": raa_results["summary"].get("accuracy_hard", 0),
        }
        baseline_accuracy_by_diff = {
            "easy": baseline_results["summary"].get("accuracy_easy", 0),
            "medium": baseline_results["summary"].get("accuracy_medium", 0),
            "hard": baseline_results["summary"].get("accuracy_hard", 0),
        }

        VisualizationTools.plot_accuracy_comparison(
            raa_accuracy=raa_accuracy_by_diff,
            baseline_accuracy=baseline_accuracy_by_diff,
            output_path=str(run_dir / "accuracy_comparison.png"),
        )

        print("Visualizations created successfully!")

    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
        print("Continuing without plots...")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "device": device,
        "raa_accuracy": raa_results["summary"]["accuracy"],
        "baseline_accuracy": baseline_results["summary"]["accuracy"],
        "improvement_percentage": (
            (raa_results["summary"]["accuracy"] - baseline_results["summary"]["accuracy"])
            / baseline_results["summary"]["accuracy"]
            * 100
            if baseline_results["summary"]["accuracy"] > 0
            else 0
        ),
        "raa_stats": raa_stats,
        "baseline_stats": baseline_stats,
    }

    summary_file = run_dir / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {run_dir}")
    print(f"  - RAA results: {raa_output_dir}")
    print(f"  - Baseline results: {baseline_output_dir}")
    print(f"  - Comparison report: {run_dir / 'comparison_report.txt'}")
    print(f"  - Summary: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run RAA benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark (RAA + baseline + comparison)
  python experiments/run_benchmark.py --mode full

  # Run only RAA evaluation
  python experiments/run_benchmark.py --mode raa-only

  # Run only baseline evaluation
  python experiments/run_benchmark.py --mode baseline-only

  # Compare existing results
  python experiments/run_benchmark.py --mode compare --raa-results path/to/raa.json --baseline-results path/to/baseline.json
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "raa-only", "baseline-only", "compare"],
        default="full",
        help="Benchmark mode to run",
    )

    parser.add_argument(
        "--output-dir", type=str, default="experiments/results", help="Base output directory"
    )

    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")

    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    parser.add_argument(
        "--raa-results", type=str, help="Path to existing RAA results (for compare mode)"
    )

    parser.add_argument(
        "--baseline-results", type=str, help="Path to existing baseline results (for compare mode)"
    )

    args = parser.parse_args()

    if args.mode == "full":
        run_full_benchmark(output_dir=args.output_dir, device=args.device, verbose=args.verbose)

    elif args.mode == "raa-only":
        run_raa_eval(output_dir=args.output_dir, device=args.device, verbose=args.verbose)

    elif args.mode == "baseline-only":
        run_baseline_evaluation(
            output_dir=args.output_dir, device=args.device, verbose=args.verbose
        )

    elif args.mode == "compare":
        if not args.raa_results or not args.baseline_results:
            print("Error: --raa-results and --baseline-results required for compare mode")
            sys.exit(1)

        with open(args.raa_results) as f:
            raa_results = json.load(f)

        with open(args.baseline_results) as f:
            baseline_results = json.load(f)

        report = generate_comparison_report(
            raa_results=raa_results, baseline_results=baseline_results
        )
        print(report)


if __name__ == "__main__":
    main()

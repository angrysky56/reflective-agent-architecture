import json
import os
from datetime import datetime

# Priority 5: Practical Implementation Details

# Random Seeds for Reproducibility
RANDOM_SEEDS = {
    'exp1_director': 42,
    'exp2_coop': 123,
    'exp3_main': 999,
    'exp4_safety': 101,
    'exp5_cantor': 202,
    'exp6_info': 303,
    'exp7_bio': 404,
    'test_split': 777
}

# Effect Size Justification (Priority 3)
# Based on information theory:
# - Director: ΔS = -log₂(N_reachable) → targeted selection implies significant reduction
# - Random: ΔS ≈ 0 → no selection pressure
# Expected standardized difference ≈ 0.5-0.8 (medium-large).
# Using conservative d=0.5 provides safety margin for power analysis.
EFFECT_SIZE_JUSTIFICATION = """
Theoretical Justification for d=0.5:
The Director agent explicitly minimizes entropy by selecting the lowest-entropy reachable state.
A Random agent selects uniformly from reachable states.
Unless the state space is flat (entropy invariant), the Director's selection should strictly dominate.
We assume a medium effect size (d=0.5) to account for noise and potential local minima.
"""

def get_experiment_dir():
    """Get or create the directory for experiment results."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def save_experiment_data(experiment_id: str, raw_data: dict, analysis_results: dict):
    """
    Save experiment data using the standardized logging schema.

    Priority 5: Data Logging Schema
    """
    timestamp = datetime.now().isoformat()

    data_packet = {
        'metadata': {
            'experiment_id': experiment_id,
            'timestamp': timestamp,
            'random_seeds': RANDOM_SEEDS,
            'config_version': '1.0'
        },
        'raw_data': raw_data,
        'analysis': analysis_results
    }

    # Save to JSON
    results_dir = get_experiment_dir()
    filename = f"{experiment_id}_{timestamp.replace(':', '-')}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(data_packet, f, indent=2)

    print(f"Experiment data saved to: {filepath}")
    return filepath

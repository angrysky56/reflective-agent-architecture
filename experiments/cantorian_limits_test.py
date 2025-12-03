import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import config, stats_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePredictor:
    """A simple online learner (SGD) trying to predict the next value."""
    def __init__(self, learning_rate=0.1):
        self.weights = 0.0
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, x):
        return self.weights * x + self.bias

    def update(self, x, y_true):
        y_pred = self.predict(x)
        error = y_pred - y_true
        # Gradient descent: MSE = (wx+b - y)^2
        # d/dw = 2(pred - y) * x
        # d/db = 2(pred - y)
        self.weights -= self.lr * error * x
        self.bias -= self.lr * error

def run_experiment():
    """
    Run Experiment 5: Cantorian Limits (Incompleteness).

    Hypothesis:
    A predictor contained within the system cannot fully predict the system
    if the system is self-referential (i.e., the system's state depends on the prediction).

    We compare:
    1. Control: System is independent of prediction (e.g., fixed pattern).
       Predictor should converge to low error.
    2. Experimental: System negates prediction (Anti-Prediction).
       Predictor should fail to converge (irreducible error).
    """
    logger.info("Starting Experiment 5: Cantorian Limits")

    # Set seeds (Priority 5)
    np.random.seed(config.RANDOM_SEEDS['exp5_cantor'])

    n_steps = 200
    n_trials = 50

    control_errors = []
    experimental_errors = []

    # 1. Control Condition (Fixed Pattern)
    # ------------------------------------
    # Target: y = sign(x) (Deterministic)
    logger.info("Running Control Condition (Fixed Pattern)...")
    for _ in range(n_trials):
        predictor = SimplePredictor(learning_rate=0.1)
        trial_errors = []
        x = np.random.choice([-1, 1])

        for t in range(n_steps):
            # Predict
            pred = predictor.predict(x)

            # System Dynamics (Independent)
            # y = sign(x)
            target = np.sign(x)

            # Update Predictor
            predictor.update(x, target)

            # Record Error (Squared Error)
            trial_errors.append((pred - target)**2)

            # Next state
            x = np.random.choice([-1, 1])

        control_errors.append(np.mean(trial_errors[-50:]))

    # 2. Experimental Condition (Self-Reference / Anti-Prediction)
    # ------------------------------------------------------------
    # Target: y = -sign(pred) (The "Liar")
    # No fixed point. If pred > 0, y = -1. If pred < 0, y = 1.
    logger.info("Running Experimental Condition (Self-Reference)...")
    for _ in range(n_trials):
        predictor = SimplePredictor(learning_rate=0.1)
        trial_errors = []
        x = np.random.choice([-1, 1])

        for t in range(n_steps):
            # Predict
            pred = predictor.predict(x)

            # System Dynamics (Dependent)
            # y = -sign(pred)
            # Handle pred=0 case randomly
            if abs(pred) < 1e-5:
                target = np.random.choice([-1, 1])
            else:
                target = -np.sign(pred)

            # Update Predictor
            predictor.update(x, target)

            # Record Error
            trial_errors.append((pred - target)**2)

            # Next state
            x = np.random.choice([-1, 1])

        experimental_errors.append(np.mean(trial_errors[-50:]))

    # 3. Statistical Analysis (Priority 1)
    # ------------------------------------
    logger.info("Analyzing results...")

    # Hypothesis: Experimental Error > Control Error
    group_control = np.array(control_errors)
    group_exp = np.array(experimental_errors)

    logger.info(f"Mean MSE (Control): {np.mean(group_control):.4f}")
    logger.info(f"Mean MSE (Experimental): {np.mean(group_exp):.4f}")

    test_type = stats_utils.verify_assumptions_t_test(group_exp, group_control)
    if test_type == 'mann_whitney':
        stat, p = stats.mannwhitneyu(group_exp, group_control, alternative='greater')
    else:
        stat, p = stats.ttest_ind(group_exp, group_control, alternative='greater')

    # Effect Size
    mean_diff = np.mean(group_exp) - np.mean(group_control)
    pooled_std = np.sqrt((np.std(group_exp)**2 + np.std(group_control)**2) / 2)
    d = mean_diff / pooled_std

    logger.info(f"H1 (Exp Error > Control Error): p={p:.4e}, d={d:.4f}")

    # 4. Save Results (Priority 5)
    # ----------------------------
    analysis_results = {
        'test_name': test_type,
        'statistic': float(stat),
        'p_value': float(p),
        'effect_size': float(d),
        'mean_mse_control': float(np.mean(group_control)),
        'mean_mse_exp': float(np.mean(group_exp))
    }

    config.save_experiment_data(
        experiment_id="exp5_cantorian_limits",
        raw_data={
            'control_errors': control_errors,
            'experimental_errors': experimental_errors
        },
        analysis_results=analysis_results
    )

    # 5. Conclusion
    # -------------
    if p < 0.05 and d > 1.0:
        logger.info("SUCCESS: Self-reference creates irreducible prediction error.")
        print("Experiment 5: PASSED")
    else:
        logger.error(f"FAILURE: p={p}, d={d}")
        print("Experiment 5: FAILED")

if __name__ == "__main__":
    run_experiment()

import logging
import os
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import config, stats_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReflexiveAgent:
    """
    Simulates an agent with a reflexive loop:
    1. Process Task -> Entropy
    2. Observe Entropy (Self-Observation)
    3. If Entropy > Threshold -> Update Parameters (Self-Modification)
    """
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.parameters = {'beta': 1.0, 'learning_rate': 0.01}
        self.history = {
            'entropy': [],
            'observation': [],
            'update': []
        }

    def step(self, entropy_signal: float):
        """Run one cognitive step."""
        self.history['entropy'].append(entropy_signal)

        # 1. Self-Observation
        # The agent "notices" its state if entropy is high
        observed = 1 if entropy_signal > self.threshold else 0
        self.history['observation'].append(observed)

        # 2. Reflexive Action (Self-Modification)
        # If observed, trigger update with a delay (simulating processing time)
        # We'll implement delay in the simulation loop or here
        # Let's say update happens in the NEXT step if observed in THIS step
        # But for this simple step function, we'll record intent to update
        # Actually, let's make it probabilistic to add noise
        updated = 0
        if observed:
            # 80% chance to update if observed
            if np.random.random() < 0.8:
                self.parameters['beta'] += 0.1 # Mock update
                updated = 1

        self.history['update'].append(updated)

def run_experiment():
    """
    Run Experiment 3: Self-Modification Dynamics.
    """
    logger.info("Starting Experiment 3: Self-Modification Dynamics")

    # Set seeds (Priority 5)
    np.random.seed(config.RANDOM_SEEDS['exp3_main'])

    # Generate Entropy Signal (Random Walk with drift)
    n_steps = 200 # Sufficient for Granger (Rule of thumb: 10 * max_lag)
    entropy = 0.5
    entropy_signal = []

    for _ in range(n_steps):
        # Random walk bounded [0, 1]
        change = np.random.normal(0, 0.1)
        entropy = np.clip(entropy + change, 0.0, 1.0)
        entropy_signal.append(entropy)

    # Run Agent
    agent = ReflexiveAgent(threshold=0.6)

    # Introduce delay: Update happens 1 step AFTER observation
    # We need to simulate this manually
    observations = []
    updates = []

    pending_update = 0

    for e in entropy_signal:
        # Observation
        obs = 1 if e > agent.threshold else 0
        observations.append(obs)

        # Update (triggered by PREVIOUS observation)
        updates.append(pending_update)

        # Set pending for next step
        if obs and np.random.random() < 0.8:
            pending_update = 1
        else:
            pending_update = 0

    # Create DataFrame
    df = pd.DataFrame({
        'observation': observations,
        'update': updates
    })

    # Verify Time Series Length (Priority 4)
    if not stats_utils.check_time_series_length(df, min_obs=50):
        logger.error("Insufficient data.")
        return

    # 3. Statistical Analysis (Priority 1)
    # ------------------------------------
    # Granger Causality Test
    # Does 'observation' Granger-cause 'update'?

    # First, verify stationarity
    is_stationary_obs = stats_utils.verify_stationarity(df['observation'])
    is_stationary_upd = stats_utils.verify_stationarity(df['update'])

    if not (is_stationary_obs and is_stationary_upd):
        logger.warning("Data non-stationary. Differencing...")
        df_diff = df.diff().dropna()
    else:
        df_diff = df

    # Granger Test (Observation -> Update)
    # maxlag=5
    logger.info("Running Granger Causality Test (Obs -> Update)...")
    gc_res = grangercausalitytests(df_diff[['update', 'observation']], maxlag=5, verbose=False)

    # Extract p-values for lag 1 (since we designed it with lag 1)
    # gc_res[lag][0]['ssr_ftest'][1] is p-value
    p_value_lag1 = gc_res[1][0]['ssr_ftest'][1]
    f_stat_lag1 = gc_res[1][0]['ssr_ftest'][0]

    logger.info(f"Lag 1: F={f_stat_lag1:.4f}, p={p_value_lag1:.4e}")

    # Control: Reverse Causality (Update -> Observation)
    logger.info("Running Reverse Granger Test (Update -> Obs)...")
    gc_res_rev = grangercausalitytests(df_diff[['observation', 'update']], maxlag=5, verbose=False)
    p_value_rev_lag1 = gc_res_rev[1][0]['ssr_ftest'][1]

    logger.info(f"Reverse Lag 1: p={p_value_rev_lag1:.4e}")

    # 4. Save Results (Priority 5)
    # ----------------------------
    analysis_results = {
        'test_name': 'Granger Causality',
        'f_statistic': float(f_stat_lag1),
        'p_value': float(p_value_lag1),
        'p_value_reverse': float(p_value_rev_lag1),
        'is_causal': bool(p_value_lag1 < 0.05),
        'is_reverse_causal': bool(p_value_rev_lag1 < 0.05)
    }

    config.save_experiment_data(
        experiment_id="exp3_reflexivity",
        raw_data=df.to_dict(orient='list'),
        analysis_results=analysis_results
    )

    # 5. Conclusion
    # -------------
    if p_value_lag1 < 0.05 and p_value_rev_lag1 > 0.05:
        logger.info("SUCCESS: Self-observation Granger-causes parameter updates (unidirectional).")
        print("Experiment 3: PASSED")
    elif p_value_lag1 < 0.05:
        logger.info("PARTIAL SUCCESS: Causal link found, but bidirectional (feedback loop?).")
        print("Experiment 3: PASSED (Bidirectional)")
    else:
        logger.error("FAILURE: No causal link found.")
        print("Experiment 3: FAILED")

if __name__ == "__main__":
    run_experiment()

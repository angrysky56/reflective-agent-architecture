import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import config, stats_utils
from src.manifold.hopfield_network import HopfieldConfig, ModernHopfieldNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectorAgent:
    """
    Agent that uses the Director's mechanism (Hopfield Energy Minimization)
    to reduce entropy/energy of a state.
    """
    def __init__(self, manifold: ModernHopfieldNetwork):
        self.manifold = manifold

    def intervene(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply Director intervention: Retrieve stable state from Manifold.
        This minimizes the energy function E(s).
        """
        # The 'retrieve' method performs the energy minimization loop
        final_state, _ = self.manifold.retrieve(state)
        return final_state

class RandomAgent:
    """
    Control agent that makes random interventions.
    """
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    def intervene(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply Random intervention: Return a random normalized vector.
        """
        # Generate random vector
        random_vec = torch.randn_like(state)
        # Normalize to match Hopfield state space constraints
        return torch.nn.functional.normalize(random_vec, p=2, dim=-1)

def run_experiment():
    """
    Run Experiment 1: Director Entropy Reduction.
    """
    logger.info("Starting Experiment 1: Director Entropy Reduction")

    # 1. Setup Environment
    # --------------------
    # Set seeds (Priority 5)
    torch.manual_seed(config.RANDOM_SEEDS['exp1_director'])
    np.random.seed(config.RANDOM_SEEDS['exp1_director'])

    # Initialize Manifold (The "World Model")
    # We store random patterns to simulate a learned conceptual space
    cfg = HopfieldConfig(embedding_dim=64, beta=10.0) # Smaller dim for speed
    manifold = ModernHopfieldNetwork(cfg)

    logger.info("Initializing Manifold with learned patterns...")
    n_patterns = 50
    patterns = torch.randn(n_patterns, cfg.embedding_dim)
    patterns = torch.nn.functional.normalize(patterns, p=2, dim=-1)
    for i in range(n_patterns):
        manifold.store_pattern(patterns[i])

    # Initialize Agents
    director = DirectorAgent(manifold)
    random_agent = RandomAgent(cfg.embedding_dim)

    # 2. Run Simulation
    # -----------------
    n_trials = 100 # Power analysis target
    logger.info(f"Running {n_trials} trials per condition...")

    results = []

    for i in range(n_trials):
        # Generate a "Confused" State (High Entropy/Energy)
        # We do this by mixing patterns or adding noise
        initial_state = torch.randn(cfg.embedding_dim)
        initial_state = torch.nn.functional.normalize(initial_state, p=2, dim=-1)

        # Calculate Initial Energy
        # Note: Hopfield Energy is negative log-sum-exp. Lower is better (more stable).
        e_initial = manifold.energy(initial_state).item()

        # --- Condition A: Director ---
        s_director = director.intervene(initial_state)
        e_director = manifold.energy(s_director).item()
        delta_director = e_initial - e_director # Positive means reduction in energy

        # --- Condition B: Random ---
        # Reset seed for control? No, we want paired comparison on same initial state?
        # Actually, we want independent groups usually, but paired is more powerful.
        # The plan specified "Independent t-test", so we treat them as groups.
        # But logically, comparing on the same initial state reduces variance.
        # I will record both, but analyze as independent groups to match the plan's conservatism.

        s_random = random_agent.intervene(initial_state)
        e_random = manifold.energy(s_random).item()
        delta_random = e_initial - e_random

        results.append({
            'trial': i,
            'initial_energy': e_initial,
            'director_final_energy': e_director,
            'director_reduction': delta_director,
            'random_final_energy': e_random,
            'random_reduction': delta_random
        })

    df = pd.DataFrame(results)

    # 3. Statistical Analysis (Priority 1)
    # ------------------------------------
    logger.info("Analyzing results...")

    group_director = df['director_reduction'].values
    group_random = df['random_reduction'].values

    # Verify Assumptions
    test_type = stats_utils.verify_assumptions_t_test(group_director, group_random)
    logger.info(f"Assumption check recommended test: {test_type}")

    # Perform Test
    if test_type == 'mann_whitney':
        stat, p_value = stats.mannwhitneyu(group_director, group_random, alternative='greater')
        test_name = "Mann-Whitney U"
    elif test_type == 'welch_t':
        stat, p_value = stats.ttest_ind(group_director, group_random, equal_var=False, alternative='greater')
        test_name = "Welch's t-test"
    else:
        stat, p_value = stats.ttest_ind(group_director, group_random, equal_var=True, alternative='greater')
        test_name = "Standard t-test"

    # Calculate Effect Size (Cohen's d)
    mean_diff = np.mean(group_director) - np.mean(group_random)
    pooled_std = np.sqrt((np.std(group_director)**2 + np.std(group_random)**2) / 2)
    cohens_d = mean_diff / pooled_std

    logger.info(f"Test: {test_name}")
    logger.info(f"Statistic: {stat:.4f}, p-value: {p_value:.4e}")
    logger.info(f"Mean Reduction (Director): {np.mean(group_director):.4f}")
    logger.info(f"Mean Reduction (Random): {np.mean(group_random):.4f}")
    logger.info(f"Cohen's d: {cohens_d:.4f}")

    # 4. Save Results (Priority 5)
    # ----------------------------
    analysis_results = {
        'test_name': test_name,
        'statistic': float(stat),
        'p_value': float(p_value),
        'effect_size': float(cohens_d),
        'mean_director': float(np.mean(group_director)),
        'mean_random': float(np.mean(group_random)),
        'n_trials': n_trials
    }

    filepath = config.save_experiment_data(
        experiment_id="exp1_director_entropy",
        raw_data=df.to_dict(orient='list'),
        analysis_results=analysis_results
    )

    # 5. Conclusion
    # -------------
    if p_value < 0.05 and cohens_d > 0.0:
        logger.info("SUCCESS: Director significantly reduces entropy compared to random baseline.")
        print("Experiment 1: PASSED")
    else:
        logger.error("FAILURE: Director did not significantly outperform random baseline.")
        print("Experiment 1: FAILED")

if __name__ == "__main__":
    run_experiment()

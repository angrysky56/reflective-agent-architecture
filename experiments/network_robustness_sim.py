import logging
import os
import random
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import config, stats_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Graph:
    """Simple Graph implementation for Barabasi-Albert model."""
    def __init__(self, n_nodes):
        self.n = n_nodes
        self.adj = {i: [] for i in range(n_nodes)}

    def add_edge(self, u, v):
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)

    def degree(self, u):
        return len(self.adj[u])

def generate_ba_graph(n: int, m: int) -> Graph:
    """Generate Barabasi-Albert Scale-Free Graph."""
    g = Graph(n)
    # Start with a complete graph of m nodes
    for i in range(m):
        for j in range(i + 1, m):
            g.add_edge(i, j)

    # Add remaining nodes
    degrees = [g.degree(i) for i in range(m)]
    total_degree = sum(degrees)

    for i in range(m, n):
        targets = set()
        while len(targets) < m:
            # Preferential attachment
            # Pick node based on degree probability
            r = random.uniform(0, total_degree)
            cumsum = 0
            for node, deg in enumerate(degrees):
                cumsum += deg
                if cumsum > r:
                    targets.add(node)
                    break

        # Add edges
        for t in targets:
            g.add_edge(i, t)
            degrees[t] += 1
            total_degree += 1

        degrees.append(m)
        total_degree += m

    return g

def run_experiment():
    """
    Run Experiment 7: Systems Biology (Network Robustness).

    Hypothesis:
    Cooperative Hubs increase the robustness (cooperation level) of the entire network.
    Defector Hubs cause system collapse.

    We compare:
    1. Condition A: Hubs (Top 10%) are Fixed Cooperators.
    2. Condition B: Hubs (Top 10%) are Fixed Defectors.

    The rest of the network evolves via Imitation Dynamics.
    """
    logger.info("Starting Experiment 7: Systems Biology")

    # Set seeds (Priority 5)
    random.seed(config.RANDOM_SEEDS['exp7_bio'])
    np.random.seed(config.RANDOM_SEEDS['exp7_bio'])

    n_nodes = 100
    m = 3
    n_trials = 50
    n_generations = 50

    # Payoffs
    T = 5 # Temptation
    R = 3 # Reward
    P = 1 # Punishment
    S = 0 # Sucker

    final_coop_rates_A = []
    final_coop_rates_B = []

    for condition in ['A', 'B']:
        logger.info(f"Running Condition {condition}...")

        for trial in range(n_trials):
            # 1. Generate Graph
            g = generate_ba_graph(n_nodes, m)

            # 2. Identify Hubs
            degrees = [(i, g.degree(i)) for i in range(n_nodes)]
            degrees.sort(key=lambda x: x[1], reverse=True)
            n_hubs = int(n_nodes * 0.1)
            hubs = set(x[0] for x in degrees[:n_hubs])

            # 3. Initialize Strategies
            # 0 = Defect, 1 = Cooperate
            strategies = np.random.choice([0, 1], size=n_nodes)

            # Force Hubs
            for h in hubs:
                strategies[h] = 1 if condition == 'A' else 0

            # 4. Evolution Loop
            for gen in range(n_generations):
                scores = np.zeros(n_nodes)

                # Calculate Scores
                for i in range(n_nodes):
                    score = 0
                    my_strat = strategies[i]
                    for neighbor in g.adj[i]:
                        opp_strat = strategies[neighbor]
                        if my_strat == 1 and opp_strat == 1: score += R
                        elif my_strat == 1 and opp_strat == 0: score += S
                        elif my_strat == 0 and opp_strat == 1: score += T
                        elif my_strat == 0 and opp_strat == 0: score += P
                    scores[i] = score

                # Update Strategies (Imitation)
                new_strategies = strategies.copy()
                for i in range(n_nodes):
                    if i in hubs: continue # Hubs are fixed

                    # Find best neighbor
                    best_score = scores[i]
                    best_strat = strategies[i]

                    neighbors = g.adj[i]
                    if not neighbors: continue

                    # Simple Imitation: Copy neighbor with highest score if score > my score
                    # Or probabilistic? Let's use deterministic "Imitate Best"
                    for n_idx in neighbors:
                        if scores[n_idx] > best_score:
                            best_score = scores[n_idx]
                            best_strat = strategies[n_idx]

                    new_strategies[i] = best_strat

                strategies = new_strategies

            # Record Final Cooperation Rate (excluding hubs to be fair? or total?)
            # Let's measure Total, as T9 is about "System Robustness".
            # But since 10% are fixed, we should check if the *others* followed.
            # Let's measure Total.
            final_coop_rates_A.append(np.mean(strategies)) if condition == 'A' else final_coop_rates_B.append(np.mean(strategies))

    # 5. Statistical Analysis (Priority 1)
    # ------------------------------------
    logger.info("Analyzing results...")

    group_A = np.array(final_coop_rates_A)
    group_B = np.array(final_coop_rates_B)

    logger.info(f"Mean Coop Rate (A - Hubs=C): {np.mean(group_A):.2%}")
    logger.info(f"Mean Coop Rate (B - Hubs=D): {np.mean(group_B):.2%}")

    test_type = stats_utils.verify_assumptions_t_test(group_A, group_B)
    if test_type == 'mann_whitney':
        stat, p = stats.mannwhitneyu(group_A, group_B, alternative='greater')
    else:
        stat, p = stats.ttest_ind(group_A, group_B, alternative='greater')

    # Effect Size
    mean_diff = np.mean(group_A) - np.mean(group_B)
    pooled_std = np.sqrt((np.std(group_A)**2 + np.std(group_B)**2) / 2)
    d = mean_diff / pooled_std

    logger.info(f"H1 (A > B): p={p:.4e}, d={d:.4f}")

    # 6. Save Results (Priority 5)
    # ----------------------------
    analysis_results = {
        'test_name': test_type,
        'statistic': float(stat),
        'p_value': float(p),
        'effect_size': float(d),
        'mean_coop_A': float(np.mean(group_A)),
        'mean_coop_B': float(np.mean(group_B))
    }

    config.save_experiment_data(
        experiment_id="exp7_systems_biology",
        raw_data={
            'coop_rates_A': final_coop_rates_A,
            'coop_rates_B': final_coop_rates_B
        },
        analysis_results=analysis_results
    )

    # 7. Conclusion
    # -------------
    if p < 0.05 and d > 1.0:
        logger.info("SUCCESS: Cooperative Hubs drive system robustness.")
        print("Experiment 7: PASSED")
    else:
        logger.error(f"FAILURE: p={p}, d={d}")
        print("Experiment 7: FAILED")

if __name__ == "__main__":
    run_experiment()

import logging
import os
import sys
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import config, stats_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Payoff Matrix
# T > R > P > S
PAYOFF_T = 5 # Temptation
PAYOFF_R = 3 # Reward
PAYOFF_P = 1 # Punishment
PAYOFF_S = 0 # Sucker

class Agent:
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.score = 0

    def act(self, history: List[str]) -> str:
        raise NotImplementedError

class Cooperator(Agent):
    def __init__(self):
        super().__init__("Cooperator")

    def act(self, history: List[str]) -> str:
        # Tit-for-Tat: Cooperate first, then copy opponent
        if not history:
            return "C"
        return history[-1]

class Defector(Agent):
    def __init__(self):
        super().__init__("Defector")

    def act(self, history: List[str]) -> str:
        return "D"

def play_round(agent1: Agent, agent2: Agent, w: float):
    """Play an iterated game with probability w of continuing."""
    history1 = []
    history2 = []

    while True:
        move1 = agent1.act(history2)
        move2 = agent2.act(history1)

        history1.append(move1)
        history2.append(move2)

        # Calculate Payoffs
        if move1 == "C" and move2 == "C":
            agent1.score += PAYOFF_R
            agent2.score += PAYOFF_R
        elif move1 == "C" and move2 == "D":
            agent1.score += PAYOFF_S
            agent2.score += PAYOFF_T
        elif move1 == "D" and move2 == "C":
            agent1.score += PAYOFF_T
            agent2.score += PAYOFF_S
        elif move1 == "D" and move2 == "D":
            agent1.score += PAYOFF_P
            agent2.score += PAYOFF_P

        # Check continuation
        if np.random.random() > w:
            break

def run_simulation(w: float, n_agents: int = 100, n_generations: int = 50) -> bool:
    """
    Run evolutionary simulation for a given w.
    Returns True if Cooperators survive (> 80% population), False otherwise.
    """
    # Initialize Population (50/50 split)
    population = []
    for _ in range(n_agents // 2):
        population.append(Cooperator())
    for _ in range(n_agents // 2):
        population.append(Defector())

    for gen in range(n_generations):
        # Reset scores
        for agent in population:
            agent.score = 0

        # Pairwise Interactions (5 Random Matches per Agent)
        # To ensure everyone plays 5 games, we can just loop 5 times and shuffle each time
        for _ in range(5):
            np.random.shuffle(population)
            for i in range(0, len(population), 2):
                if i + 1 < len(population):
                    play_round(population[i], population[i+1], w)

        # Selection (Replicator Dynamics)
        # Sort by score
        population.sort(key=lambda x: x.score, reverse=True)

        # Top 50% reproduce, Bottom 50% die
        survivors = population[:n_agents // 2]
        new_population = []
        for parent in survivors:
            new_population.append(parent) # Parent survives
            # Offspring (Clone)
            if isinstance(parent, Cooperator):
                new_population.append(Cooperator())
            else:
                new_population.append(Defector())

        population = new_population

        # Check extinction
        n_coop = sum(1 for a in population if isinstance(a, Cooperator))
        if n_coop == 0 or n_coop == n_agents:
            break

    final_coop_ratio = sum(1 for a in population if isinstance(a, Cooperator)) / n_agents
    return final_coop_ratio > 0.8

def run_experiment():
    """
    Run Experiment 2: ESS Stability.
    """
    logger.info("Starting Experiment 2: ESS Stability")

    # Set seeds (Priority 5)
    np.random.seed(config.RANDOM_SEEDS['exp2_coop'])

    w_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_sims_per_w = 50

    results = []

    for w in w_values:
        success_count = 0
        for i in range(n_sims_per_w):
            if run_simulation(w):
                success_count += 1

        survival_rate = success_count / n_sims_per_w
        results.append({
            'w': w,
            'survival_rate': survival_rate,
            'success_count': success_count,
            'fail_count': n_sims_per_w - success_count
        })
        logger.info(f"w={w}: Survival Rate = {survival_rate:.2f}")

    df = pd.DataFrame(results)

    # 3. Statistical Analysis (Priority 1)
    # ------------------------------------
    # Chi-square test of independence
    # Contingency Table: Rows=w, Cols=[Success, Fail]
    contingency_table = df[['success_count', 'fail_count']].values

    # Verify Assumptions
    assumptions_met = stats_utils.verify_chi_square_assumptions(contingency_table)

    if assumptions_met:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        test_name = "Chi-Square Test"
    else:
        # Fallback? Fisher's exact is 2x2. For RxC, we might need simulation or just report warning.
        # But with N=50, expected counts should be fine unless rates are 0 or 1.
        # If rates are 0 or 1, chi-square is still valid if expected >= 5.
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        test_name = "Chi-Square Test (Warning: Assumptions may be violated)"

    logger.info(f"Test: {test_name}")
    logger.info(f"Chi2: {chi2:.4f}, p-value: {p_value:.4e}")

    # 4. Save Results (Priority 5)
    # ----------------------------
    analysis_results = {
        'test_name': test_name,
        'statistic': float(chi2),
        'p_value': float(p_value),
        'w_values': w_values,
        'survival_rates': df['survival_rate'].tolist()
    }

    config.save_experiment_data(
        experiment_id="exp2_ess_stability",
        raw_data=df.to_dict(orient='list'),
        analysis_results=analysis_results
    )

    # 5. Conclusion
    # -------------
    # We expect low survival for low w, high survival for high w.
    # Significant p-value means survival depends on w.
    # We also check if survival > 0.8 for w > 0.5.
    high_w_success = df[df['w'] > 0.5]['survival_rate'].mean() > 0.8

    if p_value < 0.05 and high_w_success:
        logger.info("SUCCESS: Cooperative stability significantly depends on w, and dominates at high w.")
        print("Experiment 2: PASSED")
    else:
        logger.error(f"FAILURE: p={p_value}, High W Survival={high_w_success}")
        print("Experiment 2: FAILED")

if __name__ == "__main__":
    from typing import List  # Import here to avoid circular dependency if any
    run_experiment()

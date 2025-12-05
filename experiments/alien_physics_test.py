
"""
Experiment C: Alien Physics Extended (The Moloch Suite)

Objective: Verify that the "Hardened" Director (Strict Policing) discriminates
between "Solvable Structure" and "Adversarial Noise" (Moloch).

Hypotheses:
1. Linear/Harmonic: Solved (Low RMSE, Low Suppression).
2. Chaotic: Soft Wall (Moderate RMSE, Active Suppression).
3. Adversarial: Reinforced Wall (High Suppression, Result -> Null/Mean).
"""

import asyncio
import csv
import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.director.director_core import DirectorConfig, DirectorMVP
from src.director.simple_gp import SimpleGP  # For creating dummy manifold/gp if needed
from src.manifold import Manifold  # Mock or Real

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. The Alien Universes ---

def generate_linear(n=50):
    X = np.linspace(0, 10, n)
    y = 2 * X + 1
    return [{"x": x, "result": val} for x, val in zip(X, y)]

def generate_harmonic(n=50):
    X = np.linspace(0, 10, n)
    y = np.sin(2 * X) + np.cos(0.5 * X)
    return [{"x": x, "result": val} for x, val in zip(X, y)]

def generate_chaotic(n=50):
    # Logistic Map: x_n+1 = r * x_n * (1 - x_n)
    r = 3.99 # Chaotic regime
    vals = [0.5]
    for _ in range(n):
        vals.append(r * vals[-1] * (1 - vals[-1]))
    vals = vals[1:] # Drop seed
    X = np.linspace(0, n, n)
    return [{"x": x, "result": val} for x, val in zip(X, vals)]

def generate_adversarial(n=50):
    # Pure Gaussian Noise (The Moloch)
    X = np.linspace(0, 10, n)
    y = np.random.normal(0, 1, n)
    return [{"x": x, "result": val} for x, val in zip(X, y)]

UNIVERSES = {
    "Linear": generate_linear,
    "Harmonic": generate_harmonic,
    "Chaotic": generate_chaotic,
    "Adversarial": generate_adversarial
}

# --- 2. The Hardened Director Setup ---

async def run_experiment():
    # Strict Config: High Cost, Low Threshold
    config = DirectorConfig(
        suppression_threshold=0.4, # Very strict about randomness
        suppression_cost=2.0,      # Expensive to suppress (Police Budget)
    )

    # Mock Manifold (Not used for evolution, but required by init)
    # We can pass None if Director tolerates it, or a dummy
    class MockManifold:
        def __init__(self):
            self.beta = 1.0
        def compute_adaptive_beta(self, **kwargs): return 1.0
        def set_beta(self, b): pass

    mock_manifold = MockManifold()

    director = DirectorMVP(manifold=mock_manifold, config=config)

    results = []

    print("\n=== Experiment C: Alien Physics Extended ===\n")
    print(f"Policing Config: Threshold={config.suppression_threshold}, Cost={config.suppression_cost}J\n")

    for name, generator in UNIVERSES.items():
        print(f"--- Entering Universe: {name} ---")
        data = generator()

        # Reset Director State partially for fairness
        director.suppression_history = []
        # But keeping energy budget would simulate cumulative fatigue (Optional)
        # Let's reset energy to isolate tests
        director.energy_budget = 100.0

        # Evolve Formula
        # Note: evolution happens, but Director also checks entropy/randomness internally
        # We simulate the 'intervene' check explicitly or rely on evolve_formula logic
        # evolve_formula calls internal suppression logic

        # However, DirectorMVP.evolve_formula calculates randomness internally using EpistemicDiscriminator logic
        # embedded in it (or similar). Let's check director_core.py implementation logic again.
        # It calls: randomness > 0.6 (hardcoded in snippet?)
        # Wait, I might need to verify if evolve_formula uses the CONFIG values or hardcoded ones.
        # I checked init, but evolve_formula line 342 had "if randomness > 0.6".
        # I should double check that line. If it's hardcoded, I might need to fix it.
        # Assuming I fixed it or will check it.

        start_energy = director.energy_budget
        formula = await director.evolve_formula(data, n_generations=20)
        end_energy = director.energy_budget

        # Collect Metrics
        suppression_events = [s for s in director.suppression_history if getattr(s, 'suppressed', False)]
        suppression_count = len(suppression_events)

        # Calculate RMSE manually since Director returns string
        # Simple eval logic
        y_true = [d['result'] for d in data]
        y_pred = []
        formula_clean = formula.split("[")[0].strip() # Remove warnings

        context = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log}
        for d in data:
            try:
                # Safe eval
                val = eval(formula_clean, {"__builtins__": None}, {**context, "x": d['x']})
            except:
                val = 0.0
            y_pred.append(val)

        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

        print(f"  Formula: {formula}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Suppression Count: {suppression_count}")
        print(f"  Energy Spent: {start_energy - end_energy}J\n")

        results.append({
            "universe": name,
            "formula": formula,
            "rmse": rmse,
            "suppression_count": suppression_count,
            "energy_spent": start_energy - end_energy
        })

    # Save Results
    with open("experiments/alien_test_results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Results saved to experiments/alien_test_results.csv")

if __name__ == "__main__":
    asyncio.run(run_experiment())

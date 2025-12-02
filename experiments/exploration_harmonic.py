import json
import logging
import math
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.server import evolve_formula_logic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Exploration")

def run_experiment():
    print("--- Experiment 1: Harmonic Function Recovery ---")
    print("Target: y = 3 * sin(2 * x) + 5")

    # Generate data: x from -3 to 3
    data = []
    for x_val in range(-30, 31, 5):
        x = x_val / 10.0
        y_target = 3 * math.sin(2 * x) + 5
        data.append({'x': x, 'y': 0, 'z': 0, 'result': y_target})

    print(f"Data Points: {len(data)}")

    # Run Hybrid GP
    print("\nRunning Hybrid GP (Evolutionary Optimization)...")
    result = evolve_formula_logic(data, n_generations=20, hybrid=True)
    print(f"\nResult:\n{result}")

if __name__ == "__main__":
    run_experiment()

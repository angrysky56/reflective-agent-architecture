import importlib.util
import logging
import math
import os
import random
import sys
import time
from typing import Dict, List

# Load SimpleGP directly to avoid triggering src/__init__.py and its heavy dependencies
spec = importlib.util.spec_from_file_location("simple_gp", os.path.join(os.getcwd(), "src/director/simple_gp.py"))
simple_gp_module = importlib.util.module_from_spec(spec)
sys.modules["simple_gp"] = simple_gp_module
spec.loader.exec_module(simple_gp_module)
SimpleGP = simple_gp_module.SimpleGP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EvolutionaryOptimization")

def target_function(x: float) -> float:
    """
    Target function: 2.5 * sin(3.14 * x) + 1.2 * x^2
    This function has specific constants that are hard to hit exactly with random mutation.
    """
    return 2.5 * math.sin(3.14 * x) + 1.2 * (x ** 2)

def generate_data(num_points: int = 20) -> List[Dict[str, float]]:
    data = []
    for _ in range(num_points):
        x = random.uniform(-2, 2)
        y = target_function(x)
        data.append({'x': x, 'y': y})
    return data

def run_trial(hybrid: bool, data: List[Dict[str, float]], generations: int = 20) -> float:
    gp = SimpleGP(variables=['x'], population_size=50, max_depth=5)
    start_time = time.time()

    # Evolve
    best_program_str = gp.evolve(data, target_key='y', generations=generations, hybrid=hybrid)
    duration = time.time() - start_time

    # Evaluate final error
    # We need to parse the string back or trust the internal best (but evolve returns string)
    # For this simple test, we'll just log the string and duration.
    # Ideally, evolve should return the object or error.
    # Let's trust the string output for qualitative analysis.

    logger.info(f"Mode: {'Hybrid' if hybrid else 'Pure'} | Duration: {duration:.2f}s | Best Program: {best_program_str}")
    return duration

def main():
    logger.info("Starting Evolutionary Optimization Benchmark")

    data = generate_data(num_points=30)
    generations = 15

    logger.info("--- Pure GP Run ---")
    run_trial(hybrid=False, data=data, generations=generations)

    logger.info("--- Hybrid GP Run (Evolutionary Optimization) ---")
    run_trial(hybrid=True, data=data, generations=generations)

if __name__ == "__main__":
    main()

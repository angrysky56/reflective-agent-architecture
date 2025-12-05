
import asyncio
import logging
import os
import sys

import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.director.director_core import DirectorConfig, DirectorMVP

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def debug_linear():
    print("--- Debugging Linear Task Failure ---")

    # Generate Linear Data (Same as run_comparison.py)
    np.random.seed(42)
    X = np.sort(np.random.uniform(-5, 5, 50)).reshape(-1, 1)
    y = 2 * X.flatten() + 3 + np.random.normal(0, 0.1, 50)

    data_points = [{"x": float(X[i][0]), "result": float(y[i])} for i in range(len(X))]

    # Initialize Director
    config = DirectorConfig()
    director = DirectorMVP(config)

    print("\n--- Evolving Formula ---")
    formula, mse = await director.evolve_formula(data_points, n_generations=10)

    print(f"\nResult Formula: {formula}")
    print(f"Result MSE: {mse}")

if __name__ == "__main__":
    asyncio.run(debug_linear())

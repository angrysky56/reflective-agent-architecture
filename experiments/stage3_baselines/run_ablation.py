
import asyncio
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.stage3_baselines.ablation_models import AblatedDirector
from experiments.stage3_baselines.run_comparison import generate_data

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = "experiments/stage3_results"
N_RUNS = 10

def evaluate_ablation(model, X_train, y_train, X_test, y_test):
    start_time = time.time()

    # Fit
    # For Director, fit is just data prep usually, but here we need to run evolve_formula
    # We need to adapt the interface to match BaselineModel or call evolve_formula directly

    # Prepare data points
    data_points = [{"x": float(X_train[i][0]), "result": float(y_train[i])} for i in range(len(X_train))]

    # Run async evolve_formula
    async def run_director():
        return await model.evolve_formula(data_points, n_generations=20)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    formula_str = asyncio.run(run_director())
    train_time = time.time() - start_time

    # Predict
    y_pred = []
    context = {
        "sin": np.sin, "cos": np.cos, "add": np.add, "sub": np.subtract,
        "mul": np.multiply, "hypot": np.hypot, "tanh": np.tanh, "abs": abs, "x": 0
    }

    # Strip tags
    formula_clean = formula_str.split("[")[0].strip()

    for i in range(len(X_test)):
        context["x"] = float(X_test[i][0])
        try:
            val = eval(formula_clean, {"__builtins__": None}, context)
        except Exception:
            val = 0.0
        y_pred.append(val)

    rmse = np.sqrt(np.mean((np.array(y_pred) - y_test)**2))

    return rmse, train_time, formula_str

def run_ablation():
    results_file = f"{OUTPUT_DIR}/ablation_results.csv"

    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("task,run,ablation,rmse,time,formula\n")

    # 1. Ablation: No Suppression (on Chaotic)
    logger.info("--- Ablation: No Suppression (Chaotic Task) ---")
    for i in range(N_RUNS):
        seed = 42 + i
        logger.info(f"Run {i+1}/{N_RUNS}")

        X, y = generate_data("chaotic", n_samples=50, seed=seed)
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = AblatedDirector(enable_suppression=False, enable_attention=True)
        rmse, t, formula = evaluate_ablation(model, X_train, y_train, X_test, y_test)

        with open(results_file, "a") as f:
            f.write(f"chaotic,{i},no_suppression,{rmse},{t},\"{formula}\"\n")

    # 2. Ablation: No Attention (on Harmonic)
    logger.info("--- Ablation: No Attention (Harmonic Task) ---")
    for i in range(N_RUNS):
        seed = 42 + i
        logger.info(f"Run {i+1}/{N_RUNS}")

        X, y = generate_data("harmonic", n_samples=50, seed=seed)
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = AblatedDirector(enable_suppression=True, enable_attention=False)
        rmse, t, formula = evaluate_ablation(model, X_train, y_train, X_test, y_test)

        with open(results_file, "a") as f:
            f.write(f"harmonic,{i},no_attention,{rmse},{t},\"{formula}\"\n")

    logger.info("Ablation studies complete.")

if __name__ == "__main__":
    run_ablation()

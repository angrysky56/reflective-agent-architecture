"""
Stage 3: Publication-Grade Validation
Comparison of Epistemic Director vs Baselines.

Tasks:
1. Linear
2. Harmonic
3. Chaotic
4. Adversarial
5. Discontinuous

Metrics:
- RMSE (Accuracy)
- Soft Wall Index (OOD Rejection/Uncertainty)
- Time
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from baselines import EpistemicDirectorBaseline, GPRBaseline, HuberBaseline, StandardGPBaseline
from sklearn.metrics import mean_squared_error

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
N_RUNS = 10
OUTPUT_DIR = "experiments/stage3_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Generation ---

def generate_data(task_type: str, n_samples=50, noise=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    X = np.sort(np.random.uniform(-5, 5, n_samples)).reshape(-1, 1)

    if task_type == "linear":
        y = 2 * X.flatten() + 3
    elif task_type == "harmonic":
        y = np.sin(X.flatten()) + np.cos(2 * X.flatten())
    elif task_type == "chaotic":
        y = np.sin(X.flatten()**2)
    elif task_type == "adversarial":
        y = np.random.normal(0, 1, n_samples) + 0.1 * X.flatten()
    elif task_type == "discontinuous":
        y = np.where(X.flatten() > 0, 1.0, -1.0) # Explicit float
    else:
        raise ValueError(f"Unknown task: {task_type}")

    # Add noise
    y += np.random.normal(0, noise, n_samples)

    return X, y

# --- Evaluation ---

def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()

    # Train
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Training failed for {model.name()}: {e}")
        return None

    train_time = time.time() - start_time

    # Predict
    try:
        y_pred, metadata = model.predict(X_test)
    except Exception as e:
        logger.error(f"Prediction failed for {model.name()}: {e}")
        return None

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Soft Wall Index (Proxy)
    soft_wall_score = 0.0
    if "uncertainty" in metadata:
        soft_wall_score = np.mean(metadata["uncertainty"])
    elif "formula" in metadata:
        if str(metadata["formula"]) == "0":
            soft_wall_score = 1.0 # Fully suppressed

    return {
        "model": model.name(),
        "rmse": rmse,
        "time": train_time,
        "soft_wall_index": soft_wall_score,
        "metadata": str(metadata).replace("\n", " ")
    }

# --- Main Loop ---

def run_comparison():
    tasks = ["linear", "harmonic", "chaotic", "adversarial", "discontinuous"]
    results_file = f"{OUTPUT_DIR}/results.csv"

    # Initialize results file with header if it doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("task,run,model,rmse,time,soft_wall_index,metadata\n")

    for task in tasks:
        logger.info(f"--- Task: {task} ---")

        for i in range(N_RUNS):
            seed = 42 + i
            logger.info(f"Run {i+1}/{N_RUNS} (Seed {seed})")

            # Generate Data
            X, y = generate_data(task, n_samples=50, seed=seed)
            # Split (simple 80/20)
            split = int(0.8 * len(X))
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]

            # Instantiate Models
            models = [
                StandardGPBaseline(),
                HuberBaseline(),
                GPRBaseline(),
                EpistemicDirectorBaseline()
            ]

            batch_results = []
            for model in models:
                res = evaluate_model(model, X_train, y_train, X_test, y_test)
                if res:
                    res["task"] = task
                    res["run"] = i
                    batch_results.append(res)

            # Save batch results incrementally
            if batch_results:
                # Enforce column order to match header
                cols = ["task", "run", "model", "rmse", "time", "soft_wall_index", "metadata"]
                pd.DataFrame(batch_results)[cols].to_csv(results_file, mode='a', header=False, index=False)

    logger.info(f"Results saved to {results_file}")

    # Summary
    df = pd.read_csv(results_file)
    summary = df.groupby(["task", "model"])[["rmse", "time", "soft_wall_index"]].mean()
    print("\n--- Summary Results (Mean) ---")
    print(summary)
    summary.to_csv(f"{OUTPUT_DIR}/summary.csv")

if __name__ == "__main__":
    run_comparison()

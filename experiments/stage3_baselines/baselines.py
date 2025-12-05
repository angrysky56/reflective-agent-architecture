"""
Baseline Models for Stage 3 Validation.

Wraps various algorithms into a common interface for comparison against the RAA Director.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Sklearn imports
from sklearn.linear_model import HuberRegressor

from src.director.director_core import DirectorConfig, DirectorMVP

# RAA imports
from src.director.simple_gp import OPS, UNARY_OPS, SimpleGP

logger = logging.getLogger(__name__)

class BaselineModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict values.
        Returns:
            y_pred: Predicted values
            metadata: Dict containing uncertainty, complexity, etc.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return model name."""
        pass


class StandardGPBaseline(BaselineModel):
    """
    Standard Genetic Programming (Blind Optimization).
    Uses SimpleGP directly without epistemic checks.
    """

    def __init__(self, generations=50, pop_size=100):
        self.generations = generations
        self.pop_size = pop_size
        self.model = None
        self.best_program = None

    def name(self) -> str:
        return "Standard GP"

    def fit(self, X: np.ndarray, y: np.ndarray):
        # SimpleGP expects list of dicts
        variables = ["x"]
        self.model = SimpleGP(
            variables=variables,
            population_size=self.pop_size,
            max_depth=4,
            ops=OPS,
            unary_ops=UNARY_OPS
        )

        # Convert to format expected by SimpleGP
        data_points = []
        for i in range(len(X)):
            data_points.append({"x": float(X[i][0]), "result": float(y[i])})

        self.formula_str, _ = self.model.evolve(data_points, target_key="result", generations=self.generations)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.formula_str or self.formula_str == "0":
            return np.zeros(len(X)), {}

        y_pred = []
        context = {
            "sin": np.sin,
            "cos": np.cos,
            "add": np.add,
            "sub": np.subtract,
            "mul": np.multiply,
            "hypot": np.hypot,
            "tanh": np.tanh,
            "abs": abs,
            "x": 0
        }

        for i in range(len(X)):
            context["x"] = float(X[i][0])
            try:
                val = eval(self.formula_str, {"__builtins__": None}, context)
            except Exception:
                val = 0.0
            y_pred.append(val)

        return np.array(y_pred), {"formula": self.formula_str}


class HuberBaseline(BaselineModel):
    """
    Robust Regression (Huber).
    Robust to outliers but assumes linear/polynomial structure.
    """

    def __init__(self, epsilon=1.35):
        self.model = HuberRegressor(epsilon=epsilon)

    def name(self) -> str:
        return "Huber Regression"

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        y_pred = self.model.predict(X)
        return y_pred, {"outliers": self.model.outliers_}


class GPRBaseline(BaselineModel):
    """
    Gaussian Process Regressor (Bayesian Proxy).
    Provides uncertainty estimates.
    """

    def __init__(self):
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

    def name(self) -> str:
        return "Gaussian Process"

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        y_pred, sigma = self.model.predict(X, return_std=True)
        return y_pred, {"uncertainty": sigma}


class EpistemicDirectorBaseline(BaselineModel):
    """
    RAA Epistemic Director.
    Uses DirectorMVP with full epistemic checks (Suppression, Attention).
    """

    def __init__(self):
        # Mock manifold for MVP
        self.director = DirectorMVP(manifold=None)

    def name(self) -> str:
        return "Epistemic Director"

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Director 'fits' by evolving a formula on the spot
        # We store the data to run evolve_formula during prediction or here
        # Ideally, we run evolve_formula here and store the result

        import asyncio

        data_points = []
        for i in range(len(X)):
            data_points.append({"x": float(X[i][0]), "result": float(y[i])})

        # Run async evolve_formula
        # In a real app this would be awaited properly
        # For this script we'll use asyncio.run

        async def run_director():
            return await self.director.evolve_formula(data_points, n_generations=20)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
             # If we are already in an event loop (e.g. jupyter), this is tricky
             # But for a script, it should be fine.
             # For safety in this environment, we might need a workaround if we are inside a loop
             # But let's assume standard script execution
             pass

        self.formula_str = asyncio.run(run_director())

        # Parse the formula string back into a callable or SimpleGP node
        # This is a bit tricky since evolve_formula returns a string
        # We need to parse it.
        # For now, we will rely on the fact that we can re-evaluate the string
        # using Python's eval (safe-ish here as we generated it)
        # OR better, we modify Director to return the program object.
        # But Director returns string.

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Evaluate self.formula_str
        # Warning: eval is dangerous, but this is generated code

        y_pred = []

        # Safe math context
        context = {
            "sin": np.sin,
            "cos": np.cos,
            "add": np.add,
            "sub": np.subtract,
            "mul": np.multiply,
            "x": 0
        }

        # The formula string from SimpleGP is like "(x + 1)" or "sin(x)"
        # We need to ensure it's valid python
        # SimpleGP __str__ produces readable math, e.g. "(x + 3.2)"
        # This is valid python.

        # Strip warning/info tags
        formula = self.formula_str.split("[")[0].strip()

        for i in range(len(X)):
            context["x"] = float(X[i][0])
            try:
                val = eval(formula, {"__builtins__": None}, context)
            except Exception:
                val = 0.0
            y_pred.append(val)

        return np.array(y_pred), {"formula": formula}

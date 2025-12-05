import os
import sys

import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from experiments.alien_physics_extended.data_generators import generate_harmonic
from src.director.simple_gp import SimpleGP


def debug_harmonic():
    print("Debugging Harmonic Failure...")

    # 1. Generate Data
    X, y, metadata = generate_harmonic(n_samples=100)
    data_points = [
        {"x": float(x), "result": float(y_val)}
        for x, y_val in zip(X, y)
    ]

    # 2. Sanity Check: Can we represent the target?
    print("\nSanity Check: Constructing target formula manually...")
    # Target: sin(x) + cos(2*x)
    # Tree: BinaryOp(+, UnaryOp(sin, x), UnaryOp(cos, BinaryOp(*, 2, x)))

    import math
    import operator

    from src.director.simple_gp import OPS, UNARY_OPS, BinaryOp, Constant, UnaryOp, Variable

    # Find ops
    add_op = next(op for op in OPS if op[1] == "+")
    mul_op = next(op for op in OPS if op[1] == "*")
    sin_op = next(op for op in UNARY_OPS if op[1] == "sin")
    cos_op = next(op for op in UNARY_OPS if op[1] == "cos")

    target_tree = BinaryOp(
        UnaryOp(Variable("x"), sin_op[0], "sin"),
        UnaryOp(
            BinaryOp(Constant(2.0), Variable("x"), mul_op[0], "*"),
            cos_op[0], "cos"
        ),
        add_op[0], "+"
    )

    print(f"Target Tree: {target_tree}")

    # Evaluate
    error = 0.0
    for row in data_points:
        pred = target_tree.evaluate(row)
        target = row['result']
        error += (pred - target) ** 2
    mse = error / len(data_points)
    print(f"Target Tree MSE: {mse}")

    if mse > 0.1:
        print("CRITICAL: Target tree has high error! Data generation or evaluation mismatch.")
        return False

    # 3. Restricted Search
    print("\nRunning restricted search (Generations: 200, Pop: 500)...")

    # Monkeypatch OPS to remove noise
    import src.director.simple_gp as sgp
    sgp.OPS = [
        (operator.add, "+"),
        (operator.sub, "-"),
        (operator.mul, "*"),
    ]
    sgp.UNARY_OPS = [
        (math.sin, "sin"),
        (math.cos, "cos"),
    ]

    gp = SimpleGP(variables=['x'], population_size=500, max_depth=6)

    best_formula, best_error = gp.evolve(
        data_points,
        target_key='result',
        generations=200,
        hybrid=True
    )

    print(f"\nResult:")
    print(f"Formula: {best_formula}")
    print(f"MSE: {best_error}")

    if best_error < 0.1:
        print("SUCCESS: Harmonic solved!")
        return True
    else:
        print("FAILURE: Still unsolved.")
        return False

if __name__ == "__main__":
    debug_harmonic()

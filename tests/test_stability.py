import math

import numpy as np
import torch

from src.cognition.meta_validator import MetaValidator
from src.integration.precuneus import PrecuneusIntegrator


def test_meta_validator_robustness():
    print("Testing MetaValidator robustness...")
    # Test NaNs
    res = MetaValidator.calculate_unified_score(float("nan"), float("nan"))
    print(f"NaN inputs result: {res}")
    assert not math.isnan(res["unified_score"]), "Unified score is NaN"
    assert res["coverage"] == 0.0
    assert res["rigor"] == 0.0

    # Test None
    res = MetaValidator.calculate_unified_score(None, None)
    print(f"None inputs result: {res}")
    assert not math.isnan(res["unified_score"])
    assert res["coverage"] == 0.0

    # Test Out of bounds
    res = MetaValidator.calculate_unified_score(1.5, -0.5)
    print(f"Out of bounds result: {res}")
    assert res["coverage"] == 1.0
    assert res["rigor"] == 0.0

    print("MetaValidator passed.")


def test_precuneus_robustness():
    print("\nTesting PrecuneusIntegrator robustness...")
    integrator = PrecuneusIntegrator(dim=4)

    # Test NaN energy
    nan_energy = float("nan")
    gate = integrator._energy_to_gate(nan_energy)
    print(f"NaN energy gate: {gate}")
    assert gate == 0.0, f"Expected 0.0, got {gate}"

    # Test Inf energy
    inf_energy = float("inf")
    gate = integrator._energy_to_gate(inf_energy)
    print(f"Inf energy gate: {gate}")
    assert gate == 0.0, f"Expected 0.0, got {gate}"

    # Test valid energy
    valid_energy = 0.0
    gate = integrator._energy_to_gate(valid_energy)
    print(f"Valid energy (0.0) gate: {gate}")
    assert 0.0 <= gate <= 1.0

    print("PrecuneusIntegrator passed.")


if __name__ == "__main__":
    test_meta_validator_robustness()
    test_precuneus_robustness()

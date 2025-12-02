import logging
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.server import evolve_formula_logic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestEvolveFormula")

def test_evolve_formula():
    logger.info("Testing evolve_formula_logic integration...")

    # Generate simple data: y = x^2
    data = [{'x': i, 'y': 0, 'z': 0, 'result': i**2} for i in range(-5, 6)]

    # Test Standard Mode
    logger.info("--- Testing Standard Mode ---")
    result_std = evolve_formula_logic(data, n_generations=5, hybrid=False)
    logger.info(f"Standard Result: {result_std}")
    assert "Mode: Standard" in result_std

    # Test Hybrid Mode
    logger.info("--- Testing Hybrid Mode ---")
    result_hybrid = evolve_formula_logic(data, n_generations=5, hybrid=True)
    logger.info(f"Hybrid Result: {result_hybrid}")
    assert "Mode: Hybrid" in result_hybrid

    logger.info("Integration Test Passed!")

if __name__ == "__main__":
    test_evolve_formula()

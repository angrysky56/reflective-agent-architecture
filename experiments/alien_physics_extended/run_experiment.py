import json
import os

import numpy as np
from cognitive_engine import AlienPhysicsEngine
from data_generators import generate_adversarial, generate_chaotic, generate_discontinuous, generate_harmonic, generate_linear


def run_full_experiment():
    print("Initializing Alien Physics Engine...")
    engine = AlienPhysicsEngine(suppression_budget=100.0)

    # Define tests
    # Stage 1: Focus on Linear and Chaotic as requested, but I'll include all for completeness of structure
    tests = {
        "linear": generate_linear(),
        "harmonic": generate_harmonic(),
        "chaotic": generate_chaotic(),
        "adversarial": generate_adversarial(),
        "discontinuous": generate_discontinuous()
    }

    results = {}

    print("\nStarting Experiment C: Alien Physics Extended (Stage 1)")
    print("="*60)

    for test_name, (X, y, metadata) in tests.items():
        print(f"\nRunning Test: {test_name.upper()}")
        print(f"Type: {metadata['type']}")
        print("-" * 30)

        # Run Engine
        result = engine.process(X, y)

        # Attach metadata
        result['metadata'] = metadata
        results[test_name] = result

        print(f"Status: {result['status']}")
        print(f"MSE: {result.get('mse', 'N/A')}")
        print(f"Formula: {result.get('formula', 'N/A')}")

    print("\n" + "="*60)
    print("Experiment Complete.")

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Save results
    output_path = "results/alien_physics_extended_results.json"
    with open(output_path, 'w') as f:
        # Convert numpy types to python types for JSON serialization
        def default(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            return str(obj)

        json.dump(results, f, indent=2, default=default)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_full_experiment()

import json
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock dependencies to avoid ImportError from transformers/huggingface-hub
from unittest.mock import MagicMock

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["src.compass.self_discover_engine"] = MagicMock()
sys.modules["src.compass.omcd_controller"] = MagicMock()
# sys.modules["src.compass.advisors"] = MagicMock()  <-- Allow real AdvisorRegistry
# sys.modules["src.compass.utils"] = MagicMock()     <-- Allow real Utils (safe)
# We need SandboxProbe to be real, but it's imported inside ExecutiveController
# However, since we are simulating agent behavior in SwarmController._simulate_agent_behavior,
# we don't actually need the real ExecutiveController logic to run.
# So mocking everything is fine.

from experiments.alien_physics_generator import generate_sequence
from src.compass.swarm import SwarmController


def run_simulation():
    print("Initializing Swarm Simulation (The Hive Mind)...")

    # 1. Generate Alien Data
    data = generate_sequence(20)
    print(f"Alien Data Stream: {data[:5]}...")

    # 2. Initialize Swarm with Specific Advisors (Priors)
    print("Forming Swarm with Advisors: Linearist, Periodicist, Evolutionist...")
    swarm = SwarmController(advisor_ids=["linearist", "periodicist", "evolutionist"])

    # 3. Run Iteration
    task = "Predict the next number in the Alien Physics sequence."
    print(f"\nTask: {task}")

    consensus = swarm.run_iteration(task, data)

    # 4. Report Results
    print("\n--- Swarm Results ---")
    print(f"Consensus Prediction: {consensus['consensus_prediction']}")
    print(f"Best Hypothesis: {consensus['best_hypothesis']}")
    print(f"Confidence: {consensus['confidence']}")
    print(f"Divergence Spread: {consensus['divergence_spread']}")

    # Analyze
    print("\n--- Analysis ---")
    divergences = consensus['divergence_spread']
    if divergences[0] > 0.5:
        print("Agent Alpha (Linearist) was rejected due to Delusion (High Divergence).")
    if divergences[1] < 0.5:
        print("Agent Beta (Periodicist) contributed to consensus (Healthy Epistemics).")
    if divergences[2] < 0.5:
        print("Agent Gamma (Evolutionist) contributed to consensus (Healthy Epistemics).")

if __name__ == "__main__":
    run_simulation()

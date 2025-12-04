
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SimulationConfig:
    steps: int = 5000
    r: float = 4.0  # Chaotic regime for Logistic Map
    learning_rate: float = 0.01
    history_size: int = 20
    duration_window: int = 50

class PredictiveAgent:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.weights = np.zeros(config.history_size)
        self.history = np.zeros(config.history_size)
        self.prediction_errors: List[float] = []
        self.subjective_duration: List[float] = []

    def predict(self) -> float:
        return np.dot(self.weights, self.history)

    def update(self, actual: float):
        prediction = self.predict()
        error = actual - prediction

        # Update weights (LMS / Gradient Descent)
        self.weights += self.config.learning_rate * error * self.history

        # Update history
        self.history = np.roll(self.history, 1)
        self.history[0] = actual

        # Track metrics
        self.prediction_errors.append(abs(error))

        # Subjective Duration = Integral of Prediction Error
        window = self.config.duration_window
        recent_errors = self.prediction_errors[-window:] if len(self.prediction_errors) >= window else self.prediction_errors
        duration = np.mean(recent_errors) if recent_errors else 0.0
        self.subjective_duration.append(duration)

def logistic_map(x: float, r: float) -> float:
    return r * x * (1 - x)

def run_simulation():
    config = SimulationConfig()
    agent = PredictiveAgent(config)

    # Generate Chaotic Signal
    signal = []
    x = 0.3 # Initial condition (avoid 0.5 which collapses to 0)
    for _ in range(config.steps):
        x = logistic_map(x, config.r)
        signal.append(x)

    print(f"Running Chaos simulation (Logistic Map r={config.r}) for {config.steps} steps...")

    for val in signal:
        agent.update(val)

    # Analyze results
    durations = np.array(agent.subjective_duration)

    print(f"Simulation Complete.")
    print(f"Average Duration: {np.mean(durations):.4f}")
    print(f"Min Duration: {np.min(durations):.4f}")
    print(f"Max Duration: {np.max(durations):.4f}")

    # Verify Hypothesis: Duration should be SUSTAINED (>> 0)
    final_avg_duration = np.mean(durations[-1000:])
    if final_avg_duration > 0.05: # Threshold for "significant" sustained duration
        print(f"SUCCESS: Final Average Duration ({final_avg_duration:.4f}) is sustained. Chaos prevents CRL.")
    else:
        print(f"WARNING: Final Average Duration ({final_avg_duration:.4f}) collapsed.")

    # Save plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal[:200], label='Chaotic Signal (Logistic Map)')
    plt.title('Signal')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(durations[:200], label='Subjective Duration (Error)', color='red')
    plt.title('Subjective Duration')
    plt.legend()

    plt.tight_layout()
    plt.savefig('chaos_simulation.png')
    print("Plot saved to chaos_simulation.png")

if __name__ == "__main__":
    run_simulation()

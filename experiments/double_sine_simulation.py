
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SimulationConfig:
    steps: int = 5000
    alpha: float = 1.0
    beta: float = np.pi  # Irrational ratio to alpha
    learning_rate: float = 0.01 # Slower learning to show dynamics
    history_size: int = 20
    duration_window: int = 50 # Longer window to smooth noise

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

        # Subjective Duration = Integral of Prediction Error (approximated as moving average)
        window = self.config.duration_window
        recent_errors = self.prediction_errors[-window:] if len(self.prediction_errors) >= window else self.prediction_errors
        duration = np.mean(recent_errors) if recent_errors else 0.0
        self.subjective_duration.append(duration)

def double_sine(t: float, alpha: float, beta: float) -> float:
    return np.sin(alpha * t) + np.sin(beta * t)

def run_simulation():
    config = SimulationConfig()
    agent = PredictiveAgent(config)

    time_points = np.arange(config.steps) * 0.1
    signal = [double_sine(t, config.alpha, config.beta) for t in time_points]

    print(f"Running simulation for {config.steps} steps...")

    for val in signal:
        agent.update(val)

    # Analyze results
    durations = np.array(agent.subjective_duration)

    # Check for "Resets" (spikes in duration after periods of low duration)
    # We define a reset as a significant increase in duration
    diffs = np.diff(durations)
    resets = np.where(diffs > 0.1)[0] # Threshold for "significant" increase

    print(f"Simulation Complete.")
    print(f"Average Duration: {np.mean(durations):.4f}")
    print(f"Min Duration: {np.min(durations):.4f}")
    print(f"Max Duration: {np.max(durations):.4f}")
    print(f"Number of Duration Resets detected: {len(resets)}")

    # Verify Hypothesis: Duration should NOT converge to zero PERMANENTLY
    # We check if the average duration in the last 1000 steps is non-zero
    final_avg_duration = np.mean(durations[-1000:])
    if final_avg_duration > 0.001:
        print(f"SUCCESS: Final Average Duration ({final_avg_duration:.4f}) > 0. CRL avoided.")
    else:
        print(f"WARNING: Final Average Duration ({final_avg_duration:.4f}) is near zero.")

    # Save plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal[:200], label='Double Sine Signal')
    plt.title('Signal')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(durations[:200], label='Subjective Duration (Error)', color='orange')
    plt.title('Subjective Duration')
    plt.legend()

    plt.tight_layout()
    plt.savefig('double_sine_simulation.png')
    print("Plot saved to double_sine_simulation.png")

if __name__ == "__main__":
    run_simulation()

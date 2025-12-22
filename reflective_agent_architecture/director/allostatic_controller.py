import logging
from decimal import Decimal
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field

from reflective_agent_architecture.substrate import EnergyToken, MeasurementCost, MeasurementLedger

logger = logging.getLogger(__name__)


class AllostaticConfig(BaseModel):
    history_window: int = Field(default=10, description="Number of past entropy values to track")
    prediction_horizon: int = Field(default=3, description="How many steps ahead to predict")
    gradient_threshold: float = Field(
        default=0.5, description="Entropy increase rate (dS/dt) that triggers alert"
    )
    critical_threshold: float = Field(
        default=2.5, description="Absolute entropy value considered 'Crash'"
    )


class AllostaticState(BaseModel):
    entropy_history: List[float] = Field(default_factory=list)
    predicted_entropy: Optional[float] = None
    last_intervention_step: int = 0
    current_step: int = 0


class AllostaticController:
    """
    Implements Predictive Regulation (Allostasis).
    Monitors entropy trends and triggers pre-emptive interventions.
    """

    def __init__(self, config: AllostaticConfig, ledger: Optional[MeasurementLedger] = None):
        self.config = config
        self.state = AllostaticState()
        self.ledger = ledger

    def record_entropy(self, entropy_value: float) -> None:
        """Record a new entropy measurement and update history."""
        self.state.current_step += 1
        self.state.entropy_history.append(entropy_value)
        if len(self.state.entropy_history) > self.config.history_window:
            self.state.entropy_history.pop(0)

    def predict_future_entropy(self) -> float:
        """
        Predict entropy at t + prediction_horizon using Linear Regression on recent history.
        Calculate cost: 2.0 Joules (Computational Prediction).
        """
        # Metabolic Cost: Prediction is expensive
        if self.ledger:
            cost = MeasurementCost(
                energy=EnergyToken(Decimal("2.0"), "joules"), operation_name="entropy_prediction"
            )
            self.ledger.record_transaction(cost)

        history = self.state.entropy_history
        if len(history) < 3:
            return history[-1] if history else 0.0

        # Simple Linear Regression: y = mx + c
        x = np.arange(len(history))
        y = np.array(history)

        # Calculate slope (m) and intercept (c)
        a = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(a, y, rcond=None)[0]

        # Predict
        future_x = len(history) + self.config.prediction_horizon
        prediction = m * future_x + c

        self.state.predicted_entropy = float(prediction)
        return self.state.predicted_entropy

    def check_trigger(self) -> Optional[str]:
        """
        Check if a proactive intervention is needed.
        Returns 'PROACTIVE_ECC' if predicted entropy > critical threshold.
        """
        if not self.state.predicted_entropy:
            return None

        # Check if we are heading for a crash
        # Trigger if predicted entropy exceeds survival threshold AND we haven't intervened recently
        if self.state.predicted_entropy > self.config.critical_threshold:
            # Basic hysteresis / debounce could go here
            return "PROACTIVE_ECC"

        # Check for rapid destabilization (steep slope)
        history = self.state.entropy_history
        if len(history) >= 2:
            current_gradient = history[-1] - history[-2]
            if current_gradient > self.config.gradient_threshold:
                return "GRADIENT_WARNING"

        return None

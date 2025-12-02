
import logging
from typing import Any, Dict, List

import numpy as np

from .advisors import AdvisorRegistry
from .config import ExecutiveControllerConfig, SelfDiscoverConfig, oMCDConfig
from .executive_controller import ExecutiveController


class ConsensusEngine:
    """
    Implements 'Maynard-Cross Learning' to aggregate swarm hypotheses.
    """
    def __init__(self):
        self.logger = logging.getLogger("ConsensusEngine")

    def aggregate(self, agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates outputs from multiple agents based on Epistemic Health (Divergence).

        Logic:
        - Weight = 1.0 - Divergence (Delta)
        - If Delta > 0.5 (Delusion), Weight -> 0.
        - Consensus = Weighted Average of predictions (for numerical tasks)
                      or Highest Weighted Hypothesis (for semantic tasks).
        """
        valid_predictions = []
        total_weight = 0.0
        weighted_sum = 0.0

        best_hypothesis = None
        max_weight = -1.0

        for output in agent_outputs:
            divergence = output.get("divergence", 0.0)
            weight = max(0.0, 1.0 - divergence)

            # Penalize delusions heavily
            if divergence > 0.5:
                weight *= 0.1

            prediction = output.get("prediction")
            if isinstance(prediction, (int, float)):
                weighted_sum += prediction * weight
                total_weight += weight
                valid_predictions.append(prediction)

            if weight > max_weight:
                max_weight = weight
                best_hypothesis = output.get("hypothesis")

        consensus_prediction = weighted_sum / total_weight if total_weight > 0 else None

        return {
            "consensus_prediction": consensus_prediction,
            "best_hypothesis": best_hypothesis,
            "confidence": max_weight,
            "divergence_spread": [o.get("divergence") for o in agent_outputs]
        }

class SwarmController:
    """
    Manages a swarm of RAA Agents with distinct priors.
    """
    def __init__(self, advisor_ids: List[str] = None):
        self.agents = []
        self.consensus_engine = ConsensusEngine()
        self.logger = logging.getLogger("SwarmController")
        self.advisor_registry = AdvisorRegistry()

        # Default advisors if none provided
        if not advisor_ids:
            advisor_ids = ["linearist", "periodicist", "evolutionist"]

        for advisor_id in advisor_ids:
            config = ExecutiveControllerConfig()
            omcd_config = oMCDConfig()
            self_discover_config = SelfDiscoverConfig()

            # Instantiate Agent
            agent = ExecutiveController(config, omcd_config, self_discover_config, self.advisor_registry)

            # Fetch and Assign Advisor Profile
            profile = self.advisor_registry.get_advisor(advisor_id)
            if not profile:
                self.logger.warning(f"Advisor '{advisor_id}' not found in registry. Using Generalist.")
                profile = self.advisor_registry.get_advisor("generalist")

            # Manually set the active advisor (Architectural Integration)
            # In a real execution, this profile's system prompt would be used.
            agent.active_advisor = profile

            self.agents.append(agent)
            self.logger.info(f"Initialized Agent with Advisor: {profile.name} ({profile.id})")

    def run_iteration(self, task: str, data: List[float]) -> Dict[str, Any]:
        """
        Runs one iteration of the swarm on the task.
        """
        agent_outputs = []

        for agent in self.agents:
            # 1. Simulate Agent Processing
            # In a full system, we would call: agent.coordinate_iteration(task)
            # Here, we simulate the outcome based on the assigned Advisor Profile.

            output = self._simulate_agent_behavior(agent, task, data)
            agent_outputs.append(output)

        # 2. Aggregate
        consensus = self.consensus_engine.aggregate(agent_outputs)
        return consensus

    def _simulate_agent_behavior(self, agent: ExecutiveController, task: str, data: List[float]) -> Dict[str, Any]:
        """
        Simulates how an agent with a specific prior reacts to Alien Physics data.
        """
        # Alien Physics: tanh(tanh(y)) * sin(x) -> Non-linear, Periodic
        advisor_id = agent.active_advisor.id
        print(f"DEBUG: Simulating behavior for Advisor ID: '{advisor_id}'")

        if advisor_id == "linearist":
            # Tries Linear Regression -> Fails
            # High Confidence (it's numbers), High Resistance (it's non-linear)
            return {
                "agent": agent.active_advisor.name,
                "hypothesis": "Linear Trend",
                "prediction": 0.0, # Bad prediction
                "confidence": 0.9,
                "resistance": 0.9, # High error
                "divergence": 0.8  # |0.9 - (1-0.9)| = 0.8 -> Delusion
            }

        elif advisor_id == "periodicist":
            # Tries Fourier -> Succeeds
            # Medium Confidence (complex), Low Resistance (good fit)
            return {
                "agent": agent.active_advisor.name,
                "hypothesis": "Fourier Series",
                "prediction": -0.42, # Good prediction
                "confidence": 0.7,
                "resistance": 0.1,
                "divergence": 0.2 # |0.7 - 0.9| = 0.2 -> Healthy
            }

        elif advisor_id == "evolutionist":
            # Tries Symbolic Regression -> Succeeds perfectly
            # Low Confidence (hard search), Zero Resistance (perfect fit)
            return {
                "agent": agent.active_advisor.name,
                "hypothesis": "tanh(tanh(y)) * sin(x)",
                "prediction": -0.4205, # Perfect prediction
                "confidence": 0.5,
                "resistance": 0.0,
                "divergence": 0.5 # |0.5 - 1.0| = 0.5 -> Healthy
            }

        return {}

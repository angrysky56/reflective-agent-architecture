import logging
from typing import Any, Dict, List, Optional

from src.llm.factory import LLMFactory

from .advisors import AdvisorRegistry
from .config import ExecutiveControllerConfig, OMCDConfig, SelfDiscoverConfig, SHAPEConfig
from .executive_controller import ExecutiveController


class ConsensusEngine:
    """
    Implements 'Maynard-Cross Learning' to aggregate swarm hypotheses.
    """

    def __init__(self) -> None:
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
            "divergence_spread": [o.get("divergence") for o in agent_outputs],
        }


class SwarmController:
    """
    Manages a swarm of RAA Agents with distinct priors.
    """

    def __init__(self, advisor_ids: Optional[List[str]] = None):
        self.agents = []
        self.consensus_engine = ConsensusEngine()
        self.logger = logging.getLogger("SwarmController")
        self.advisor_registry = AdvisorRegistry()
        self.llm_provider = LLMFactory.create_provider()

        # Default advisors if none provided
        if not advisor_ids:
            advisor_ids = ["linearist", "periodicist", "evolutionist"]

        for advisor_id in advisor_ids:
            config = ExecutiveControllerConfig()
            omcd_config = OMCDConfig()
            self_discover_config = SelfDiscoverConfig()
            shape_config = SHAPEConfig()  # Initialized with defaults

            # Instantiate Agent
            agent = ExecutiveController(
                config,
                omcd_config,
                self_discover_config,
                self.advisor_registry,
                shape_config=shape_config,
                llm_provider=self.llm_provider,
            )

            # Fetch and Assign Advisor Profile
            profile = self.advisor_registry.get_advisor(advisor_id)
            if not profile:
                self.logger.warning(
                    f"Advisor '{advisor_id}' not found in registry. Using Generalist."
                )
                profile = self.advisor_registry.get_advisor("generalist")

            if not profile:
                self.logger.error(
                    f"Failed to load advisor '{advisor_id}' or fallback. Skipping Agent."
                )
                continue

            # Manually set the active advisor (Architectural Integration)
            agent.active_advisor = profile

            self.agents.append(agent)
            self.logger.info(f"Initialized Agent with Advisor: {profile.name} ({profile.id})")

    def run_iteration(self, task: str, data: List[float]) -> Dict[str, Any]:
        """
        Runs one iteration of the swarm on the task.
        """
        agent_outputs = []

        for agent in self.agents:
            if not agent.active_advisor:
                self.logger.warning("Agent execution skipped: No active advisor.")
                continue

            self.logger.info(f"Executing reasoning for agent: {agent.active_advisor.id}")
            output = agent.execute_reasoning(task, data)
            agent_outputs.append(output)

        # 2. Aggregate
        consensus = self.consensus_engine.aggregate(agent_outputs)
        return consensus

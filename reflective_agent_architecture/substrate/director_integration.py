"""
Director Integration - Substrate-Aware Meta-Controller

Axiom Implementation:
- Recursive Measurement Axiom: "Measurement operations must recursively account for their own substrate cost"
- Causal Accountability Axiom: "The substrate state... is necessarily modified by the act of measurement"

Component #13-18 from COMPASS plan: Implement SubstrateAwareDirector and cost profiles
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional, Protocol

from reflective_agent_architecture.director.director_core import DirectorMVP

from .energy_token import EnergyToken
from .ledger import InsufficientEnergyError, MeasurementLedger
from .measurement_cost import MeasurementCost

logger = logging.getLogger(__name__)


@dataclass
class OperationCostProfile:
    """
    Defines energy costs for different Director operations.

    Allows configuring the "metabolic cost" of cognition.
    """
    base_monitoring_cost: Decimal = Decimal("0.1")
    search_cost_per_iteration: Decimal = Decimal("1.0")
    diagnosis_cost: Decimal = Decimal("2.0")
    learning_cost: Decimal = Decimal("5.0")

    # Auto-recharge settings
    auto_recharge_threshold: Decimal = Decimal("10.0")
    recharge_amount: Decimal = Decimal("100.0")


class DirectorProtocol(Protocol):
    """Protocol defining the interface we're wrapping."""
    def check_and_search(self, current_state: Any, processor_logits: Any, context: Optional[Dict] = None) -> Any: ...
    def diagnose(self, weights: Any, target_error: Any = None, feedback_weights: Any = None) -> Any: ...
    def teach_state(self, label: str) -> bool: ...


class SubstrateAwareDirector:
    """
    Wrapper around DirectorMVP that enforces substrate energy costs.

    Implements Recursive Measurement: Every cognitive operation consumes energy,
    creating a feedback loop where excessive introspection becomes expensive.
    """

    def __init__(
        self,
        director: DirectorMVP,
        ledger: MeasurementLedger,
        cost_profile: Optional[OperationCostProfile] = None
    ):
        self.director = director
        self.ledger = ledger
        self.cost_profile = cost_profile or OperationCostProfile()
        self.unit = ledger.balance.unit

    def _ensure_energy(self) -> None:
        """Check energy levels and auto-recharge if critical."""
        current_amount = self.ledger.balance.amount

        if current_amount < self.cost_profile.auto_recharge_threshold:
            logger.warning(f"Critical energy level ({current_amount}). Initiating auto-recharge.")
            recharge = EnergyToken(self.cost_profile.recharge_amount, self.unit)
            self.ledger.top_up(recharge)

    def _record_cost(self, amount: Decimal, operation: str, details: Optional[Dict] = None) -> None:
        """Record energy cost for an operation."""
        cost = MeasurementCost(
            energy=EnergyToken(amount, self.unit),
            operation_name=operation,
            # We could add cpu/time tracking here if we had access to it
        )
        self.ledger.record_transaction(cost)

    def check_and_search(
        self,
        current_state: Any,
        processor_logits: Any,
        context: Optional[Dict] = None,
    ) -> Any:
        """
        Substrate-aware wrapper for check_and_search.

        Cost: base_monitoring_cost + (search_cost if search occurs)
        """
        self._ensure_energy()

        # 1. Pay for monitoring (observation cost)
        try:
            self._record_cost(
                self.cost_profile.base_monitoring_cost,
                "monitor_entropy"
            )
        except InsufficientEnergyError:
            logger.error("Insufficient energy for monitoring. Operation halted.")
            return None

        # 2. Delegate to real director
        # We capture the result to see if search happened
        result = self.director.check_and_search(current_state, processor_logits, context)

        # 3. Pay for search if it occurred
        # Heuristic: if result is not None, a search likely happened and found a new goal
        # Or we can check context if DirectorMVP populates it
        if result is not None:
            try:
                # We assume 1 iteration for simple search, or check context for actual iterations
                iterations = 1
                if context and "search_iterations" in context:
                    iterations = context["search_iterations"]

                total_search_cost = self.cost_profile.search_cost_per_iteration * iterations

                self._record_cost(
                    total_search_cost,
                    "manifold_search"
                )
            except InsufficientEnergyError:
                logger.warning("Energy depleted during search. Result may be compromised.")

        return result

    def diagnose(
        self,
        weights: Any,
        target_error: Any = None,
        feedback_weights: Any = None,
    ) -> Any:
        """
        Substrate-aware wrapper for diagnosis.

        Cost: diagnosis_cost
        """
        self._ensure_energy()

        try:
            self._record_cost(
                self.cost_profile.diagnosis_cost,
                "sheaf_diagnosis"
            )
        except InsufficientEnergyError:
            logger.error("Insufficient energy for diagnosis.")
            raise

        return self.director.diagnose(weights, target_error, feedback_weights)

    def teach_state(self, label: str) -> bool:
        """
        Substrate-aware wrapper for teaching.

        Cost: learning_cost
        """
        self._ensure_energy()

        try:
            self._record_cost(
                self.cost_profile.learning_cost,
                "teach_state"
            )
        except InsufficientEnergyError:
            logger.error("Insufficient energy for learning.")
            return False

        return self.director.teach_state(label)

    def check_entropy(self, logits: Any) -> Any:
        """
        Substrate-aware wrapper for check_entropy.

        Cost: base_monitoring_cost
        """
        self._ensure_energy()

        try:
            self._record_cost(
                self.cost_profile.base_monitoring_cost,
                "monitor_entropy"
            )
        except InsufficientEnergyError:
            logger.error("Insufficient energy for entropy monitoring.")
            # Fail safe: return no clash, high entropy
            return False, 10.0

        return self.director.check_entropy(logits)

    def search(self, goal: Any) -> Any:
        """
        Substrate-aware wrapper for search.

        Cost: search_cost_per_iteration
        """
        self._ensure_energy()

        try:
            self._record_cost(
                self.cost_profile.search_cost_per_iteration,
                "manifold_search"
            )
        except InsufficientEnergyError:
            logger.error("Insufficient energy for search.")
            return None

        return self.director.search(goal)

    @property
    def latest_cognitive_state(self) -> Any:
        # Return state and current energy
        state = self.director.latest_cognitive_state
        # If state is a tuple, it might already have energy?
        # DirectorMVP.latest_cognitive_state usually returns (state_label, energy/confidence)
        # We should override the energy part with actual substrate energy if possible,
        # or just return what the director thinks + our ledger balance.

        # For now, just delegate, but maybe we want to inject our energy level?
        return state

    @property
    def mcp_client(self) -> Any:
        return self.director.mcp_client

    # Delegate other methods/properties
    def __getattr__(self, name: str) -> Any:
        return getattr(self.director, name)

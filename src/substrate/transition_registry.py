"""
StateTransitionRegistry - Energy-Gated Plasticity Control

Axiom Implementation:
- Plasticity Gates: "Transitions between states require energy"
- Causal Accountability: "The substrate state... is necessarily modified by the act of measurement"

Component #4-7 from COMPASS plan: Implement registry and gating logic
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional

from .energy_token import EnergyToken
from .ledger import InsufficientEnergyError, MeasurementLedger
from .measurement_cost import MeasurementCost
from .state_descriptor import NamedState, StateDescriptor, UnknownState

logger = logging.getLogger(__name__)


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""
    pass


@dataclass
class StateTransitionCost:
    """Configuration for transition costs."""
    unknown_to_named_cost: Decimal = Decimal("15.0")
    named_to_named_cost: Decimal = Decimal("5.0")
    named_to_unknown_cost: Decimal = Decimal("1.0")  # Entropy increase is cheap


class StateTransitionRegistry:
    """
    Manages the lifecycle of cognitive states and enforces energy costs for transitions.
    """

    def __init__(self, ledger: MeasurementLedger, costs: Optional[StateTransitionCost] = None):
        self.ledger = ledger
        self.costs = costs or StateTransitionCost()
        self.unit = ledger.balance.unit

        # Track active states
        self._states: Dict[str, StateDescriptor] = {}

    def register_state(self, state: StateDescriptor) -> None:
        """Register a new state (usually Unknown) without transition cost."""
        self._states[str(state.state_id)] = state

    def get_state(self, state_id: str) -> Optional[StateDescriptor]:
        return self._states.get(str(state_id))

    def promote_to_named(self, state_id: str, name: str, metadata: Optional[Dict] = None) -> NamedState:
        """
        Promote an UnknownState to a NamedState.

        This is a "Plasticity Gate" operation that requires significant energy.
        """
        current_state = self.get_state(state_id)
        if not current_state:
            raise ValueError(f"State {state_id} not found")

        if isinstance(current_state, NamedState):
            raise InvalidTransitionError(f"State {state_id} is already named '{current_state.name}'")

        # Calculate cost
        cost_amount = self.costs.unknown_to_named_cost

        # Check and deduct energy
        cost = MeasurementCost(
            energy=EnergyToken(cost_amount, self.unit),
            operation_name=f"promote_to_{name}"
        )

        try:
            self.ledger.record_transaction(cost)
        except InsufficientEnergyError:
            logger.error(f"Insufficient energy to promote state {state_id} to '{name}'")
            raise

        # Create new state
        new_metadata = current_state.metadata.copy()
        if metadata:
            new_metadata.update(metadata)

        new_state = NamedState(
            state_id=current_state.state_id, # Maintain identity
            created_at=current_state.created_at,
            metadata=new_metadata,
            name=name
        )

        # Update registry
        self._states[str(state_id)] = new_state
        logger.info(f"Promoted state {state_id} to NamedState '{name}'")

        return new_state

    def transition_to_unknown(self, state_id: str) -> UnknownState:
        """
        Demote a NamedState to UnknownState (increase entropy).

        Cheap operation.
        """
        current_state = self.get_state(state_id)
        if not current_state:
            raise ValueError(f"State {state_id} not found")

        if isinstance(current_state, UnknownState):
            return current_state

        # Calculate cost
        cost_amount = self.costs.named_to_unknown_cost

        cost = MeasurementCost(
            energy=EnergyToken(cost_amount, self.unit),
            operation_name="demote_to_unknown"
        )

        self.ledger.record_transaction(cost)

        new_state = UnknownState(
            state_id=current_state.state_id,
            created_at=current_state.created_at,
            metadata=current_state.metadata,
            entropy=1.0
        )

        self._states[str(state_id)] = new_state
        return new_state

"""
StateTransitionRegistry - Manages State Lifecycle Transitions

Implements energy-gated validation for Unknown→Named transitions.
Integrates with MeasurementLedger to enforce substrate precondition.

Axioms Implemented:
- Energy-Gated Transition: "State transitions require explicit substrate expenditure"
- Teleological Validity: "State transitions contain purpose-gated validity conditions"
- Causal Accountability: "Substrate state is modified by transition act"
"""

from typing import Dict, Optional
from uuid import UUID

from .state_descriptor import StateDescriptor, StateType, StateTransitionCost
from .ledger import MeasurementLedger, InsufficientEnergyError
from .measurement_cost import MeasurementCost


class InvalidTransitionError(Exception):
    """Raised when attempting an invalid state transition."""
    pass


class StateTransitionRegistry:
    """
    Central registry for managing state lifecycle transitions.
    
    Enforces energy gating and causal accountability for all transitions.
    """
    
    def __init__(
        self,
        ledger: MeasurementLedger,
        transition_costs: Optional[StateTransitionCost] = None
    ):
        """
        Initialize transition registry.
        
        Args:
            ledger: MeasurementLedger for energy accounting
            transition_costs: Custom transition costs (uses defaults if None)
        """
        self.ledger = ledger
        self.transition_costs = transition_costs or StateTransitionCost.default()
        
        # Track active states by UUID
        self._states: Dict[UUID, StateDescriptor] = {}
    
    def register_state(self, descriptor: StateDescriptor) -> None:
        """
        Register a state in the registry.
        
        Args:
            descriptor: StateDescriptor to register
        """
        self._states[descriptor.state_id] = descriptor
    
    def get_state(self, state_id: UUID) -> Optional[StateDescriptor]:
        """
        Retrieve a state by ID.
        
        Args:
            state_id: UUID of the state to retrieve
            
        Returns:
            StateDescriptor if found, None otherwise
        """
        return self._states.get(state_id)
    
    def promote_to_named(
        self,
        state_id: UUID,
        metadata: Optional[Dict] = None
    ) -> StateDescriptor:
        """
        Promote an Unknown state to Named, consuming substrate energy.
        
        Implements Energy-Gated Transition axiom - transition is blocked
        if insufficient energy available.
        
        Args:
            state_id: UUID of the state to promote
            metadata: Additional metadata for the Named state
            
        Returns:
            New Named StateDescriptor
            
        Raises:
            ValueError: If state not found or already Named
            InvalidTransitionError: If transition not valid
            InsufficientEnergyError: If insufficient energy for transition
        """
        # Retrieve current state
        current = self._states.get(state_id)
        if current is None:
            raise ValueError(f"State {state_id} not found in registry")
        
        # Validate transition is possible
        if not current.is_unknown():
            raise InvalidTransitionError(
                f"Cannot promote state {state_id}: already {current.state_type.value}"
            )
        
        # Calculate and record energy cost
        cost = MeasurementCost(
            energy=self.transition_costs.unknown_to_named,
            operation_name=f"unknown_to_named_{str(state_id)[:8]}"
        )
        
        # Record transaction (may raise InsufficientEnergyError)
        self.ledger.record_transaction(cost)
        
        # Perform transition
        new_state = current.promote_to_named(metadata)
        
        # Update registry
        self._states[state_id] = new_state
        
        return new_state
    
    def count_by_type(self, state_type: StateType) -> int:
        """
        Count states of a given type.
        
        Args:
            state_type: StateType to count
            
        Returns:
            Number of states with the given type
        """
        return sum(
            1 for state in self._states.values()
            if state.state_type == state_type
        )
    
    def get_diagnostics(self) -> Dict:
        """
        Get registry diagnostics.
        
        Returns:
            Dictionary with:
            - total_states: Total registered states
            - unknown_count: Number of Unknown states
            - named_count: Number of Named states
            - ledger_balance: Current energy balance
        """
        return {
            "total_states": len(self._states),
            "unknown_count": self.count_by_type(StateType.UNKNOWN),
            "named_count": self.count_by_type(StateType.NAMED),
            "ledger_balance": self.ledger.balance
        }

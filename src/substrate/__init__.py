"""
Substrate API - Phase 1: Primitive Types

Implements the foundational substrate types without gating logic,
resolving the circular dependency identified in meta-controller analysis.

Based on COMPASS-generated implementation plan and 10 Triadic Kernel axioms:
1. StateDescriptor Identity
2. Substrate Quantification
3. Measurement Cost
4. Energy-Gated Transition
5. Metric Namespace Isolation
6. Recursive Measurement
7. Identity Stability
8. Boundary Integrity
9. Teleological Validity
10. Causal Accountability

Phase 1 Components (from COMPASS plan #1-3):
- EnergyToken: Immutable value object for energy accounting
- SubstrateQuantity: Measurement with uncertainty
- MeasurementCost: Energy cost metadata structure
"""

from .director_integration import OperationCostProfile, SubstrateAwareDirector
from .energy_token import EnergyToken
from .ledger import InsufficientEnergyError, MeasurementLedger
from .measurement_cost import MeasurementCost
from .state_descriptor import NamedState, StateDescriptor, StateTransitionCost, UnknownState
from .substrate_quantity import SubstrateQuantity
from .transition_registry import InvalidTransitionError, StateTransitionRegistry

__all__ = [
    "EnergyToken",
    "SubstrateQuantity",
    "MeasurementCost",
    "MeasurementLedger",
    "InsufficientEnergyError",
    "StateDescriptor",
    "UnknownState",
    "NamedState",
    "StateTransitionRegistry",
    "StateTransitionCost",
    "InvalidTransitionError",
    "SubstrateAwareDirector",
    "OperationCostProfile",
]

__version__ = "0.1.0-phase1"

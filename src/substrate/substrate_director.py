"""
Substrate-Aware Director - Phase 3 Integration

Integrates MeasurementLedger with RAA Director to create substrate-accountable
meta-controller that recursively measures its own measurement operations.

Axioms Implemented:
- Energy-Gated Transition: Operations blocked when energy insufficient
- Recursive Measurement: Meta-controller tracks cost of its own monitoring
- Causal Accountability: Each cognitive operation modifies substrate state
"""

from decimal import Decimal
from typing import Dict, Optional
from dataclasses import dataclass

from ..substrate import (
    EnergyToken,
    MeasurementLedger,
    MeasurementCost,
    SubstrateQuantity,
    InsufficientEnergyError
)


@dataclass
class OperationCostProfile:
    """Energy cost profile for RAA cognitive operations."""
    
    # Base costs per operation type
    deconstruct: EnergyToken = EnergyToken(Decimal("5.0"), "cognitive_units")
    hypothesize: EnergyToken = EnergyToken(Decimal("8.0"), "cognitive_units")
    synthesize: EnergyToken = EnergyToken(Decimal("10.0"), "cognitive_units")
    check_state: EnergyToken = EnergyToken(Decimal("1.0"), "cognitive_units")
    
    # Scaling factors based on cognitive load
    entropy_multiplier: float = 2.0  # High entropy = higher cost
    manifold_energy_multiplier: float = 1.5  # High manifold energy = higher cost


class SubstrateAwareDirector:
    """
    Wraps RAA Director with substrate-based energy accounting.
    
    Implements recursive measurement: tracks energy cost of monitoring
    operations, creating substrate accountability for meta-cognitive processes.
    """
    
    def __init__(
        self,
        ledger: MeasurementLedger,
        cost_profile: Optional[OperationCostProfile] = None,
        critical_threshold: Optional[EnergyToken] = None
    ):
        """
        Initialize substrate-aware director.
        
        Args:
            ledger: MeasurementLedger for energy accounting
            cost_profile: Energy costs for operations (uses defaults if None)
            critical_threshold: Energy level triggering auto-recharge
        """
        self.ledger = ledger
        self.cost_profile = cost_profile or OperationCostProfile()
        self.critical_threshold = critical_threshold or EnergyToken(
            Decimal("10.0"), "cognitive_units"
        )
        
        # Track operation counts for diagnostics
        self._operation_counts: Dict[str, int] = {
            "deconstruct": 0,
            "hypothesize": 0,
            "synthesize": 0,
            "check_state": 0,
            "recharge": 0
        }
    
    def gate_operation(
        self,
        operation_type: str,
        entropy: Optional[float] = None,
        manifold_energy: Optional[float] = None
    ) -> bool:
        """
        Check if operation is allowed given current energy state.
        
        Implements Energy-Gated Transition axiom.
        
        Args:
            operation_type: Type of operation to gate
            entropy: Current cognitive entropy (for cost scaling)
            manifold_energy: Current manifold energy (for cost scaling)
            
        Returns:
            True if operation allowed, False if gated
        """
        # Calculate operation cost with scaling
        base_cost = self._get_base_cost(operation_type)
        scaled_cost = self._scale_cost(base_cost, entropy, manifold_energy)
        
        # Check if sufficient energy
        return self.ledger.check_balance(scaled_cost)
    
    def record_operation(
        self,
        operation_type: str,
        entropy: Optional[float] = None,
        manifold_energy: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Record energy cost of completed operation.
        
        Implements Recursive Measurement and Causal Accountability axioms.
        
        Args:
            operation_type: Type of operation completed
            entropy: Cognitive entropy during operation
            manifold_energy: Manifold energy during operation
            metadata: Additional operation metadata
            
        Raises:
            InsufficientEnergyError: If balance insufficient for operation
        """
        # Calculate actual cost
        base_cost = self._get_base_cost(operation_type)
        scaled_cost = self._scale_cost(base_cost, entropy, manifold_energy)
        
        # Create measurement cost record
        # Note: cpu_cycles/memory_bytes would be added here if we're tracking
        # actual hardware resources. For now, scaled_cost already accounts for
        # cognitive load via entropy and manifold energy multipliers.
        cost = MeasurementCost(
            energy=scaled_cost,
            operation_name=f"raa_{operation_type}"
        )
        
        # Record transaction (may raise InsufficientEnergyError)
        self.ledger.record_transaction(cost)
        
        # Update operation count
        self._operation_counts[operation_type] = \
            self._operation_counts.get(operation_type, 0) + 1
        
        # Check for critical energy and auto-recharge
        if self.ledger.balance < self.critical_threshold:
            self._auto_recharge()
    
    def get_diagnostics(self) -> Dict:
        """
        Get energy consumption diagnostics.
        
        Returns:
            Dictionary with:
            - current_balance: Current energy level
            - operation_counts: Count of each operation type
            - total_operations: Total operations performed
            - history_length: Number of transactions recorded
        """
        return {
            "current_balance": self.ledger.balance,
            "operation_counts": dict(self._operation_counts),
            "total_operations": sum(self._operation_counts.values()),
            "history_length": len(self.ledger.history),
            "critical_threshold": self.critical_threshold
        }
    
    def _get_base_cost(self, operation_type: str) -> EnergyToken:
        """Get base energy cost for operation type."""
        costs = {
            "deconstruct": self.cost_profile.deconstruct,
            "hypothesize": self.cost_profile.hypothesize,
            "synthesize": self.cost_profile.synthesize,
            "check_state": self.cost_profile.check_state
        }
        return costs.get(operation_type, EnergyToken(Decimal("1.0"), "cognitive_units"))
    
    def _scale_cost(
        self,
        base: EnergyToken,
        entropy: Optional[float],
        manifold_energy: Optional[float]
    ) -> EnergyToken:
        """Scale base cost by cognitive load factors."""
        multiplier = Decimal("1.0")
        
        # Scale by entropy (high entropy = confused = higher cost)
        if entropy is not None and entropy > 0.5:
            multiplier *= Decimal(str(self.cost_profile.entropy_multiplier))
        
        # Scale by manifold energy (high energy = unstable = higher cost)
        if manifold_energy is not None and abs(manifold_energy) > 0.4:
            multiplier *= Decimal(str(self.cost_profile.manifold_energy_multiplier))
        
        return base * multiplier
    
    def _auto_recharge(self) -> None:
        """
        Automatic energy recharge when critical threshold reached.
        
        Implements substrate replenishment policy.
        """
        recharge_amount = EnergyToken(Decimal("50.0"), "cognitive_units")
        self.ledger.top_up(recharge_amount)
        self._operation_counts["recharge"] = \
            self._operation_counts.get("recharge", 0) + 1

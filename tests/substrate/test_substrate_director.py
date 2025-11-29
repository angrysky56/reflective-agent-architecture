"""
Phase 3 Tests - Substrate-Aware Director

Validates integration of MeasurementLedger with RAA Director pattern.
Tests recursive measurement, energy gating, and causal accountability.
"""

from decimal import Decimal
import pytest

from src.substrate import (
    EnergyToken,
    MeasurementLedger,
    InsufficientEnergyError
)
from src.substrate.substrate_director import (
    SubstrateAwareDirector,
    OperationCostProfile
)


class TestSubstrateAwareDirector:
    """Test substrate-based energy gating and recursive measurement."""
    
    def test_initialization(self):
        """Director initializes with ledger and default cost profile."""
        initial_energy = EnergyToken(Decimal("100.0"), "cognitive_units")
        ledger = MeasurementLedger(initial_energy)
        director = SubstrateAwareDirector(ledger)
        
        assert director.ledger.balance == initial_energy
        assert director.cost_profile is not None
        assert director.critical_threshold == EnergyToken(Decimal("10.0"), "cognitive_units")
    
    def test_gate_operation_sufficient_energy(self):
        """Operations allowed when sufficient energy available."""
        initial_energy = EnergyToken(Decimal("100.0"), "cognitive_units")
        ledger = MeasurementLedger(initial_energy)
        director = SubstrateAwareDirector(ledger)
        
        # Should allow deconstruct (cost: 5.0)
        assert director.gate_operation("deconstruct") is True
        
        # Should allow synthesize (cost: 10.0)
        assert director.gate_operation("synthesize") is True
    
    def test_gate_operation_insufficient_energy(self):
        """Operations gated when energy insufficient."""
        low_energy = EnergyToken(Decimal("3.0"), "cognitive_units")
        ledger = MeasurementLedger(low_energy)
        director = SubstrateAwareDirector(ledger)
        
        # Should gate deconstruct (cost: 5.0 > balance: 3.0)
        assert director.gate_operation("deconstruct") is False
    
    def test_record_operation_deducts_energy(self):
        """Recording operation deducts energy from ledger."""
        initial_energy = EnergyToken(Decimal("100.0"), "cognitive_units")
        ledger = MeasurementLedger(initial_energy)
        director = SubstrateAwareDirector(ledger)
        
        director.record_operation("deconstruct")
        
        # 100 - 5 = 95
        expected = EnergyToken(Decimal("95.0"), "cognitive_units")
        assert ledger.balance == expected
    
    def test_record_operation_tracks_count(self):
        """Recording operations updates diagnostic counters."""
        initial_energy = EnergyToken(Decimal("100.0"), "cognitive_units")
        ledger = MeasurementLedger(initial_energy)
        director = SubstrateAwareDirector(ledger)
        
        director.record_operation("deconstruct")
        director.record_operation("deconstruct")
        director.record_operation("hypothesize")
        
        diagnostics = director.get_diagnostics()
        assert diagnostics["operation_counts"]["deconstruct"] == 2
        assert diagnostics["operation_counts"]["hypothesize"] == 1
        assert diagnostics["total_operations"] == 3
    
    def test_cost_scaling_by_entropy(self):
        """High entropy increases operation cost (Entropy Multiplier axiom)."""
        initial_energy = EnergyToken(Decimal("100.0"), "cognitive_units")
        ledger = MeasurementLedger(initial_energy)
        director = SubstrateAwareDirector(ledger)
        
        # Record operation with high entropy (> 0.5)
        director.record_operation("deconstruct", entropy=0.8)
        
        # Cost should be scaled: 5.0 * 2.0 (entropy_multiplier) = 10.0
        # 100 - 10 = 90
        expected = EnergyToken(Decimal("90.0"), "cognitive_units")
        assert ledger.balance == expected
    
    def test_cost_scaling_by_manifold_energy(self):
        """High manifold energy increases operation cost."""
        initial_energy = EnergyToken(Decimal("100.0"), "cognitive_units")
        ledger = MeasurementLedger(initial_energy)
        director = SubstrateAwareDirector(ledger)
        
        # Record operation with high manifold energy (> 0.4)
        director.record_operation("deconstruct", manifold_energy=-0.45)
        
        # Cost should be scaled: 5.0 * 1.5 (manifold_multiplier) = 7.5
        # 100 - 7.5 = 92.5
        expected = EnergyToken(Decimal("92.5"), "cognitive_units")
        assert ledger.balance == expected
    
    def test_combined_cost_scaling(self):
        """High entropy AND manifold energy compound scaling."""
        initial_energy = EnergyToken(Decimal("100.0"), "cognitive_units")
        ledger = MeasurementLedger(initial_energy)
        director = SubstrateAwareDirector(ledger)
        
        # Both entropy and manifold energy high
        director.record_operation("deconstruct", entropy=0.8, manifold_energy=-0.45)
        
        # Cost: 5.0 * 2.0 * 1.5 = 15.0
        # 100 - 15 = 85
        expected = EnergyToken(Decimal("85.0"), "cognitive_units")
        assert ledger.balance == expected
    
    def test_auto_recharge_on_critical_threshold(self):
        """System auto-recharges when energy hits critical threshold."""
        # Start with energy just above critical
        initial_energy = EnergyToken(Decimal("15.0"), "cognitive_units")
        ledger = MeasurementLedger(initial_energy)
        
        critical = EnergyToken(Decimal("10.0"), "cognitive_units")
        director = SubstrateAwareDirector(ledger, critical_threshold=critical)
        
        # This operation costs 10.0, bringing balance to 5.0 (below critical)
        director.record_operation("synthesize")
        
        # Should have auto-recharged +50
        # 15 - 10 = 5, then +50 = 55
        expected = EnergyToken(Decimal("55.0"), "cognitive_units")
        assert ledger.balance == expected
        
        # Recharge should be tracked
        diagnostics = director.get_diagnostics()
        assert diagnostics["operation_counts"]["recharge"] == 1
    
    def test_insufficient_energy_raises_error(self):
        """Recording operation with insufficient energy raises error."""
        low_energy = EnergyToken(Decimal("3.0"), "cognitive_units")
        ledger = MeasurementLedger(low_energy)
        director = SubstrateAwareDirector(ledger)
        
        with pytest.raises(InsufficientEnergyError):
            director.record_operation("deconstruct")  # Costs 5.0
        
        # Balance should remain unchanged
        assert ledger.balance == low_energy
    
    def test_diagnostics_complete(self):
        """Diagnostics provide complete operational view."""
        initial_energy = EnergyToken(Decimal("100.0"), "cognitive_units")
        ledger = MeasurementLedger(initial_energy)
        director = SubstrateAwareDirector(ledger)
        
        director.record_operation("deconstruct")
        director.record_operation("hypothesize")
        
        diagnostics = director.get_diagnostics()
        
        assert "current_balance" in diagnostics
        assert "operation_counts" in diagnostics
        assert "total_operations" in diagnostics
        assert diagnostics["total_operations"] == 2
        assert "history_length" in diagnostics
        assert diagnostics["history_length"] == 2

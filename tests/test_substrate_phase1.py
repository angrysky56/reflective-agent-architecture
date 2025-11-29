"""
Phase 1 Unit Tests - Primitive Substrate Types

Tests Components #4-6 from COMPASS plan:
- #4: EnergyToken arithmetic operations
- #5: SubstrateQuantity comparison operations  
- #6: MeasurementCost serialization

Validates Triadic Kernel axioms:
- Substrate Quantification: finite, consumable, deterministic
- Measurement Cost: non-zero consumption
"""

import pytest
from decimal import Decimal
from src.substrate import EnergyToken, SubstrateQuantity, MeasurementCost


class TestEnergyToken:
    """Component #4: Validate EnergyToken arithmetic operations"""
    
    def test_creation_valid(self):
        """EnergyToken can be created with valid amount"""
        token = EnergyToken(amount=Decimal("100.0"))
        assert token.amount == Decimal("100.0")
        assert token.unit == "energy_units"
    
    def test_creation_negative_fails(self):
        """EnergyToken rejects negative amounts (Substrate Quantification axiom)"""
        with pytest.raises(ValueError, match="non-negative"):
            EnergyToken(amount=Decimal("-10.0"))
    
    def test_addition(self):
        """Energy tokens can be added"""
        t1 = EnergyToken(amount=Decimal("50.0"))
        t2 = EnergyToken(amount=Decimal("30.0"))
        result = t1 + t2
        assert result.amount == Decimal("80.0")
    
    def test_addition_incompatible_units(self):
        """Cannot add tokens with different units"""
        t1 = EnergyToken(amount=Decimal("50.0"), unit="joules")
        t2 = EnergyToken(amount=Decimal("30.0"), unit="watts")
        with pytest.raises(ValueError, match="different units"):
            t1 + t2
    
    def test_subtraction(self):
        """Energy tokens can be subtracted"""
        t1 = EnergyToken(amount=Decimal("50.0"))
        t2 = EnergyToken(amount=Decimal("30.0"))
        result = t1 - t2
        assert result.amount == Decimal("20.0")
    
    def test_subtraction_underflow_fails(self):
        """Cannot subtract more energy than available"""
        t1 = EnergyToken(amount=Decimal("30.0"))
        t2 = EnergyToken(amount=Decimal("50.0"))
        with pytest.raises(ValueError, match="negative amount"):
            t1 - t2
    
    def test_multiplication(self):
        """Energy can be scaled by scalar"""
        token = EnergyToken(amount=Decimal("10.0"))
        result = token * 3
        assert result.amount == Decimal("30.0")
    
    def test_division(self):
        """Energy can be divided by scalar"""
        token = EnergyToken(amount=Decimal("30.0"))
        result = token / 3
        assert result.amount == Decimal("10.0")
    
    def test_division_by_zero_fails(self):
        """Cannot divide energy by zero"""
        token = EnergyToken(amount=Decimal("30.0"))
        with pytest.raises(ZeroDivisionError):
            token / 0
    
    def test_comparison(self):
        """Energy tokens can be compared"""
        t1 = EnergyToken(amount=Decimal("50.0"))
        t2 = EnergyToken(amount=Decimal("30.0"))
        assert t1 > t2
        assert t2 < t1
        assert t1 == EnergyToken(amount=Decimal("50.0"))
    
    def test_immutability(self):
        """EnergyToken is immutable (Identity Stability axiom)"""
        token = EnergyToken(amount=Decimal("100.0"))
        with pytest.raises(AttributeError):
            token.amount = Decimal("200.0")  # type: ignore


class TestSubstrateQuantity:
    """Component #5: Validate SubstrateQuantity comparison operations"""
    
    def test_creation(self):
        """SubstrateQuantity can be created with magnitude and uncertainty"""
        q = SubstrateQuantity(magnitude=Decimal("10.5"), uncertainty=Decimal("0.1"))
        assert q.magnitude == Decimal("10.5")
        assert q.uncertainty == Decimal("0.1")
    
    def test_negative_uncertainty_fails(self):
        """Uncertainty must be non-negative"""
        with pytest.raises(ValueError, match="non-negative"):
            SubstrateQuantity(magnitude=Decimal("10.0"), uncertainty=Decimal("-0.1"))
    
    def test_addition_error_propagation(self):
        """Addition propagates uncertainty correctly"""
        q1 = SubstrateQuantity(magnitude=Decimal("10.0"), uncertainty=Decimal("0.3"))
        q2 = SubstrateQuantity(magnitude=Decimal("5.0"), uncertainty=Decimal("0.4"))
        result = q1 + q2
        assert result.magnitude == Decimal("15.0")
        # σ = √(0.3² + 0.4²) = 0.5
        assert abs(result.uncertainty - Decimal("0.5")) < Decimal("0.001")
    
    def test_subtraction_error_propagation(self):
        """Subtraction propagates uncertainty correctly"""
        q1 = SubstrateQuantity(magnitude=Decimal("10.0"), uncertainty=Decimal("0.3"))
        q2 = SubstrateQuantity(magnitude=Decimal("5.0"), uncertainty=Decimal("0.4"))
        result = q1 - q2
        assert result.magnitude == Decimal("5.0")
        assert abs(result.uncertainty - Decimal("0.5")) < Decimal("0.001")
    
    def test_scalar_multiplication(self):
        """Scaling preserves relative uncertainty structure"""
        q = SubstrateQuantity(magnitude=Decimal("10.0"), uncertainty=Decimal("0.5"))
        result = q * 2
        assert result.magnitude == Decimal("20.0")
        assert result.uncertainty == Decimal("1.0")
    
    def test_comparison(self):
        """SubstrateQuantity magnitudes can be compared"""
        q1 = SubstrateQuantity(magnitude=Decimal("10.0"))
        q2 = SubstrateQuantity(magnitude=Decimal("5.0"))
        assert q1 > q2
        assert q2 < q1
    
    def test_overlap_detection(self):
        """Can detect overlapping uncertainty bounds"""
        q1 = SubstrateQuantity(magnitude=Decimal("10.0"), uncertainty=Decimal("1.0"))
        q2 = SubstrateQuantity(magnitude=Decimal("10.5"), uncertainty=Decimal("1.0"))
        assert q1.overlaps(q2)
        
        q3 = SubstrateQuantity(magnitude=Decimal("15.0"), uncertainty=Decimal("0.5"))
        assert not q1.overlaps(q3)


class TestMeasurementCost:
    """Component #6: Validate MeasurementCost serialization and cost calculation"""
    
    def test_creation_minimal(self):
        """MeasurementCost can be created with just energy"""
        energy = EnergyToken(amount=Decimal("5.0"))
        cost = MeasurementCost(energy=energy, operation_name="test_op")
        assert cost.energy == energy
        assert cost.operation_name == "test_op"
    
    def test_creation_full(self):
        """MeasurementCost can include all cost components"""
        energy = EnergyToken(amount=Decimal("5.0"))
        cpu = SubstrateQuantity(magnitude=Decimal("1000.0"))
        mem = SubstrateQuantity(magnitude=Decimal("2048.0"))
        time = SubstrateQuantity(magnitude=Decimal("500.0"))
        
        cost = MeasurementCost(
            energy=energy,
            cpu_cycles=cpu,
            memory_bytes=mem,
            time_nanos=time,
            operation_name="full_measurement"
        )
        
        assert cost.cpu_cycles == cpu
        assert cost.memory_bytes == mem
        assert cost.time_nanos == time
    
    def test_negative_component_fails(self):
        """Cost components must be non-negative (Measurement Cost axiom)"""
        energy = EnergyToken(amount=Decimal("5.0"))
        negative_cpu = SubstrateQuantity(magnitude=Decimal("-100.0"))
        
        with pytest.raises(ValueError, match="non-negative"):
            MeasurementCost(energy=energy, cpu_cycles=negative_cpu)
    
    def test_total_energy_calculation(self):
        """Total energy implements weighted formula from Component #10"""
        energy = EnergyToken(amount=Decimal("5.0"))
        cpu = SubstrateQuantity(magnitude=Decimal("1000.0"))
        mem = SubstrateQuantity(magnitude=Decimal("2000.0"))
        time = SubstrateQuantity(magnitude=Decimal("1000.0"))  # 1 microsecond
        
        cost = MeasurementCost(
            energy=energy,
            cpu_cycles=cpu,
            memory_bytes=mem,
            time_nanos=time
        )
        
        # total = 5.0 + 0.4*1000 + 0.3*2000 + 0.3*1 = 5.0 + 400 + 600 + 0.3 = 1005.3
        total = cost.total_energy(cpu_weight=0.4, mem_weight=0.3, time_weight=0.3)
        expected = Decimal("5.0") + Decimal("400.0") + Decimal("600.0") + Decimal("0.3")
        assert abs(total.amount - expected) < Decimal("0.01")
    
    def test_immutability(self):
        """MeasurementCost is immutable"""
        energy = EnergyToken(amount=Decimal("5.0"))
        cost = MeasurementCost(energy=energy)
        with pytest.raises(AttributeError):
            cost.operation_name = "modified"  # type: ignore

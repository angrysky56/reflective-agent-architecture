
from decimal import Decimal

import pytest

from src.substrate import EnergyToken, InsufficientEnergyError, MeasurementCost, MeasurementLedger, SubstrateQuantity


def test_ledger_initialization():
    initial = EnergyToken(Decimal("100.0"), "joules")
    ledger = MeasurementLedger(initial)
    assert ledger.balance == initial
    assert len(ledger.history) == 0

def test_record_transaction_success():
    initial = EnergyToken(Decimal("100.0"), "joules")
    ledger = MeasurementLedger(initial)

    cost = MeasurementCost(
        energy=EnergyToken(Decimal("10.0"), "joules"),
        operation_name="test_op"
    )

    ledger.record_transaction(cost)

    # 100 - 10 = 90
    expected = EnergyToken(Decimal("90.0"), "joules")
    assert ledger.balance == expected
    assert len(ledger.history) == 1
    assert ledger.history[0] == cost

def test_record_transaction_insufficient_funds():
    initial = EnergyToken(Decimal("5.0"), "joules")
    ledger = MeasurementLedger(initial)

    cost = MeasurementCost(
        energy=EnergyToken(Decimal("10.0"), "joules"),
        operation_name="expensive_op"
    )

    with pytest.raises(InsufficientEnergyError):
        ledger.record_transaction(cost)

    # Balance should remain unchanged
    assert ledger.balance == initial
    assert len(ledger.history) == 0

def test_check_balance():
    initial = EnergyToken(Decimal("50.0"), "joules")
    ledger = MeasurementLedger(initial)

    assert ledger.check_balance(EnergyToken(Decimal("40.0"), "joules")) is True
    assert ledger.check_balance(EnergyToken(Decimal("50.0"), "joules")) is True
    assert ledger.check_balance(EnergyToken(Decimal("50.1"), "joules")) is False

def test_top_up():
    initial = EnergyToken(Decimal("10.0"), "joules")
    ledger = MeasurementLedger(initial)

    ledger.top_up(EnergyToken(Decimal("20.0"), "joules"))

    expected = EnergyToken(Decimal("30.0"), "joules")
    assert ledger.balance == expected

def test_transaction_with_metadata_cost():
    initial = EnergyToken(Decimal("100.0"), "joules")
    ledger = MeasurementLedger(initial)

    # Cost with CPU usage
    # total = 10 + 0.4 * 100 = 50
    cost = MeasurementCost(
        energy=EnergyToken(Decimal("10.0"), "joules"),
        cpu_cycles=SubstrateQuantity(Decimal("100.0")),
        operation_name="complex_op"
    )

    ledger.record_transaction(cost)

    expected = EnergyToken(Decimal("50.0"), "joules")
    assert ledger.balance == expected

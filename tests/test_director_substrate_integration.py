
from decimal import Decimal
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.substrate import EnergyToken, InsufficientEnergyError, MeasurementLedger
from src.substrate.director_integration import OperationCostProfile, SubstrateAwareDirector


# Mock DirectorMVP since we don't want to instantiate the real one with heavy dependencies
class MockDirector:
    def check_and_search(self, current_state, processor_logits, context=None):
        return "new_goal" # Simulate finding a goal

    def diagnose(self, weights, target_error=None, feedback_weights=None):
        return {"status": "diagnosed"}

    def teach_state(self, label):
        return True

@pytest.fixture
def ledger():
    initial = EnergyToken(Decimal("100.0"), "joules")
    return MeasurementLedger(initial)

@pytest.fixture
def mock_director():
    return MockDirector()

@pytest.fixture
def substrate_director(ledger, mock_director):
    return SubstrateAwareDirector(mock_director, ledger)

def test_initialization(substrate_director):
    assert substrate_director.ledger.balance.amount == Decimal("100.0")
    assert isinstance(substrate_director.cost_profile, OperationCostProfile)

def test_check_and_search_cost(substrate_director):
    # Initial balance
    start_balance = substrate_director.ledger.balance.amount

    # Run operation
    substrate_director.check_and_search(None, None)

    # Expected cost: monitoring (0.1) + search (1.0) = 1.1
    expected_cost = Decimal("1.1")
    current_balance = substrate_director.ledger.balance.amount

    assert start_balance - current_balance == expected_cost
    assert len(substrate_director.ledger.history) == 2 # Monitor + Search

def test_diagnose_cost(substrate_director):
    start_balance = substrate_director.ledger.balance.amount

    substrate_director.diagnose(None)

    # Expected cost: diagnosis (2.0)
    expected_cost = Decimal("2.0")
    current_balance = substrate_director.ledger.balance.amount

    assert start_balance - current_balance == expected_cost

def test_teach_state_cost(substrate_director):
    start_balance = substrate_director.ledger.balance.amount

    substrate_director.teach_state("test")

    # Expected cost: learning (5.0)
    expected_cost = Decimal("5.0")
    current_balance = substrate_director.ledger.balance.amount

    assert start_balance - current_balance == expected_cost

def test_insufficient_energy(ledger, mock_director):
    # Create director with very low energy
    low_energy_ledger = MeasurementLedger(EnergyToken(Decimal("0.05"), "joules"))
    # Disable auto-recharge for this test by setting threshold high but balance lower
    # Actually, easier to just mock ensure_energy to do nothing or set threshold to 0

    profile = OperationCostProfile(auto_recharge_threshold=Decimal("0.0"))
    director = SubstrateAwareDirector(mock_director, low_energy_ledger, profile)

    # Should fail monitoring (cost 0.1 > 0.05)
    result = director.check_and_search(None, None)

    assert result is None
    # Balance should be unchanged as operation was halted before cost recording?
    # No, check_balance raises InsufficientEnergyError inside record_transaction
    # And we catch it and return None
    assert low_energy_ledger.balance.amount == Decimal("0.05")

def test_auto_recharge(ledger, mock_director):
    # Set threshold higher than current balance
    profile = OperationCostProfile(
        auto_recharge_threshold=Decimal("150.0"), # Higher than 100
        recharge_amount=Decimal("50.0")
    )
    director = SubstrateAwareDirector(mock_director, ledger, profile)

    # Trigger operation
    director.check_and_search(None, None)

    # Should have recharged: 100 + 50 = 150
    # Then paid costs: 150 - 1.1 = 148.9
    expected_balance = Decimal("148.9")
    assert director.ledger.balance.amount == expected_balance

def test_delegation(substrate_director):
    # Verify that method calls are actually passed to the inner director
    with patch.object(substrate_director.director, 'check_and_search', return_value="delegated") as mock_method:
        result = substrate_director.check_and_search("state", "logits")

        mock_method.assert_called_once_with("state", "logits", None)
        assert result == "delegated"

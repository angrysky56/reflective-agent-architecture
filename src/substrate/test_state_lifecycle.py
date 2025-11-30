

from decimal import Decimal

import pytest

from src.substrate import EnergyToken, InsufficientEnergyError, InvalidTransitionError, MeasurementCost, MeasurementLedger, NamedState, StateDescriptor, StateTransitionRegistry, UnknownState


@pytest.fixture
def ledger():
    initial = EnergyToken(Decimal("100.0"), "joules")
    return MeasurementLedger(initial)

@pytest.fixture
def registry(ledger):
    return StateTransitionRegistry(ledger)

def test_state_creation():
    state = UnknownState()
    assert state.is_named is False
    assert state.entropy == 1.0
    assert isinstance(state, StateDescriptor)

def test_register_state(registry):
    state = UnknownState()
    registry.register_state(state)

    retrieved = registry.get_state(str(state.state_id))
    assert retrieved == state

def test_promote_to_named_success(registry, ledger):
    # Setup
    state = UnknownState()
    registry.register_state(state)
    initial_balance = ledger.balance.amount

    # Action
    named_state = registry.promote_to_named(str(state.state_id), "Focused")

    # Verification
    assert isinstance(named_state, NamedState)
    assert named_state.name == "Focused"
    assert named_state.state_id == state.state_id # Identity preserved

    # Cost verification (default cost is 15.0)
    expected_balance = initial_balance - Decimal("15.0")
    assert ledger.balance.amount == expected_balance

    # Registry update verification
    retrieved = registry.get_state(str(state.state_id))
    assert isinstance(retrieved, NamedState)
    assert retrieved.name == "Focused"

def test_promote_insufficient_energy(registry, ledger):
    # Drain ledger
    drain = MeasurementCost(
        energy=EnergyToken(Decimal("90.0"), "joules"),
        operation_name="drain"
    )
    ledger.record_transaction(drain)
    assert ledger.balance.amount == Decimal("10.0") # Less than 15.0 cost

    state = UnknownState()
    registry.register_state(state)

    with pytest.raises(InsufficientEnergyError):
        registry.promote_to_named(str(state.state_id), "Focused")

    # State should remain Unknown
    retrieved = registry.get_state(str(state.state_id))
    assert isinstance(retrieved, UnknownState)

def test_invalid_transition(registry):
    state = NamedState(name="Existing")
    registry.register_state(state)

    # Cannot promote already named state
    with pytest.raises(InvalidTransitionError):
        registry.promote_to_named(str(state.state_id), "NewName")

def test_demote_to_unknown(registry, ledger):
    state = NamedState(name="Stuck")
    registry.register_state(state)
    initial_balance = ledger.balance.amount

    unknown = registry.transition_to_unknown(str(state.state_id))

    assert isinstance(unknown, UnknownState)
    assert unknown.state_id == state.state_id

    # Cost verification (default cost is 1.0)
    expected_balance = initial_balance - Decimal("1.0")
    assert ledger.balance.amount == expected_balance

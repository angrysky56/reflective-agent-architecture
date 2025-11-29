"""
MeasurementLedger - Energy Accounting and Gating

Axiom Implementation:
- Energy-Gated Transition Axiom: "State transitions require explicit substrate expenditure"
- Recursive Measurement Axiom: "Measurement operations must recursively account for their own substrate cost"
- Causal Accountability Axiom: "The substrate state... is necessarily modified by the act of measurement"

Component #7-12 from COMPASS plan: Implement MeasurementLedger with transaction history and gating logic
"""

import threading
from typing import Callable, List, Optional

from .energy_token import EnergyToken
from .measurement_cost import MeasurementCost


class InsufficientEnergyError(Exception):
    """Raised when an operation requires more energy than available."""
    pass


class MeasurementLedger:
    """
    Central ledger for tracking substrate energy consumption and enforcing limits.

    Thread-safe implementation using a lock for transaction recording.
    """

    def __init__(self, initial_balance: EnergyToken):
        """
        Initialize ledger with starting energy balance.

        Args:
            initial_balance: Initial energy available to the system
        """
        self._balance = initial_balance
        self._history: List[MeasurementCost] = []
        self._lock = threading.RLock()
        self._on_transaction: Optional[Callable[[MeasurementCost, EnergyToken], None]] = None

    def set_transaction_callback(self, callback: Callable[[MeasurementCost, EnergyToken], None]) -> None:
        """Set a callback to be invoked after each successful transaction."""
        with self._lock:
            self._on_transaction = callback

    @property
    def balance(self) -> EnergyToken:
        """Current energy balance."""
        with self._lock:
            return self._balance

    @property
    def history(self) -> List[MeasurementCost]:
        """Transaction history."""
        with self._lock:
            return list(self._history)

    def check_balance(self, required: EnergyToken) -> bool:
        """
        Check if sufficient energy is available.

        Args:
            required: Amount of energy required

        Returns:
            True if balance >= required, False otherwise
        """
        with self._lock:
            # EnergyToken handles unit compatibility check in comparison
            return self._balance >= required

    def record_transaction(self, cost: MeasurementCost) -> None:
        """
        Record a measurement cost and deduct energy.

        Args:
            cost: The cost of the operation to record

        Raises:
            InsufficientEnergyError: If balance is insufficient
            ValueError: If cost unit doesn't match ledger unit
        """
        with self._lock:
            total_cost = cost.total_energy()

            if not self.check_balance(total_cost):
                raise InsufficientEnergyError(
                    f"Insufficient energy: required {total_cost}, available {self._balance}"
                )

            # Deduct energy
            self._balance = self._balance - total_cost
            self._history.append(cost)

            # Notify callback if set
            if self._on_transaction:
                try:
                    self._on_transaction(cost, self._balance)
                except Exception as e:
                    # Don't let callback failure break the transaction
                    print(f"Transaction callback failed: {e}")

    def top_up(self, amount: EnergyToken) -> None:
        """
        Add energy to the ledger (e.g., from external source or recharge).

        Args:
            amount: Energy to add
        """
        with self._lock:
            self._balance = self._balance + amount

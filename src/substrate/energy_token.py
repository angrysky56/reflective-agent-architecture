"""
EnergyToken - Immutable Value Object for Substrate Energy Accounting

Axiom Implementation: Substrate Quantification Axiom
"Substrate exists as a finite, consumable resource measured in deterministic energy units."

Component #1 from COMPASS plan: Design EnergyToken as immutable value object with unit and amount fields
"""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class EnergyToken:
    """
    Immutable representation of substrate energy.

    Attributes:
        amount: Precise energy quantity using Decimal for deterministic arithmetic
        unit: Energy unit descriptor (e.g., 'joules', 'compute_cycles', 'tokens')

    Invariants:
        - amount must be non-negative (enforced by __post_init__)
        - instances are immutable (frozen=True)
        - arithmetic operations return new instances
    """

    amount: Decimal
    unit: str = "energy_units"

    def __post_init__(self) -> None:
        """Validate amount is non-negative."""
        if self.amount < 0:
            raise ValueError(f"EnergyToken amount must be non-negative, got {self.amount}")

    def __add__(self, other: "EnergyToken") -> "EnergyToken":
        """Add two energy tokens with unit compatibility check."""
        if self.unit != other.unit:
            raise ValueError(f"Cannot add tokens with different units: {self.unit} vs {other.unit}")
        return EnergyToken(amount=self.amount + other.amount, unit=self.unit)

    def __sub__(self, other: "EnergyToken") -> "EnergyToken":
        """Subtract energy tokens with unit compatibility check."""
        if self.unit != other.unit:
            raise ValueError(
                f"Cannot subtract tokens with different units: {self.unit} vs {other.unit}"
            )
        result_amount = self.amount - other.amount
        if result_amount < 0:
            raise ValueError(f"Energy subtraction would result in negative amount: {result_amount}")
        return EnergyToken(amount=result_amount, unit=self.unit)

    def __mul__(self, scalar: Decimal | float | int) -> "EnergyToken":
        """Multiply energy by scalar."""
        return EnergyToken(amount=self.amount * Decimal(str(scalar)), unit=self.unit)

    def __truediv__(self, scalar: Decimal | float | int) -> "EnergyToken":
        """Divide energy by scalar."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide energy by zero")
        return EnergyToken(amount=self.amount / Decimal(str(scalar)), unit=self.unit)

    def __lt__(self, other: "EnergyToken") -> bool:
        """Compare energy amounts with unit compatibility check."""
        if self.unit != other.unit:
            raise ValueError(
                f"Cannot compare tokens with different units: {self.unit} vs {other.unit}"
            )
        return self.amount < other.amount

    def __le__(self, other: "EnergyToken") -> bool:
        if self.unit != other.unit:
            raise ValueError(
                f"Cannot compare tokens with different units: {self.unit} vs {other.unit}"
            )
        return self.amount <= other.amount

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnergyToken):
            return False
        return self.amount == other.amount and self.unit == other.unit

    def __repr__(self) -> str:
        return f"EnergyToken(amount={self.amount}, unit='{self.unit}')"

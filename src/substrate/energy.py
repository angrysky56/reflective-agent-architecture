import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional


class EnergyUnit(Enum):
    JOULES = "joules"

@dataclass
class EnergyToken:
    amount: Decimal
    unit: str = "joules"

    def __add__(self, other: 'EnergyToken') -> 'EnergyToken':
        if self.unit != other.unit:
            raise ValueError(f"Cannot add different units: {self.unit} vs {other.unit}")
        return EnergyToken(self.amount + other.amount, self.unit)

    def __sub__(self, other: 'EnergyToken') -> 'EnergyToken':
        if self.unit != other.unit:
            raise ValueError(f"Cannot subtract different units: {self.unit} vs {other.unit}")
        return EnergyToken(self.amount - other.amount, self.unit)

    def __lt__(self, other: 'EnergyToken') -> bool:
        return self.amount < other.amount

@dataclass
class MeasurementCost:
    energy: EnergyToken
    operation_name: str
    timestamp: float = field(default_factory=time.time)

class EnergyDepletionError(Exception):
    """Raised when the agent runs out of energy."""
    pass

class MetabolicLedger:
    def __init__(self, max_energy: float = 100.0):
        self.max_energy = Decimal(str(max_energy))
        self.current_energy = self.max_energy
        self.transactions: List[MeasurementCost] = []

    def record_transaction(self, cost: MeasurementCost):
        """Deduct energy and record transaction."""
        if self.current_energy < cost.energy.amount:
            raise EnergyDepletionError(
                f"Insufficient energy for {cost.operation_name}. "
                f"Required: {cost.energy.amount}, Available: {self.current_energy}"
            )

        self.current_energy -= cost.energy.amount
        self.transactions.append(cost)

    def recharge(self, amount: Optional[float] = None):
        """Recharge energy (e.g., after sleep)."""
        if amount is None:
            self.current_energy = self.max_energy
        else:
            self.current_energy = min(self.max_energy, self.current_energy + Decimal(str(amount)))

    def get_status(self) -> Dict[str, str]:
        return {
            "current_energy": str(self.current_energy),
            "max_energy": str(self.max_energy),
            "percentage": f"{(self.current_energy / self.max_energy) * 100:.1f}%"
        }

"""
MeasurementCost - Energy Cost Metadata for Operations

Axiom Implementation: Measurement Cost + Recursive Measurement Axioms
"All measurement operations incur non-zero substrate consumption."
"Measurement operations must recursively account for their own substrate cost."

Component #3 from COMPASS plan: Define MeasurementCost struct containing EnergyToken and metadata fields
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .energy_token import EnergyToken
from .substrate_quantity import SubstrateQuantity


@dataclass(frozen=True)
class MeasurementCost:
    """
    Records the substrate cost of a measurement operation.

    Attributes:
        energy: Energy consumed by the operation
        cpu_cycles: Computational cycles used (optional detail)
        memory_bytes: Memory consumed (optional detail)
        time_nanos: Wall-clock time in nanoseconds (optional detail)
        timestamp: When the measurement occurred
        operation_name: Description of the operation

    Invariants:
        - energy must be non-negative
        - timestamp is immutable
        - all cost components are non-negative
    """

    energy: EnergyToken
    cpu_cycles: SubstrateQuantity | None = None
    memory_bytes: SubstrateQuantity | None = None
    time_nanos: SubstrateQuantity | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operation_name: str = "unnamed_measurement"

    def __post_init__(self) -> None:
        """Validate cost components are non-negative."""
        # Energy validation is handled by EnergyToken itself

        for component_name, component in [
            ("cpu_cycles", self.cpu_cycles),
            ("memory_bytes", self.memory_bytes),
            ("time_nanos", self.time_nanos),
        ]:
            if component is not None and component.magnitude < 0:
                raise ValueError(
                    f"{component_name} must be non-negative, got {component.magnitude}"
                )

    def total_energy(
        self, cpu_weight: float = 0.4, mem_weight: float = 0.3, time_weight: float = 0.3
    ) -> EnergyToken:
        """
        Calculate total energy from all cost components.

        Formula from Component #10: total_energy = α·cpu + β·memory + γ·time

        Args:
            cpu_weight: α coefficient for CPU contribution
            mem_weight: β coefficient for memory contribution
            time_weight: γ coefficient for time contribution

        Returns:
            Weighted sum as EnergyToken
        """
        from decimal import Decimal

        total = self.energy.amount

        if self.cpu_cycles is not None:
            total += Decimal(str(cpu_weight)) * self.cpu_cycles.magnitude

        if self.memory_bytes is not None:
            total += Decimal(str(mem_weight)) * self.memory_bytes.magnitude

        if self.time_nanos is not None:
            # Convert nanoseconds to normalized units (e.g., microseconds)
            total += Decimal(str(time_weight)) * (self.time_nanos.magnitude / Decimal("1000"))

        return EnergyToken(amount=total, unit=self.energy.unit)

    def __repr__(self) -> str:
        components = [f"energy={self.energy}"]
        if self.cpu_cycles:
            components.append(f"cpu={self.cpu_cycles.magnitude}")
        if self.memory_bytes:
            components.append(f"mem={self.memory_bytes.magnitude}")
        if self.time_nanos:
            components.append(f"time={self.time_nanos.magnitude}ns")

        return f"MeasurementCost({', '.join(components)}, op='{self.operation_name}')"

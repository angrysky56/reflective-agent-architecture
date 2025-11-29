"""
SubstrateQuantity - Measurement with Uncertainty

Axiom Implementation: Measurement Cost Axiom
"All measurement operations incur non-zero substrate consumption."

Component #2 from COMPASS plan: Implement SubstrateQuantity with magnitude and uncertainty properties
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Final


@dataclass(frozen=True)
class SubstrateQuantity:
    """
    Represents a measured quantity with inherent uncertainty.
    
    Attributes:
        magnitude: The measured value
        uncertainty: Measurement precision bound (±)
        
    Invariants:
        - magnitude can be any finite value
        - uncertainty must be non-negative
        - instances are immutable
    """
    
    magnitude: Decimal
    uncertainty: Decimal = Decimal("0.0")
    
    def __post_init__(self) -> None:
        """Validate uncertainty is non-negative."""
        if self.uncertainty < 0:
            raise ValueError(f"Uncertainty must be non-negative, got {self.uncertainty}")
    
    def __add__(self, other: "SubstrateQuantity") -> "SubstrateQuantity":
        """
        Add quantities with error propagation.
        σ(A+B) = √(σ_A² + σ_B²)
        """
        new_magnitude = self.magnitude + other.magnitude
        new_uncertainty = (self.uncertainty ** 2 + other.uncertainty ** 2).sqrt()
        return SubstrateQuantity(magnitude=new_magnitude, uncertainty=new_uncertainty)
    
    def __sub__(self, other: "SubstrateQuantity") -> "SubstrateQuantity":
        """
        Subtract quantities with error propagation.
        σ(A-B) = √(σ_A² + σ_B²)
        """
        new_magnitude = self.magnitude - other.magnitude
        new_uncertainty = (self.uncertainty ** 2 + other.uncertainty ** 2).sqrt()
        return SubstrateQuantity(magnitude=new_magnitude, uncertainty=new_uncertainty)
    
    def __mul__(self, scalar: Decimal | float | int) -> "SubstrateQuantity":
        """Scale magnitude and uncertainty by scalar."""
        s = Decimal(str(scalar))
        return SubstrateQuantity(
            magnitude=self.magnitude * s,
            uncertainty=self.uncertainty * abs(s)
        )
    
    def __lt__(self, other: "SubstrateQuantity") -> bool:
        """
        Compare magnitudes.
        Note: Does not account for overlapping uncertainty bounds.
        """
        return self.magnitude < other.magnitude
    
    def __le__(self, other: "SubstrateQuantity") -> bool:
        return self.magnitude <= other.magnitude
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubstrateQuantity):
            return False
        return (self.magnitude == other.magnitude and 
                self.uncertainty == other.uncertainty)
    
    def overlaps(self, other: "SubstrateQuantity") -> bool:
        """Check if uncertainty bounds overlap."""
        lower_self = self.magnitude - self.uncertainty
        upper_self = self.magnitude + self.uncertainty
        lower_other = other.magnitude - other.uncertainty
        upper_other = other.magnitude + other.uncertainty
        
        return not (upper_self < lower_other or upper_other < lower_self)
    
    def relative_uncertainty(self) -> Decimal:
        """Calculate relative uncertainty (uncertainty / magnitude)."""
        if self.magnitude == 0:
            return Decimal("inf")
        return abs(self.uncertainty / self.magnitude)
    
    def __repr__(self) -> str:
        return f"SubstrateQuantity(magnitude={self.magnitude}, uncertainty=±{self.uncertainty})"

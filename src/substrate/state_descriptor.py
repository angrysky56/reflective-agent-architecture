"""
StateDescriptor - Immutable Identity for Cognitive States

Axiom Implementation:
- Plasticity Gates: "Transitions between states require energy"
- Recursive Measurement: States track their own creation metadata

Component #1-3 from COMPASS plan: Implement StateDescriptor hierarchy
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict
from uuid import UUID, uuid4

from .energy_token import EnergyToken


class StateType(Enum):
    """Enumeration of possible state types."""

    UNKNOWN = "unknown"
    NAMED = "named"


@dataclass(frozen=True)
class StateTransitionCost:
    """Configuration for state transition energy costs."""

    unknown_to_named: EnergyToken = field(default_factory=lambda: EnergyToken(Decimal("15.0")))
    named_to_named: EnergyToken = field(default_factory=lambda: EnergyToken(Decimal("5.0")))
    named_to_unknown: EnergyToken = field(default_factory=lambda: EnergyToken(Decimal("1.0")))

    @classmethod
    def default(cls) -> "StateTransitionCost":
        return cls()


@dataclass(frozen=True)
class StateDescriptor:
    """
    Base class for all cognitive states.

    Immutable value object representing a snapshot of the agent's cognitive configuration.
    """

    state_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def is_named(self) -> bool:
        """Whether this state has a recognized identity."""
        return False

    @property
    def state_type(self) -> StateType:
        """The type of this state."""
        return StateType.NAMED if self.is_named else StateType.UNKNOWN

    def is_unknown(self) -> bool:
        """Helper to check if state is unknown."""
        return not self.is_named

    def promote_to_named(self, metadata: Dict | None = None) -> "NamedState":
        """
        Promote this state to a NamedState.
        Only valid for UnknownState.
        """
        raise NotImplementedError("Base StateDescriptor cannot be promoted directly.")


@dataclass(frozen=True)
class UnknownState(StateDescriptor):
    """
    Represents a transient, unrecognized, or unstable cognitive state.

    This is the default state during exploration or high-entropy phases.
    """

    entropy: float = 1.0

    @property
    def is_named(self) -> bool:
        return False

    def promote_to_named(self, metadata: Dict | None = None) -> "NamedState":
        """Promote UnknownState to NamedState."""
        new_meta = self.metadata.copy()
        if metadata:
            new_meta.update(metadata)

        name = new_meta.get("name", "Unnamed")
        return NamedState(
            state_id=self.state_id,
            created_at=datetime.now(timezone.utc),
            metadata=new_meta,
            name=name,
            stability=1.0 - (self.entropy * 0.5),
        )

    def __repr__(self) -> str:
        return f"UnknownState(id={str(self.state_id)[:8]}, entropy={self.entropy})"


@dataclass(frozen=True)
class NamedState(StateDescriptor):
    """
    Represents a crystallized, recognized cognitive state (e.g., "Focused", "Creative").

    Transitioning to a NamedState requires energy expenditure (Plasticity Gate).
    """

    name: str = "Unnamed"
    stability: float = 1.0

    @property
    def is_named(self) -> bool:
        return True

    @property
    def entropy(self) -> float:
        """
        Entropy is inverse of stability.
        High stability (1.0) -> Low entropy (0.0)
        """
        return max(0.0, 1.0 - self.stability)

    def __repr__(self) -> str:
        return f"NamedState(name='{self.name}', id={str(self.state_id)[:8]}, entropy={self.entropy:.2f})"

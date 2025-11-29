"""
StateDescriptor - Immutable Identity for Cognitive States

Axiom Implementation:
- Plasticity Gates: "Transitions between states require energy"
- Recursive Measurement: States track their own creation metadata

Component #1-3 from COMPASS plan: Implement StateDescriptor hierarchy
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Dict
from uuid import UUID, uuid4


@dataclass(frozen=True)
class StateDescriptor:
    """
    Base class for all cognitive states.

    Immutable value object representing a snapshot of the agent's cognitive configuration.
    """
    state_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def is_named(self) -> bool:
        """Whether this state has a recognized identity."""
        return False


@dataclass(frozen=True)
class UnknownState(StateDescriptor):
    """
    Represents a transient, unrecognized, or unstable cognitive state.

    This is the default state during exploration or high-entropy phases.
    """
    entropy: float = 1.0

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

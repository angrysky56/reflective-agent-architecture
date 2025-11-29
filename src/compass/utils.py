"""
Utility functions for COMPASS framework.

Provides mathematical functions, logging, validation, and helper utilities
used across different components of the integrated cognitive system.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ============================================================================
# Mathematical Functions
# ============================================================================


def sigmoid(x: float, k: float = 1.0, c: float = 0.0) -> float:
    """
    Sigmoid activation function.

    Args:
        x: Input value
        k: Steepness parameter
        c: Center point

    Returns:
        Sigmoid output in range (0, 1)
    """
    return 1.0 / (1.0 + math.exp(-k * (x - c)))


def confidence_from_delta_mu(delta_mu: float, variance: float, lambda_conf: float = 1.0) -> float:
    """
    Calculate decision confidence from value mode difference (oMCD model).

    Args:
        delta_mu: Difference in value modes
        variance: Combined variance
        lambda_conf: Confidence scaling factor

    Returns:
        Confidence level in [0, 1]
    """
    if variance <= 0:
        raise ValueError("Variance must be positive")

    numerator = lambda_conf * abs(delta_mu)
    denominator = math.sqrt(1.0 + 0.5 * (lambda_conf**2) * variance)

    return sigmoid(numerator / denominator)


def calculate_cost(z: float, alpha: float, nu: float) -> float:
    """
    Calculate cognitive resource cost (oMCD model).

    Args:
        z: Amount of cognitive resources invested
        alpha: Unitary effort cost
        nu: Cost power

    Returns:
        Cost of resource allocation
    """
    if z < 0:
        raise ValueError("Resource allocation must be non-negative")

    return alpha * (z**nu)


def calculate_benefit(z: float, r: float, confidence: float) -> float:
    """
    Calculate benefit of resource allocation (oMCD model).

    Args:
        z: Amount of cognitive resources invested
        r: Importance weight
        confidence: Decision confidence

    Returns:
        Benefit of resource allocation
    """
    return r * confidence


def update_precision(initial_precision: float, resources: float, beta: float) -> float:
    """
    Update precision of value representation (oMCD model).

    Args:
        initial_precision: Initial precision (1/σ₀)
        resources: Amount of resources invested
        beta: Type #1 effort efficacy

    Returns:
        Updated precision (1/σ)
    """
    return initial_precision + beta * resources


def advancement_score(truth: float, scrutiny: float, improvement: float, alpha: float = 0.4, beta: float = 0.6) -> float:
    """
    Calculate advancement score (SLAP model).

    Args:
        truth: Base truth value
        scrutiny: Scrutiny measure
        improvement: Improvement measure
        alpha: Weight for scrutiny
        beta: Weight for improvement

    Returns:
        Advancement score
    """
    return truth + (alpha * scrutiny) + (beta * improvement)


def entropy(probabilities: List[float]) -> float:
    """
    Calculate Shannon entropy.

    Args:
        probabilities: List of probability values

    Returns:
        Entropy value
    """
    if not np.isclose(sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1.0")

    return -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)


# ============================================================================
# Logging Utilities
# ============================================================================


class COMPASSLogger:
    """Centralized logger for COMPASS framework."""

    def __init__(self, name: str = "COMPASS", level: str = "INFO"):
        """
        Initialize logger.

        Args:
            name: Logger name
            level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Create console handler if not already exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs)

    def log_trajectory(self, trajectory: List[Dict], filename: Optional[str] = None):
        """
        Log a complete trajectory.

        Args:
            trajectory: List of action-observation pairs
            filename: Optional file to save trajectory
        """
        self.info(f"Trajectory with {len(trajectory)} steps")

        if filename:
            with open(filename, "w") as f:
                json.dump(trajectory, f, indent=2, default=str)
            self.info(f"Trajectory saved to {filename}")


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Trajectory:
    """Represents a sequence of actions and observations."""

    steps: List[tuple[Any, Any]]  # List of (action, observation) pairs
    score: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def add_step(self, action: Any, observation: Any):
        """Add a step to the trajectory."""
        self.steps.append((action, observation))

    def __len__(self):
        return len(self.steps)

    def to_dict(self) -> Dict:
        """Convert trajectory to dictionary."""
        return {"steps": self.steps, "score": self.score, "timestamp": self.timestamp.isoformat(), "length": len(self.steps)}


@dataclass
class SelfReflection:
    """Represents a self-reflection from the Self-Discover framework."""

    trajectory_id: int
    content: str
    insights: List[str]
    improvements: List[str]
    context_awareness: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        """Convert reflection to dictionary."""
        return {"trajectory_id": self.trajectory_id, "content": self.content, "insights": self.insights, "improvements": self.improvements, "context_awareness": self.context_awareness, "timestamp": self.timestamp.isoformat()}


@dataclass
class Goal:
    """Represents a reasoning goal for the Executive Controller."""

    id: str
    description: str
    priority: float  # 0.0 to 1.0
    created_at: datetime = None
    status: str = "active"  # active, completed, suspended, failed
    parent_id: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    progress: float = 0.0

    def to_dict(self) -> Dict:
        return {"id": self.id, "description": self.description, "priority": self.priority, "created_at": self.created_at.isoformat(), "status": self.status, "parent_id": self.parent_id, "subgoals": self.subgoals, "progress": self.progress}


@dataclass
class RepresentationState:
    """Tracks the current representation state."""

    current_type: str
    confidence: float
    history: List[str]
    reason_for_selection: str


@dataclass
class ObjectiveState:
    """Represents the state of a SMART objective."""

    name: str
    description: str
    metric: str
    target_value: float
    current_value: float
    deadline: datetime
    is_feasible: bool = True
    is_relevant: bool = True

    @property
    def progress(self) -> float:
        """Calculate progress as percentage."""
        if self.target_value == 0:
            return 0.0
        return min(100.0, (self.current_value / self.target_value) * 100.0)

    @property
    def is_on_track(self) -> bool:
        """Check if objective is on track."""
        # Simple heuristic: should be proportional to time elapsed
        # This could be enhanced with more sophisticated tracking
        return self.progress >= 50.0  # Simplified for now

    def to_dict(self) -> Dict:
        """Convert objective to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "metric": self.metric,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "deadline": self.deadline.isoformat(),
            "progress": self.progress,
            "is_feasible": self.is_feasible,
            "is_relevant": self.is_relevant,
            "is_on_track": self.is_on_track,
        }


# ============================================================================
# Validation Helpers
# ============================================================================


def validate_probability(p: float, name: str = "probability") -> None:
    """
    Validate that a value is a valid probability.

    Args:
        p: Value to validate
        name: Name of the parameter for error messages

    Raises:
        ValueError: If p is not in [0, 1]
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p}")


def validate_positive(value: float, name: str = "value") -> None:
    """
    Validate that a value is positive.

    Args:
        value: Value to validate
        name: Name of the parameter for error messages

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str = "value") -> None:
    """
    Validate that a value is non-negative.

    Args:
        value: Value to validate
        name: Name of the parameter for error messages

    Raises:
        ValueError: If value is negative
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


# ============================================================================
# File I/O Helpers
# ============================================================================


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Dict, filepath: str) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        filepath: Output file path
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict:
    """
    Load data from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, "r") as f:
        return json.load(f)


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Robustly extract JSON from text, handling markdown blocks and thinking tags.

    Args:
        text: Raw text from LLM

    Returns:
        Parsed dictionary or None if extraction failed
    """
    try:
        # 1. Try finding markdown blocks
        if "```json" in text:
            content = text.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        elif "```" in text:
            content = text.split("```")[1].split("```")[0].strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass # Continue to other methods

        # 2. Try finding the first '{' and last '}'
        start_idx = text.find("{")
        end_idx = text.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content = text[start_idx : end_idx + 1]
            return json.loads(content)

        # 3. Try parsing the whole text
        return json.loads(text)

    except (json.JSONDecodeError, ValueError):
        return None


# ============================================================================
# Performance Utilities
# ============================================================================


class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = "Operation", logger: Optional[COMPASSLogger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        message = f"{self.name} completed in {duration:.3f}s"

        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0

        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

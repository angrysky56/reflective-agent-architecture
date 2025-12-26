"""
Putnamian Evolution Framework - Phase 1 Implementation Starter
================================================================

This module implements the Stress Sensor and Volatile Buffer for detecting
"semantic desire lines" - high-utility, high-cost reasoning patterns that
are candidates for structural crystallization during sleep cycles.

Based on: /docs/putnamian_evolution_framework.md
Author: RAA Evolution Project
Date: December 10, 2025
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class ReasoningTrace:
    """Represents a complete reasoning chain from input to output"""

    input_pattern: str
    steps: List[str]  # Sequence of operations/tools used
    result: Any
    timestamp: datetime
    token_count: int = 0
    step_count: int = 0

    def __post_init__(self) -> None:
        self.step_count = len(self.steps)


@dataclass
class StressVector:
    """
    Encapsulates the "topological stress" of a reasoning pattern.

    Stress = Utility × MetabolicCost

    High stress indicates a frequently-used, expensive operation
    that is a candidate for crystallization (paving the desire line).
    """

    pattern: str
    trace: ReasoningTrace
    utility_score: float  # 0-1: How valuable was this output?
    energy_cost: float  # Joules consumed
    stress: float  # utility × cost
    timestamp: datetime

    @classmethod
    def from_trace(cls, trace: "ReasoningTrace", utility: float, energy: float) -> "StressVector":
        """Factory method to create StressVector from trace"""
        stress = utility * energy
        return cls(
            pattern=trace.input_pattern,
            trace=trace,
            utility_score=utility,
            energy_cost=energy,
            stress=stress,
            timestamp=trace.timestamp,
        )


class VolatileBuffer:
    """
    Temporary storage for high-stress traces during wakefulness.

    This buffer accumulates "desire lines" that will be processed
    during the next sleep cycle for potential crystallization.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.pattern_counts: Dict[str, int] = {}

    def add_desire_line(self, stress_vector: StressVector) -> None:
        """Add a high-stress trace to the buffer"""
        self.buffer.append(stress_vector)

        # Track pattern frequency
        pattern_key = stress_vector.pattern
        self.pattern_counts[pattern_key] = self.pattern_counts.get(pattern_key, 0) + 1

    def get_high_frequency_patterns(self, min_frequency: int = 3) -> List[str]:
        """Identify patterns that appear frequently (potential desire lines)"""
        return [pattern for pattern, count in self.pattern_counts.items() if count >= min_frequency]

    def get_vectors_for_pattern(self, pattern: str) -> List[StressVector]:
        """Retrieve all stress vectors matching a specific pattern"""
        return [v for v in self.buffer if v.pattern == pattern]

    def get_top_stress_patterns(self, n: int = 10) -> List[tuple]:
        """Get the N patterns with highest average stress"""
        pattern_stress = {}
        pattern_counts = {}

        for vector in self.buffer:
            pattern = vector.pattern
            if pattern not in pattern_stress:
                pattern_stress[pattern] = 0
                pattern_counts[pattern] = 0
            pattern_stress[pattern] += vector.stress
            pattern_counts[pattern] += 1

        # Calculate average stress per pattern
        avg_stress = {p: pattern_stress[p] / pattern_counts[p] for p in pattern_stress}

        # Sort by average stress
        sorted_patterns = sorted(avg_stress.items(), key=lambda x: x[1], reverse=True)

        return sorted_patterns[:n]

    def clear(self) -> None:
        """Clear buffer after sleep cycle processing"""
        self.buffer.clear()
        self.pattern_counts.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if not self.buffer:
            return {"size": 0, "unique_patterns": 0, "avg_stress": 0.0, "max_stress": 0.0}

        stresses = [v.stress for v in self.buffer]
        return {
            "size": len(self.buffer),
            "unique_patterns": len(self.pattern_counts),
            "avg_stress": np.mean(stresses),
            "max_stress": np.max(stresses),
            "total_energy_cost": sum(v.energy_cost for v in self.buffer),
        }


class StressSensor:
    """
    The "Nerve Ending" - monitors reasoning efficiency in real-time.

    Embedded in the Director agent, this sensor tracks the computational
    cost of reasoning chains and identifies high-stress patterns that
    are candidates for structural optimization.
    """

    def __init__(
        self, stress_threshold: float = 5.0, volatile_buffer: Optional[VolatileBuffer] = None
    ) -> None:
        self.stress_threshold = stress_threshold
        self.volatile_buffer = volatile_buffer or VolatileBuffer()
        self.observation_count = 0

    def observe_inference(
        self,
        trace: ReasoningTrace,
        utility_evaluator: Callable[[Any], float],
        energy_calculator: Callable[[ReasoningTrace], float],
    ) -> Optional[StressVector]:
        """
        Monitor a reasoning chain and potentially flag it as a desire line.

        Args:
            trace: The complete reasoning trace
            utility_evaluator: Function that scores output quality (0-1)
            energy_calculator: Function that computes metabolic cost

        Returns:
            StressVector if stress exceeds threshold, None otherwise
        """
        self.observation_count += 1

        # Evaluate the trace
        utility = utility_evaluator(trace.result)
        energy_cost = energy_calculator(trace)

        # Create stress vector
        stress_vector = StressVector.from_trace(trace, utility, energy_cost)

        # Check if this is a high-stress pattern (desire line candidate)
        if stress_vector.stress > self.stress_threshold:
            self.volatile_buffer.add_desire_line(stress_vector)
            return stress_vector

        return None

    def get_desire_line_candidates(
        self, min_frequency: int = 3, min_avg_stress: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Identify the most promising patterns for crystallization.

        A good candidate has:
        - High frequency (used often)
        - High average stress (valuable but expensive)

        Returns:
            List of candidate patterns with metadata
        """
        candidates = []

        # Get high-frequency patterns
        frequent_patterns = self.volatile_buffer.get_high_frequency_patterns(min_frequency)

        # Calculate statistics for each
        for pattern in frequent_patterns:
            vectors = self.volatile_buffer.get_vectors_for_pattern(pattern)

            avg_stress = np.mean([v.stress for v in vectors])
            avg_utility = np.mean([v.utility_score for v in vectors])
            avg_cost = np.mean([v.energy_cost for v in vectors])

            if avg_stress >= min_avg_stress:
                candidates.append(
                    {
                        "pattern": pattern,
                        "frequency": len(vectors),
                        "avg_stress": avg_stress,
                        "avg_utility": avg_utility,
                        "avg_cost": avg_cost,
                        "total_cost": sum(v.energy_cost for v in vectors),
                        "sample_traces": [v.trace for v in vectors[:3]],
                    }
                )

        # Sort by stress (highest first)
        candidates.sort(key=lambda x: x["avg_stress"], reverse=True)

        return candidates

    def get_statistics(self) -> Dict[str, Any]:
        """Get sensor statistics"""
        buffer_stats = self.volatile_buffer.get_stats()

        return {
            "observations": self.observation_count,
            "buffer": buffer_stats,
            "stress_threshold": self.stress_threshold,
        }


if __name__ == "__main__":
    print("Putnamian Evolution - Phase 1 Demo")
    print("=" * 60)

    # Create stress sensor
    sensor = StressSensor(stress_threshold=3.0)

    # Simulate reasoning traces
    print("\nSimulating reasoning chains...")

    # Pattern 1: Frequent, expensive (DESIRE LINE)
    for i in range(5):
        trace = ReasoningTrace(
            input_pattern="Analyze complex JSON data",
            steps=["parse_json", "extract_fields", "transform", "validate", "aggregate"],
            result={"status": "success", "data": [1, 2, 3]},
            timestamp=datetime.now(),
            token_count=150,
        )
        sensor.observe_inference(
            trace, utility_evaluator=lambda r: 0.9, energy_calculator=lambda t: 8.0
        )

    # Get statistics
    print("\n" + "=" * 60)
    print("SENSOR STATISTICS:")
    stats = sensor.get_statistics()
    print(f"  Total observations: {stats['observations']}")
    print(f"  Desire lines flagged: {stats['buffer']['size']}")
    print(f"  Unique patterns: {stats['buffer']['unique_patterns']}")
    print(f"  Avg stress: {stats['buffer']['avg_stress']:.2f}")
    print(f"  Total energy cost: {stats['buffer']['total_energy_cost']:.2f} J")

    # Identify crystallization candidates
    print("\n" + "=" * 60)
    print("DESIRE LINE CANDIDATES:")
    candidates = sensor.get_desire_line_candidates(min_frequency=3, min_avg_stress=5.0)

    for i, candidate in enumerate(candidates, 1):
        print(f"\n{i}. {candidate['pattern']}")
        print(f"   Frequency: {candidate['frequency']} times")
        print(f"   Avg Stress: {candidate['avg_stress']:.2f}")
        print(f"   Total Cost: {candidate['total_cost']:.2f} J")

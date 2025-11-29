"""
COMPASS Integration Layer for RAA
Provides lightweight interface between RAA cognitive primitives and COMPASS orchestration.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .config import COMPASSConfig
from .omcd_controller import oMCDController


@dataclass
class CognitiveState:
    """Unified representation of RAA cognitive state for COMPASS."""
    energy: float                    # Hopfield energy level
    entropy: float                   # Confusion metric
    confidence: float                # Belief certainty
    stability: str                   # "Stable" | "Unstable"
    state_type: str                  # "Focused" | "Broad" | "Looping"
    available_resources: float = 100.0


@dataclass
class ResourceAllocation:
    """oMCD resource allocation decision."""
    optimal_resources: float
    expected_benefit: float
    expected_cost: float
    confidence_at_optimal: float
    recommendation: str


class COMPASSOrchestrator:
    """
    Lightweight COMPASS orchestration for RAA.

    Phase 1 Integration: oMCD resource allocation only.
    Future phases will add Self-Discover reflection, SLAP scoring, etc.
    """

    def __init__(self, config: Optional[COMPASSConfig] = None):
        """Initialize COMPASS with configuration."""
        self.config = config or COMPASSConfig()

        # Phase 1: Initialize oMCD only
        self.omcd = oMCDController(self.config.omcd)

        print("[COMPASS] Orchestrator initialized (Phase 1: oMCD resource allocation)")

    def allocate_resources(
        self,
        cognitive_state: CognitiveState,
        task_complexity: float,
        importance: float = 10.0
    ) -> ResourceAllocation:
        """
        Determine optimal cognitive resource allocation using oMCD model.

        Args:
            cognitive_state: Current RAA cognitive state
            task_complexity: Estimated difficulty (0.0-1.0)
            importance: Decision significance (R parameter)

        Returns:
            Resource allocation recommendation
        """
        # Map RAA state to oMCD state representation
        # High complexity = Low value difference (hard to distinguish options)
        omcd_state = {
            'value_difference': max(0.01, 1.0 - task_complexity),
            'variance': 1.0 - cognitive_state.confidence,
            'precision': 1.0 / max(0.01, 1.0 - cognitive_state.confidence)
        }

        # Compute optimal allocation
        allocation_result = self.omcd.determine_resource_allocation(
            current_state=omcd_state,
            importance=importance,
            available_resources=cognitive_state.available_resources
        )

        # Generate recommendation
        resources = allocation_result['optimal_resources']
        if resources < 30:
            recommendation = "LOW_EFFORT: Simple heuristic sufficient"
        elif resources < 70:
            recommendation = "MODERATE_EFFORT: Standard reasoning required"
        else:
            recommendation = "HIGH_EFFORT: Deep analysis or System 3 escalation recommended"

        return ResourceAllocation(
            optimal_resources=allocation_result['optimal_resources'],
            expected_benefit=allocation_result['expected_benefit'],
            expected_cost=allocation_result['expected_cost'],
            confidence_at_optimal=allocation_result.get('confidence_at_optimal', 0.5),
            recommendation=recommendation
        )

    def should_escalate_to_system3(
        self,
        allocation: ResourceAllocation,
        external_model_cost: float = 100.0
    ) -> bool:
        """
        Determine if task should escalate to external model (System 3).

        Args:
            allocation: Current resource allocation
            external_model_cost: Cost of calling external model

        Returns:
            True if escalation recommended
        """
        # Escalate if high effort required and benefit-cost ratio poor
        if allocation.optimal_resources > 80:
            benefit_cost_ratio = allocation.expected_benefit / allocation.expected_cost
            if benefit_cost_ratio < 1.5:
                return True

        return False

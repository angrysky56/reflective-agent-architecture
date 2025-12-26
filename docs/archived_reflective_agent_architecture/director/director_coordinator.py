"""
Director Coordinator: Integration of RAA's Existing Monitoring Components
===========================================================================

Coordinates RAA's existing internal monitoring systems to provide complete
systems understanding WITHOUT losing entropy monitoring.

Integrates:
- entropy_monitor.py (Confusion detection)
- matrix_monitor.py (Cognitive proprioception - thought state classification)
- intervention_tracker.py (Action logging)
- meta_pattern_analyzer.py (Recurring pattern recognition)
- adaptive_criterion.py (Dynamic thresholds)
- Plus: Hopfield manifold, Metabolic ledger, Precuneus fusion

Key Principle: AUGMENT entropy monitoring with complementary internal states,
don't replace it. Entropy reveals confusion; other signals reveal WHY and WHAT TO DO.

Author: RAA Enhancement (Reflective Agent Architecture)
Date: December 10, 2025
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class IntegratedPhase(Enum):
    """
    Holistic cognitive phases combining multiple monitoring signals.
    Based on actual RAA architecture components.
    """

    EXPLORING_HEALTHY = "exploring_healthy"  # High entropy + good energy + novel patterns
    FOCUSED_PRODUCTIVE = "focused_productive"  # Low entropy + stable attractor + balanced
    CONFUSED_GENUINE = "confused_genuine"  # High entropy + high curvature (complexity)
    CONFUSED_STUCK = "confused_stuck"  # High entropy + looping + energy waste
    FATIGUE_COGNITIVE = "fatigue_cognitive"  # Rising entropy + depleting resources
    CRYSTALLIZING = "crystallizing"  # Decreasing entropy + pattern stabilization
    LOOPING_BEHAVIORAL = "looping_behavioral"  # Repetitive ops despite resources
    FRAGMENTED = "fragmented"  # Imbalanced tripartite fusion


@dataclass
class DirectorState:
    """
    Complete systems state combining all RAA monitoring components.

    This is what the Director ACTUALLY knows about itself, not theory.
    """

    # From entropy_monitor.py
    entropy: float
    entropy_trend: float
    entropy_sample_size: int

    # From matrix_monitor.py (Cognitive Proprioception)
    matrix_state: str  # "Focused", "Broad", "Looping", "Unknown"
    hopfield_energy: float

    # From metabolic ledger (substrate/)
    metabolic_pct: float
    energy_raw: Dict[str, str]

    # From intervention_tracker.py / operation patterns
    is_looping: bool
    dominant_operation: str
    operation_counts: Dict[str, int]

    # From evolution buffer (stress_sensor.py - Putnamian framework)
    evolution_observations: int
    stress_buffer_size: int
    avg_stress: float
    max_stress: float
    stress_threshold: float

    # From Precuneus fusion (integration/)
    fusion_weights: Dict[str, float]  # context, perspective, operation
    fusion_balance: float
    dominant_stream: str

    # Current active goal
    current_goal: Optional[str]

    # From director_interoception.py (Vector-Based Interoception)
    adjunction_tension: float = 0.0  # ||Vector(Goal) - Vector(Result)||
    adjunction_stress: float = 0.0  # utility × tension


@dataclass
class DirectorAssessment:
    """
    Integrated assessment combining all monitoring signals.
    Provides ACTIONABLE guidance, not just classification.
    """

    phase: IntegratedPhase
    confidence: float

    # Core question: Is this healthy or needs intervention?
    is_healthy: bool
    requires_intervention: bool

    # Diagnostic reasoning
    entropy_interpretation: str  # What does current entropy MEAN in context?
    matrix_interpretation: str  # What does proprioceptive state reveal?
    metabolic_interpretation: str  # Resource status and implications
    pattern_interpretation: str  # What do operation patterns indicate?

    # Signal coherence
    signals_agree: List[str]  # Which signals converge on same diagnosis?
    signals_conflict: List[str]  # Which signals contradict? (This is information!)

    # Actionable guidance
    warnings: List[str]
    immediate_actions: List[str]  # What to do RIGHT NOW
    strategic_recommendations: List[str]  # Longer-term adjustments

    # Affective State (New Dec 10)
    valence: float = 0.0


class DirectorCoordinator:
    """
    Coordinates RAA's existing monitoring components for complete understanding.

    Design Philosophy:
    1. Entropy monitoring is PRIMARY - keeps existing value
    2. Other signals provide CONTEXT - explain what entropy means
    3. Integration reveals patterns invisible to individual monitors
    4. Output is ACTIONABLE - specific recommendations, not just labels

    This is NOT a replacement for existing components.
    This is a coordination layer that makes them work together.
    """

    def __init__(self) -> None:
        self.history: List[DirectorAssessment] = []

    def assess_complete_state(self, state: DirectorState) -> DirectorAssessment:
        """
        Main integration point - takes current state from all monitors,
        produces holistic assessment with actionable guidance.

        This is where RAA's "complete systems understanding" happens.
        """

        # Step 0: Calculate Signal Coherence early (needed for valence)
        agreements, conflicts = self._analyze_signal_coherence(state)

        # Step 1: Interpret entropy IN CONTEXT of other signals
        entropy_meaning = self._interpret_entropy_contextually(
            state.entropy,
            state.entropy_trend,
            state.matrix_state,
            state.hopfield_energy,
            state.is_looping,
            state.operation_counts,
        )

        # Step 1.5: Calculate Affective Valence (New Dec 10)
        valence = self._calculate_valence(state, conflicts)

        # Step 2: Interpret matrix monitor (cognitive proprioception)
        matrix_meaning = self._interpret_matrix_state(
            state.matrix_state, state.hopfield_energy, state.entropy, state.adjunction_tension
        )

        # Step 3: Interpret metabolic state (energy/fatigue)
        metabolic_meaning = self._interpret_metabolic_state(
            state.metabolic_pct, state.operation_counts, state.is_looping
        )

        # Step 4: Interpret operation patterns
        pattern_meaning = self._interpret_operation_patterns(
            state.dominant_operation, state.operation_counts, state.is_looping, state.entropy
        )

        # Step 5: Detect signal agreements and conflicts (Already done in Step 0)
        # agreements, conflicts = self._analyze_signal_coherence(state)

        # Step 6: Classify integrated phase
        phase, confidence = self._classify_integrated_phase(state, entropy_meaning, agreements)

        # Step 7: Determine health and intervention needs
        is_healthy, requires_intervention = self._assess_health(phase, state, conflicts)

        # Step 8: Generate actionable guidance
        warnings, immediate, strategic = self._generate_actionable_guidance(
            phase, state, conflicts, entropy_meaning
        )

        assessment = DirectorAssessment(
            phase=phase,
            confidence=confidence,
            is_healthy=is_healthy,
            requires_intervention=requires_intervention,
            entropy_interpretation=entropy_meaning,
            matrix_interpretation=matrix_meaning,
            metabolic_interpretation=metabolic_meaning,
            pattern_interpretation=pattern_meaning,
            signals_agree=agreements,
            signals_conflict=conflicts,
            warnings=warnings,
            immediate_actions=immediate,
            strategic_recommendations=strategic,
            valence=valence,
        )

        self.history.append(assessment)
        return assessment

    def _interpret_entropy_contextually(
        self,
        entropy: float,
        trend: float,
        matrix_state: str,
        hopfield_energy: float,
        is_looping: bool,
        op_counts: Dict[str, int],
    ) -> str:
        """
        THIS IS KEY: Entropy alone doesn't tell you enough.
        Context reveals what high/low entropy actually MEANS.
        """

        # Low entropy cases
        if entropy < 0.3:
            if is_looping:
                return "Low entropy from BEHAVIORAL LOCK - stuck in repetitive pattern, not focused thinking"
            elif matrix_state == "Focused" and hopfield_energy < -0.4:
                return "Low entropy from DEEP ATTRACTOR - system has converged on stable pattern (healthy focus)"
            else:
                return "Low entropy from LOCAL MINIMUM - may be surface-level pattern without deep understanding"

        # High entropy cases
        elif entropy > 0.7:
            if matrix_state == "Focused" and hopfield_energy < -0.4:
                return "High entropy despite stable attractor - GENUINE COMPLEXITY detected (high-curvature domain)"
            elif "deconstruct" in op_counts and op_counts["deconstruct"] > 3:
                return "High entropy from ACTIVE EXPLORATION - system is deliberately fragmenting problem (healthy)"
            elif is_looping:
                return "High entropy from CONFUSION - system is stuck searching without convergence"
            else:
                return (
                    "High entropy from BROAD SEARCH - system is exploring widely (may be unfocused)"
                )

        # Medium entropy
        else:
            if trend > 0.1:
                return "Rising entropy - system complexity increasing OR focus degrading (monitor closely)"
            elif trend < -0.1:
                return "Falling entropy - system converging on solution OR entering local minimum"
            else:
                return "Stable moderate entropy - balanced exploration/exploitation"

    def _interpret_matrix_state(
        self,
        matrix_state: str,
        hopfield_energy: float,
        entropy: float,
        adjunction_tension: float = 0.0,
    ) -> str:
        """
        Interpret what the Matrix Monitor (cognitive proprioception) reveals.
        This is RAA's internal "feeling" of its own thought state.

        Now augmented with adjunction_tension for hallucination detection.
        """

        # Hallucination Check: Low entropy + High tension = confident but wrong
        if entropy < 0.3 and adjunction_tension > 0.5:
            return "HALLUCINATION RISK: Low entropy (confident) but high adjunction tension (misaligned with goal)"

        if matrix_state == "Focused":
            if adjunction_tension > 0.4:
                return "Focused but drifting - goal/result misalignment detected"
            elif hopfield_energy < -0.6:
                return "Deep focused state - strong attractor basin (may be over-fitted)"
            elif hopfield_energy > -0.3:
                return "Shallow focused state - weak attractor (fragile convergence)"
            else:
                return "Healthy focused state - moderate attractor depth"

        elif matrix_state == "Broad":
            return "Broad exploration state - high attention diversity (good for discovery, bad for execution)"

        elif matrix_state == "Looping":
            return "Looping state detected - cognitive repetition without progress"

        else:  # "Unknown"
            return "Unknown cognitive state - pattern doesn't match trained archetypes (may indicate novel situation)"

    def _interpret_metabolic_state(
        self, metabolic_pct: float, op_counts: Dict[str, int], is_looping: bool
    ) -> str:
        """
        Interpret energy levels in context of activity.
        Same energy level means different things depending on what system is doing.
        """

        if metabolic_pct < 20:
            return "CRITICAL energy depletion - cognitive performance severely degraded"

        elif metabolic_pct < 40:
            if is_looping:
                return "Low energy + looping = FATIGUE-DRIVEN REPETITION (needs sleep cycle)"
            else:
                return "Low energy but productive - final push before depletion"

        elif metabolic_pct > 80:
            total_ops = sum(op_counts.values())
            if total_ops < 5:
                return "High energy + low activity = FRESH START (good time for complex work)"
            else:
                return "High energy despite activity = EFFICIENT OPERATION (well-optimized)"

        else:  # 40-80%
            return f"Moderate energy ({metabolic_pct}%) - sustainable operation range"

    def _interpret_operation_patterns(
        self, dominant_op: str, op_counts: Dict[str, int], is_looping: bool, entropy: float
    ) -> str:
        """
        What do tool usage patterns reveal about cognitive strategy?
        """

        total_ops = sum(op_counts.values())
        if total_ops == 0:
            return "No operations yet - system initialization"

        # Calculate operation diversity
        max_count = max(op_counts.values())
        diversity = 1 - (max_count / total_ops)

        if diversity < 0.3:
            if is_looping:
                return f"Very low diversity + looping - STUCK on {dominant_op} operation"
            else:
                return f"Very low diversity - FOCUSED STRATEGY using primarily {dominant_op}"

        elif diversity > 0.7:
            if entropy > 0.6:
                return "High diversity + high entropy - SCATTERED EXPLORATION (may need focus)"
            else:
                return "High diversity + low entropy - SYSTEMATIC MULTI-STRATEGY APPROACH"

        else:
            return f"Moderate diversity - BALANCED APPROACH with emphasis on {dominant_op}"

    def _analyze_signal_coherence(self, state: DirectorState) -> tuple[List[str], List[str]]:
        """
        Detect where monitoring signals agree vs conflict.
        Conflicts are INFORMATION - they reveal nuanced states.
        """

        agreements = []
        conflicts = []

        # Agreement: Low entropy + Focused matrix + Stable energy
        if state.entropy < 0.3 and state.matrix_state == "Focused" and state.hopfield_energy < -0.4:
            agreements.append("Entropy + Matrix + Hopfield all indicate FOCUSED state")

        # Conflict: Low entropy + Looping behavior
        if state.entropy < 0.3 and state.is_looping:
            conflicts.append("Low entropy BUT looping - not focused thinking, behavioral lock")

        # Conflict: High entropy + Focused matrix
        if state.entropy > 0.7 and state.matrix_state == "Focused":
            conflicts.append(
                "High entropy BUT focused attractor - genuine complexity not confusion"
            )

        # Agreement: High entropy + High energy + Broad matrix
        if state.entropy > 0.6 and state.metabolic_pct > 70 and state.matrix_state == "Broad":
            agreements.append("Entropy + Energy + Matrix all indicate EXPLORING state")

        # Conflict: Low energy + High activity
        if state.metabolic_pct < 30 and len(state.operation_counts) > 10:
            conflicts.append("Low energy BUT high activity - may be thrashing")

        # Agreement: Looping detection + Low diversity
        op_diversity = self._calculate_diversity(state.operation_counts)
        if state.is_looping and op_diversity < 0.3:
            agreements.append("Looping + Low diversity confirms REPETITIVE PATTERN")

        # Conflict: High diversity + Looping
        if not state.is_looping and op_diversity > 0.7 and state.entropy < 0.3:
            conflicts.append(
                "High operation diversity BUT low entropy - inefficient redundant search"
            )

        return agreements, conflicts

    def _calculate_diversity(self, op_counts: Dict[str, int]) -> float:
        """Shannon diversity of operations"""
        if not op_counts:
            return 0.5
        total = sum(op_counts.values())
        max_count = max(op_counts.values())
        return 1 - (max_count / total)

    def _classify_integrated_phase(
        self, state: DirectorState, entropy_meaning: str, agreements: List[str]
    ) -> tuple[IntegratedPhase, float]:
        """
        Classify phase using ALL signals, not just entropy.
        Confidence reflects agreement across signals.
        """

        # EXPLORING_HEALTHY
        if (
            state.entropy > 0.6
            and state.metabolic_pct > 60
            and "EXPLORATION" in entropy_meaning.upper()
            and not state.is_looping
        ):
            confidence = 0.85 if len(agreements) > 0 else 0.65
            return IntegratedPhase.EXPLORING_HEALTHY, confidence

        # FOCUSED_PRODUCTIVE
        if (
            state.entropy < 0.3
            and state.matrix_state == "Focused"
            and state.hopfield_energy < -0.4
            and not state.is_looping
        ):
            # Check if first agreement mentions FOCUSED
            has_focused_agreement = agreements and "FOCUSED" in agreements[0]
            confidence = 0.9 if has_focused_agreement else 0.7
            return IntegratedPhase.FOCUSED_PRODUCTIVE, confidence

        # CONFUSED_GENUINE (high curvature complexity)
        if (
            state.entropy > 0.7
            and state.matrix_state == "Focused"
            and "COMPLEXITY" in entropy_meaning.upper()
        ):
            confidence = 0.8
            return IntegratedPhase.CONFUSED_GENUINE, confidence

        # CONFUSED_STUCK
        if (
            state.entropy > 0.6
            and (state.is_looping or "CONFUSION" in entropy_meaning.upper())
            and state.metabolic_pct < 50
        ):
            confidence = 0.75
            return IntegratedPhase.CONFUSED_STUCK, confidence

        # LOOPING_BEHAVIORAL
        if state.is_looping:
            confidence = (
                0.9 if any("LOOPING" in a or "REPETITIVE" in a for a in agreements) else 0.7
            )
            return IntegratedPhase.LOOPING_BEHAVIORAL, confidence

        # FATIGUE_COGNITIVE
        if state.metabolic_pct < 30 or (state.entropy_trend > 0.1 and state.metabolic_pct < 50):
            confidence = 0.8
            return IntegratedPhase.FATIGUE_COGNITIVE, confidence

        # CRYSTALLIZING
        if state.entropy_trend < -0.05 and state.entropy < 0.4 and state.hopfield_energy < -0.4:
            confidence = 0.75
            return IntegratedPhase.CRYSTALLIZING, confidence

        # FRAGMENTED
        if state.fusion_balance < 0.8:
            confidence = 0.7
            return IntegratedPhase.FRAGMENTED, confidence

        # Default: EXPLORING with low confidence
        return IntegratedPhase.EXPLORING_HEALTHY, 0.4

    def _assess_health(
        self, phase: IntegratedPhase, state: DirectorState, conflicts: List[str]
    ) -> tuple[bool, bool]:
        """
        Is system healthy? Does it need intervention?

        Returns: (is_healthy, requires_intervention)
        """

        # Unhealthy phases
        unhealthy_phases = {
            IntegratedPhase.CONFUSED_STUCK,
            IntegratedPhase.FATIGUE_COGNITIVE,
            IntegratedPhase.LOOPING_BEHAVIORAL,
        }

        if phase in unhealthy_phases:
            return False, True

        # Critical energy
        if state.metabolic_pct < 15:
            return False, True

        # Too many signal conflicts
        if len(conflicts) > 2:
            return False, True

        # Otherwise healthy
        needs_intervention = (
            state.metabolic_pct < 30  # Approaching depletion
            or len(conflicts) > 0  # Some signal disagreement
        )

        return True, needs_intervention

    def _generate_actionable_guidance(
        self,
        phase: IntegratedPhase,
        state: DirectorState,
        conflicts: List[str],
        entropy_meaning: str,
    ) -> tuple[List[str], List[str], List[str]]:
        """
        Generate SPECIFIC, ACTIONABLE recommendations.
        Not vague advice - concrete tool calls and strategies.
        """

        warnings = []
        immediate = []
        strategic = []

        # Phase-specific guidance
        if phase == IntegratedPhase.FOCUSED_PRODUCTIVE:
            if state.hopfield_energy < -0.7:
                warnings.append("Very deep attractor - may be over-fitted")
                strategic.append("Test generalization: Set new related goal to check flexibility")
            immediate.append("Optimal state for synthesis - call synthesize() now")

        elif phase == IntegratedPhase.CONFUSED_GENUINE:
            warnings.append("High-curvature domain detected - inherent complexity")
            immediate.append("Use deconstruct() to fragment into simpler sub-problems")
            strategic.append("Consider consult_compass() for multi-step reasoning")

        elif phase == IntegratedPhase.CONFUSED_STUCK:
            warnings.append("STUCK: High entropy + Looping + Low energy")
            immediate.append("BREAK PATTERN: set_intentionality(mode='adaptation') OR set new goal")
            immediate.append("Consider run_sleep_cycle() to reset and consolidate")

        elif phase == IntegratedPhase.LOOPING_BEHAVIORAL:
            warnings.append(f"Repetitive {state.dominant_operation} without progress")
            immediate.append(f"Stop calling {state.dominant_operation} - try different approach")
            immediate.append("Use hypothesize() or explore_for_utility() to find new paths")

        elif phase == IntegratedPhase.FATIGUE_COGNITIVE:
            warnings.append(f"Energy at {state.metabolic_pct}% - performance degrading")
            immediate.append("REQUIRED: run_sleep_cycle(epochs=1) to recharge")
            strategic.append("After sleep, resume with simplified goal")

        elif phase == IntegratedPhase.EXPLORING_HEALTHY:
            if state.metabolic_pct < 40:
                warnings.append("Exploring while low energy - may deplete soon")
                strategic.append("Consider crystallizing findings before energy depletes")

        elif phase == IntegratedPhase.CRYSTALLIZING:
            immediate.append("Healthy convergence - continue current approach")
            if state.metabolic_pct > 70:
                immediate.append("Good energy + convergence = Optimal for final synthesis")

        elif phase == IntegratedPhase.FRAGMENTED:
            warnings.append(f"Imbalanced tripartite fusion: {state.fusion_balance:.2f}")
            immediate.append("Use deconstruct() to re-balance context/perspective/operation")

        # Conflict-based guidance
        for conflict in conflicts:
            if "behavioral lock" in conflict.lower():
                warnings.append(conflict)
                immediate.append("Behavioral lock despite resources - change strategy immediately")
            elif "complexity" in conflict.lower():
                warnings.append(conflict)
                strategic.append("High-curvature domain - multiple approaches may be needed")

        return warnings, immediate, strategic

    def _calculate_valence(self, state: DirectorState, conflicts: List[str]) -> float:
        """
        Calculate scalar emotional valence (-1.0 to 1.0) from multi-signal state.
        See: design_utility_aware_search.md

        High Positive = Flow, resonance, healthy engagement (Deepens Energy Wells)
        High Negative = Distress, dissonance, confusion (Flattens Energy Wells)
        """
        valence = 0.0

        # 1. Entropy (Confusion vs Flow)
        if state.entropy > 0.7:
            valence -= 0.3  # Analysis paralysis / Overwhelmed
        elif state.entropy < 0.3 and not state.is_looping:
            valence += 0.2  # Flow state
        elif state.entropy < 0.3 and state.is_looping:
            valence -= 0.1  # Stuckness (frustrating)

        # 2. Metabolic (Energy / Vitality)
        if state.metabolic_pct < 20:
            valence -= 0.4  # Depletion / Pain
        elif state.metabolic_pct < 40:
            valence -= 0.2  # Fatigue
        elif state.metabolic_pct > 80:
            valence += 0.2  # Vitality / Eagerness

        # 3. Adjunction Tension (Truth / Hallucination)
        # If tension is high, we are confident but wrong -> Dissonance
        if state.adjunction_tension > 0.8:
            valence -= 0.5  # Cognitive Dissonance
        elif state.adjunction_tension < 0.2:
            valence += 0.3  # Resonance / Alignment

        # 4. Signal Coherence (Internal Conflict)
        # Each conflict is a point of internal friction
        valence -= len(conflicts) * 0.1

        # Clamp to [-1.0, 1.0]
        return max(-1.0, min(1.0, valence))

    def get_complete_report(self, assessment: DirectorAssessment) -> str:
        """
        Human-readable complete systems understanding report.
        Shows integrated reasoning across ALL monitoring components.
        """
        report = []
        report.append("╔═══════════════════════════════════════════════════════════╗")
        report.append("║   DIRECTOR COMPLETE SYSTEMS STATE                         ║")
        report.append("╚═══════════════════════════════════════════════════════════╝\n")

        report.append(f"Phase: {assessment.phase.value.upper().replace('_', ' ')}")
        report.append(f"Confidence: {assessment.confidence:.0%}")
        report.append(f"Health: {'✓ HEALTHY' if assessment.is_healthy else '✗ NEEDS ATTENTION'}")
        report.append(
            f"Intervention Required: {'YES' if assessment.requires_intervention else 'No'}\n"
        )

        report.append("─── INTEGRATED SIGNAL INTERPRETATIONS ───")
        report.append(f"\nEntropy: {assessment.entropy_interpretation}")
        report.append(f"\nMatrix (Proprioception): {assessment.matrix_interpretation}")
        report.append(f"\nMetabolic: {assessment.metabolic_interpretation}")
        report.append(f"\nOperations: {assessment.pattern_interpretation}\n")

        if assessment.signals_agree:
            report.append("─── SIGNALS AGREE ───")
            for agreement in assessment.signals_agree:
                report.append(f"  ✓ {agreement}")
            report.append("")

        if assessment.signals_conflict:
            report.append("─── SIGNALS CONFLICT (Nuanced State) ───")
            for conflict in assessment.signals_conflict:
                report.append(f"  ⚠ {conflict}")
            report.append("")

        if assessment.warnings:
            report.append("─── WARNINGS ───")
            for warning in assessment.warnings:
                report.append(f"  ⚠ {warning}")
            report.append("")

        if assessment.immediate_actions:
            report.append("─── IMMEDIATE ACTIONS ───")
            for i, action in enumerate(assessment.immediate_actions, 1):
                report.append(f"  {i}. {action}")
            report.append("")

        if assessment.strategic_recommendations:
            report.append("─── STRATEGIC RECOMMENDATIONS ───")
            for i, rec in enumerate(assessment.strategic_recommendations, 1):
                report.append(f"  {i}. {rec}")

        return "\n".join(report)


# =============================================================================
# Example Integration with Actual RAA check_cognitive_state Output
# =============================================================================

if __name__ == "__main__":
    # Simulate actual state from current check_cognitive_state
    state = DirectorState(
        entropy=0.10,
        entropy_trend=0.0,
        entropy_sample_size=100,
        matrix_state="Focused",
        hopfield_energy=-0.49,
        metabolic_pct=89.0,
        energy_raw={"current_energy": "89.0", "max_energy": "100.0", "percentage": "89.0%"},
        is_looping=False,
        dominant_operation="hypothesize",
        operation_counts={"deconstruct": 4, "synthesize": 4, "hypothesize": 12},
        evolution_observations=30,
        stress_buffer_size=0,
        avg_stress=0.0,
        max_stress=0.0,
        stress_threshold=5.0,
        fusion_weights={"context": 0.340, "perspective": 0.330, "operation": 0.330},
        fusion_balance=0.995,
        dominant_stream="context",
        current_goal="Design Multi-Signal Cognitive Integration module...",
    )

    coordinator = DirectorCoordinator()
    assessment = coordinator.assess_complete_state(state)

    print(coordinator.get_complete_report(assessment))
    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("This demonstrates COMPLETE SYSTEMS UNDERSTANDING by coordinating")
    print("RAA's existing monitoring components (entropy, matrix, metabolic, patterns)")
    print("rather than replacing them with theoretical redesigns.")

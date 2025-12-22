"""
Grok-Lang: Empathetic Communication as Relational Type-Checking

This module implements the Grok-Depth Score - a quantifiable metric for measuring
"grok success" (empathetic alignment) between two mind-states across six evolutionary
levels of language understanding.

Based on the formalization:
    Γ_A, Γ_B; Δ_A, Δ_B ⊢_E φ_A ≈ φ_B :: (M_A, M_B)

Where alignment is measured as weighted similarity across:
    L0: Signal → Somatic Resonance
    L1: Symbol → Lexical Alignment
    L2: Syntax → Pattern Recognition
    L3: Semantics → Perspective Taking
    L4: Pragmatics → Functional Empathy
    L5: Meta → Joint Attention
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class CognitiveLevel(Enum):
    """The six evolutionary levels of Grok-Lang."""

    SIGNAL = 0  # L0: Raw sensory data (tone, rhythm, pace)
    SYMBOL = 1  # L1: Naming things (lexical units)
    SYNTAX = 2  # L2: Grammar and structure
    SEMANTICS = 3  # L3: Meaning and truth
    PRAGMATICS = 4  # L4: Context and intent
    META = 5  # L5: Language about language


@dataclass
class AffectVector:
    """Affective state representation (VAD model)."""

    valence: float = 0.0  # -1 (negative) to +1 (positive)
    arousal: float = 0.0  # 0 (calm) to 1 (excited)
    dominance: float = 0.5  # 0 (submissive) to 1 (dominant)

    def to_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.dominance])

    def similarity(self, other: "AffectVector") -> float:
        """Cosine similarity of affect vectors."""
        a, b = self.to_array(), other.to_array()
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / norm) if norm > 0 else 0.0


class Intent(Enum):
    """Speech act intent taxonomy (extended Searle classification)."""

    # Core speech acts
    ASSERT = "assert"  # Stating a fact
    INFORM = "inform"  # Providing information
    QUESTION = "question"  # Requesting information
    REQUEST = "request"  # Asking for action
    PROMISE = "promise"  # Committing to action
    EXPRESS = "express"  # Expressing emotion
    DECLARE = "declare"  # Making something so by saying it
    # Extended for empathetic communication
    REASSURE = "reassure"
    DEFEND = "defend"
    APOLOGIZE = "apologize"
    CHALLENGE = "challenge"
    ALIGN = "align"
    META_REPAIR = "meta_repair"


@dataclass
class MindState:
    """
    The Mind-State M = (Γ, Δ, Σ) for Grok-Lang.

    This is a computational representation of a cognitive context that
    encapsulates an agent's:
    - Γ (gamma): Typing context - embeddings + symbolic bindings
    - Δ (delta): Active derivation/transition state
    - Σ (sigma): Semantic mappings - goals, beliefs, world model
    """

    # Identity
    agent_id: str

    # Γ: Typing Context (Resonant)
    gamma_embedding: Optional[np.ndarray] = None  # Vector representation
    gamma_symbols: Dict[str, Any] = field(default_factory=dict)  # Symbolic bindings

    # Affective State (E)
    affect: AffectVector = field(default_factory=AffectVector)

    # Intent (I)
    intent: Intent = Intent.INFORM
    intent_confidence: float = 1.0

    # Δ: Current cognitive level being processed
    current_level: CognitiveLevel = CognitiveLevel.SEMANTICS

    # Σ: Semantic mappings
    beliefs: Dict[str, float] = field(default_factory=dict)  # belief -> confidence
    goals: List[str] = field(default_factory=list)

    # Relational state (Δ_rel)
    trust: float = 0.5  # 0-1
    intimacy: float = 0.3  # 0-1
    power_differential: float = 0.0  # -1 (other dominant) to +1 (self dominant)


@dataclass
class Utterance:
    """
    An utterance φ in Grok-Lang: ⟨Content∣State⟩

    Pairs the content (what is said) with the state indices (how it's meant).
    """

    content: str

    # Level-specific representations
    signal_features: Optional[np.ndarray] = None  # Acoustic/prosodic features
    symbol_tokens: List[str] = field(default_factory=list)
    syntax_structure: Optional[Dict] = None  # Parse tree or similar
    semantic_embedding: Optional[np.ndarray] = None  # Meaning vector
    pragmatic_intent: Intent = Intent.INFORM
    meta_commentary: Optional[str] = None

    # The speaker's mind-state at utterance time
    speaker_state: Optional[MindState] = None


class GrokDepthCalculator:
    """
    Computes the Grok-Depth Score: a weighted alignment metric across six levels.

    GrokDepth(M_A, M_B, φ_A, φ_B) = Σ (weight_i × alignment_score(level_i))

    Higher weights are assigned to higher levels (meta being heaviest),
    reflecting that deep understanding requires alignment at abstract levels.
    """

    # Default weights: meta-heavy (higher levels matter more for "grokking")
    DEFAULT_WEIGHTS = {
        CognitiveLevel.SIGNAL: 0.05,
        CognitiveLevel.SYMBOL: 0.10,
        CognitiveLevel.SYNTAX: 0.10,
        CognitiveLevel.SEMANTICS: 0.20,
        CognitiveLevel.PRAGMATICS: 0.25,
        CognitiveLevel.META: 0.30,
    }

    def __init__(
        self,
        weights: Optional[Dict[CognitiveLevel, float]] = None,
        embedding_model: Optional[Any] = None,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.embedding_model = embedding_model

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def compute_grok_depth(
        self,
        mind_a: MindState,
        mind_b: MindState,
        utterance_a: Optional[Utterance] = None,
        utterance_b: Optional[Utterance] = None,
    ) -> Dict[str, Any]:
        """
        Compute the Grok-Depth Score between two mind-states.

        Returns:
            Dictionary with:
            - total_score: Overall grok-depth (0-1)
            - level_scores: Per-level alignment scores
            - diagnosis: Human-readable assessment
            - alignment_vector: Raw scores per level
        """
        level_scores = {}

        # L0: Signal → Somatic Resonance
        level_scores[CognitiveLevel.SIGNAL] = self._signal_alignment(
            mind_a, mind_b, utterance_a, utterance_b
        )

        # L1: Symbol → Lexical Alignment
        level_scores[CognitiveLevel.SYMBOL] = self._symbol_alignment(
            mind_a, mind_b, utterance_a, utterance_b
        )

        # L2: Syntax → Pattern Recognition
        level_scores[CognitiveLevel.SYNTAX] = self._syntax_alignment(
            mind_a, mind_b, utterance_a, utterance_b
        )

        # L3: Semantics → Perspective Taking
        level_scores[CognitiveLevel.SEMANTICS] = self._semantic_alignment(
            mind_a, mind_b, utterance_a, utterance_b
        )

        # L4: Pragmatics → Functional Empathy
        level_scores[CognitiveLevel.PRAGMATICS] = self._pragmatic_alignment(
            mind_a, mind_b, utterance_a, utterance_b
        )

        # L5: Meta → Joint Attention
        level_scores[CognitiveLevel.META] = self._meta_alignment(
            mind_a, mind_b, utterance_a, utterance_b
        )

        # Compute weighted total
        total_score = sum(self.weights[level] * score for level, score in level_scores.items())

        # Find strongest and weakest levels
        sorted_levels = sorted(level_scores.items(), key=lambda x: x[1])
        weakest = sorted_levels[0]
        strongest = sorted_levels[-1]

        # Find critical gaps (levels below 0.4)
        critical_gaps = [level for level, score in level_scores.items() if score < 0.4]

        # Generate diagnosis
        diagnosis = self._generate_diagnosis(level_scores, total_score)

        return {
            "total_score": total_score,
            "level_scores": {level.name: score for level, score in level_scores.items()},
            "per_level": level_scores,  # Keep CognitiveLevel keys for internal use
            "diagnosis": diagnosis,
            "strongest_level": strongest[0],
            "weakest_level": weakest[0],
            "critical_gaps": critical_gaps,
            "alignment_vector": [level_scores[CognitiveLevel(i)] for i in range(6)],
            "weights_used": {level.name: w for level, w in self.weights.items()},
        }

    def _signal_alignment(
        self, m_a: MindState, m_b: MindState, u_a: Optional[Utterance], u_b: Optional[Utterance]
    ) -> float:
        """
        Somatic Resonance: "I feel your physical state before I know your words."

        Measures affective state similarity (valence, arousal, dominance).
        """
        return m_a.affect.similarity(m_b.affect)

    def _symbol_alignment(
        self, m_a: MindState, m_b: MindState, u_a: Optional[Utterance], u_b: Optional[Utterance]
    ) -> float:
        """
        Lexical Alignment: "Does 'pain' mean the same thing to you as to me?"

        Measures overlap in symbolic bindings (shared vocabulary/concepts).
        """
        if not m_a.gamma_symbols or not m_b.gamma_symbols:
            return 0.5  # Neutral if no symbols

        shared = set(m_a.gamma_symbols.keys()) & set(m_b.gamma_symbols.keys())
        total = set(m_a.gamma_symbols.keys()) | set(m_b.gamma_symbols.keys())

        return len(shared) / len(total) if total else 0.5

    def _syntax_alignment(
        self, m_a: MindState, m_b: MindState, u_a: Optional[Utterance], u_b: Optional[Utterance]
    ) -> float:
        """
        Pattern Recognition: "I see how you structure your reality."

        Measures structural similarity in how information is organized.
        """
        # For now: use embedding similarity of gamma if available
        if m_a.gamma_embedding is not None and m_b.gamma_embedding is not None:
            return self._cosine_similarity(m_a.gamma_embedding, m_b.gamma_embedding)
        return 0.5

    def _semantic_alignment(
        self, m_a: MindState, m_b: MindState, u_a: Optional[Utterance], u_b: Optional[Utterance]
    ) -> float:
        """
        Perspective Taking: "I understand this is true from your view."

        Measures belief overlap and semantic embedding similarity.
        """
        # Belief alignment
        if m_a.beliefs and m_b.beliefs:
            shared_beliefs = set(m_a.beliefs.keys()) & set(m_b.beliefs.keys())
            if shared_beliefs:
                # Compute confidence alignment for shared beliefs
                alignments = [1.0 - abs(m_a.beliefs[k] - m_b.beliefs[k]) for k in shared_beliefs]
                belief_score = sum(alignments) / len(alignments)
            else:
                belief_score = 0.3
        else:
            belief_score = 0.5

        # Utterance semantic similarity (if available)
        if (
            u_a
            and u_b
            and u_a.semantic_embedding is not None
            and u_b.semantic_embedding is not None
        ):
            semantic_score = self._cosine_similarity(u_a.semantic_embedding, u_b.semantic_embedding)
            return 0.5 * belief_score + 0.5 * semantic_score

        return belief_score

    def _pragmatic_alignment(
        self, m_a: MindState, m_b: MindState, u_a: Optional[Utterance], u_b: Optional[Utterance]
    ) -> float:
        """
        Functional Empathy: "I know WHY you are telling me this now."

        Measures intent alignment and goal compatibility.
        """
        # Intent alignment
        intent_match = 1.0 if m_a.intent == m_b.intent else 0.3

        # Goal compatibility
        if m_a.goals and m_b.goals:
            shared_goals = set(m_a.goals) & set(m_b.goals)
            goal_score = len(shared_goals) / max(len(m_a.goals), len(m_b.goals))
        else:
            goal_score = 0.5

        # Trust and intimacy alignment
        relational_score = (
            1.0 - (abs(m_a.trust - m_b.trust) + abs(m_a.intimacy - m_b.intimacy)) / 2.0
        )

        return 0.4 * intent_match + 0.3 * goal_score + 0.3 * relational_score

    def _meta_alignment(
        self, m_a: MindState, m_b: MindState, u_a: Optional[Utterance], u_b: Optional[Utterance]
    ) -> float:
        """
        Joint Attention: "We are now discussing HOW we are communicating."

        Measures meta-cognitive alignment - awareness of shared context
        and willingness to repair misalignment.
        """
        # Both agents at meta level indicates joint attention
        level_match = (
            1.0
            if (
                m_a.current_level == CognitiveLevel.META
                and m_b.current_level == CognitiveLevel.META
            )
            else 0.5
        )

        # Check for meta-repair intent
        repair_bonus = (
            0.2 if (m_a.intent == Intent.META_REPAIR or m_b.intent == Intent.META_REPAIR) else 0.0
        )

        # High trust + intimacy indicates established joint attention
        relational_base = (m_a.trust + m_b.trust + m_a.intimacy + m_b.intimacy) / 4.0

        return min(1.0, level_match * 0.4 + relational_base * 0.4 + repair_bonus)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a.flatten(), b.flatten()) / norm)

    def _generate_diagnosis(
        self, level_scores: Dict[CognitiveLevel, float], total_score: float
    ) -> str:
        """Generate a human-readable diagnosis of the grok-depth."""

        # Find weakest and strongest levels
        sorted_levels = sorted(level_scores.items(), key=lambda x: x[1])
        weakest = sorted_levels[0]
        strongest = sorted_levels[-1]

        # Overall assessment
        if total_score >= 0.8:
            assessment = "Deep grokking achieved - minds are well-aligned"
        elif total_score >= 0.6:
            assessment = "Partial grokking - alignment exists but gaps remain"
        elif total_score >= 0.4:
            assessment = "Surface understanding - significant perspective gaps"
        else:
            assessment = "Minimal grokking - fundamental misalignment detected"

        # Level-specific recommendations
        if weakest[1] < 0.4:
            if weakest[0] == CognitiveLevel.SIGNAL:
                rec = "Somatic mis-resonance: affective states diverge. Try emotional attunement."
            elif weakest[0] == CognitiveLevel.SYMBOL:
                rec = "Lexical misalignment: you may be using words differently. Define terms."
            elif weakest[0] == CognitiveLevel.PRAGMATICS:
                rec = "Functional misalignment: unclear on intent. Clarify 'why' of communication."
            elif weakest[0] == CognitiveLevel.META:
                rec = "No joint attention: consider meta-communication about the conversation."
            else:
                rec = f"Low alignment at {weakest[0].name} level. Consider explicit alignment."
        else:
            rec = "No critical gaps detected."

        return (
            f"{assessment}. Strongest: {strongest[0].name} ({strongest[1]:.2f}). "
            f"Weakest: {weakest[0].name} ({weakest[1]:.2f}). {rec}"
        )


def demonstrate_grok_lang() -> Dict[str, Any]:
    """Demonstrate the Grok-Lang system with the 'Fine' example."""

    # Create two mind-states for the "Fine" scenario
    speaker = MindState(
        agent_id="speaker",
        affect=AffectVector(valence=-0.3, arousal=0.4, dominance=0.3),
        intent=Intent.DEFEND,  # Actually hiding distress
        intent_confidence=0.7,
        beliefs={"relationship_stable": 0.4},
        goals=["avoid_conflict", "receive_comfort"],
        trust=0.6,
        intimacy=0.5,
    )

    listener = MindState(
        agent_id="listener",
        affect=AffectVector(valence=0.1, arousal=0.2, dominance=0.5),
        intent=Intent.INFORM,  # Taking "fine" at face value
        intent_confidence=0.9,
        beliefs={"relationship_stable": 0.8},
        goals=["check_in"],
        trust=0.6,
        intimacy=0.5,
    )

    utterance = Utterance(
        content="Fine.",
        pragmatic_intent=Intent.DEFEND,
        meta_commentary="Acknowledging relational strain",
    )

    # Compute grok-depth
    calculator = GrokDepthCalculator()
    result = calculator.compute_grok_depth(speaker, listener, utterance, None)

    print("=== Grok-Lang Analysis: 'Fine.' ===")
    print(f"\nTotal Grok-Depth: {result['total_score']:.3f}")
    print("\nPer-Level Alignment:")
    for level, score in result["level_scores"].items():
        print(f"  {level}: {score:.3f}")
    print(f"\nDiagnosis: {result['diagnosis']}")

    return result


if __name__ == "__main__":
    demonstrate_grok_lang()

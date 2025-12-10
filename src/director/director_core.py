"""
Director Core: Integration of Monitoring and Search

Implements the full Director loop:
1. Monitor entropy from Processor output
2. Detect "clash" (high entropy state)
3. Search Manifold for alternative framing
4. Return new goal for Pointer update

This is the Phase 1 (MVP) implementation following SEARCH_MECHANISM_DESIGN.md.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from src.compass.adapters import RAALLMProvider
from src.compass.compass_framework import COMPASS

if TYPE_CHECKING:
    from src.integration.continuity_service import ContinuityService
    from src.manifold import Manifold

import numpy as np

from src.director.entropy_monitor import EntropyMonitor
from src.director.process_logger import logger as process_logger  # Integrated Process Logger

from .epistemic_discriminator import EpistemicDiscriminator
from .epistemic_metrics import estimate_complexity, estimate_randomness
from .hybrid_search import HybridSearchConfig, HybridSearchStrategy
from .ltn_refiner import LTNConfig, LTNRefiner
from .matrix_monitor import MatrixMonitor, MatrixMonitorConfig
from .plasticity_modulator import PlasticityModulator
from .recursive_observer import RecursiveObserver
from .reflexive_closure_engine import ReflexiveClosureEngine
from .search_mvp import SearchResult
from .sheaf_diagnostics import SheafAnalyzer, SheafConfig, SheafDiagnostics
from .simple_gp import TRIG_OPS, TRIG_UNARY_OPS, SimpleGP
from .thought_suppression import SuppressionResult, SuppressionStrategy, ThoughtSuppressor

logger = logging.getLogger(__name__)


@dataclass
class Intervention:
    """Represents a proactive intervention in the cognitive process."""

    type: str
    source: str
    target: str
    content: str
    priority: float
    energy_cost: float


@dataclass
class DirectorConfig:
    """Configuration for Director."""

    # Entropy monitoring
    entropy_threshold_percentile: float = 0.75
    entropy_history_size: int = 100
    default_entropy_threshold: float = 0.8  # Increased from 0.2 to reduce false positives
    enable_system2: bool = True  # Master switch for Time Dilation / GP

    # Reflexive Closure
    enable_reflexive_closure: bool = True
    reflexive_analysis_interval: int = 50

    # Search parameters
    search_k: int = 5  # Number of neighbors to retrieve
    search_metric: str = "cosine"  # Distance metric
    exclude_threshold: float = 0.95  # Similarity threshold for excluding current basin
    use_energy_aware_search: bool = True  # Use energy-aware selection (aligns with Hopfield theory)

    # Control parameters
    max_search_attempts: int = 3  # Maximum search attempts before giving up
    min_entropy_reduction: float = 0.1  # Minimum entropy reduction to accept new goal

    # Logging
    log_search_episodes: bool = True
    device: str = "cpu"

    # Sheaf Diagnostics
    sheaf_config: Optional[SheafConfig] = None

    # Matrix Monitor
    matrix_monitor_config: Optional[MatrixMonitorConfig] = None

    # Hybrid Search
    hybrid_search_config: Optional[HybridSearchConfig] = None

    # Adaptive Temperature
    adaptive_temp_min: float = 0.2
    adaptive_temp_max: float = 0.8

    # Policing (Thought Suppression)
    suppression_threshold: float = 0.6
    suppression_cost: float = 2.0


class DirectorMVP:
    """
    Director MVP: Metacognitive Monitor + Search Engine.

    The Director is the core innovation of RAA. It detects confusion via
    entropy monitoring and triggers search in the Manifold for alternative
    conceptual framings.
    """

    def __init__(
        self,
        manifold: "Manifold",
        config: Optional[DirectorConfig] = None,
        embedding_fn: Optional[Callable[[str], torch.Tensor]] = None,
        mcp_client: Optional[Any] = None,
        continuity_service: Optional["ContinuityService"] = None,
        llm_provider: Optional[Any] = None,
        work_history: Optional[Any] = None,
        precuneus: Optional[Any] = None,
    ):
        """
        Initialize Director.

        Args:
            manifold: Manifold (Modern Hopfield Network) to search
            config: Director configuration
            embedding_fn: Function to embed text for LTN constraints
            mcp_client: Optional MCP client for tool execution
            continuity_service: Optional ContinuityService for anchoring milestones
            work_history: Optional WorkHistory for long-term entropy recall
            precuneus: Optional PrecuneusIntegrator for state modulation
        """
        self.manifold = manifold
        self.config = config or DirectorConfig()
        self.mcp_client = mcp_client
        self.continuity_service = continuity_service
        self.work_history = work_history
        self.precuneus = precuneus

        # Entropy monitor
        self.monitor = EntropyMonitor(
            threshold_percentile=self.config.entropy_threshold_percentile,
            history_size=self.config.entropy_history_size,
            default_threshold=self.config.default_entropy_threshold,
        )

        # Sheaf Analyzer
        sheaf_config = self.config.sheaf_config or SheafConfig(device=self.config.device)
        self.sheaf_analyzer = SheafAnalyzer(sheaf_config)

        # Initialize Reflexive Closure Engine
        self.reflexive_engine = None
        if self.config.enable_reflexive_closure:
            self.reflexive_engine = ReflexiveClosureEngine(
                analysis_interval=self.config.reflexive_analysis_interval,
                criterion=None,  # Will use default which loads from disk
            )
            logger.info("Reflexive Closure Engine initialized.")

        # Initialize Recursive Observer (Meta-Layer)
        self.observer = RecursiveObserver()
        logger.info("Recursive Observer initialized.")

        # 3. Matrix Monitor (Cognitive Proprioception)
        self.matrix_monitor = MatrixMonitor(
            config=self.config.matrix_monitor_config
            or MatrixMonitorConfig(device=self.config.device)
        )
        self.matrix_monitor.seed_defaults()

        # 4. Hybrid Search Strategy (Operator C)
        hybrid_cfg = self.config.hybrid_search_config or HybridSearchConfig(
            knn_k=self.config.search_k,
            knn_metric=self.config.search_metric,
            knn_exclude_threshold=self.config.exclude_threshold,
        )

        # Initialize LTN Refiner
        # Use provided embedding_fn or a dummy that warns
        def dummy_embed(text: str) -> torch.Tensor:
            logger.warning("Using dummy embedding function. Constraints will fail.")
            # Return a 1D tensor of size 1 just to satisfy type checks,
            # though it will likely fail dimension checks if used.
            # Ideally we should raise an error if constraints are used.
            return torch.zeros(1, device=self.config.device)

        ltn_config = hybrid_cfg.ltn_config or LTNConfig(device=self.config.device)
        self.ltn_refiner = LTNRefiner(embedding_fn=embedding_fn or dummy_embed, config=ltn_config)

        self.hybrid_search = HybridSearchStrategy(
            manifold=self.manifold,
            ltn_refiner=self.ltn_refiner,
            sheaf_analyzer=self.sheaf_analyzer,
            config=hybrid_cfg,
        )

        # 5. COMPASS Framework (Metacognitive Orchestration)
        # Initialize with RAA LLM adapter and MCP client
        # Use provided llm_provider or create default
        _llm_provider = llm_provider if llm_provider is not None else RAALLMProvider()
        self.compass = COMPASS(llm_provider=_llm_provider, mcp_client=self.mcp_client)

        # 6. Agent Factory (Dynamic Escalation)

        # Initialize with the same LLM provider and MCP client executor
        # Import here to avoid circular dependency
        from src.integration.agent_factory import AgentFactory

        self.agent_factory = AgentFactory(
            llm_provider=_llm_provider,
            tool_executor=self.mcp_client.call_tool if self.mcp_client else None,
        )

        # Initialize Allostatic Controller (Predictive)
        from src.director.allostatic_controller import AllostaticConfig, AllostaticController

        self.allostatic_controller = AllostaticController(
            AllostaticConfig(), ledger=None
        )  # Ledger wired later
        # 7. Swarm Controller (Hive Mind)
        from src.integration.swarm_controller import SwarmController

        self.swarm_controller = SwarmController(self.agent_factory)

        # Cognitive State
        self.latest_cognitive_state: tuple[str, float] = ("Unknown", 0.0)
        self.latest_diagnostics: dict[str, Any] = {}
        self.latest_attention_weights: Optional[torch.Tensor] = None

        # Search episode logging
        self.search_episodes: List[Dict[str, Any]] = []

        # Energy Budget (Thermodynamic Constraint)
        self.energy_budget = 100.0

        # --- Research Integrations (Moloch Suite) ---

        # 1. Thought Suppressor (Experiment B: Police Strategy)
        self.thought_suppressor = ThoughtSuppressor(
            suppression_threshold=self.config.suppression_threshold,
            suppression_cost=self.config.suppression_cost,
            quarantine_threshold=0.8,
            quarantine_cost=0.5,
        )
        self.suppression_history: List[SuppressionResult] = []
        logger.info("ThoughtSuppressor initialized (active policing mode)")

        # 2. Epistemic Discriminator (Experiment C)
        self.epistemic_discriminator = EpistemicDiscriminator(
            complexity_threshold=0.6, randomness_threshold=0.2, dissonance_threshold=0.5
        )
        logger.info("EpistemicDiscriminator initialized")

        # 3. Plasticity Modulator (Phenomenal Time)
        self.plasticity_modulator = PlasticityModulator(
            min_p=0.1, max_p=1.0, tau_min=0.1, tau_max=2.0
        )
        logger.info("PlasticityModulator initialized")

    def get_current_confidence(self) -> float:
        """Estimate current confidence from entropy."""
        if not self.monitor.entropy_history:
            return 0.5
        last_entropy = self.monitor.entropy_history[-1]
        # Rough inverse mapping: 0 entropy -> 1 confidence, 2 entropy -> 0 confidence
        return max(0.0, 1.0 - (last_entropy / 2.0))

    async def intervene(self, current_entropy: float, current_goal: str) -> Optional[bool]:
        """
        Intervene in the cognitive process based on entropy and epistemic state.
        Returns False if propagation should be blocked (Suppression).
        """
        # 1. Thought Suppression (Policing)
        if current_entropy > self.thought_suppressor.suppression_threshold:
            suppression_result = self.thought_suppressor.evaluate_thought(
                thought_id=f"thought_{int(time.time())}",
                entropy=current_entropy,
                energy_budget=self.energy_budget,
                graph_handle=self.manifold,  # Assuming manifold can be passed, or None
            )

            self.suppression_history.append(suppression_result)

            if suppression_result.suppressed:
                self.energy_budget -= suppression_result.energy_cost
                logger.info(
                    f"[SUPPRESSION] Strategy: {suppression_result.strategy.value}, "
                    f"Cost: {suppression_result.energy_cost:.2f}J, "
                    f"Reason: {suppression_result.reason}"
                )

                if suppression_result.strategy == SuppressionStrategy.SUPPRESS:
                    return False  # Block propagation

        # 2. Epistemic Discrimination
        # Use long-term history if available, otherwise short-term monitor history
        if self.work_history:
            # Fetch last 100 points from DB (Long Term)
            history_list = self.work_history.get_entropy_history(limit=100)
            # Append current
            history_list.append(current_entropy)
            entropy_history = np.array(history_list)
        else:
            # Fallback to in-memory monitor (Short Term)
            entropy_history = np.array(self.monitor.entropy_history)

        if len(entropy_history) > 10:
            assessment = self.epistemic_discriminator.assess(entropy_history)
            logger.info(
                f"Epistemic Assessment: {assessment.recommendation} (Conf: {assessment.confidence:.2f})"
            )

            if assessment.recommendation == "trigger_dissonance":
                logger.warning("Epistemic Dissonance Triggered - Halting path")
                return False

        # 3. Plasticity Modulation
        p_state = self.plasticity_modulator.compute_p(
            energy=self.energy_budget,
            confidence=self.get_current_confidence(),
            entropy=current_entropy,
            manifold_stability=(
                self.matrix_monitor.get_stability()
                if hasattr(self.matrix_monitor, "get_stability")
                else 1.0
            ),
        )

        logger.info(f"Plasticity: P={p_state.value:.2f} ({p_state.mode})")

        # Apply to Precuneus if available (placeholder)
        if self.precuneus is not None:
            if hasattr(self.precuneus, "set_integration_rate"):
                self.precuneus.set_integration_rate(p_state.value)

        return True

    def map_entropy_to_generations(self, entropy: float) -> int:
        """
        Map entropy (uncertainty) to computational time (generations).

        Time Perception:
        - Low Entropy (0.2) -> 10 generations (Fast, System 1)
        - High Entropy (0.8+) -> 500+ generations (Slow, System 2)

        Formula: 10 + (entropy * 600)
        """
        # Clamp entropy between 0 and 1
        entropy = max(0.0, min(1.0, entropy))
        generations = int(10 + (entropy * 600))
        return generations

    async def evolve_formula(
        self, data_points: List[Dict[str, float]], n_generations: int = 10
    ) -> str:
        """
        Evolve a mathematical formula to fit the data points.
        Uses Genetic Programming (System 2) with Epistemic Discrimination.

        Implements the "Diamond Proof" mechanisms:
        1. Complexity Estimation -> Attention (Focused Search)
        2. Randomness Estimation -> Suppression (Noise Filtering)
        3. Discontinuity Detection -> Epistemic Honesty (Warning)
        """
        logger.info(
            f"Director: Evolving formula for {len(data_points)} points over {n_generations} generations..."
        )

        if not data_points:
            return "0"

        # Extract target values for epistemic analysis
        y_values = [d.get("result", 0.0) for d in data_points]

        # --- Epistemic Analysis ---
        complexity_info = estimate_complexity(y_values)
        randomness_info = estimate_randomness(y_values)

        complexity = complexity_info["complexity_score"]
        randomness = randomness_info["randomness_score"]

        logger.info(
            f"Director: [Epistemic] Complexity: {complexity:.3f} ({complexity_info['type']})"
        )
        logger.info(
            f"Director: [Epistemic] Randomness: {randomness:.3f} ({randomness_info['type']})"
        )

        if self.reflexive_engine:
            import uuid

            episode_id = str(uuid.uuid4())

            # Dynamic Threshold Logic (Integration of Reflexive Learning)
            # Get global learned threshold
            reflexive_threshold = self.reflexive_engine.get_threshold(
                self.latest_cognitive_state[0]
            )
            # Use the higher of local config or learned threshold (Conservative/Safe)
            current_threshold = max(self.config.suppression_threshold, reflexive_threshold)

            logger.debug(
                f"Director: Using composed threshold: {current_threshold:.3f} (Config: {self.config.suppression_threshold}, Reflexive: {reflexive_threshold:.3f})"
            )

            self.reflexive_engine.record_intervention_start(
                episode_id=episode_id,
                entropy=randomness,
                energy=self.energy_budget,
                cognitive_state=self.latest_cognitive_state[0],
                goal="policing_check",
                intervention_type="epistemic_filter",
                intervention_source="director_core",
                threshold=current_threshold,
                parameters={"complexity": complexity},
            )
        else:
            current_threshold = self.config.suppression_threshold

        # --- 1. Suppression (Policing Entropy) ---
        suppression_active = False
        if randomness > current_threshold:
            logger.info(
                f"Director: [Action] HIGH RANDOMNESS ({randomness:.2f} > {current_threshold:.2f}) -> Activating Suppression (Noise Filtering)"
            )

            # Log to suppression history

            self.suppression_history.append(
                SuppressionResult(
                    entropy_before=randomness,
                    energy_cost=self.config.suppression_cost,
                    strategy=SuppressionStrategy.SUPPRESS,
                    suppressed=True,
                    reason="High Randomness (Epistemic Filtering)",
                )
            )
            self.energy_budget -= self.config.suppression_cost

            # Reflexive Feedback: Suppression Event
            if self.reflexive_engine:
                # Suppression is the "Safe Bet" (Quality 0.5)
                self.reflexive_engine.record_intervention_end(
                    episode_id=episode_id,
                    entropy_after=0.0,  # Effectively zeroed
                    energy_after=self.energy_budget,
                    task_success=True,  # We successfully suppressed
                    outcome_quality=0.5,  # Neutral/Safe outcome
                    metadata={"strategy": "suppress", "cost": self.config.suppression_cost},
                )

            # Simple moving average smoothing
            y_smooth = np.convolve(y_values, np.ones(5) / 5, mode="same")
            for i, d in enumerate(data_points):
                d["result"] = float(y_smooth[i])
            suppression_active = True

            # If we suppress, we usually simplify the problem and continue, OR we abort.
            # In Experiment C, we continued with smoothing.
            # But "Suppression" implies blocking the original chaotic signal.

        else:
            # We ACCEPTED the signal.
            pass

        # --- 2. Attention (Focused Search) ---
        focused_search = False
        ops: Optional[List[Tuple[Callable[[float, float], float], str]]] = None
        unary_ops: Optional[List[Tuple[Callable[[float], float], str]]] = None

        if complexity > 0.7 and complexity_info["type"] != "discontinuous":
            logger.info(
                "Director: [Action] HIGH COMPLEXITY -> Activating Focused Search (Trig Primitives)"
            )
            focused_search = True
            ops = TRIG_OPS
            unary_ops = TRIG_UNARY_OPS
            n_generations = max(n_generations, 100)

        # --- 3. Epistemic Honesty ---
        warning_msg = ""
        if complexity_info["type"] == "discontinuous":
            logger.info("Director: [Action] DISCONTINUITY DETECTED -> Flagging for Segmentation")
            warning_msg = " [WARNING: Discontinuity Detected - Approximation Only]"

        # GP Setup
        first_point = data_points[0]
        variables = [k for k in first_point.keys() if k != "result"]
        pop_size = 200 if (focused_search or suppression_active) else 50

        gp = SimpleGP(
            variables=variables,
            population_size=pop_size,
            max_depth=6 if focused_search else 4,
            ops=ops,
            unary_ops=unary_ops,
        )

        # Evolve
        # [PROCESS LOGGING]
        try:
            process_logger.log(
                "THOUGHT",
                {
                    "action": "run_reflexive_loop",
                    "step": "start",
                    "context": "Reflexive Evolution Loop",
                    "cognitive_state": (
                        self.latest_cognitive_state[0]
                        if hasattr(self, "latest_cognitive_state")
                        else "Unknown"
                    ),
                },
            )
        except Exception as e:
            logger.warning(f"Director: Failed to log reflexive loop start: {e}")
        best_formula, best_error = gp.evolve(
            data_points, target_key="result", generations=n_generations, hybrid=True
        )

        # Calculate Compute Cost
        # Heuristic: 0.1J per generation * population_ratio
        compute_cost = (n_generations * 0.1) * (pop_size / 50.0)
        self.energy_budget -= compute_cost

        # Reflexive Feedback: Acceptance Outcome
        if self.reflexive_engine and not suppression_active:
            # Quality driven by Error (MSE)
            # Low MSE (< 0.1) -> High Quality (1.0)
            # High MSE (> 1.0) -> Low Quality (0.0) -> implies we accepted noise

            mse = best_error
            if mse < 0.1:
                quality = 1.0
            elif mse > 10.0:
                quality = 0.0
            else:
                # Linear interpolation 0.1..10.0 -> 1.0..0.0
                quality = max(0.0, 1.0 - (mse - 0.1) / 9.9)

            self.reflexive_engine.record_intervention_end(
                episode_id=episode_id,
                entropy_after=mse,  # Uncertainty remains high if error is high
                energy_after=self.energy_budget,
                task_success=(quality > 0.6),
                outcome_quality=quality,
                metadata={"strategy": "accept", "mse": mse, "cost": compute_cost},
            )

        logger.info(
            f"Director: Evolution complete. Formula: {best_formula} (MSE: {best_error:.4f})"
        )

        result_str = best_formula + warning_msg
        if suppression_active:
            result_str += " [Suppressed Noise]"

        return result_str

    async def process_task_with_time_gate(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a task with the "Time Gate" (Dynamic Inference Budgeting).
        If entropy is high, we allocate more compute (generations) to solve it.
        """
        logger.info(f"Director: Processing task with Time Gate: {task[:5000]}...")

        # 1. Immediate Prediction (System 1)
        # We use COMPASS to get the initial thought/attempt
        result = await self.compass.process_task(task, context)

        # 2. Measure Entropy (Uncertainty)
        # Use EntropyCalculator to get formal Shannon entropy
        from src.integration.entropy_calculator import EntropyCalculator

        calc = EntropyCalculator()

        # Create pseudo-logits from confidence: [confidence, 1-confidence]
        confidence = result.get("score", 0.5)
        scores = torch.tensor([confidence, 1.0 - confidence], dtype=torch.float32)
        logits = torch.log(scores + 1e-8)
        entropy_score = calc.compute_entropy(logits)

        logger.info(
            f"Director: System 1 Confidence: {confidence:.2f}, Entropy (Bits): {entropy_score:.2f}"
        )
        if context:
            logger.info(f"Director: Context keys: {list(context.keys())}")
            logger.info(f"Director: force_time_gate: {context.get('force_time_gate')}")

        # 3. Temporal Decision
        # 2.9 Cognitive State Check (Proprioception)
        # We consult the agent's internal state to decide on System 2 usage.
        should_engage_system2 = False
        cognitive_state_label = "Unknown"

        if self.config.enable_system2:
            try:
                # Call internal tool to get state
                # We assume self.compass.mcp_client can call internal tools
                # Or we use the context's tool execution capability if available
                # ideally: state = await self.compass.mcp_client.call_tool("mcp_check_cognitive_state", {})
                # For safety, we wrap this in a try/except or skip if client not ready
                pass
                # actually, better to rely on simpler logic for now or implement direct call if possible.
                # Given strict instructions to "call mcp_check_cognitive_state in Director", we need access.
                # Director has self.compass.mcp_client (RAAMCPClient)
                if self.compass and self.compass.mcp_client:
                    state_result = await self.compass.mcp_client.call_tool(
                        "mcp_check_cognitive_state", {}
                    )
                    # Parse result: "State: Focused (Stability: 0.8)"
                    state_str = str(state_result)
                    if "Looping" in state_str or "Stuck" in state_str:
                        should_engage_system2 = True
                        cognitive_state_label = "Looping/Stuck"
                        logger.info(
                            f"Director: Cognitive State is {cognitive_state_label}. FORCING System 2."
                        )
                    elif "Focused" in state_str or "Flow" in state_str:
                        cognitive_state_label = "Focused/Flow"

            except Exception as e:
                logger.warning(f"Director: Failed to check cognitive state: {e}")

        # 3. Temporal Decision
        # Threshold: 0.8 bits
        # Check for forced Time Gate in context
        force_gate = context.get("force_time_gate", False) if context else False

        # Logic:
        # 1. If System 2 Disabled -> System 1
        # 2. If Forced -> System 2
        # 3. If Cognitive State says STUCK -> System 2
        # 4. If Entropy High (>0.8) AND NOT Focused -> System 2
        # 5. If Entropy SUPER High (>0.95) -> System 2 (even if Focused)

        engage_reason = None

        if not self.config.enable_system2 and not force_gate:
            logger.info(
                f"Director: High Entropy ({entropy_score:.2f}) but System 2 disabled. Using System 1."
            )
            return result

        if force_gate:
            engage_reason = "Forced by User"
        elif should_engage_system2:
            engage_reason = f"Cognitive State ({cognitive_state_label})"
        elif entropy_score >= 0.95:
            engage_reason = f"Critical Entropy ({entropy_score:.2f})"
        elif entropy_score >= 0.8 and cognitive_state_label != "Focused/Flow":
            engage_reason = f"High Entropy ({entropy_score:.2f})"

        if not engage_reason:
            logger.info("Director: Low Entropy. Trusting System 1 (Fast Time).")
            return result

        else:
            # High Entropy: Distort Time. Enter the "Temporal Buffer".
            logger.info(f"Director: {engage_reason} detected. Engaging System 2 (Time Dilation).")

            # --- NEW: Intervention Check ---
            should_continue = await self.intervene(entropy_score, task)
            if not should_continue:
                logger.info("Director: Intervention blocked propagation (Suppression/Dissonance).")
                return {"status": "suppressed", "reason": "High Entropy/Randomness"}
            # -------------------------------

            # A. Allocation of Time (Compute)
            pondering_budget = self.map_entropy_to_generations(entropy_score)
            logger.info(
                f"Director: Allocating {pondering_budget} generations to 'evolve_formula'..."
            )

            # B. The "Slow" Path (Evolutionary Optimization)
            # We need to extract data points from context to run evolve_formula
            # Assuming context contains 'data_points' or we can extract them
            data_points = context.get("data_points") if context else None

            # If no explicit data points, we might need to parse them or skip
            if not data_points and context and "data" in str(context):
                # Try to find 'data' key loosely
                data_points = context.get("data")

            if data_points:
                try:
                    # Call evolve_formula directly (System 2 function)
                    evolution_text = await self.evolve_formula(
                        data_points, n_generations=pondering_budget
                    )

                    # C. Insight Integration
                    # Force the agent to accept the "future" result
                    logger.info("Director: Integrating Evolutionary Insight...")

                    # We revise the original result with the new insight
                    revision_prompt = (
                        f"Original Answer: {result.get('solution')}\n\n"
                        f"New Evolutionary Insight (High Confidence): {evolution_text}\n\n"
                        "Task: Integrate this mathematical law into the final answer. "
                        "Replace any guessed formulas with this one. "
                        "Mark the output with [System 2 Intervention]."
                    )

                    # Use COMPASS to synthesize the final answer
                    # We can reuse process_task but with the new prompt and high confidence context
                    revised_result = await self.compass.process_task(revision_prompt, context)
                    return revised_result

                except Exception as e:
                    logger.error(f"Director: Evolutionary Loop failed: {e}")
                    return result  # Fallback to System 1

            else:
                # NEW: Context Retrieval Loop (Progress-Oriented)
                # If high entropy but no data_points, use tools to gather context and retry.
                logger.info(
                    "Director: High entropy but no 'data_points'. Initiating Context Retrieval Loop."
                )

                # Step 1: Build a retrieval prompt that instructs COMPASS to use graph tools
                retrieval_prompt = (
                    f"SYSTEM ALERT: The agent is stuck on: '{task[:300]}...'. "
                    "Cognitive State: {state}. Entropy: {entropy:.2f}. "
                    "Use `inspect_graph` to find related ThoughtNodes in the knowledge graph. "
                    "Use `search_knowledge` if the graph is sparse. "
                    "Return the most relevant context to help solve the original task. "
                    "DO NOT attempt to solve the task yet—just retrieve context."
                ).format(
                    state=cognitive_state_label,
                    entropy=entropy_score,
                )

                try:
                    retrieval_result = await self.compass.process_task(retrieval_prompt, context)
                    retrieved_context = retrieval_result.get("solution", "")

                    if retrieved_context and len(retrieved_context) > 50:
                        logger.info(
                            f"Director: Retrieved {len(retrieved_context)} chars of context. Re-attempting task."
                        )

                        # Step 2: Inject retrieved context and re-attempt
                        enriched_context = (context or {}).copy()
                        enriched_context["retrieved_knowledge"] = retrieved_context
                        enriched_context["_retrieval_attempt"] = True  # Prevent infinite loop

                        # Step 3: Re-attempt original task with enriched context
                        revised_result = await self.compass.process_task(task, enriched_context)
                        revised_result["_context_retrieval_used"] = True
                        return revised_result

                    else:
                        logger.warning("Director: Context retrieval returned insufficient data.")
                        return result

                except Exception as e:
                    logger.error(f"Director: Context Retrieval Loop failed: {e}")
                    return result  # Fallback to System 1

    async def check_proactive_interventions(self) -> Optional[Intervention]:
        """
        Check for proactive interventions (Allostasis).

        Returns:
            Intervention object if triggered, None otherwise.
        """
        if not self.allostatic_controller:
            return None

        predicted_entropy = self.allostatic_controller.predict_future_entropy()
        allostatic_trigger = self.allostatic_controller.check_trigger()

        if allostatic_trigger == "PROACTIVE_ECC":
            logger.warning(
                f"Allostatic Alert! Predicted Entropy {predicted_entropy:.2f} > Critical. Deploying ECC Swarm."
            )
            return await self._handle_proactive_ecc(predicted_entropy)

        return None

    async def _handle_proactive_ecc(self, predicted_entropy: float) -> Optional[Intervention]:
        """
        Deploy ECC Swarm (Redundancy, Parity, Syndrome) to preemptively stabilize the system.
        """
        logger.info(
            f"Deploying ECC Swarm for Allostatic Regulation (Predicted Entropy: {predicted_entropy:.2f})"
        )

        # 1. Define ECC Task
        ecc_task = (
            f"SYSTEM WARNING: Entropy is predicted to reach {predicted_entropy:.2f} (CRITICAL). "
            f"The logic is becoming unstable. "
            f"DEPLOYING ERROR CORRECTION CODE. "
            f"Redundancy Agent: Stabilize axioms. "
            f"Parity Agent: Check for contradictions. "
            f"Syndrome Agent: Diagnose the noise source."
        )
        task = f"ECC TRIGGERED: Entropy gradient {predicted_entropy:.2f} > 0.05. Check system logic."  # Using predicted_entropy as a proxy for gradient here

        try:
            process_logger.log(
                "SWARM",
                {"action": "ecc_trigger", "gradient": float(predicted_entropy), "task": task},
            )
        except Exception as e:
            logger.warning(f"Director: Failed to log ECC trigger: {e}")

        # 2. Summon ECC Advisors
        advisor_ids = ["redundancy_agent", "parity_agent", "syndrome_agent"]

        # 3. Run Swarm
        # We pass an empty context or current manifold state as context
        swarm_result = await self.swarm_controller.run_swarm(
            task=ecc_task,
            context=f"Predicted Entropy: {predicted_entropy}",
            advisor_ids=advisor_ids,
        )

        # 4. Construct Intervention
        # We treat this as a "Top-Down" correction
        intervention = Intervention(
            type="allostatic_correction",
            source="Swarm_ECC",
            target="manifold",
            content=swarm_result,
            priority=1.0,  # High priority prevention
            energy_cost=5.0,  # Expensive but cheaper than a crash
        )

        return intervention

    def _check_entropy(self, entropy: float, energy: float) -> bool:
        """
        Check if entropy exceeds threshold.

        Uses Reflexive Closure (if enabled) to get dynamic threshold.
        """
        # Get local context threshold (percentile-based)
        monitor_threshold = self.monitor.get_threshold()

        if self.reflexive_engine:
            # Get global learned threshold (success-based)
            reflexive_threshold = self.reflexive_engine.get_threshold(
                self.latest_cognitive_state[0]
            )

            # Composition: Use the higher of the two.
            # - If local context is noisy (high monitor_threshold), we wait for even higher entropy.
            # - If local context is quiet but learned experience says "don't intervene below X", we wait for X.
            threshold = max(monitor_threshold, reflexive_threshold)

            logger.debug(
                f"Threshold Composition: Monitor={monitor_threshold:.3f}, "
                f"Reflexive={reflexive_threshold:.3f} -> Effective={threshold:.3f}"
            )
        else:
            threshold = monitor_threshold

        return entropy > threshold

    def check_entropy(self, logits: torch.Tensor) -> tuple[bool, float]:
        """
        Check if entropy indicates a clash.

        Args:
            logits: Processor output logits (batch, vocab_size) or (batch, seq_len, vocab_size)

        Returns:
            is_clash: Whether clash was detected
            entropy_value: Computed entropy
        """
        # Get raw entropy and monitor's opinion (which we might override)
        _, entropy_value = self.monitor.check_logits(logits)

        # Re-evaluate using our dynamic logic
        # Use latest known energy
        energy = self.latest_cognitive_state[1]
        is_clash = self._check_entropy(entropy_value, energy)

        return is_clash, entropy_value

    def search(
        self,
        current_state: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None,
        metric: Optional[str] = None,
    ) -> Optional[SearchResult]:
        """
        Search Manifold for alternative goal framing.

        Args:
            current_state: Current goal/state embedding
            context: Optional context information for logging
            k: Optional override for number of neighbors
            metric: Optional override for distance metric

        Returns:
            SearchResult if alternative found, None if search failed
        """
        # Delegate to Hybrid Search Strategy
        # This handles both fast k-NN and slow LTN refinement
        try:
            result = self.hybrid_search.search(
                current_state=current_state,
                evidence=None,  # Director search is usually unsupervised/intrinsic, unless context provides evidence
                constraints=[],
                context=context,
                k=k,
                metric=metric,
            )

            # Log search episode
            if self.config.log_search_episodes and result:
                self._log_search_episode(current_state, result, context)

            return result

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None

    def check_and_search(
        self,
        current_state: torch.Tensor,
        processor_logits: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Main Director loop: monitor -> detect -> compute adaptive beta -> search -> return.

        This is the primary interface used by the RAA integration.

        The Director now dynamically adjusts exploration based on confusion:
        - High entropy (confusion) -> low beta -> more exploratory search
        - Low entropy (confidence) -> high beta -> more focused search

        Args:
            current_state: Current goal state from Pointer
            processor_logits: Output logits from Processor
            context: Optional context (query, history, etc.)

        Returns:
            new_goal: New goal vector if search successful, None if no intervention needed
        """
        # Step 1: Monitor entropy
        is_clash, entropy_value = self.check_entropy(processor_logits)

        # [META-LAYER] Observe current state
        self.observer.observe(
            f"Monitoring entropy: {entropy_value:.3f} (Threshold: {self.monitor.get_threshold():.3f})",
            level=0,
            metadata={"entropy": entropy_value, "is_clash": is_clash},
        )

        # Add entropy to context for logging
        if context is None:
            context = {}
        context["entropy"] = entropy_value
        context["threshold"] = self.monitor.get_threshold()

        # --- COMPASS Resource Allocation (oMCD) ---
        # Map Director state to oMCD state
        # High entropy = Low value difference (hard decision)
        # Precision = 1/variance (approximate)

        # Heuristic mapping:
        # value_difference ~ 1 / (1 + entropy)
        # precision ~ 1.0 (default)

        omcd_state = {
            "value_difference": 1.0 / (1.0 + entropy_value),
            "precision": 1.0,
            "variance": 1.0,
        }

        # Determine allocation
        allocation = self.compass.omcd_controller.determine_resource_allocation(
            current_state=omcd_state,
            importance=10.0,  # Default importance
            available_resources=100.0,  # Default available
        )

        logger.info(
            f"COMPASS oMCD Allocation: {allocation['amount']:.2f} resources (Confidence: {allocation['confidence']:.3f})"
        )
        context["compass_allocation"] = allocation

        # If allocation is very high, we might want to trigger full COMPASS processing
        # For now, we just log it and use it to inform search (potentially)
        if allocation["amount"] > 80.0:
            logger.warning(
                f"High entropy/allocation detected ({allocation['amount']:.2f}). Triggering COMPASS intervention."
            )

            # Trigger COMPASS asynchronously
            try:
                # We need to get the running loop or create a task
                # Since check_and_search is sync, we assume there's an event loop running (e.g. in the server)
                loop = asyncio.get_running_loop()

                task_desc = f"High Entropy Intervention: Entropy={entropy_value:.2f}, Allocation={allocation['amount']:.2f}. Analyze context and intervene."
                # Pass current context copy
                task_ctx = context.copy() if context else {}

                loop.create_task(self.compass.process_task(task_desc, task_ctx))
                logger.info("COMPASS intervention task scheduled.")

            except RuntimeError:
                logger.warning("No running event loop. Skipping async COMPASS trigger.")
            except Exception as e:
                logger.error(f"Failed to trigger COMPASS: {e}")

        # --- Dynamic Agent Escalation ---
        # If entropy is extremely high (e.g. > 2.5) or Sheaf Diagnostics recommend escalation
        # we spawn a specialized agent.
        if entropy_value > 2.5:
            # [META-LAYER] Trigger immediate reflection
            reflection = self.observer.reflect()
            if reflection:
                logger.info(f"Meta-Reflection: {reflection}")
                self._apply_reflection_action(reflection)

            logger.warning(
                f"Critical Entropy ({entropy_value:.2f}). Triggering Dynamic Agent Escalation."
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._escalate_to_agent(
                        signal_type="High Entropy",
                        context=f"Entropy: {entropy_value:.2f}. Processor is confused. Context: {str(context)[:200]}",
                    )
                )
            except RuntimeError:
                logger.warning("No running event loop. Skipping agent escalation.")
            except Exception as e:
                logger.error(f"Failed to trigger agent escalation: {e}")
        # --------------------------------

        # ------------------------------------------

        # Step 2: Check if intervention needed and perform search
        new_goal = None
        if is_clash:
            # [META-LAYER] Observe clash
            self.observer.observe(
                f"Clash detected! Entropy {entropy_value:.3f} exceeds threshold.",
                level=1,
                metadata={"type": "clash"},
            )

            logger.info(
                f"Clash detected! Entropy={entropy_value:.3f} > "
                f"threshold={self.monitor.get_threshold():.3f}"
            )

            # Start Intervention Tracking
            episode_id = None
            if self.reflexive_engine:
                episode_id = self.reflexive_engine.record_intervention_start(
                    episode_id=f"ep_{int(time.time() * 1000)}",
                    entropy=entropy_value,
                    energy=(
                        self.manifold.energy
                        if hasattr(self.manifold, "energy")
                        else self.energy_budget
                    ),  # Real energy from substrate/manifold
                    cognitive_state=self.latest_cognitive_state[0],
                    goal=str(self.current_goal) if hasattr(self, "current_goal") else "Unknown",
                    intervention_type="search",
                    intervention_source="entropy",
                    threshold=self.reflexive_engine.get_threshold(self.latest_cognitive_state[0]),
                )

            # Update Allostatic Controller with current entropy
            self.allostatic_controller.record_entropy(
                entropy_value
            )  # Using entropy_value from monitor

            # 1. Check for Proactive ECC (Allostasis)
            # This is now async, so we assume fire-and-forget in the main loop context,
            # OR we check the trigger synchronously and only spawn if needed.
            # For safety in this sync method, we'll check the trigger synchronously first.

            # Since check_proactive_interventions is now async, in a sync context we should schedule it
            # if we had a loop.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.check_proactive_interventions())
            except RuntimeError:
                pass  # No loop, skip proactive check

            # --- ADAPTIVE BETA LOGIC ---
            # Store original beta to reset it after search
            original_beta = self.manifold.beta

            search_result = None

            try:
                # Step 3: Compute adaptive beta based on confusion (entropy)
                # Let compute_adaptive_beta calculate max_entropy internally
                adaptive_beta = self.manifold.compute_adaptive_beta(
                    entropy=entropy_value, max_entropy=None
                )

                logger.info(
                    f"Setting adaptive beta: {adaptive_beta:.3f} (original: {original_beta:.3f}, "
                    f"entropy: {entropy_value:.3f})"
                )

                # Temporarily set the manifold's beta for this search
                self.manifold.set_beta(adaptive_beta)

                # Add adaptive beta to context for logging
                context["adaptive_beta"] = adaptive_beta
                context["original_beta"] = original_beta

                # --- DYNAMIC PARAMETER RETRIEVAL ---
                search_k = self.config.search_k
                search_metric = self.config.search_metric

                if self.reflexive_engine:
                    # Get dynamic parameters from Reflexive Closure
                    dyn_k = self.reflexive_engine.get_parameter(
                        "search_k", self.latest_cognitive_state[0]
                    )
                    dyn_metric = self.reflexive_engine.get_parameter(
                        "search_metric", self.latest_cognitive_state[0]
                    )

                    if dyn_k is not None:
                        search_k = int(dyn_k)
                    if dyn_metric is not None:
                        search_metric = str(dyn_metric)

                    logger.debug(
                        f"Using dynamic search params: k={search_k}, metric={search_metric}"
                    )
                # -----------------------------------

                # Step 4: Search for alternative using the adaptive beta and dynamic params
                search_result = self.search(
                    current_state, context, k=search_k, metric=search_metric
                )

            except Exception as e:
                logger.error(f"Search with adaptive beta failed: {e}")
                # The 'finally' block will still run to clean up beta
            finally:
                # Step 5: CRITICAL - Reset beta to its original value
                # This ensures the next generation step doesn't use the temporary exploratory beta
                logger.debug(f"Resetting beta to original value: {original_beta:.3f}")
                self.manifold.set_beta(original_beta)

            # --- END ADAPTIVE BETA LOGIC ---

            if search_result is None:
                logger.warning("Search failed to find alternative")
                # Record failure if tracking
                if self.reflexive_engine and episode_id:
                    self.reflexive_engine.record_intervention_end(
                        episode_id=episode_id,
                        entropy_after=entropy_value,
                        energy_after=0.0,
                        task_success=False,
                        outcome_quality=0.0,
                    )
                return None

            # Step 6: Return new goal
            new_goal = search_result.best_pattern

            logger.info(
                f"Search successful. Selected neighbor with score={search_result.selection_score:.3f}"
            )

            # Record success if tracking
            if self.reflexive_engine and episode_id:
                self.reflexive_engine.record_intervention_end(
                    episode_id=episode_id,
                    entropy_after=entropy_value * 0.8,  # Mock reduction
                    energy_after=0.0,
                    task_success=True,
                    outcome_quality=search_result.selection_score,
                )

            # Anchor this milestone (Continuity)
            if self.continuity_service:
                try:
                    # Convert tensor to numpy for continuity service
                    state_np = (
                        new_goal.cpu().numpy() if isinstance(new_goal, torch.Tensor) else new_goal
                    )
                    # Ensure it's 1D or flatten it
                    if len(state_np.shape) > 1:
                        state_np = state_np.flatten()

                    self.continuity_service.add_anchor(
                        agent_id="system",
                        vector=state_np,
                        metadata={
                            "type": "intervention",
                            "trigger": "clash",
                            "entropy": float(entropy_value),
                            "source": "director_search",
                        },
                    )
                    logger.info("Anchored new goal state in Continuity Field.")
                except Exception as e:
                    logger.warning(f"Failed to anchor state: {e}")
        else:
            logger.debug(f"No clash detected (entropy={entropy_value:.3f})")
            return None

        return new_goal

    def _apply_reflection_action(self, action: Dict[str, Any]) -> None:
        """
        Execute a self-modification action triggered by the Recursive Observer.
        """
        action_type = action.get("action_type", "NONE")
        params = action.get("parameters", {})

        logger.info(f"Applying Meta-Action: {action_type} with params {params}")

        if action_type == "SWITCH_STRATEGY":
            # Example: Force a specific search metric or k
            if "metric" in params:
                self.config.search_metric = params["metric"]
                logger.info(f"Switched search metric to {params['metric']}")
            if "k" in params:
                self.config.search_k = int(params["k"])
                logger.info(f"Switched search k to {params['k']}")

        elif action_type == "ADJUST_THRESHOLD":
            # Example: Adjust entropy threshold
            if "multiplier" in params:
                current = self.monitor.get_threshold()
                new_threshold = current * float(params["multiplier"])
                # We can't easily set the threshold directly on the monitor without a setter
                # But we can add a manual override or adjust the base
                # For now, let's just log it as a proof of concept
                logger.info(f"Requested threshold adjustment: {current} -> {new_threshold}")

        elif action_type == "TRIGGER_SLEEP":
            logger.info("Meta-Layer requested sleep cycle. (Not yet implemented)")

    def get_remedial_action(self, state: str, energy: float, entropy: float) -> Dict[str, Any]:
        """
        Determine the best remedial action based on cognitive state.

        This replaces hardcoded logic in server.py with a centralized,
        potentially adaptive decision.
        """
        warnings = []
        advice = "Continue current line of reasoning."

        # 1. State-based Logic
        if state == "Looping":
            warnings.append(f"WARNING: Agent is in a '{state}' state.")
            advice = "Stop. Use 'diagnose_pointer' or 'hypothesize' to break the loop."
        elif state == "Confused":
            warnings.append(f"WARNING: Agent is in a '{state}' state.")
            advice = "High entropy detected. Use 'deconstruct' to break down the problem."
        elif state == "Scattered":
            warnings.append(f"WARNING: Agent is in a '{state}' state.")
            advice = "Focus required. Use 'synthesize' to merge scattered thoughts or 'take_nap' to consolidate."

        # 2. Energy-based Logic (Stability)
        if energy > -0.8 and state != "Unknown":
            warnings.append("Note: State is unstable (high energy).")
            if state == "Flow":
                advice = "Flow state is fragile. Proceed with caution."

        # 3. Entropy-based Logic (Uncertainty)
        if entropy > 0.8:
            warnings.append(f"High Entropy ({entropy:.2f}).")
            if "deconstruct" not in advice:
                advice += " Consider 'deconstruct' or 'consult_compass'."

        return {
            "warnings": warnings,
            "advice": advice,
            "suggested_tools": [],  # Could be populated dynamically
        }

    def diagnose(
        self,
        weights: list[torch.Tensor],
        target_error: Optional[torch.Tensor] = None,
        feedback_weights: Optional[list[torch.Tensor]] = None,
    ) -> SheafDiagnostics:
        """
        Perform sheaf-theoretic diagnosis of the network.

        Args:
            weights: Network weights
            target_error: Optional target error
            feedback_weights: Optional feedback weights

        Returns:
            SheafDiagnostics result
        """
        return self.sheaf_analyzer.full_diagnosis(weights, target_error, feedback_weights)

    def monitor_thought_process(self, attention_weights: torch.Tensor) -> tuple[str, float]:
        """
        Monitor the 'topology' of thought by analyzing attention matrices.
        Returns the cognitive state label and its energy (stability).
        """
        # Store raw weights for potential feedback/teaching
        self.latest_attention_weights = attention_weights

        state, energy, diagnostics = self.matrix_monitor.check_state(attention_weights)
        self.latest_cognitive_state = (state, energy)
        self.latest_diagnostics = diagnostics
        logger.debug(f"Director monitored thought process. State: {state}")
        return state, energy

    def teach_state(self, label: str) -> bool:
        """
        Teach the Director that the *last* monitored thought corresponds to 'label'.
        """
        if self.latest_attention_weights is None:
            logger.warning("Cannot teach state: No recent thought process recorded.")
            return False

        self.matrix_monitor.register_state(self.latest_attention_weights, label)
        logger.info(f"Director learned new state: {label}")
        return True

    def get_known_states(self) -> Dict[int, str]:
        """List all known cognitive states."""
        return self.matrix_monitor.state_labels.copy()

    def get_adaptive_temperature(self) -> float:
        """
        Calculate adaptive temperature based on cognitive energy.

        Logic:
        - Low Energy (Stable) -> Low Temperature (Exploitation)
        - High Energy (Unstable) -> High Temperature (Exploration)
        """
        _, energy = self.latest_cognitive_state

        # Map Energy (-5.0 to 0.0) to (min_temp, max_temp)
        # Energy is typically -5 (very stable) to 0 (very unstable)
        # We want a sigmoid-like mapping or linear clamping

        # Clamp energy to expected range
        clamped_energy = max(-5.0, min(0.0, float(energy)))

        # Normalize to 0.0 (stable) - 1.0 (unstable)
        # -5 -> 0.0
        # 0 -> 1.0
        normalized_instability = (clamped_energy + 5.0) / 5.0

        # Map to temp range
        min_t = self.config.adaptive_temp_min
        max_t = self.config.adaptive_temp_max

        temperature = min_t + (max_t - min_t) * normalized_instability

        logger.debug(f"Adaptive Temperature: {temperature:.2f} (Energy: {energy:.2f})")
        return temperature

    def visualize_last_thought(self) -> str:
        """Get ASCII visualization of the last thought."""
        if self.latest_attention_weights is None:
            return "No recent thought to visualize."
        return self.matrix_monitor.visualize_topology(self.latest_attention_weights)

    def _log_search_episode(
        self,
        current_state: torch.Tensor,
        result: SearchResult,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Log search episode for analysis and future policy learning."""
        episode = {
            "current_state": current_state.detach().cpu(),
            "new_state": result.best_pattern.detach().cpu(),
            "neighbor_indices": result.neighbor_indices,
            "distances": result.neighbor_distances.detach().cpu(),
            "selection_score": result.selection_score,
            "entropy": context.get("entropy") if context else None,
            "threshold": context.get("threshold") if context else None,
            "adaptive_beta": context.get("adaptive_beta") if context else None,
            "original_beta": context.get("original_beta") if context else None,
        }

        self.search_episodes.append(episode)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get Director statistics.

        Returns:
            Dictionary with entropy stats and search episode count
        """
        entropy_stats = self.monitor.get_statistics()

        return {
            "entropy": entropy_stats,
            "num_search_episodes": len(self.search_episodes),
            "config": {
                "search_k": self.config.search_k,
                "metric": self.config.search_metric,
                "threshold_percentile": self.config.entropy_threshold_percentile,
            },
            "sheaf_diagnostics": {
                "h1_threshold": self.sheaf_analyzer.config.h1_escalation_threshold,
                "overlap_threshold": self.sheaf_analyzer.config.overlap_warning_threshold,
            },
        }

    def reset(self) -> None:
        """Reset Director state."""
        self.monitor.reset()
        self.search_episodes.clear()

    async def _escalate_to_agent(self, signal_type: str, context: str) -> None:
        """
        Async helper to spawn and execute a specialized agent or Swarm.
        """
        try:
            logger.info(f"Escalating to specialized agent for {signal_type}...")

            # If High Entropy, trigger Swarm
            if signal_type == "High Entropy":
                logger.info("CRITICAL ENTROPY -> Triggering SWARM CONSENSUS.")
                advisors = ["linearist", "periodicist", "evolutionist", "thermodynamicist"]
                synthesis = await self.swarm_controller.run_swarm(
                    task=f"Resolve High Entropy Clash. Context: {context}",
                    advisor_ids=advisors,
                    context=context,
                )
                logger.info(f"Swarm Synthesis Result:\n{synthesis}")
                # TODO: Apply synthesis to goal?

            else:
                # Default Single Agent Escalation
                # 1. Spawn Agent (Generates Persona)
                tool_name = await self.agent_factory.spawn_agent(signal_type, context)

                # 2. Execute Agent
                query = f"The system is stuck with {signal_type}. Please analyze the context and provide a new goal or framing to resolve the obstruction.\nContext: {context}"

                result = await self.agent_factory.execute_agent(tool_name, {"query": query})

                logger.info(f"Specialized Agent {tool_name} Result:\n{result}")

        except Exception as e:
            logger.error(f"Agent escalation failed: {e}")

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"DirectorMVP(threshold={stats['entropy']['threshold']:.3f}, "
            f"searches={stats['num_search_episodes']})"
        )

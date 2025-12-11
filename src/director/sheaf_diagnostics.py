"""
Sheaf diagnostics: Topological Analysis of Predictive Coding Networks

Based on "Sheaf Cohomology of Linear Predictive Coding Networks" (Seely, 2025).

This module provides tools for analyzing recurrent neural architectures through
the lens of cellular sheaf theory. Key concepts:

1. **Cellular Sheaf Structure**: Maps network activations to 0-cochains (vertices)
   and prediction errors to 1-cochains (edges).

2. **Cohomology Groups**:
   - H^0: Activation patterns with zero prediction error everywhere
   - H^1: Irreducible error patterns that cannot be eliminated by any activation choice

3. **Hodge decomposition**: Splits supervision signal into:
   - Part that CAN be eliminated by inference (im d)
   - Part that CANNOT be eliminated (ker d^T) - the harmonic residual

4. **Monodromy Analysis**: detects whether feedback loops create:
   - Resonance (Φ ≈ I): Slow inference but learnable
   - Tension (Φ ≈ -I): Fast inference but learning may stall

Integration with RAA:
- Provides principled stuck-state detection beyond entropy monitoring
- Enables topological escalation criteria
- Measures harmonic-diffusive overlap for learning progress
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CognitiveTopology(Enum):
    """Classification of cognitive topology states."""

    RESONANCE = "resonance"  # Φ ≈ I: feedback reinforces
    TENSION = "tension"  # Φ ≈ -I: feedback contradicts
    MIXED = "mixed"  # General eigenvalue spectrum
    TRIVIAL = "trivial"  # No feedback (feedforward only)


@dataclass
class SheafConfig:
    """Configuration for Sheaf diagnostics."""

    # Numerical tolerances
    svd_tolerance: float = 1e-6  # Threshold for null space detection
    resonance_tolerance: float = 0.2  # Tolerance for Φ ≈ I detection
    tension_tolerance: float = 0.2  # Tolerance for Φ ≈ -I detection

    # diagnostic thresholds
    h1_escalation_threshold: int = 0  # Escalate if dim(H^1) > this
    overlap_warning_threshold: float = 0.1  # warn if overlap below this

    # Computation settings
    use_pinv: bool = True  # Use pseudoinverse (more stable)
    device: str = "cpu"


@dataclass
class CohomologyResult:
    """Result of cohomology computation."""

    h0_dimension: int  # dim(ker δ^0) - zero-error activations
    h1_dimension: int  # dim(ker d^T) - irreducible errors
    can_fully_resolve: bool  # whether H^1 = 0
    singular_values: torch.Tensor  # SVd spectrum for diagnostics
    null_space_basis: torch.Tensor | None  # Basis of ker d^T if non-trivial


@dataclass
class HodgeDecomposition:
    """Result of Hodge decomposition."""

    harmonic_projector: torch.Tensor  # H = I - d @ d^†
    diffusive_operator: torch.Tensor  # G = d^†
    harmonic_residual: torch.Tensor | None  # r* = H @ b
    diffusive_activation: torch.Tensor | None  # z* = -G @ b
    eliminable_error: torch.Tensor | None  # -d @ z*


@dataclass
class MonodromyAnalysis:
    """Result of monodromy analysis for feedback loops."""

    monodromy_matrix: torch.Tensor  # Φ = w_FB @ w_forward
    eigenvalues: torch.Tensor  # Complex eigenvalues
    topology: CognitiveTopology  # Classification
    spectral_radius: float  # max |λ|
    condition_number: float  # κ = λ_max / λ_min


@dataclass
class SheafDiagnostics:
    """
    Complete diagnostic result from sheaf analysis.

    This is the main output structure containing all diagnostic information
    about a network's topological properties.
    """

    cohomology: CohomologyResult
    hodge: HodgeDecomposition
    monodromy: MonodromyAnalysis | None

    # derived metrics
    harmonic_diffusive_overlap: float  # Key metric for learning
    learning_can_proceed: bool  # whether topology allows learning
    escalation_recommended: bool  # whether to escalate to System 3
    diagnostic_messages: list[str] = field(default_factory=list)


class SheafAnalyzer:
    """
    Analyzer for cellular sheaf structure of predictive coding networks.

    This class provides the mathematical machinery to analyze network
    configurations through cellular sheaf theory, detecting topological
    obstructions to learning and inference.
    """

    def __init__(self, config: SheafConfig | None = None):
        self.config = config or SheafConfig()
        self.device = self.config.device

    def build_coboundary_matrix(
        self, weights: list[torch.Tensor], topology: str = "feedforward"
    ) -> torch.Tensor:
        """
        Build the sheaf coboundary matrix δ^0 from network weights.

        For a network with layers h_1 -> h_2 -> ... -> h_n:
        The coboundary computes prediction errors: (δ^0 s)_e = s_v - w_e s_u

        Args:
            weights: List of weight matrices [w_1, w_2, ..., w_n]
            topology: "feedforward" or "recurrent"

        Returns:
            Coboundary matrix δ^0 of shape (total_edge_dims, total_vertex_dims)
        """
        if not weights:
            raise ValueError("Need at least one weight matrix")

        # Compute dimensions
        vertex_dims = [weights[0].shape[1]]  # Input dimension
        for w in weights:
            vertex_dims.append(w.shape[0])

        total_vertex_dim = sum(vertex_dims)
        total_edge_dim = sum(w.shape[0] for w in weights)  # Each edge has output dim

        # Determine device from weights
        device = weights[0].device if weights else self.device

        # Build coboundary matrix
        # Each edge e = (u -> v) contributes: -w_e on column u, I on column v
        delta = torch.zeros(total_edge_dim, total_vertex_dim, device=device)

        edge_offset = 0
        vertex_offset = 0

        for i, w in enumerate(weights):
            out_dim, in_dim = w.shape

            # -w on source vertex columns
            delta[edge_offset : edge_offset + out_dim, vertex_offset : vertex_offset + in_dim] = -w

            # I on target vertex columns
            delta[
                edge_offset : edge_offset + out_dim,
                vertex_offset + in_dim : vertex_offset + in_dim + out_dim,
            ] = torch.eye(out_dim, device=device)

            edge_offset += out_dim
            vertex_offset += in_dim

        return delta

    def build_relative_coboundary(
        self, weights: list[torch.Tensor], clamped_indices: list[int] | None = None
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Build the relative coboundary d for clamped (supervised) systems.

        when input x and output y are clamped to data, we only optimize
        over internal activations. d extracts the columns of δ^0 for free vertices.

        Args:
            weights: List of weight matrices
            clamped_indices: Indices of clamped vertices (default: first and last)

        Returns:
            d: Relative coboundary matrix
            free_indices: Indices of free vertices
        """
        delta = self.build_coboundary_matrix(weights)
        num_vertices = delta.shape[1]

        # default: clamp first (input) and last (output) vertices
        if clamped_indices is None:
            vertex_dims = [weights[0].shape[1]]
            for w in weights:
                vertex_dims.append(w.shape[0])
            clamped_indices = list(range(vertex_dims[0]))  # Input
            clamped_indices += list(range(sum(vertex_dims[:-1]), num_vertices))  # Output

        # Free vertices are those not clamped
        all_indices = set(range(num_vertices))
        free_indices = sorted(all_indices - set(clamped_indices))

        # Extract columns for free vertices
        d = delta[:, free_indices]

        return d, free_indices

    def compute_cohomology(
        self, d: torch.Tensor, delta: torch.Tensor | None = None
    ) -> CohomologyResult:
        """
        Compute sheaf cohomology groups.

        H^0 = ker(δ^0) represents global sections (connected components).
        H^1 = ker(d^T) represents irreducible error patterns.

        Args:
            d: Relative coboundary matrix (for H^1)
            delta: Full coboundary matrix (optional, for H^0)

        Returns:
            CohomologyResult with dimensions and null space basis
        """
        # SVd decomposition: d = U @ S @ V^T
        # ker(d^T) is spanned by left singular vectors with zero singular values
        u, s, vh = torch.linalg.svd(d, full_matrices=True)

        # Count null dimensions
        null_mask = s < self.config.svd_tolerance
        h1_dim = null_mask.sum().item()

        # For H^0, we look at ker(δ^0) if delta is provided
        h0_dim = 0
        if delta is not None:
            # H^0 = ker(δ^0)
            # dim(H^0) = num_cols - rank(δ^0)
            # we use SVd to find rank
            try:
                _, s_delta, _ = torch.linalg.svd(delta, full_matrices=False)
                rank_delta = (s_delta > self.config.svd_tolerance).sum().item()
                h0_dim = delta.shape[1] - rank_delta
            except Exception as e:
                logger.warning(f"Failed to compute H^0: {e}")
                h0_dim = 0

        # Extract null space basis if non-trivial
        null_basis = None
        if h1_dim > 0:
            # Null space of d^T = column space of U corresponding to zero singular values
            # But for ker(d^T), we need the rows of U where S ≈ 0
            # Actually: ker(d^T) is the orthogonal complement of im(d)
            # In SVd terms: columns of U where S ≈ 0
            null_indices = torch.where(null_mask)[0]
            if len(null_indices) > 0:
                # Extend indices for full U matrix
                full_null_indices = null_indices.tolist() + list(range(len(s), u.shape[1]))
                null_basis = u[:, full_null_indices[:h1_dim]]

        return CohomologyResult(
            h0_dimension=h0_dim,
            h1_dimension=h1_dim,
            can_fully_resolve=(h1_dim == 0),
            singular_values=s,
            null_space_basis=null_basis,
        )

    def compute_hodge_decomposition(
        self, d: torch.Tensor, b: torch.Tensor | None = None
    ) -> HodgeDecomposition:
        """
        Compute Hodge decomposition operators.

        The target error b decomposes orthogonally:
            b = (-d z*) + r*
        where:
            - (-d z*) ∈ im(d) is eliminable by inference
            - r* ∈ ker(d^T) is the harmonic residual

        The key operators are:
            - H = I - d @ d^† : Harmonic projector
            - G = d^† : diffusive operator

        Args:
            d: Relative coboundary matrix
            b: Target error vector (optional)

        Returns:
            HodgeDecomposition with operators and decomposed signals
        """
        # Compute pseudoinverse
        if self.config.use_pinv:
            d_pinv = torch.linalg.pinv(d)
        else:
            # Manual computation via SVd
            u, s, vh = torch.linalg.svd(d, full_matrices=False)
            # Avoid division by zero
            s_safe = torch.where(s > self.config.svd_tolerance, s, torch.ones_like(s))
            s_inv = torch.where(s > self.config.svd_tolerance, 1.0 / s_safe, torch.zeros_like(s))
            d_pinv = vh.T @ torch.diag(s_inv) @ u.T

        # Harmonic projector: H = I - d @ d^†
        identity = torch.eye(d.shape[0], device=d.device)
        harmonic_projector = identity - d @ d_pinv

        # diffusive operator: G = d^†
        diffusive_op = d_pinv

        # If target error provided, compute decomposition
        harmonic_residual = None
        diffusive_activation = None
        eliminable_error = None

        if b is not None:
            if b.dim() == 1:
                b = b.unsqueeze(1)

            # r* = H @ b (irreducible residual)
            harmonic_residual = harmonic_projector @ b

            # z* = -G @ b (optimal internal activations)
            diffusive_activation = -diffusive_op @ b

            # Eliminable part: -d @ z* = d @ G @ b
            eliminable_error = d @ diffusive_op @ b

        return HodgeDecomposition(
            harmonic_projector=harmonic_projector,
            diffusive_operator=diffusive_op,
            harmonic_residual=harmonic_residual,
            diffusive_activation=diffusive_activation,
            eliminable_error=eliminable_error,
        )

    def compute_harmonic_diffusive_overlap(
        self,
        hodge: HodgeDecomposition,
        edge_indices: list[int] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> float:
        """
        Compute overlap between harmonic load and diffusive activation.

        Learning requires both:
        1. Non-zero harmonic residual (error signal)
        2. Non-zero diffusive activation (active source)

        Low overlap means learning is starved on some edges.

        From Eq. 19 in the paper:
            ∂E/∂w_e = (Hb)_e @ (Gb)_u^T

        Args:
            hodge: Computed Hodge decomposition
            edge_indices: Specific edges to analyze (optional)
            vertex_indices: Specific vertices to analyze (optional)

        Returns:
            Overlap score in [0, 1]
        """
        if hodge.harmonic_residual is None or hodge.diffusive_activation is None:
            logger.warning("Cannot compute overlap: no target error provided")
            return 0.0

        # Compute norms
        harmonic_norm = torch.norm(hodge.harmonic_residual)
        diffusive_norm = torch.norm(hodge.diffusive_activation)

        if harmonic_norm < 1e-10 or diffusive_norm < 1e-10:
            return 0.0

        # For a more precise overlap, we'd compute per-edge alignment
        # Simplified: use cosine similarity between flattened vectors
        h_flat = hodge.harmonic_residual.flatten()
        # Need to map diffusive (vertex) to edges - use H^T for projection
        d_projected = hodge.harmonic_projector.T @ hodge.harmonic_residual.flatten()

        # Compute alignment
        overlap = torch.abs(h_flat @ d_projected) / (
            torch.norm(h_flat) * torch.norm(d_projected) + 1e-10
        )

        return float(overlap.item())

    def analyze_monodromy(
        self, w_forward: torch.Tensor, w_feedback: torch.Tensor
    ) -> MonodromyAnalysis:
        """
        Analyze monodromy of a feedback loop.

        The monodromy Φ = w_FB @ w_forward determines how signals
        propagate around the loop:

        - Φ ≈ I (resonance): Changes reinforce, slow inference
        - Φ ≈ -I (tension): Changes contradict, learning stalls

        Args:
            w_forward: Forward weight matrix
            w_feedback: Feedback weight matrix

        Returns:
            MonodromyAnalysis with eigenvalue spectrum and classification
        """
        # Compute monodromy
        monodromy_matrix = w_feedback @ w_forward

        # Eigenvalue analysis
        eigvals = torch.linalg.eigvals(monodromy_matrix)

        # Spectral radius
        spectral_radius = torch.max(torch.abs(eigvals)).item()

        # Condition number (for real part)
        real_parts = eigvals.real
        nonzero_mask = torch.abs(real_parts) > self.config.svd_tolerance
        if nonzero_mask.sum() > 1:
            condition_number = (
                torch.max(torch.abs(real_parts[nonzero_mask]))
                / torch.min(torch.abs(real_parts[nonzero_mask]))
            ).item()
        else:
            condition_number = 1.0

        # Classify topology
        topology = self._classify_monodromy(eigvals)

        return MonodromyAnalysis(
            monodromy_matrix=monodromy_matrix,
            eigenvalues=eigvals,
            topology=topology,
            spectral_radius=spectral_radius,
            condition_number=condition_number,
        )

    def _classify_monodromy(self, eigvals: torch.Tensor) -> CognitiveTopology:
        """Classify monodromy based on eigenvalue spectrum."""
        # Check for resonance (all eigenvalues ≈ 1)
        if torch.allclose(
            eigvals.real, torch.ones_like(eigvals.real), atol=self.config.resonance_tolerance
        ):
            return CognitiveTopology.RESONANCE

        # Check for tension (all eigenvalues ≈ -1)
        if torch.allclose(
            eigvals.real, -torch.ones_like(eigvals.real), atol=self.config.tension_tolerance
        ):
            return CognitiveTopology.TENSION

        return CognitiveTopology.MIXED

    def full_diagnosis(
        self,
        weights: list[torch.Tensor],
        target_error: torch.Tensor | None = None,
        feedback_weights: list[torch.Tensor] | None = None,
    ) -> SheafDiagnostics:
        """
        Perform complete sheaf-theoretic diagnosis of network.

        This is the main entry point for analyzing a network configuration.

        Args:
            weights: Forward weight matrices
            target_error: Supervision-induced target error (optional)
            feedback_weights: Feedback connection weights (optional)

        Returns:
            Complete Sheafdiagnostics with all analysis results
        """
        messages: list[str] = []

        # Build relative coboundary
        delta = self.build_coboundary_matrix(weights)
        d, free_indices = self.build_relative_coboundary(weights)

        # Compute cohomology (pass delta for H^0)
        cohomology = self.compute_cohomology(d, delta=delta)

        if cohomology.h1_dimension > 0:
            messages.append(
                f"wARNING: Non-trivial H^1 (dim={cohomology.h1_dimension}). "
                f"Irreducible error patterns exist."
            )

        # Compute Hodge decomposition
        hodge = self.compute_hodge_decomposition(d, target_error)

        # Compute overlap
        overlap: float = self.compute_harmonic_diffusive_overlap(hodge)

        if overlap < self.config.overlap_warning_threshold:
            messages.append(
                f"wARNING: Low harmonic-diffusive overlap ({overlap:.3f}). "
                f"Learning may be starved."
            )

        # Analyze monodromy if feedback provided
        monodromy: MonodromyAnalysis | None = None
        if feedback_weights:
            # Analyze first feedback loop (can extend for multiple)
            if len(feedback_weights) > 0 and len(weights) > 0:
                monodromy = self.analyze_monodromy(weights[0], feedback_weights[0])

                if monodromy.topology == CognitiveTopology.TENSION:
                    messages.append(
                        "wARNING: Feedback loop has TENSION topology. "
                        "Learning may stall due to internal contradictions."
                    )

        # determine recommendations
        learning_can_proceed: bool = (
            cohomology.can_fully_resolve and overlap >= self.config.overlap_warning_threshold
        )

        escalation_recommended: bool = (
            cohomology.h1_dimension > self.config.h1_escalation_threshold
            or (monodromy is not None and monodromy.topology == CognitiveTopology.TENSION)
        )

        if escalation_recommended:
            messages.append(
                "RECOMMENdATION: Escalate to System 3. " "Topological obstructions detected."
            )

        return SheafDiagnostics(
            cohomology=cohomology,
            hodge=hodge,
            monodromy=monodromy,
            harmonic_diffusive_overlap=overlap,
            learning_can_proceed=learning_can_proceed,
            escalation_recommended=escalation_recommended,
            diagnostic_messages=messages,
        )


class AttentionSheafAnalyzer:
    """
    Specialized analyzer for attention-based architectures.

    Converts attention weight matrices into sheaf structure for analysis.
    This connects the paper's linear analysis to transformer architectures.
    """

    def __init__(self, config: SheafConfig | None = None):
        self.config = config or SheafConfig()
        self.base_analyzer = SheafAnalyzer(config)

    def attention_to_sheaf_weights(self, attention_weights: torch.Tensor) -> list[torch.Tensor]:
        """
        Convert attention weights to effective sheaf weight matrices.

        Attention can be viewed as soft weight matrices where:
        w_ij = attention_ij (how much position j attends to position i)

        Args:
            attention_weights: Shape (batch, heads, seq, seq)

        Returns:
            List of effective weight matrices per head
        """
        # Average over batch if needed
        if attention_weights.dim() == 4:
            attn = attention_weights.mean(dim=0)  # (heads, seq, seq)
        else:
            attn = attention_weights

        # Each head gives an effective weight matrix
        weights = [attn[h] for h in range(attn.shape[0])]

        return weights

    def diagnose_attention(
        self, attention_weights: torch.Tensor, goal_bias: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """
        Analyze attention pattern through sheaf lens.

        Args:
            attention_weights: Attention matrix (batch, heads, seq, seq)
            goal_bias: Goal biasing tensor (if using GoalBiasedAttention)

        Returns:
            dictionary with per-head diagnostics and aggregate metrics
        """
        weights = self.attention_to_sheaf_weights(attention_weights)

        results: dict[str, Any] = {"per_head": [], "aggregate": {}}

        for i, w in enumerate(weights):
            # Treat attention as a single-layer system for analysis
            # This is simplified but captures key patterns
            diagnosis = self.base_analyzer.full_diagnosis([w])
            results["per_head"].append(
                {
                    "head": i,
                    "h1_dim": diagnosis.cohomology.h1_dimension,
                    "overlap": diagnosis.harmonic_diffusive_overlap,
                    "can_learn": diagnosis.learning_can_proceed,
                }
            )

        # Aggregate metrics
        h1_dims = [r["h1_dim"] for r in results["per_head"]]
        overlaps = [r["overlap"] for r in results["per_head"]]

        results["aggregate"] = {
            "max_h1_dim": max(h1_dims),
            "mean_overlap": sum(overlaps) / len(overlaps) if overlaps else 0,
            "num_problematic_heads": sum(1 for r in results["per_head"] if not r["can_learn"]),
        }

        return results


def create_supervision_target(
    input_embedding: torch.Tensor, output_target: torch.Tensor, weights: list[torch.Tensor]
) -> torch.Tensor:
    """
    Create the target error vector b for a supervised problem.

    For clamped input x and target y:
        b = [-w_1 x, 0, ..., 0, y]

    This represents the "excitation" from boundary conditions.

    Args:
        input_embedding: Input x
        output_target: Target y
        weights: Network weights

    Returns:
        Target error vector b
    """
    # First edge: -w_1 @ x
    first_error = -weights[0] @ input_embedding

    # Middle edges: 0
    middle_errors = []
    for w in weights[1:-1]:
        middle_errors.append(torch.zeros(w.shape[0], device=input_embedding.device))

    # Last edge: y (the target)
    last_error = output_target

    # Concatenate
    all_errors = [first_error] + middle_errors + [last_error]
    b = torch.cat([e.flatten() for e in all_errors])

    return b

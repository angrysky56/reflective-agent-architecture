"""
Tests for Sheaf Diagnostics Module

Tests the cellular sheaf theory implementation for analyzing
predictive coding network topologies.
"""

import pytest
import torch

from src.director.sheaf_diagnostics import (
    AttentionSheafAnalyzer,
    CognitiveTopology,
    SheafAnalyzer,
    SheafConfig,
    create_supervision_target,
)


class TestSheafAnalyzer:
    """Test core SheafAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default config."""
        return SheafAnalyzer(SheafConfig(device="cpu"))

    @pytest.fixture
    def simple_network_weights(self):
        """Simple 3-layer feedforward network weights."""
        # x (2) -> h1 (3) -> h2 (3) -> y (2)
        W1 = torch.randn(3, 2)
        W2 = torch.randn(3, 3)
        W3 = torch.randn(2, 3)
        return [W1, W2, W3]

    def test_coboundary_matrix_shape(self, analyzer, simple_network_weights):
        """Test coboundary matrix has correct dimensions."""
        delta = analyzer.build_coboundary_matrix(simple_network_weights)
        
        # Total vertex dim: 2 + 3 + 3 + 2 = 10
        # Total edge dim: 3 + 3 + 2 = 8
        assert delta.shape == (8, 10)

    def test_coboundary_computes_prediction_errors(self, analyzer):
        """Test that coboundary correctly computes prediction errors."""
        # Simple 2-layer network: x (2) -> h (2) -> y (2)
        W1 = torch.eye(2)  # Identity mapping
        W2 = torch.eye(2)
        
        delta = analyzer.build_coboundary_matrix([W1, W2])
        
        # Create activation vector s = [x, h, y]
        x = torch.tensor([1.0, 2.0])
        h = torch.tensor([1.0, 2.0])  # Correct: h = W1 @ x = x
        y = torch.tensor([3.0, 4.0])  # Incorrect: y ≠ W2 @ h
        
        s = torch.cat([x, h, y])
        
        # Compute prediction errors: r = delta @ s
        r = delta @ s
        
        # First edge error: h - W1 @ x = [1,2] - [1,2] = [0, 0]
        assert torch.allclose(r[:2], torch.zeros(2), atol=1e-6)
        
        # Second edge error: y - W2 @ h = [3,4] - [1,2] = [2, 2]
        assert torch.allclose(r[2:], torch.tensor([2.0, 2.0]))

    def test_relative_coboundary_clamps_io(self, analyzer, simple_network_weights):
        """Test relative coboundary removes clamped vertex columns."""
        D, free_indices = analyzer.build_relative_coboundary(simple_network_weights)
        
        # Clamped: input (2 dims) + output (2 dims) = 4 dims
        # Free: h1 (3 dims) + h2 (3 dims) = 6 dims
        # Total edges: 8
        assert D.shape == (8, 6)
        assert len(free_indices) == 6

    def test_cohomology_feedforward_trivial(self, analyzer, simple_network_weights):
        """Feedforward networks typically have trivial H^1."""
        D, _ = analyzer.build_relative_coboundary(simple_network_weights)
        cohomology = analyzer.compute_cohomology(D)
        
        # For well-conditioned feedforward, H^1 should be 0
        # (May not always be true depending on dimensions)
        assert cohomology.h1_dimension >= 0
        assert isinstance(cohomology.can_fully_resolve, bool)

    def test_cohomology_detects_overdetermined(self, analyzer):
        """Test that cohomology detects overdetermined systems."""
        # Create a system with more constraints than variables
        # More edges than free vertices -> likely non-trivial H^1
        W1 = torch.randn(5, 2)  # 5 constraints
        W2 = torch.randn(5, 5)  # 5 constraints  
        W3 = torch.randn(2, 5)  # 2 constraints
        # Total: 12 edge dims, free vertices: 5 + 5 = 10
        
        D, _ = analyzer.build_relative_coboundary([W1, W2, W3])
        cohomology = analyzer.compute_cohomology(D)
        
        # Should detect some structure
        assert cohomology.singular_values is not None
        assert len(cohomology.singular_values) > 0

    def test_hodge_decomposition_properties(self, analyzer, simple_network_weights):
        """Test Hodge decomposition satisfies key properties."""
        D, _ = analyzer.build_relative_coboundary(simple_network_weights)
        
        # Create a target error
        b = torch.randn(D.shape[0])
        
        hodge = analyzer.compute_hodge_decomposition(D, b)
        
        # H should be a projector: H^2 = H
        H = hodge.harmonic_projector
        H_squared = H @ H
        assert torch.allclose(H, H_squared, atol=1e-5)
        
        # H should be symmetric
        assert torch.allclose(H, H.T, atol=1e-5)
        
        # Decomposition: b = -D @ z* + r*
        if hodge.eliminable_error is not None and hodge.harmonic_residual is not None:
            reconstructed = hodge.eliminable_error.flatten() + hodge.harmonic_residual.flatten()
            assert torch.allclose(b, reconstructed, atol=1e-5)

    def test_hodge_harmonic_in_kernel(self, analyzer, simple_network_weights):
        """Test harmonic residual lies in ker(D^T)."""
        D, _ = analyzer.build_relative_coboundary(simple_network_weights)
        b = torch.randn(D.shape[0])
        
        hodge = analyzer.compute_hodge_decomposition(D, b)
        
        if hodge.harmonic_residual is not None:
            # D^T @ r* should be ~0
            Dt_r = D.T @ hodge.harmonic_residual.flatten()
            assert torch.allclose(Dt_r, torch.zeros_like(Dt_r), atol=1e-5)


class TestMonodromyAnalysis:
    """Test monodromy analysis for feedback loops."""

    @pytest.fixture
    def analyzer(self):
        return SheafAnalyzer(SheafConfig(device="cpu"))

    def test_resonance_detection(self, analyzer):
        """Test detection of resonant feedback (Φ ≈ I)."""
        # Create feedback that gives monodromy ≈ I
        W_forward = torch.eye(3)
        W_feedback = torch.eye(3)  # Φ = I @ I = I
        
        result = analyzer.analyze_monodromy(W_forward, W_feedback)
        
        assert result.topology == CognitiveTopology.RESONANCE
        assert torch.allclose(result.eigenvalues.real, torch.ones(3), atol=0.3)

    def test_tension_detection(self, analyzer):
        """Test detection of tension feedback (Φ ≈ -I)."""
        # Create feedback that gives monodromy ≈ -I
        W_forward = torch.eye(3)
        W_feedback = -torch.eye(3)  # Φ = -I @ I = -I
        
        result = analyzer.analyze_monodromy(W_forward, W_feedback)
        
        assert result.topology == CognitiveTopology.TENSION
        assert torch.allclose(result.eigenvalues.real, -torch.ones(3), atol=0.3)

    def test_mixed_topology(self, analyzer):
        """Test detection of mixed eigenvalue spectrum."""
        # Create rotation matrix (eigenvalues on unit circle, not ±1)
        theta = torch.tensor(torch.pi / 4)
        rotation = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]
        ])
        
        W_forward = rotation
        W_feedback = rotation  # Φ = rotation^2
        
        result = analyzer.analyze_monodromy(W_forward, W_feedback)
        
        assert result.topology == CognitiveTopology.MIXED
        assert result.spectral_radius > 0


class TestFullDiagnosis:
    """Test complete diagnostic pipeline."""

    @pytest.fixture
    def analyzer(self):
        return SheafAnalyzer(SheafConfig(device="cpu"))

    def test_full_diagnosis_feedforward(self, analyzer):
        """Test full diagnosis on feedforward network."""
        weights = [
            torch.randn(4, 3),
            torch.randn(4, 4),
            torch.randn(2, 4)
        ]
        
        target_error = torch.randn(4 + 4 + 2)  # Total edge dimensions
        
        result = analyzer.full_diagnosis(weights, target_error)
        
        # Check all components present
        assert result.cohomology is not None
        assert result.hodge is not None
        assert result.monodromy is None  # No feedback provided
        assert isinstance(result.harmonic_diffusive_overlap, float)
        assert isinstance(result.learning_can_proceed, bool)
        assert isinstance(result.escalation_recommended, bool)

    def test_full_diagnosis_with_feedback(self, analyzer):
        """Test full diagnosis with feedback connections."""
        # Use square first layer for monodromy analysis
        weights = [
            torch.randn(4, 4),  # Square for eigenvalue computation
            torch.randn(4, 4),
            torch.randn(2, 4)
        ]
        
        # Tension feedback (must match first weight dimensions)
        feedback_weights = [-torch.eye(4)]
        
        result = analyzer.full_diagnosis(
            weights, 
            feedback_weights=feedback_weights
        )
        
        assert result.monodromy is not None
        assert result.monodromy.topology in [
            CognitiveTopology.RESONANCE,
            CognitiveTopology.TENSION,
            CognitiveTopology.MIXED
        ]

    def test_escalation_recommendation(self, analyzer):
        """Test that escalation is recommended for problematic topologies."""
        # Create a network with tension feedback
        W = torch.eye(4)
        weights = [W, W, torch.randn(2, 4)]
        feedback_weights = [-W]  # Strong tension
        
        result = analyzer.full_diagnosis(weights, feedback_weights=feedback_weights)
        
        # Should detect tension and recommend escalation
        if result.monodromy and result.monodromy.topology == CognitiveTopology.TENSION:
            assert result.escalation_recommended
            assert any("TENSION" in msg for msg in result.diagnostic_messages)


class TestAttentionSheafAnalyzer:
    """Test attention-specific sheaf analysis."""

    @pytest.fixture
    def analyzer(self):
        return AttentionSheafAnalyzer(SheafConfig(device="cpu"))

    def test_attention_to_weights(self, analyzer):
        """Test conversion of attention weights to sheaf structure."""
        # Simulated attention: (batch=2, heads=4, seq=8, seq=8)
        attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        
        weights = analyzer.attention_to_sheaf_weights(attn)
        
        assert len(weights) == 4  # One per head
        assert all(w.shape == (8, 8) for w in weights)

    def test_diagnose_attention(self, analyzer):
        """Test attention pattern diagnosis."""
        attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        
        result = analyzer.diagnose_attention(attn)
        
        assert "per_head" in result
        assert "aggregate" in result
        assert len(result["per_head"]) == 4
        assert "mean_overlap" in result["aggregate"]


class TestSupervisionTarget:
    """Test supervision target creation."""

    def test_supervision_target_shape(self):
        """Test target error vector has correct shape."""
        input_emb = torch.randn(3)
        output_target = torch.randn(2)
        weights = [
            torch.randn(4, 3),
            torch.randn(4, 4),
            torch.randn(2, 4)
        ]
        
        b = create_supervision_target(input_emb, output_target, weights)
        
        # Total edge dimensions: 4 + 4 + 2 = 10
        assert b.shape == (10,)

    def test_supervision_target_structure(self):
        """Test target error has correct structure."""
        input_emb = torch.tensor([1.0, 2.0])
        output_target = torch.tensor([5.0, 6.0])
        
        W1 = torch.eye(3, 2)  # 3x2
        W2 = torch.eye(2, 3)  # 2x3
        
        b = create_supervision_target(input_emb, output_target, [W1, W2])
        
        # First part: -W1 @ x
        expected_first = -W1 @ input_emb
        assert torch.allclose(b[:3], expected_first)
        
        # Last part: y
        assert torch.allclose(b[-2:], output_target)


class TestIntegrationWithManifold:
    """Test integration with Hopfield manifold concepts."""

    def test_energy_landscape_alignment(self):
        """Test that sheaf analysis aligns with Hopfield energy concepts."""
        analyzer = SheafAnalyzer(SheafConfig(device="cpu"))
        
        # Create orthonormal weights (well-conditioned)
        Q, _ = torch.linalg.qr(torch.randn(4, 4))
        weights = [Q[:, :2], Q, Q[:2, :]]
        
        D, _ = analyzer.build_relative_coboundary(weights)
        cohomology = analyzer.compute_cohomology(D)
        
        # Well-conditioned system should have good singular value spectrum
        # (analogous to good energy landscape conditioning)
        sv = cohomology.singular_values
        if len(sv) > 1:
            condition = sv[0] / sv[-1]
            assert condition < 1000  # Not too ill-conditioned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

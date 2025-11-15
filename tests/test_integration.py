"""
Integration Tests for RAA Reasoning Loop

Tests that components compose correctly via the integration layer.
"""

import pytest
import torch

from src.manifold import Manifold, HopfieldConfig
from src.director import Director, DirectorConfig
from src.pointer import GoalController, PointerConfig
from src.integration import RAAReasoningLoop, ReasoningConfig


@pytest.fixture
def embedding_dim():
    """Standard embedding dimension for tests."""
    return 128


@pytest.fixture
def device():
    """Standard device for tests."""
    return "cpu"


@pytest.fixture
def manifold(embedding_dim, device):
    """Create and initialize Manifold with patterns."""
    config = HopfieldConfig(
        embedding_dim=embedding_dim,
        beta=1.0,
        max_patterns=100,
        update_steps=5,
        device=device
    )
    manifold = Manifold(config=config)
    
    # Store some prototype patterns (clustered)
    num_clusters = 5
    patterns_per_cluster = 4
    
    for cluster_id in range(num_clusters):
        center = torch.randn(embedding_dim)
        center = torch.nn.functional.normalize(center, p=2, dim=0)
        
        for _ in range(patterns_per_cluster):
            noise = torch.randn(embedding_dim) * 0.1
            pattern = center + noise
            pattern = torch.nn.functional.normalize(pattern, p=2, dim=0)
            manifold.store_pattern(pattern)
    
    return manifold


@pytest.fixture
def director(manifold, device):
    """Create Director."""
    config = DirectorConfig(
        entropy_threshold_percentile=0.70,
        search_k=3,
        use_energy_aware_search=True,
        device=device
    )
    return Director(manifold=manifold, config=config)


@pytest.fixture
def pointer(embedding_dim, device):
    """Create Pointer."""
    config = PointerConfig(
        embedding_dim=embedding_dim,
        hidden_dim=embedding_dim,
        num_layers=2,
        controller_type="gru",
        device=device
    )
    return GoalController(config=config)


@pytest.fixture
def reasoning_loop(manifold, director, pointer, device):
    """Create complete reasoning loop."""
    config = ReasoningConfig(
        max_steps=10,
        max_reframing_attempts=3,
        energy_threshold=0.1,
        device=device
    )
    return RAAReasoningLoop(
        manifold=manifold,
        director=director,
        pointer=pointer,
        config=config
    )


def test_reasoning_loop_initialization(reasoning_loop):
    """Test that reasoning loop initializes correctly."""
    assert reasoning_loop.manifold is not None
    assert reasoning_loop.director is not None
    assert reasoning_loop.pointer is not None
    assert reasoning_loop.config is not None
    
    # Check metrics are initialized
    assert "energy_trajectory" in reasoning_loop.metrics
    assert "entropy_trajectory" in reasoning_loop.metrics
    assert "num_reframings" in reasoning_loop.metrics


def test_reasoning_loop_single_step(reasoning_loop, embedding_dim):
    """Test single reasoning step executes without errors."""
    # Create input
    input_embedding = torch.randn(embedding_dim)
    input_embedding = torch.nn.functional.normalize(input_embedding, p=2, dim=0)
    
    # Initialize pointer
    reasoning_loop.pointer.reset(batch_size=1)
    reasoning_loop.pointer.set_goal(input_embedding)
    current_state = reasoning_loop.pointer.get_current_goal()
    
    # Execute single step
    new_state, step_metrics = reasoning_loop.reason_step(current_state, step=0)
    
    # Verify outputs
    assert new_state is not None
    assert new_state.shape[0] == embedding_dim
    assert "step" in step_metrics
    assert "energy" in step_metrics
    assert "reframing_triggered" in step_metrics


def test_reasoning_loop_full_cycle(reasoning_loop, embedding_dim):
    """Test complete reasoning cycle from input to solution."""
    # Create input
    input_embedding = torch.randn(embedding_dim)
    input_embedding = torch.nn.functional.normalize(input_embedding, p=2, dim=0)
    
    # Run reasoning loop
    solution, metrics = reasoning_loop.reason(
        input_embeddings=input_embedding,
        return_trajectory=False
    )
    
    # Verify solution
    assert solution is not None
    assert solution.shape[0] == embedding_dim
    
    # Verify metrics
    assert "total_steps" in metrics
    assert "num_reframings" in metrics
    assert "final_energy" in metrics
    assert "final_entropy" in metrics
    assert metrics["total_steps"] <= reasoning_loop.config.max_steps


def test_reasoning_loop_convergence(reasoning_loop, embedding_dim):
    """Test that reasoning loop can converge before max_steps."""
    # Create input very close to a stored pattern (should converge quickly)
    patterns = reasoning_loop.manifold.get_patterns()
    if patterns.shape[0] > 0:
        # Use first pattern with small perturbation
        input_embedding = patterns[0] + torch.randn(embedding_dim) * 0.01
        input_embedding = torch.nn.functional.normalize(input_embedding, p=2, dim=0)
        
        # Run reasoning
        solution, metrics = reasoning_loop.reason(input_embeddings=input_embedding)
        
        # Should converge before max steps
        assert metrics["total_steps"] < reasoning_loop.config.max_steps
        assert "convergence_reason" in metrics


def test_reasoning_loop_reframing(reasoning_loop, embedding_dim):
    """Test that Director can trigger reframing during reasoning."""
    # Create input far from any pattern (high energy, should trigger search)
    input_embedding = torch.randn(embedding_dim) * 3.0  # Large magnitude
    input_embedding = torch.nn.functional.normalize(input_embedding, p=2, dim=0)
    
    # Run reasoning
    solution, metrics = reasoning_loop.reason(input_embeddings=input_embedding)
    
    # Verify reframing can happen (though not guaranteed)
    assert "num_reframings" in metrics
    assert "reframing_events" in metrics
    # Note: Reframing might not trigger if Director threshold not exceeded
    # This is expected behavior


def test_reasoning_loop_metrics_tracking(reasoning_loop, embedding_dim):
    """Test that metrics are properly tracked throughout reasoning."""
    input_embedding = torch.randn(embedding_dim)
    input_embedding = torch.nn.functional.normalize(input_embedding, p=2, dim=0)
    
    solution, metrics = reasoning_loop.reason(
        input_embeddings=input_embedding,
        return_trajectory=True  # Request full trajectory
    )
    
    # Verify trajectory tracking
    assert "state_trajectory" in metrics
    assert len(metrics["state_trajectory"]) == metrics["total_steps"] + 1  # +1 for initial
    
    # Verify energy trajectory
    assert "energy_trajectory" in metrics
    assert len(metrics["energy_trajectory"]) > 0


def test_reasoning_loop_pseudo_logits(reasoning_loop, embedding_dim):
    """Test pseudo-logits computation for Director."""
    current_state = torch.randn(1, embedding_dim)
    current_state = torch.nn.functional.normalize(current_state, p=2, dim=-1)
    
    pseudo_logits = reasoning_loop._compute_pseudo_logits(current_state)
    
    # Verify shape and properties
    assert pseudo_logits is not None
    assert pseudo_logits.dim() == 2  # (batch, num_patterns)
    assert pseudo_logits.shape[0] == 1  # batch size


def test_reasoning_loop_energy_threshold(reasoning_loop, embedding_dim):
    """Test that reasoning stops when energy threshold is reached."""
    # Set very high energy threshold (should never stop early)
    reasoning_loop.config.energy_threshold = 10.0
    
    input_embedding = torch.randn(embedding_dim)
    input_embedding = torch.nn.functional.normalize(input_embedding, p=2, dim=0)
    
    solution, metrics = reasoning_loop.reason(input_embeddings=input_embedding)
    
    # Should run to max_steps
    assert metrics["total_steps"] == reasoning_loop.config.max_steps
    
    # Now test with very low threshold (should stop early)
    reasoning_loop.config.energy_threshold = 0.0001
    reasoning_loop.reset_metrics()
    
    solution, metrics = reasoning_loop.reason(input_embeddings=input_embedding)
    
    # Might stop early if energy gets low enough
    # (not guaranteed, depends on manifold patterns)
    assert metrics["total_steps"] <= reasoning_loop.config.max_steps


def test_reasoning_loop_component_integration(manifold, director, pointer):
    """Test that components are properly integrated and communicate."""
    # Create reasoning loop
    loop = RAAReasoningLoop(
        manifold=manifold,
        director=director,
        pointer=pointer
    )
    
    # Verify components are accessible
    assert loop.manifold is manifold
    assert loop.director is director
    assert loop.pointer is pointer
    
    # Verify they can interact
    embedding_dim = manifold.embedding_dim
    test_embedding = torch.randn(embedding_dim)
    test_embedding = torch.nn.functional.normalize(test_embedding, p=2, dim=0)
    
    # Pointer can update
    pointer.set_goal(test_embedding)
    goal = pointer.get_current_goal()
    assert goal.shape[-1] == embedding_dim
    
    # Manifold can retrieve
    retrieved, _ = manifold.retrieve(test_embedding)
    assert retrieved.shape[-1] == embedding_dim
    
    # Director can compute pseudo-logits
    pseudo_logits = loop._compute_pseudo_logits(test_embedding.unsqueeze(0))
    assert pseudo_logits is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

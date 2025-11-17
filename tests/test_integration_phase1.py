"""
Tests for RAA-CWD Integration Module (Phase 1)

Tests the infrastructure components:
- Embedding mapper
- Entropy calculator
- Bridge initialization
- Basic monitoring workflow
"""

import pytest
import torch

from src.director import Director, DirectorConfig
from src.integration import (
    CWDRAABridge,
    EmbeddingMapper,
    EntropyCalculator,
    cwd_to_logits,
)
from src.manifold import HopfieldConfig, Manifold


@pytest.fixture
def embedding_dim():
    """Standard embedding dimension."""
    return 512


@pytest.fixture
def device():
    """Standard device."""
    return "cpu"


@pytest.fixture
def embedding_mapper(embedding_dim, device):
    """Create embedding mapper."""
    return EmbeddingMapper(
        embedding_dim=embedding_dim,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=device,
    )


@pytest.fixture
def entropy_calculator():
    """Create entropy calculator."""
    return EntropyCalculator(temperature=1.0)


@pytest.fixture
def manifold(embedding_dim, device):
    """Create Manifold."""
    config = HopfieldConfig(
        embedding_dim=embedding_dim,
        beta=1.0,
        device=device,
    )
    return Manifold(config=config)


@pytest.fixture
def director(manifold, device):
    """Create Director."""
    config = DirectorConfig(
        entropy_threshold_percentile=0.70,
        device=device,
    )
    return Director(manifold=manifold, config=config)


# ============================================================================
# Embedding Mapper Tests
# ============================================================================


def test_embedding_mapper_initialization(embedding_mapper, embedding_dim):
    """Test embedding mapper initializes correctly."""
    assert embedding_mapper.embedding_dim == embedding_dim
    assert embedding_mapper.device == "cpu"


def test_cwd_node_to_vector(embedding_mapper, embedding_dim):
    """Test converting CWD node to embedding vector."""
    node_id = "node_123"
    node_content = "This is a test thought about consciousness"

    vector = embedding_mapper.cwd_node_to_vector(
        node_id=node_id,
        node_content=node_content,
    )

    # Check shape
    assert vector.shape == (embedding_dim,)

    # Check normalization
    norm = torch.norm(vector, p=2)
    assert torch.isclose(norm, torch.tensor(1.0), atol=1e-5)


def test_tool_to_vector(embedding_mapper, embedding_dim):
    """Test converting CWD tool to embedding vector."""
    tool_id = "tool_lock_and_key"
    tool_description = "Apply lever-and-catch mechanism to locked problems"
    tool_use_cases = ["Locks", "Latches", "Constraints"]

    vector = embedding_mapper.tool_to_vector(
        tool_id=tool_id,
        tool_description=tool_description,
        tool_use_cases=tool_use_cases,
    )

    assert vector.shape == (embedding_dim,)
    assert torch.isclose(torch.norm(vector, p=2), torch.tensor(1.0), atol=1e-5)


def test_embedding_similarity(embedding_mapper):
    """Test similarity computation between embeddings."""
    # Similar content should have high similarity
    vec1 = embedding_mapper.cwd_node_to_vector("n1", "Machine learning with neural networks")
    vec2 = embedding_mapper.cwd_node_to_vector("n2", "Deep learning using neural nets")

    similarity = embedding_mapper.compute_similarity(vec1, vec2)
    assert similarity > 0.5  # Should be fairly similar

    # Dissimilar content should have lower similarity
    vec3 = embedding_mapper.cwd_node_to_vector("n3", "Cooking pasta with tomato sauce")

    dissimilarity = embedding_mapper.compute_similarity(vec1, vec3)
    assert dissimilarity < similarity  # Should be less similar


# ============================================================================
# Entropy Calculator Tests
# ============================================================================


def test_entropy_calculator_initialization(entropy_calculator):
    """Test entropy calculator initializes correctly."""
    assert entropy_calculator.temperature == 1.0


def test_hypothesize_to_logits(entropy_calculator):
    """Test converting hypothesize results to logits."""
    # High confidence hypothesis
    high_conf_results = [
        {"hypothesis": "H1", "confidence": 0.9},
        {"hypothesis": "H2", "confidence": 0.1},
    ]

    logits = entropy_calculator.hypothesize_to_logits(high_conf_results)
    assert logits.shape[0] == 2

    # Compute entropy (should be low)
    entropy = entropy_calculator.compute_entropy(logits)
    assert entropy < 1.0  # Low entropy for high confidence

    # Uniform confidence hypothesis (should have higher entropy)
    uniform_results = [
        {"hypothesis": "H1", "confidence": 0.5},
        {"hypothesis": "H2", "confidence": 0.5},
        {"hypothesis": "H3", "confidence": 0.5},
    ]

    logits_uniform = entropy_calculator.hypothesize_to_logits(uniform_results)
    entropy_uniform = entropy_calculator.compute_entropy(logits_uniform)
    assert entropy_uniform > entropy  # More uniform = higher entropy


def test_synthesize_to_logits(entropy_calculator):
    """Test converting synthesize results to logits."""
    # Good synthesis (high quality)
    good_synthesis = {
        "synthesis": "Well-formed synthesis",
        "quality": {
            "coverage": 0.9,
            "coherence": 0.9,
            "fidelity": 0.9,
        },
    }

    logits = entropy_calculator.synthesize_to_logits(good_synthesis)
    entropy = entropy_calculator.compute_entropy(logits)

    # Good synthesis should have low entropy (confident)
    assert entropy < 0.5

    # Poor synthesis (low quality)
    poor_synthesis = {
        "synthesis": "Poor synthesis",
        "quality": {
            "coverage": 0.3,
            "coherence": 0.4,
            "fidelity": 0.3,
        },
    }

    logits_poor = entropy_calculator.synthesize_to_logits(poor_synthesis)
    entropy_poor = entropy_calculator.compute_entropy(logits_poor)

    # Poor synthesis should still have low entropy (just on the "poor" side)
    assert entropy_poor < 1.0


def test_cwd_to_logits_convenience(entropy_calculator):
    """Test convenience function for all operations."""
    # Test hypothesize
    hyp_result = [{"hypothesis": "Test", "confidence": 0.8}]
    logits = cwd_to_logits("hypothesize", hyp_result, entropy_calculator)
    assert logits.shape[0] == 1

    # Test synthesize
    syn_result = {
        "synthesis": "Test",
        "quality": {"coverage": 0.8, "coherence": 0.7, "fidelity": 0.9},
    }
    logits = cwd_to_logits("synthesize", syn_result, entropy_calculator)
    assert logits.shape[0] == 2

    # Test constrain
    con_result = {"valid": True, "satisfaction_score": 0.95}
    logits = cwd_to_logits("constrain", con_result, entropy_calculator)
    assert logits.shape[0] == 2


# ============================================================================
# Bridge Integration Tests
# ============================================================================


def test_bridge_initialization(manifold, director):
    """Test bridge initializes with all components."""
    bridge = CWDRAABridge(
        cwd_server=None,  # Mock for now
        raa_director=director,
        manifold=manifold,
    )

    assert bridge.cwd_server is None  # Mock
    assert bridge.raa_director is director
    assert bridge.manifold is manifold
    assert bridge.embedding_mapper is not None
    assert bridge.entropy_calculator is not None


def test_bridge_monitored_operation(manifold, director):
    """Test executing monitored CWD operation."""
    bridge = CWDRAABridge(
        cwd_server=None,
        raa_director=director,
        manifold=manifold,
    )

    # Execute operation (uses mock data in Phase 1)
    result = bridge.execute_monitored_operation(
        operation="hypothesize",
        params={"node_a_id": "n1", "node_b_id": "n2"},
    )

    # Check result structure
    assert isinstance(result, list)
    assert len(result) > 0

    # Check metrics updated
    metrics = bridge.get_metrics()
    assert metrics["operations_monitored"] == 1


def test_bridge_entropy_monitoring(manifold, director):
    """Test that bridge monitors entropy correctly."""
    bridge = CWDRAABridge(
        cwd_server=None,
        raa_director=director,
        manifold=manifold,
    )

    # Execute multiple operations
    for _ in range(5):
        bridge.execute_monitored_operation(
            operation="hypothesize",
            params={},
        )

    metrics = bridge.get_metrics()
    assert metrics["operations_monitored"] == 5
    assert len(metrics["integration_events"]) == 5


def test_bridge_metrics_reset(manifold, director):
    """Test metrics can be reset."""
    bridge = CWDRAABridge(
        cwd_server=None,
        raa_director=director,
        manifold=manifold,
    )

    # Execute operation
    bridge.execute_monitored_operation(operation="hypothesize", params={})

    # Reset
    bridge.reset_metrics()

    metrics = bridge.get_metrics()
    assert metrics["operations_monitored"] == 0
    assert len(metrics["integration_events"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

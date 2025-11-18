"""
CWD-RAA Integration Example: Entropy-Triggered Search

This example demonstrates the complete RAA + CWD integration with:
1. Real Ollama-based CWD server (not mocks)
2. Entropy monitoring of CWD operations
3. Director search triggered on high entropy
4. Pointer goal updates from search results

Prerequisites:
- Ollama running locally (http://localhost:11434)
- Neo4j running (bolt://localhost:7687)
- .env file with NEO4J_PASSWORD set
- Models: qwen3:latest, qwen3-embedding:0.6b (or modify .env)

Install dependencies:
    pip install ollama neo4j chromadb sentence-transformers python-dotenv

Usage:
    python examples/cwd_integration_example.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.director import Director, DirectorConfig
from src.integration.cwd_raa_bridge import BridgeConfig, CWDRAABridge
from src.manifold import HopfieldConfig, Manifold
from src.pointer import GoalController, PointerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate CWD-RAA integration with entropy-triggered search."""
    print("=" * 70)
    print("CWD-RAA Integration Example: Entropy-Triggered Search")
    print("=" * 70)

    # Configuration
    embedding_dim = 384  # nomic-embed-text dimension
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")

    # Initialize RAA components
    print("\n1. Initializing RAA components...")

    manifold_config = HopfieldConfig(
        embedding_dim=embedding_dim,
        beta=10.0,
        adaptive_beta=True,
        beta_min=5.0,
        beta_max=50.0,
        device=device,
    )
    manifold = Manifold(manifold_config)
    print(f"   ✓ Manifold: {manifold}")

    pointer_config = PointerConfig(
        embedding_dim=embedding_dim,
        controller_type="gru",
        device=device,
    )
    pointer = GoalController(pointer_config)
    print(f"   ✓ Pointer: {pointer}")

    director_config = DirectorConfig(
        search_k=5,
        entropy_threshold_percentile=0.75,
        use_energy_aware_search=True,
        device=device,
    )
    director = Director(manifold, director_config)
    print(f"   ✓ Director: {director}")

    # Seed Manifold with concept patterns
    print("\n2. Seeding Manifold with concepts...")
    concepts = [
        "mathematics",
        "physics",
        "biology",
        "chemistry",
        "computer_science",
        "philosophy",
        "psychology",
        "linguistics",
        "economics",
        "sociology",
    ]

    for concept in concepts:
        # Create random embedding (in real use, use sentence transformer)
        pattern = torch.randn(embedding_dim)
        pattern = torch.nn.functional.normalize(pattern, p=2, dim=-1)
        manifold.store_pattern(pattern)

    print(f"   ✓ Stored {len(concepts)} concept patterns")

    # Set initial goal
    print("\n3. Setting initial goal...")
    initial_goal = torch.randn(embedding_dim)
    initial_goal = torch.nn.functional.normalize(initial_goal, p=2, dim=-1)
    pointer.set_goal(initial_goal)
    print(f"   ✓ Initial goal shape: {pointer.get_current_goal().shape}")

    # Initialize CWD server (lazy import to check dependencies)
    print("\n4. Initializing CWD server...")
    try:
        # Import CWD components from integrated server package
        from src.server import CognitiveWorkspace, CWDConfig

        # Load config from .env
        cwd_config = CWDConfig()  # type: ignore[call-arg]
        cwd_workspace = CognitiveWorkspace(cwd_config)
        print(f"   ✓ CWD connected to Neo4j: {cwd_config.neo4j_uri}")
        print(f"   ✓ Using LLM: {cwd_config.llm_model}")

    except ImportError as e:
        print(f"   ✗ CWD import failed: {e}")
        print("   Ensure project deps are installed (uv sync or pip install -e .)")
        return
    except Exception as e:
        print(f"   ✗ CWD initialization failed: {e}")
        print("   Check .env file and ensure Neo4j/Ollama are running")
        return

    # Initialize CWD-RAA Bridge
    print("\n5. Initializing CWD-RAA Bridge...")
    bridge_config = BridgeConfig(
        embedding_dim=embedding_dim,
        entropy_threshold=0.6,  # FIXED: Binary distributions produce 0.0-1.0 bits
        enable_monitoring=True,
        search_on_confusion=True,  # Enable Phase 2 search
        log_integration_events=True,
        device=device,
    )

    bridge = CWDRAABridge(
        cwd_server=cwd_workspace,
        raa_director=director,
        manifold=manifold,
        config=bridge_config,
        pointer=pointer,  # Pass pointer for goal updates
    )
    print("   ✓ Bridge initialized with pointer integration")

    # Prepare workspace nodes for operations
    print("\n6. Preparing workspace nodes...")
    try:
        decomp = cwd_workspace.deconstruct(
            problem="Explore connections between mathematics and biology",
            max_depth=3,
        )
        # Collect available node IDs
        node_ids = [decomp.get("root_id")] + decomp.get("component_ids", [])
        node_ids = [nid for nid in node_ids if nid]
        print(f"   ✓ Created {len(node_ids)} nodes for testing")
    except Exception as e:
        print(f"   ✗ Failed to prepare nodes: {e}")
        node_ids = []

    # Execute monitored CWD operations
    print("\n6. Executing monitored CWD operations...")
    print("   (This will trigger entropy monitoring and search on high entropy)\n")

    # Operation 1: Hypothesize (high variance → high entropy)
    try:
        # Pick two nodes if available, else fall back (will mock)
        node_a = node_ids[0] if len(node_ids) > 0 else "concept_1"
        node_b = node_ids[1] if len(node_ids) > 1 else "concept_2"
        result = bridge.execute_monitored_operation(
            operation="hypothesize",
            params={
                "node_a_id": node_a,
                "node_b_id": node_b,
                "context": "Find connections between mathematics and biology",
            },
        )
        print(f"   Hypothesize result: {result}")
    except Exception as e:
        logger.warning(f"Hypothesize operation failed: {e}")

    # Operation 2: Synthesize (quality variance → moderate entropy)
    try:
        synth_sources = (
            node_ids[:3] if len(node_ids) >= 3 else ["concept_1", "concept_2", "concept_3"]
        )
        result = bridge.execute_monitored_operation(
            operation="synthesize",
            params={
                "node_ids": synth_sources,
                "goal": "Merge related concepts",
            },
        )
        print(f"   Synthesize result: {result}")
    except Exception as e:
        logger.warning(f"Synthesize operation failed: {e}")

    # Display metrics
    print("\n7. Integration Metrics:")
    metrics = bridge.get_metrics()
    print(f"   Operations monitored: {metrics['operations_monitored']}")
    print(f"   Entropy spikes detected: {metrics['entropy_spikes_detected']}")
    print(f"   Searches triggered: {metrics['searches_triggered']}")
    print(f"   Alternatives found: {metrics['alternatives_found']}")

    # Show entropy history
    print("\n8. Director Entropy Statistics:")
    stats = director.monitor.get_statistics()
    print(f"   Mean entropy: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"   Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"   Current threshold: {stats['threshold']:.3f}")
    print(f"   Samples: {stats['num_samples']}")

    # Show goal state
    print("\n9. Final Goal State:")
    final_goal = pointer.get_current_goal()
    print(f"   Shape: {final_goal.shape}")
    print(f"   Norm: {torch.norm(final_goal).item():.3f}")

    print("\n" + "=" * 70)
    print("Integration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

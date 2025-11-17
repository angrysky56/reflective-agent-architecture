"""
Full System Test: ReflectiveAgentArchitecture with Token Generation

This tests the COMPLETE RAA system including:
- Processor (Transformer decoder) for actual token generation
- Manifold for associative memory
- Pointer for goal state management
- Director for metacognitive monitoring and search

This is the proper way to test RAA - with real logits that have meaningful variance.
"""

import logging

import torch

from src.director import DirectorConfig
from src.integration import RAAConfig, ReflectiveAgentArchitecture
from src.manifold import HopfieldConfig
from src.pointer import PointerConfig
from src.processor import ProcessorConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def test_generation_task():
    """
    Test RAA on a simple next-token prediction task.

    This creates actual token logits with real variance, allowing:
    - Director to detect meaningful entropy changes
    - Search to find alternative framings
    - Reframing to impact generation quality
    """
    print("=" * 70)
    print("Full System Test: RAA with Token Generation")
    print("=" * 70)

    # Configuration
    embedding_dim = 256
    vocab_size = 1000  # Small vocab for demo
    device = "cpu"

    # CRITICAL: Use higher beta range for meaningful entropy variation
    config = RAAConfig(
        hopfield_config=HopfieldConfig(
            embedding_dim=embedding_dim,
            beta=10.0,  # Start higher than default 1.0
            adaptive_beta=True,  # Enable adaptive beta
            beta_min=5.0,  # Increased from 0.5
            beta_max=50.0,  # Increased from 2.0 (needs ~10x change for effect)
            max_patterns=100,
            device=device,
        ),
        processor_config=ProcessorConfig(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=3,
            num_heads=4,
            device=device,
        ),
        pointer_config=PointerConfig(
            embedding_dim=embedding_dim,
            controller_type="gru",
            device=device,
        ),
        director_config=DirectorConfig(
            search_k=5,
            entropy_threshold_percentile=0.75,
            use_energy_aware_search=True,
            device=device,
        ),
        max_reframing_attempts=3,
        enable_metacognition=True,
        device=device,
    )

    print("\n1. Initializing Full RAA System...")
    raa = ReflectiveAgentArchitecture(config)
    print(raa)

    # Store diverse concept patterns
    print("\n2. Seeding Manifold with concept patterns...")
    num_concepts = 20
    for i in range(num_concepts):
        concept = torch.randn(embedding_dim)
        concept = torch.nn.functional.normalize(concept, p=2, dim=-1)
        raa.store_concept(concept, label=f"concept_{i}")
    print(f"Stored {num_concepts} concepts")

    # Set initial goal
    print("\n3. Setting initial goal state...")
    initial_goal = torch.randn(embedding_dim)
    initial_goal = torch.nn.functional.normalize(initial_goal, p=2, dim=-1)
    raa.set_initial_goal(initial_goal)

    # Test single generation step
    print("\n4. Testing single generation step...")
    input_ids = torch.randint(0, vocab_size, (1, 10))
    result = raa.generate_step(input_ids, temperature=1.0)

    print(f"  Generated token: {result['next_token'].item()}")
    print(f"  Entropy: {result['entropy']:.4f}")
    print(f"  Goal reframed: {result['reframed']}")
    print(f"  Reframing attempts: {result.get('reframing_attempts', 0)}")

    # Test sequence generation
    print("\n5. Testing sequence generation (20 tokens)...")
    output = raa.generate(
        input_ids=input_ids,
        max_length=20,
        temperature=1.0,
        return_history=True,
    )

    print(f"  Generated {output['output_ids'].shape[1]} tokens")
    print(f"  Number of reframings: {output['num_reframings']}")

    if "entropy_history" in output:
        entropies = output["entropy_history"]
        print(
            f"  Entropy trajectory: min={min(entropies):.3f}, max={max(entropies):.3f}, mean={sum(entropies)/len(entropies):.3f}"
        )

    # Statistics
    print("\n6. System Statistics:")
    stats = raa.get_statistics()
    print(f"  Manifold patterns: {stats['manifold']['num_patterns']}")
    print(f"  Manifold embedding dim: {stats['manifold']['embedding_dim']}")
    # Access beta directly from manifold object
    print(f"  Manifold beta: {raa.manifold.beta:.2f}")
    print(f"  Adaptive beta enabled: {raa.manifold.config.adaptive_beta}")
    if raa.manifold.config.adaptive_beta:
        print(
            f"  Adaptive beta range: [{raa.manifold.config.beta_min:.1f}, {raa.manifold.config.beta_max:.1f}]"
        )
    print(f"  Director entropy threshold: {stats['director']['entropy']['threshold']:.4f}")
    print(f"  Search episodes: {stats['director']['num_search_episodes']}")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

    return raa, stats


def compare_beta_ranges():
    """
    Demonstrate the impact of different beta ranges on entropy dynamics.
    """
    print("\n" + "=" * 70)
    print("Beta Range Comparison")
    print("=" * 70)

    vocab_size = 1000
    embedding_dim = 256

    # Test different beta configurations
    configs = [
        ("Default (0.5-2.0)", 1.0, 0.5, 2.0),
        ("Medium (5.0-20.0)", 10.0, 5.0, 20.0),
        ("High (10.0-50.0)", 20.0, 10.0, 50.0),
    ]

    for name, beta, beta_min, beta_max in configs:
        print(f"\n{name}:")
        print(f"  Beta: {beta:.1f}, Range: [{beta_min:.1f}, {beta_max:.1f}]")

        # Create simple RAA
        config = RAAConfig(
            hopfield_config=HopfieldConfig(
                embedding_dim=embedding_dim,
                beta=beta,
                adaptive_beta=True,
                beta_min=beta_min,
                beta_max=beta_max,
                device="cpu",
            ),
            processor_config=ProcessorConfig(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_layers=2,
                num_heads=4,
                device="cpu",
            ),
            pointer_config=PointerConfig(embedding_dim=embedding_dim, device="cpu"),
            director_config=DirectorConfig(device="cpu"),
            device="cpu",
        )

        raa = ReflectiveAgentArchitecture(config)

        # Store patterns
        for i in range(10):
            pattern = torch.randn(embedding_dim)
            pattern = torch.nn.functional.normalize(pattern, p=2, dim=-1)
            raa.store_concept(pattern)

        # Set goal
        goal = torch.randn(embedding_dim)
        goal = torch.nn.functional.normalize(goal, p=2, dim=-1)
        raa.set_initial_goal(goal)

        # Generate a few tokens
        input_ids = torch.randint(0, vocab_size, (1, 5))
        entropies = []
        reframings = 0

        for _ in range(5):
            result = raa.generate_step(input_ids, temperature=1.0)
            entropies.append(result["entropy"])
            if result["reframed"]:
                reframings += 1
            # next_token is already (batch, 1), so just unsqueeze to (batch, 1, 1) then squeeze to (batch, 1)
            next_token = result["next_token"].view(1, 1)  # Ensure (batch, 1) shape
            input_ids = torch.cat([input_ids, next_token], dim=1)

        print(f"  Entropy range: [{min(entropies):.3f}, {max(entropies):.3f}]")
        print(f"  Entropy std: {torch.std(torch.tensor(entropies)).item():.4f}")
        print(f"  Reframings: {reframings}/5")


if __name__ == "__main__":
    # Run full system test
    raa, stats = test_generation_task()

    # Compare beta configurations
    compare_beta_ranges()

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("1. With Processor (NN), we get REAL token logits with natural variance")
    print("2. Beta needs 10x range (e.g., 5-50) for meaningful entropy modulation")
    print("3. Higher beta → sharper distributions → lower entropy")
    print("4. Adaptive beta allows Director to tune exploration vs exploitation")
    print("5. This is the PROPER way to test RAA - as an integrated system")
    print("=" * 70)

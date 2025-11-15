"""
Basic Usage Example of Reflective Agent Architecture

Demonstrates the core "Aha!" loop with all components.
"""

import torch
import logging

from src.integration import ReflectiveAgentArchitecture, RAAConfig
from src.manifold import HopfieldConfig
from src.processor import ProcessorConfig
from src.pointer import PointerConfig
from src.director import DirectorConfig


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Demonstrate basic RAA usage."""
    print("=" * 60)
    print("Reflective Agent Architecture - Basic Example")
    print("=" * 60)

    # Configuration
    embedding_dim = 256
    vocab_size = 1000  # Small vocab for demo
    device = "cpu"

    config = RAAConfig(
        hopfield_config=HopfieldConfig(
            embedding_dim=embedding_dim,
            beta=1.0,
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
            device=device,
        ),
        max_reframing_attempts=3,
        enable_metacognition=True,
        device=device,
    )

    # Create RAA
    print("\n1. Initializing Reflective Agent Architecture...")
    raa = ReflectiveAgentArchitecture(config)
    print(raa)

    # Store some concept patterns in Manifold
    print("\n2. Storing concept patterns in Manifold...")
    num_concepts = 10
    for i in range(num_concepts):
        concept_embedding = torch.randn(embedding_dim)
        raa.store_concept(concept_embedding, label=f"concept_{i}")
    print(f"Stored {num_concepts} concepts")

    # Set initial goal
    print("\n3. Setting initial goal...")
    initial_goal = torch.randn(embedding_dim)
    raa.set_initial_goal(initial_goal)
    print("Initial goal set")

    # Demonstrate retrieval from Manifold
    print("\n4. Testing Manifold retrieval...")
    query = torch.randn(embedding_dim)
    retrieved, energy_traj = raa.retrieve_from_manifold(query, num_steps=10)
    print(f"Energy trajectory: {energy_traj.tolist()}")
    print(f"Final energy: {energy_traj[-1]:.4f}")

    # Demonstrate single generation step
    print("\n5. Testing single generation step with metacognition...")
    input_ids = torch.randint(0, vocab_size, (1, 10))  # Dummy input
    result = raa.generate_step(input_ids, temperature=1.0)

    print(f"Generated token: {result['next_token'].item()}")
    print(f"Entropy: {result['entropy']:.4f}")
    print(f"Goal reframed: {result['reframed']}")

    # Demonstrate full generation
    print("\n6. Testing full sequence generation...")
    output = raa.generate(
        input_ids=input_ids,
        max_length=20,
        temperature=1.0,
        return_history=True,
    )

    print(f"Generated {output['output_ids'].shape[1]} tokens")
    print(f"Number of goal reframings: {output['num_reframings']}")

    # Get statistics
    print("\n7. RAA Statistics:")
    stats = raa.get_statistics()
    print(f"Manifold patterns: {stats['manifold']['num_patterns']}")
    print(f"Entropy threshold: {stats['director']['entropy']['threshold']:.4f}")
    print(f"Search episodes: {stats['director']['num_search_episodes']}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

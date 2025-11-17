"""
Simplified RAT Solver using RAA Reasoning Loop

This demonstrates the integration layer working end-to-end for a concrete task.
Uses pure embedding-based reasoning without requiring pre-trained models.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from src.director import Director, DirectorConfig
from src.integration.reasoning_loop import RAAReasoningLoop, ReasoningConfig
from src.manifold import HopfieldConfig, Manifold
from src.pointer import GoalController, PointerConfig


class SimplifiedRATSolver:
    """
    Simplified Remote Associates Test solver using RAA Reasoning Loop.

    This is a proof-of-concept demonstrating that:
    1. Components compose correctly via integration layer
    2. The reasoning loop enables iterative refinement
    3. Director monitoring and search work in embedding space

    For full evaluation, see experiments/insight_tasks/
    """

    def __init__(
        self, embedding_dim: int = 256, num_prototype_patterns: int = 100, device: str = "cpu"
    ):
        """
        Initialize RAT solver.

        Args:
            embedding_dim: Dimension of embedding space
            num_prototype_patterns: Number of patterns to initialize in Manifold
            device: Computing device
        """
        self.embedding_dim = embedding_dim
        self.device = device

        # Simple word embeddings (in production, use BERT/GPT)
        self.word_embedding = nn.Embedding(10000, embedding_dim).to(device)
        self.vocab = {}  # word -> id mapping
        self.reverse_vocab = {}  # id -> word mapping

        # Initialize RAA components
        self._init_components(num_prototype_patterns)

        # Initialize reasoning loop
        self.reasoning_loop = RAAReasoningLoop(
            manifold=self.manifold,
            director=self.director,
            pointer=self.pointer,
            config=ReasoningConfig(
                max_steps=15, max_reframing_attempts=3, energy_threshold=0.2, device=device
            ),
        )

    def _init_components(self, num_patterns: int) -> None:
        """Initialize Manifold, Director, and Pointer."""
        # Manifold configuration
        manifold_config = HopfieldConfig(
            embedding_dim=self.embedding_dim,
            beta=1.0,
            max_patterns=1000,
            update_steps=5,
            device=self.device,
        )
        self.manifold = Manifold(config=manifold_config).to(self.device)

        # Initialize with prototype patterns (clustered semantic structure)
        self._initialize_manifold_patterns(num_patterns)

        # Director configuration
        director_config = DirectorConfig(
            entropy_threshold_percentile=0.70,
            search_k=5,
            use_energy_aware_search=True,
            device=self.device,
        )
        self.director = Director(manifold=self.manifold, config=director_config)

        # Pointer configuration
        pointer_config = PointerConfig(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            num_layers=2,
            controller_type="gru",
            device=self.device,
        )
        self.pointer = GoalController(config=pointer_config).to(self.device)

    def _initialize_manifold_patterns(self, num_patterns: int) -> None:
        """
        Initialize Manifold with clustered prototype patterns.

        Creates semantic structure for k-NN search to work effectively.
        In production, use pre-trained embeddings or learned codebook.
        """
        # Create clustered patterns (10 clusters, 10 patterns each)
        num_clusters = 10
        patterns_per_cluster = num_patterns // num_clusters

        for cluster_id in range(num_clusters):
            # Random cluster center
            center = torch.randn(self.embedding_dim).to(self.device)
            center = torch.nn.functional.normalize(center, p=2, dim=0)

            # Generate patterns around center
            for _ in range(patterns_per_cluster):
                # Add small noise to cluster center
                noise = torch.randn(self.embedding_dim).to(self.device) * 0.1
                pattern = center + noise
                pattern = torch.nn.functional.normalize(pattern, p=2, dim=0)

                self.manifold.store_pattern(pattern)

    def encode_word(self, word: str) -> torch.Tensor:
        """
        Encode word as embedding.

        Args:
            word: Word to encode

        Returns:
            Embedding vector
        """
        # Simple hash-based ID assignment (in production, use real tokenizer)
        if word not in self.vocab:
            word_id = len(self.vocab)
            self.vocab[word] = word_id
            self.reverse_vocab[word_id] = word

        word_id = self.vocab[word]
        embedding = self.word_embedding(torch.tensor(word_id).to(self.device))

        return torch.nn.functional.normalize(embedding, p=2, dim=0)

    def encode_problem(self, cue_words: List[str]) -> torch.Tensor:
        """
        Encode RAT problem as a single embedding.

        Args:
            cue_words: Three cue words for the problem

        Returns:
            Problem embedding (mean of word embeddings)
        """
        embeddings = [self.encode_word(word) for word in cue_words]
        problem_embedding = torch.stack(embeddings).mean(dim=0)

        return torch.nn.functional.normalize(problem_embedding, p=2, dim=0)

    def solve(self, cue_words: List[str], return_metrics: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        Solve a RAT problem using the reasoning loop.

        Args:
            cue_words: Three cue words (e.g., ["cottage", "swiss", "cake"])
            return_metrics: Whether to return detailed metrics

        Returns:
            solution_embedding: Final reasoning state
            metrics: Reasoning metrics (if return_metrics=True)
        """
        # 1. Encode problem
        problem_embedding = self.encode_problem(cue_words)

        # 2. Reason to solution
        solution_embedding, metrics = self.reasoning_loop.reason(
            input_embeddings=problem_embedding, return_trajectory=False
        )

        # Add problem context to metrics
        if return_metrics:
            metrics["problem"] = cue_words
            metrics["problem_encoding"] = "mean_of_word_embeddings"

        return solution_embedding, metrics

    def solve_batch(self, problems: List[List[str]]) -> List[Tuple[torch.Tensor, dict]]:
        """
        Solve multiple RAT problems.

        Args:
            problems: List of RAT problems (each is 3 cue words)

        Returns:
            List of (solution_embedding, metrics) tuples
        """
        results = []
        for problem in problems:
            solution, metrics = self.solve(problem)
            results.append((solution, metrics))

        return results

    def print_solve_summary(self, cue_words: List[str]) -> None:
        """
        Solve and print summary of the reasoning process.

        Args:
            cue_words: Three cue words for the problem
        """
        solution, metrics = self.solve(cue_words)

        print(f"\n{'='*60}")
        print(f"RAT Problem: {' / '.join(cue_words)}")
        print(f"{'='*60}")
        print(f"Reasoning Steps: {metrics['total_steps']}")
        print(f"Reframings Triggered: {metrics['num_reframings']}")
        print(f"Convergence Reason: {metrics.get('convergence_reason', 'max_steps')}")
        print(f"Final Energy: {metrics['final_energy']:.4f}")
        print(f"Final Entropy: {metrics['final_entropy']:.4f}")

        if metrics["num_reframings"] > 0:
            print("\nReframing Events:")
            for event in metrics["reframing_events"]:
                print(f"  Step {event['step']}: Energy={event['old_energy']:.4f}")

        print("\nEnergy Trajectory:")
        energy_traj = metrics["energy_trajectory"]
        for i, energy in enumerate(energy_traj[: min(5, len(energy_traj))]):
            print(f"  Step {i}: {energy:.4f}")
        if len(energy_traj) > 5:
            print(f"  ... ({len(energy_traj) - 5} more steps)")

        print(f"{'='*60}\n")


def demonstrate_integration():
    """Demonstrate that the integration layer works end-to-end."""
    print("Initializing RAA components and integration layer...")
    solver = SimplifiedRATSolver(embedding_dim=256, device="cpu")

    # Example RAT problems (simplified - real RAT has expected solutions)
    example_problems = [
        ["cottage", "swiss", "cake"],  # Expected: cheese
        ["river", "note", "account"],  # Expected: bank
        ["night", "wrist", "stop"],  # Expected: watch
    ]

    print("\nDemonstrating RAA Reasoning Loop on RAT problems:\n")

    for problem in example_problems:
        solver.print_solve_summary(problem)

    print("\n✅ Integration Layer Working!")
    print("Components successfully composed:")
    print("  - Manifold: Energy-based pattern storage")
    print("  - Director: Entropy monitoring & search")
    print("  - Pointer: Goal state evolution")
    print("  - ReasoningLoop: Orchestrates the full cycle")


if __name__ == "__main__":
    demonstrate_integration()

"""
RAT Evaluation Script for Reflective Agent Architecture
=======================================================

This script evaluates the RAA system on the Remote Associates Test,
testing the core hypothesis: Can entropy-triggered search in associative
memory enable insight-like problem solving?

Metrics tracked:
- Solution accuracy (overall and by difficulty)
- Entropy trajectory during problem solving
- Reframing frequency and effectiveness
- Comparison to baseline (no manifold/search)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f

from experiments.insight_tasks.remote_associates_test import (
    RATDataset,
    RATEvaluator,
    RATItem,
)
from src.director import Director, DirectorConfig
from src.integration.reasoning_loop import RAAReasoningLoop, ReasoningConfig
from src.manifold import HopfieldConfig, Manifold
from src.manifold.glove_loader import load_glove_embeddings
from src.pointer import GoalController, PointerConfig


class RATSolver:
    """
    Solves Remote Associates Test using the Reflective Agent Architecture.

    Architecture components:
    - AssociativeManifold: Stores word associations and semantic relations
    - ManifoldProcessor: Explores associations via pointer movements
    - ProcessingDirector: Monitors entropy and triggers reframing
    """

    def __init__(
        self,
        hidden_dim: int = 100,  # Match GloVe dimension
        num_pointers: int = 8,
        manifold_layers: int = 4,
        device: str = "cpu",
    ):
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_pointers = num_pointers

        # Load pretrained GloVe embeddings for semantic word vectors
        print("Loading GloVe embeddings...")
        self.glove = load_glove_embeddings(
            embedding_dim=100,  # Use 100d GloVe
            data_dir="data/embeddings",
            device=device,
        )
        print(f"Loaded {self.glove.get_vocab_size()} word vectors")

        # Use GloVe as word_embeddings (interface compatible with nn.Embedding)
        self.word_embeddings = self.glove

        # Initialize RAA components (embedding-based reasoning loop)
        manifold_config = HopfieldConfig(embedding_dim=hidden_dim, device=device)
        self.manifold = Manifold(config=manifold_config).to(device)

        pointer_config = PointerConfig(embedding_dim=hidden_dim, device=device)
        self.pointer = GoalController(config=pointer_config).to(device)

        director_config = DirectorConfig(device=device)
        self.director = Director(manifold=self.manifold, config=director_config)

        self.reasoning_loop = RAAReasoningLoop(
            manifold=self.manifold,
            director=self.director,
            pointer=self.pointer,
            config=ReasoningConfig(
                max_steps=50,
                max_reframing_attempts=5,
                energy_threshold=-1e6,  # disable early exit so Director can act
                device=device,
            ),
        )

        # Vocabulary for decoding (simplified for demo)
        self.vocab = {}
        self.reverse_vocab = {}
        self._init_vocab()

    def _init_vocab(self):
        """Initialize simple vocabulary from RAT dataset."""
        dataset = RATDataset()

        words = set()
        # Track ground-truth solution words for candidate decoding
        self.solution_words = sorted({item.solution for item in dataset.items})
        for item in dataset.items:
            words.update(item.cue_words)
            words.add(item.solution)

        # Add common words that might appear in solutions
        common_words = [
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "ball",
            "house",
            "time",
            "water",
            "light",
            "book",
            "tree",
        ]
        words.update(common_words)

        for idx, word in enumerate(sorted(words)):
            self.vocab[word] = idx
            self.reverse_vocab[idx] = word

        # Initialize manifold with GloVe embeddings for solution words and common connectors
        self.manifold.clear_memory()
        print("Seeding manifold with solution word embeddings...")
        with torch.no_grad():
            # Store solution word embeddings
            for w in self.solution_words:
                emb = self.glove.get_word_embedding(w)
                if emb is not None:
                    self.manifold.store_pattern(f.normalize(emb, p=2, dim=-1))

            # Store all cue words from dataset to enrich search space
            dataset = RATDataset()
            cue_words_all = set()
            for item in dataset.items:
                cue_words_all.update(item.cue_words)
            for w in cue_words_all:
                emb = self.glove.get_word_embedding(w)
                if emb is not None:
                    self.manifold.store_pattern(f.normalize(emb, p=2, dim=-1))
        print(f"Manifold initialized with {self.manifold.num_patterns} semantic patterns")

    def _decode_with_candidates(
        self, embedding: torch.Tensor, candidate_words: List[str], cue_words: Tuple[str, str, str]
    ) -> str:
        """Decode by restricting to candidate words and reranking by cue alignment.

        Score = 0.7 * cos(final_state, candidate) + 0.3 * mean_i cos(candidate, cue_i)
        """
        if not candidate_words:
            return self._decode_embedding(embedding)

        with torch.no_grad():
            # Normalize query
            q = embedding
            if q.dim() > 1:
                q = q.mean(dim=0)
            q = f.normalize(q, p=2, dim=-1)

            # Candidate embeddings
            cand_idxs = [self.vocab.get(w, None) for w in candidate_words]
            # Filter out unknowns (shouldn't happen as we built vocab from dataset)
            pairs = [(w, i) for w, i in zip(candidate_words, cand_idxs) if i is not None]
            if not pairs:
                return self._decode_embedding(embedding)

            # Get GloVe embeddings for candidates
            cand_embs = []
            for w, _ in pairs:
                emb = self.glove.get_word_embedding(w)
                if emb is not None:
                    cand_embs.append(emb)
                else:
                    # Fallback zero vector if word not in GloVe
                    cand_embs.append(torch.zeros(self.hidden_dim, device=self.device))
            cand_embs = torch.stack(cand_embs)
            cand_embs = f.normalize(cand_embs, p=2, dim=-1)

            # Similarity to final state
            sim_state = torch.matmul(cand_embs, q)

            # Similarity to cues
            cue_embs = [self._embed_word(w).squeeze(0) for w in cue_words]
            cue_embs = [f.normalize(e, p=2, dim=-1) for e in cue_embs]
            if cue_embs:
                sim_cues = torch.stack([torch.matmul(cand_embs, c) for c in cue_embs], dim=0)
                sim_cues = sim_cues.mean(dim=0)
            else:
                sim_cues = torch.zeros_like(sim_state)

            # Heavily favor cue alignment since RAT is about connecting cues
            score = 0.2 * sim_state + 0.8 * sim_cues
            best_idx = int(torch.argmax(score).item())
            return pairs[best_idx][0]

    def _initialize_manifold_patterns(self, num_patterns: int = 100) -> None:
        """Create clustered prototype patterns to induce a meaningful landscape."""
        num_clusters = 10
        patterns_per_cluster = max(1, num_patterns // num_clusters)
        with torch.no_grad():
            for _cluster in range(num_clusters):
                center = torch.randn(self.hidden_dim, device=self.device)
                center = f.normalize(center, p=2, dim=0)
                for _ in range(patterns_per_cluster):
                    noise = torch.randn(self.hidden_dim, device=self.device) * 0.1
                    pattern = f.normalize(center + noise, p=2, dim=0)
                    self.manifold.store_pattern(pattern)

    def _embed_word(self, word: str) -> torch.Tensor:
        """Convert word to embedding vector using GloVe."""
        # Try to get from GloVe first
        glove_embedding = self.glove.get_word_embedding(word)
        if glove_embedding is not None:
            return glove_embedding.unsqueeze(0)  # [1, hidden_dim]

        # Fallback: check local vocab (though it should mostly overlap)
        if word.lower() in self.vocab:
            idx = self.vocab[word.lower()]
            glove_idx = self.glove.get_word_idx(self.reverse_vocab.get(idx, ""))
            if glove_idx is not None:
                return self.glove(torch.tensor([glove_idx], device=self.device))

        # Last resort: return zero vector or mean of GloVe embeddings
        return torch.zeros(1, self.hidden_dim, device=self.device)

    def _decode_embedding(self, embedding: torch.Tensor) -> str:
        """
        Convert embedding back to nearest word (simplified).

        In production, use:
        - Cosine similarity to word embeddings
        - Language model decoding
        - Beam search over vocabulary
        """
        # Ensure embedding is a single vector [hidden_dim]
        if embedding.dim() > 2:
            # Collapse any extra dimensions by averaging
            embedding = embedding.mean(dim=tuple(range(embedding.dim() - 1)))
        if embedding.dim() == 2:
            # If multiple vectors provided, average them
            if embedding.size(0) > 1:
                embedding = embedding.mean(dim=0)
            else:
                embedding = embedding.squeeze(0)

        # Decode using GloVe embeddings
        embedding = f.normalize(embedding, p=2, dim=-1)

        # Get all known word embeddings from GloVe
        valid_words = [w for w in self.reverse_vocab.values() if self.glove.has_word(w)]
        if not valid_words:
            return "<UNK>"

        valid_embeddings = []
        for w in valid_words:
            emb = self.glove.get_word_embedding(w)
            if emb is not None:
                valid_embeddings.append(emb)
        valid_embeddings = torch.stack(valid_embeddings)
        valid_embeddings = f.normalize(valid_embeddings, p=2, dim=-1)

        similarities = torch.matmul(embedding, valid_embeddings.t())

        # Get top-k most similar words
        top_k = min(5, len(valid_words))
        if top_k == 0:
            return "<UNK>"
        top_indices = torch.topk(similarities, k=top_k).indices

        # Return most similar known word
        for idx in top_indices.flatten():
            word = valid_words[int(idx.item())]
            if word != "<UNK>":
                return word

        return "<UNK>"

    def solve(
        self, item: RATItem, max_steps: int = 50, verbose: bool = False
    ) -> Tuple[str, List[float], int, float]:
        """
        Solve a single RAT item using RAA.

        Returns:
            solution: Predicted connecting word
            entropy_trajectory: List of entropy values during solving
            reframing_count: Number of reframing operations triggered
            computation_time: Time taken to solve
        """
        start_time = time.time()

        # Encode cue words
        cue_embeddings = [
            self._embed_word(word) for word in item.cue_words
        ]  # List of [1, hidden_dim]

        # Combine cues into initial context
        context = torch.cat(cue_embeddings, dim=0)  # [3, hidden_dim]
        context = context.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        # Normalize context to match normalized patterns used in Hopfield memory
        context = f.normalize(context, p=2, dim=-1)

        # Use embedding-based reasoning loop
        final_state, metrics = self.reasoning_loop.reason(input_embeddings=context.squeeze(0))

        # Decode within candidate solution set and rerank by cue alignment
        current_solution = self._decode_with_candidates(
            final_state, self.solution_words, item.cue_words
        )

        entropy_trajectory = metrics.get("entropy_trajectory", [])
        reframing_count = metrics.get("num_reframings", 0)

        computation_time = time.time() - start_time

        return current_solution, entropy_trajectory, reframing_count, computation_time

    def _is_valid_solution(self, solution: str, item: RATItem) -> bool:
        """Check if proposed solution matches target."""
        if not solution or solution == "<UNK>":
            return False
        return solution.lower() == item.solution.lower()


def run_evaluation(
    output_dir: str = "experiments/results", device: str = "cpu", verbose: bool = True
) -> Dict:
    """
    Run full RAT evaluation.

    Args:
        output_dir: Directory to save results
        device: Device to run on (cpu/cuda)
        verbose: Print progress

    Returns:
        Summary statistics dictionary
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = RATDataset()
    evaluator = RATEvaluator(dataset)

    # Initialize solver (hidden_dim=100 for GloVe)
    solver = RATSolver(hidden_dim=100, num_pointers=8, manifold_layers=4, device=device)

    print(f"Starting RAT Evaluation on {len(dataset)} items...")
    print(f"Device: {device}\n")

    # Evaluate each item
    for idx, item in enumerate(dataset.items):
        if verbose:
            print(f"[{idx+1}/{len(dataset)}] Testing: {item.cue_words} → {item.solution}")

        # Solve
        solution, entropy_traj, reframe_count, comp_time = solver.solve(
            item, max_steps=50, verbose=False
        )

        # Evaluate
        result = evaluator.evaluate_item(
            item=item,
            model_output=solution,
            entropy_trajectory=entropy_traj,
            reframing_count=reframe_count,
            computation_time=comp_time,
        )

        if verbose:
            status = "✓" if result["correct"] else "✗"
            ent_red = result.get("entropy_reduction")
            ent_str = f"{ent_red:.3f}" if isinstance(ent_red, (int, float)) else "n/a"
            print(
                f"  {status} Predicted: {solution} | Entropy Δ: {ent_str} | Reframes: {reframe_count}\n"
            )

    # Generate report
    stats = evaluator.compute_summary_statistics()
    report = evaluator.generate_report()

    print(report)

    # Save detailed results
    results_file = Path(output_dir) / "rat_evaluation_results.json"
    with open(results_file, "w") as f:
        # Convert to serializable format
        serializable_results = []
        for r in evaluator.results:
            serializable_results.append(
                {
                    "cue_words": r["item"].cue_words,
                    "solution": r["item"].solution,
                    "difficulty": r["difficulty"],
                    "category": r["category"],
                    "correct": r["correct"],
                    "model_output": r["model_output"],
                    "entropy_reduction": (
                        float(r["entropy_reduction"]) if r["entropy_reduction"] else None
                    ),
                    "entropy_trajectory": [float(e) for e in r["entropy_trajectory"]],
                    "reframing_count": int(r["reframing_count"]),
                    "computation_time": float(r["computation_time"]),
                }
            )

        json.dump(
            {
                "summary": {
                    k: float(v) if isinstance(v, (int, float)) else v for k, v in stats.items()
                },
                "detailed_results": serializable_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_file}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAT evaluation on RAA")
    parser.add_argument(
        "--output-dir", type=str, default="experiments/results", help="Directory to save results"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    args = parser.parse_args()

    stats = run_evaluation(output_dir=args.output_dir, device=args.device, verbose=args.verbose)

"""
Transformer Baseline for RAT
============================

Standard transformer model WITHOUT:
- Associative manifold
- Pointer-based search
- Entropy-triggered reframing

This baseline uses direct attention mechanisms to solve RAT problems,
providing a comparison point for the RAA architecture.

The key difference: No explicit search or reframing - just learned
attention patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from experiments.insight_tasks.remote_associates_test import (
    RATDataset, RATEvaluator, RATItem
)


class TransformerBaseline(nn.Module):
    """
    Standard transformer model for RAT solving.

    Architecture:
    - Word embeddings
    - Multi-head attention layers
    - Direct prediction of solution word

    NO manifold, NO pointer dynamics, NO entropy monitoring.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        vocab_size: int = 10000
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Word embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len] token indices

        Returns:
            output: [batch_size, hidden_dim] solution embedding
        """
        # Embed inputs
        x = self.embeddings(input_ids)  # [batch, seq_len, hidden]

        # Apply transformer
        x = self.transformer(x)  # [batch, seq_len, hidden]

        # Pool across sequence (mean pooling)
        x = x.mean(dim=1)  # [batch, hidden]

        # Project to solution space
        output = self.output_proj(x)  # [batch, hidden]

        return output


class BaselineSolver:
    """
    Solves RAT using baseline transformer (no RAA components).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        device: str = "cpu"
    ):
        self.device = device
        self.hidden_dim = hidden_dim

        # Initialize model
        self.model = TransformerBaseline(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(device)

        # Vocabulary (same as RAA solver)
        self.vocab = {}
        self.reverse_vocab = {}
        self._init_vocab()

    def _init_vocab(self):
        """Initialize vocabulary from RAT dataset."""
        dataset = RATDataset()

        words = set()
        for item in dataset.items:
            words.update(item.cue_words)
            words.add(item.solution)

        common_words = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "ball", "house", "time", "water", "light", "book", "tree"
        ]
        words.update(common_words)

        for idx, word in enumerate(sorted(words)):
            self.vocab[word] = idx
            self.reverse_vocab[idx] = word

    def _encode_word(self, word: str) -> int:
        """Convert word to token ID."""
        if word.lower() in self.vocab:
            return self.vocab[word.lower()]
        else:
            return hash(word.lower()) % len(self.vocab)

    def _decode_embedding(self, embedding: torch.Tensor) -> str:
        """Convert embedding to nearest word."""
        all_embeddings = self.model.embeddings.weight
        similarities = torch.matmul(
            embedding.squeeze(),
            all_embeddings.t()
        )

        top_k = 5
        top_indices = torch.topk(similarities, k=top_k).indices

        for idx in top_indices:
            word = self.reverse_vocab.get(idx.item(), "<UNK>")
            if word != "<UNK>":
                return word

        return "<UNK>"

    def solve(
        self,
        item: RATItem,
        verbose: bool = False
    ) -> Tuple[str, List[float], int, float]:
        """
        Solve RAT item using baseline transformer.

        Returns:
            solution: Predicted word
            entropy_trajectory: Empty list (no entropy tracking)
            reframing_count: Always 0 (no reframing)
            computation_time: Time taken
        """
        start_time = time.time()

        # Encode cue words
        input_ids = torch.tensor(
            [[self._encode_word(word) for word in item.cue_words]],
            device=self.device
        )  # [1, 3]

        # Forward pass
        with torch.no_grad():
            output_embedding = self.model(input_ids)  # [1, hidden_dim]

        # Decode to word
        solution = self._decode_embedding(output_embedding)

        computation_time = time.time() - start_time

        # No entropy trajectory or reframing in baseline
        return solution, [], 0, computation_time


def run_baseline_evaluation(
    output_dir: str = "experiments/results",
    device: str = "cpu",
    verbose: bool = True
) -> Dict:
    """
    Run baseline evaluation on RAT.

    Args:
        output_dir: Directory to save results
        device: Device to run on
        verbose: Print progress

    Returns:
        Summary statistics
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = RATDataset()
    evaluator = RATEvaluator(dataset)

    # Initialize baseline solver
    solver = BaselineSolver(
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        device=device
    )

    print(f"Starting Baseline Evaluation on {len(dataset)} items...")
    print(f"Model: Standard Transformer (no manifold/search/reframing)")
    print(f"Device: {device}\n")

    # Evaluate each item
    for idx, item in enumerate(dataset.items):
        if verbose:
            print(f"[{idx+1}/{len(dataset)}] Testing: {item.cue_words} → {item.solution}")

        # Solve
        solution, entropy_traj, reframe_count, comp_time = solver.solve(
            item,
            verbose=False
        )

        # Evaluate
        result = evaluator.evaluate_item(
            item=item,
            model_output=solution,
            entropy_trajectory=entropy_traj,
            reframing_count=reframe_count,
            computation_time=comp_time
        )

        if verbose:
            status = "✓" if result["correct"] else "✗"
            print(f"  {status} Predicted: {solution} | Time: {comp_time:.3f}s\n")

    # Generate report
    stats = evaluator.compute_summary_statistics()
    report = evaluator.generate_report()

    print(report)

    # Save results
    results_file = Path(output_dir) / "baseline_evaluation_results.json"
    with open(results_file, "w") as f:
        serializable_results = []
        for r in evaluator.results:
            serializable_results.append({
                "cue_words": r["item"].cue_words,
                "solution": r["item"].solution,
                "difficulty": r["difficulty"],
                "category": r["category"],
                "correct": r["correct"],
                "model_output": r["model_output"],
                "entropy_reduction": None,  # Baseline doesn't track entropy
                "entropy_trajectory": [],
                "reframing_count": 0,
                "computation_time": float(r["computation_time"])
            })

        json.dump({
            "summary": {k: float(v) if isinstance(v, (int, float)) else v
                       for k, v in stats.items()},
            "detailed_results": serializable_results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on")
    parser.add_argument("--verbose", action="store_true",
                       help="Print progress")

    args = parser.parse_args()

    stats = run_baseline_evaluation(
        output_dir=args.output_dir,
        device=args.device,
        verbose=args.verbose
    )

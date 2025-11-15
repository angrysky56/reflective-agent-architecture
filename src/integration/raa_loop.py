"""
Reflective Agent Architecture: Main Loop Implementation

Integrates Manifold, Processor, Pointer, and Director into the complete
metacognitive loop for insight-like problem solving.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from ..director import Director, DirectorConfig
from ..manifold import HopfieldConfig, Manifold
from ..pointer import Pointer, PointerConfig
from ..processor import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


@dataclass
class RAAConfig:
    """Complete configuration for Reflective Agent Architecture."""

    # Component configs
    hopfield_config: HopfieldConfig
    processor_config: ProcessorConfig
    pointer_config: PointerConfig
    director_config: DirectorConfig

    # Integration parameters
    max_reframing_attempts: int = 3  # Maximum goal reframings per generation
    enable_metacognition: bool = True  # Enable Director monitoring
    device: str = "cpu"


class ReflectiveAgentArchitecture:
    """
    Complete Reflective Agent Architecture.

    Implements the full "Aha!" loop integrating associative memory,
    metacognitive monitoring, and dynamic goal reframing.
    """

    def __init__(self, config: RAAConfig):
        """
        Initialize RAA with all components.

        Args:
            config: Complete RAA configuration
        """
        self.config = config
        self.device = config.device

        # Initialize components
        self.manifold = Manifold(config.hopfield_config)
        self.processor = Processor(config.processor_config)
        self.pointer = Pointer(config.pointer_config)
        self.director = Director(self.manifold, config.director_config)

        # Move to device
        self.manifold.to(self.device)
        self.processor.to(self.device)
        self.pointer.to(self.device)

        logger.info("Reflective Agent Architecture initialized")

    def set_initial_goal(self, goal_vector: torch.Tensor) -> None:
        """
        Set initial goal for the task.

        Args:
            goal_vector: Initial goal embedding
        """
        self.pointer.set_goal(goal_vector)
        logger.debug("Initial goal set")

    def generate_step(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Single generation step with optional metacognitive intervention.

        This implements the core "Aha!" loop:
        1. Get current goal from Pointer
        2. Generate with Processor (goal-biased)
        3. Director monitors entropy
        4. If clash detected, search and update goal
        5. Regenerate with new goal

        CRITICAL: Enforces max_reframing_attempts to prevent infinite loops.

        Args:
            input_ids: Input token sequence (batch, seq_length)
            temperature: Sampling temperature

        Returns:
            Dictionary containing:
                - next_token: Generated token
                - entropy: Output entropy
                - reframed: Whether goal was reframed
                - new_goal: New goal if reframed
                - reframing_attempts: Number of reframing attempts made
        """
        # Step 1: Get current goal
        current_goal = self.pointer.get_current_goal()

        # Step 2: Generate with Processor
        next_token, logits, entropy = self.processor.generate_next_token(
            input_ids,
            goal_state=current_goal,
            temperature=temperature,
        )

        result = {
            "next_token": next_token,
            "logits": logits,
            "entropy": entropy,
            "reframed": False,
            "new_goal": None,
            "search_result": None,
            "reframing_attempts": 0,
        }

        # Step 3-6: Metacognitive intervention with bounded attempts
        if self.config.enable_metacognition:
            attempts = 0
            current_entropy = entropy
            best_result = (next_token, logits, entropy)

            while attempts < self.config.max_reframing_attempts:
                new_goal = self.director.check_and_search(
                    current_state=current_goal,
                    processor_logits=logits,
                    context={"input_ids": input_ids, "attempt": attempts},
                )

                if new_goal is None:
                    # No intervention needed or search failed
                    break

                attempts += 1
                logger.info(
                    f"Metacognitive intervention: Reframing goal (attempt {attempts}/{self.config.max_reframing_attempts})"
                )

                # Update Pointer with new goal
                self.pointer.set_goal(new_goal)
                current_goal = new_goal

                # Regenerate with new goal
                next_token_new, logits_new, entropy_new = self.processor.generate_next_token(
                    input_ids,
                    goal_state=new_goal,
                    temperature=temperature,
                )

                logger.info(f"Reframing result: entropy {current_entropy:.3f} → {entropy_new:.3f}")

                # Check if entropy improved
                if entropy_new < current_entropy:
                    # Accept improvement
                    best_result = (next_token_new, logits_new, entropy_new)
                    current_entropy = entropy_new

                    result.update(
                        {
                            "next_token": next_token_new,
                            "logits": logits_new,
                            "entropy": entropy_new,
                            "reframed": True,
                            "new_goal": new_goal,
                            "reframing_attempts": attempts,
                        }
                    )

                    logger.info("Entropy reduced! Accepting new goal.")
                    break  # Success - exit loop
                else:
                    # No improvement - continue searching
                    logger.warning("Reframing did not reduce entropy. Continuing search...")
                    logits = logits_new  # Update for next entropy check

            if attempts >= self.config.max_reframing_attempts:
                logger.warning(
                    f"Reached max reframing attempts ({self.config.max_reframing_attempts}). Using best result."
                )
                # Keep the best result found (could be original or an intermediate)

        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        return_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate sequence with metacognitive monitoring.

        Args:
            input_ids: Input token sequence (batch, seq_length)
            max_length: Maximum generation length
            temperature: Sampling temperature
            return_history: Whether to return full generation history

        Returns:
            Dictionary containing:
                - output_ids: Generated sequence
                - generation_history: Optional list of step results
                - num_reframings: Number of goal reframings
        """
        current_ids = input_ids.clone()
        history = []
        num_reframings = 0

        for step in range(max_length):
            # Generate next token
            step_result = self.generate_step(current_ids, temperature)

            # Track reframings
            if step_result["reframed"]:
                num_reframings += 1

            # Append token
            next_token = step_result["next_token"]
            current_ids = torch.cat([current_ids, next_token.unsqueeze(-1)], dim=-1)

            if return_history:
                history.append(step_result)

            # Early stopping (optional)
            # Could add stopping criteria here

        result = {
            "output_ids": current_ids,
            "num_reframings": num_reframings,
        }

        if return_history:
            result["generation_history"] = history

        logger.info(f"Generation complete. Reframings: {num_reframings}")

        return result

    def store_concept(
        self,
        embedding: torch.Tensor,
        label: Optional[str] = None,
    ) -> None:
        """
        Store a concept in the Manifold.

        Args:
            embedding: Concept embedding vector
            label: Optional human-readable label
        """
        self.manifold.store_pattern(embedding)
        logger.debug(f"Stored concept: {label if label else 'unlabeled'}")

    def retrieve_from_manifold(
        self,
        query: torch.Tensor,
        num_steps: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve from associative memory.

        Args:
            query: Query embedding
            num_steps: Number of Hopfield update iterations

        Returns:
            retrieved_state: Converged state
            energy_trajectory: Energy at each step
        """
        return self.manifold.retrieve(query, num_steps)

    def reset_goal(self) -> None:
        """Reset Pointer to initial state."""
        self.pointer.reset()
        logger.debug("Goal state reset")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all components.

        Returns:
            Dictionary with component statistics
        """
        return {
            "manifold": {
                "num_patterns": self.manifold.num_patterns,
                "embedding_dim": self.manifold.embedding_dim,
            },
            "director": self.director.get_statistics(),
            "config": {
                "max_reframing_attempts": self.config.max_reframing_attempts,
                "metacognition_enabled": self.config.enable_metacognition,
            },
        }

    def __repr__(self) -> str:
        return (
            f"ReflectiveAgentArchitecture(\n"
            f"  Manifold: {self.manifold}\n"
            f"  Processor: {self.processor}\n"
            f"  Pointer: {self.pointer}\n"
            f"  Director: {self.director}\n"
            f")"
        )


def create_default_raa(
    embedding_dim: int = 512,
    vocab_size: int = 50257,
    device: str = "cpu",
) -> ReflectiveAgentArchitecture:
    """
    Create RAA with default configuration.

    Args:
        embedding_dim: Embedding dimension for all components
        vocab_size: Vocabulary size for Processor
        device: Device to run on

    Returns:
        Configured RAA instance
    """
    config = RAAConfig(
        hopfield_config=HopfieldConfig(
            embedding_dim=embedding_dim,
            device=device,
        ),
        processor_config=ProcessorConfig(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            device=device,
        ),
        pointer_config=PointerConfig(
            embedding_dim=embedding_dim,
            device=device,
        ),
        director_config=DirectorConfig(
            device=device,
        ),
        device=device,
    )

    return ReflectiveAgentArchitecture(config)

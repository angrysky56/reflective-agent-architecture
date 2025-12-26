"""
RAA Reasoning Loop: Embedding-Based Reasoning Integration

This module implements pure embedding-space reasoning for tasks like
Remote Associates Test, analogical reasoning, and conceptual problem solving.

Unlike the Generation Loop (token-based), this works entirely in embedding
space for faster reasoning without vocabulary constraints.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from reflective_agent_architecture.director import Director, DirectorConfig
from reflective_agent_architecture.manifold import HopfieldConfig, Manifold
from reflective_agent_architecture.pointer import GoalController, PointerConfig


@dataclass
class ReasoningConfig:
    """Configuration for RAA Reasoning Loop."""

    max_steps: int = 20  # Maximum reasoning iterations
    max_reframing_attempts: int = 5  # Maximum Director search attempts per step
    energy_threshold: float = (
        -10.0
    )  # Accept solution if energy below this (disabled by default with very low value)
    convergence_tolerance: float = (
        1e-4  # State change threshold for convergence (very tight for normalized vectors)
    )

    # Component configs
    manifold_config: Optional[HopfieldConfig] = None
    pointer_config: Optional[PointerConfig] = None
    director_config: Optional[DirectorConfig] = None

    device: str = "cpu"


class RAAReasoningLoop:
    """
    Pure embedding-based reasoning loop for insight tasks.

    This loop composes Manifold, Director, and Pointer for reasoning without
    token generation. Designed for tasks like:
    - Remote Associates Test (RAT)
    - Analogical reasoning (A:B::C:?)
    - Conceptual blending
    - Any task where solution is an embedding

    Architecture Flow:
    1. Initialize goal state from input embeddings (Pointer)
    2. Retrieve from Manifold based on current goal
    3. Compute energy and pseudo-logits for entropy
    4. Director monitors and triggers search if needed
    5. Update goal state (reframing or gradual evolution)
    6. Repeat until convergence or max steps
    """

    def __init__(
        self,
        manifold: Manifold,
        director: Director,
        pointer: GoalController,
        config: Optional[ReasoningConfig] = None,
    ):
        """
        Initialize RAA Reasoning Loop.

        Args:
            manifold: Manifold (Modern Hopfield Network) for pattern storage/retrieval
            director: Director for metacognitive monitoring and search
            pointer: Pointer (Goal Controller) for goal state management
            config: Reasoning loop configuration
        """
        self.manifold = manifold
        self.director = director
        self.pointer = pointer
        self.config = config or ReasoningConfig()

        self.device = self.config.device

        # Metrics tracking
        self.reset_metrics()

    def reset_metrics(self) -> None:
        """Reset metrics tracking."""
        self.metrics: dict[str, Any] = {
            "energy_trajectory": [],
            "entropy_trajectory": [],
            "reframing_events": [],
            "convergence_step": None,
            "num_reframings": 0,
            "final_energy": None,
            "final_entropy": None,
        }

    def _compute_pseudo_logits(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Compute pseudo-logits for Director entropy monitoring.

        In reasoning mode, we don't have a vocabulary, so we use the
        pattern attention distribution as a proxy for "prediction uncertainty".

        Args:
            current_state: Current goal state embedding

        Returns:
            Pseudo-logits (pattern similarities)
        """
        # Get pattern similarities (attention distribution before softmax)
        patterns = self.manifold.get_patterns()

        if patterns.shape[0] == 0:
            # No patterns: return uniform pseudo-logits
            return torch.zeros(1, 100).to(self.device)  # Arbitrary vocab size

        # Compute similarities (unnormalized attention)
        similarities = torch.matmul(current_state, patterns.T)  # (batch, num_patterns)

        # Scale by beta to match Hopfield attention mechanism
        scaled_similarities = self.manifold.beta * similarities

        return scaled_similarities

    def _check_convergence(self, current_state: torch.Tensor, previous_state: torch.Tensor) -> bool:
        """
        Check if reasoning has converged.

        Args:
            current_state: Current goal state
            previous_state: Previous goal state

        Returns:
            True if converged (state change below tolerance)
        """
        state_change = torch.norm(current_state - previous_state).item()
        return bool(state_change < self.config.convergence_tolerance)

    def reason_step(self, current_state: torch.Tensor, step: int) -> tuple[torch.Tensor, dict]:
        """
        Single reasoning step with metacognitive monitoring.

        Args:
            current_state: Current goal state embedding (batch, embedding_dim)
            step: Current step number

        Returns:
            new_state: Updated state after reasoning (batch, embedding_dim)
            step_metrics: Metrics for this step
        """
        step_metrics = {
            "step": step,
            "reframing_triggered": False,
            "num_search_attempts": 0,
        }

        # 1. Retrieve from Manifold based on current goal
        retrieved_state, energy_trajectory = self.manifold.retrieve(current_state)
        energy = energy_trajectory[-1].item()  # Final energy

        step_metrics["energy"] = energy
        self.metrics["energy_trajectory"].append(energy)

        # 2. Compute pseudo-logits for Director and entropy
        pseudo_logits = self._compute_pseudo_logits(current_state)

        # Compute entropy from pseudo-logits
        probs = torch.softmax(pseudo_logits, dim=-1)
        entropy = (-probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean().item()

        step_metrics["entropy"] = entropy
        self.metrics["entropy_trajectory"].append(entropy)

        # 3. Check if energy is EXTREMELY low (near-perfect match to stored pattern)
        # Note: Typical converged energies are -2 to -3, so we only stop if exceptionally low
        if energy < self.config.energy_threshold:
            step_metrics["solution_found"] = True
            self.metrics["convergence_step"] = step
            return retrieved_state, step_metrics

        # 4. Director monitors entropy and possibly triggers search
        new_goal = None
        search_attempts = 0

        while search_attempts < self.config.max_reframing_attempts:
            new_goal = self.director.check_and_search(
                current_state=current_state,
                processor_logits=pseudo_logits,
                context={"step": step, "energy": energy, "attempt": search_attempts},
            )

            search_attempts += 1

            if new_goal is None:
                # No clash detected or search failed
                break

            # Reframing triggered
            step_metrics["reframing_triggered"] = True
            self.metrics["num_reframings"] += 1
            self.metrics["reframing_events"].append(
                {
                    "step": step,
                    "old_energy": energy,
                    "attempt": search_attempts,
                }
            )

            # Evaluate new goal's energy
            new_retrieved, new_energy_traj = self.manifold.retrieve(new_goal)
            new_energy = new_energy_traj[-1].item()

            # Accept if energy improved
            if new_energy < energy:
                step_metrics["energy_improved"] = True
                step_metrics["energy_improvement"] = energy - new_energy
                current_state = new_goal
                retrieved_state = new_retrieved
                energy = new_energy
                break
            else:
                # Energy didn't improve, try another search
                step_metrics["energy_improved"] = False
                continue

        step_metrics["num_search_attempts"] = search_attempts

        # 5. Update Pointer with new goal (either from search or retrieved)
        if new_goal is not None:
            self.pointer.set_goal(new_goal)
        else:
            # Gradual evolution: update based on retrieved state
            self.pointer.update(retrieved_state)

        # 6. Get updated goal state
        updated_state = self.pointer.get_current_goal()

        return updated_state, step_metrics

    def reason(
        self, input_embeddings: torch.Tensor, return_trajectory: bool = False
    ) -> tuple[torch.Tensor, dict]:
        """
        Full reasoning loop from input to solution.

        Args:
            input_embeddings: Input problem embedding(s)
                - Shape: (embedding_dim,) for single input
                - Shape: (batch, embedding_dim) for batch
            return_trajectory: If True, return full state trajectory

        Returns:
            solution_state: Final reasoning state (same shape as input)
            metrics: Comprehensive metrics dict
        """
        # Reset metrics
        self.reset_metrics()

        # Track if input was unbatched
        squeeze_output = input_embeddings.dim() == 1
        if squeeze_output:
            input_embeddings = input_embeddings.unsqueeze(0)  # Add batch dim

        # Initialize Pointer with input
        batch_size = input_embeddings.shape[0]
        self.pointer.reset(batch_size=batch_size)
        self.pointer.set_goal(input_embeddings)

        current_state = self.pointer.get_current_goal()
        previous_state = current_state.clone()

        state_trajectory = [current_state.detach().cpu()] if return_trajectory else None

        # Reasoning loop
        step = -1
        for step in range(self.config.max_steps):
            # Note: reason_step handles batching internally, but we pass batched state
            current_state, step_metrics = self.reason_step(current_state, step)

            if return_trajectory and state_trajectory is not None:
                state_trajectory.append(current_state.detach().cpu())

            # Check convergence
            if step_metrics.get("solution_found", False):
                self.metrics["convergence_reason"] = "energy_threshold"
                break

            # Only check state convergence after first step (need valid previous state)
            if step > 0 and self._check_convergence(current_state, previous_state):
                self.metrics["convergence_step"] = step
                self.metrics["convergence_reason"] = "state_stability"
                break

            previous_state = current_state.clone()

        # Final metrics
        final_energy = self.manifold.energy(current_state).item()
        final_pseudo_logits = self._compute_pseudo_logits(current_state)
        probs = torch.softmax(final_pseudo_logits, dim=-1)
        final_entropy = (-probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean().item()

        self.metrics["final_energy"] = final_energy
        self.metrics["final_entropy"] = final_entropy
        self.metrics["total_steps"] = step + 1

        if return_trajectory:
            self.metrics["state_trajectory"] = state_trajectory

        # Remove batch dim if input was unbatched
        if squeeze_output:
            current_state = current_state.squeeze(0)

        return current_state, self.metrics

    def __repr__(self) -> str:
        return (
            f"RAAReasoningLoop(max_steps={self.config.max_steps}, "
            f"max_reframing={self.config.max_reframing_attempts})"
        )

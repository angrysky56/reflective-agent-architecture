"""
RAA Reasoning Loop: Embedding-Based Reasoning Integration (src)

This module mirrors experiments/insight_tasks/reasoning_loop.py but lives under src/integration
so downstream code can import via src.integration without referencing experiments.

It composes Manifold, Director, and Pointer to perform pure embedding-space reasoning for
insight tasks like RAT.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from src.director import Director, DirectorConfig
from src.manifold import HopfieldConfig, Manifold
from src.pointer import GoalController, PointerConfig


@dataclass
class ReasoningConfig:
    """Configuration for RAA Reasoning Loop."""

    max_steps: int = 20
    max_reframing_attempts: int = 5
    energy_threshold: float = 0.1
    convergence_tolerance: float = 1e-4

    # Component configs
    manifold_config: Optional[HopfieldConfig] = None
    pointer_config: Optional[PointerConfig] = None
    director_config: Optional[DirectorConfig] = None

    device: str = "cpu"


class RAAReasoningLoop:
    """
    Pure embedding-based reasoning loop for insight tasks.
    """

    def __init__(
        self,
        manifold: Manifold,
        director: Director,
        pointer: GoalController,
        config: Optional[ReasoningConfig] = None,
    ):
        self.manifold = manifold
        self.director = director
        self.pointer = pointer
        self.config = config or ReasoningConfig()
        self.device = self.config.device
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.metrics = {
            "energy_trajectory": [],
            "entropy_trajectory": [],
            "reframing_events": [],
            "convergence_step": None,
            "num_reframings": 0,
            "final_energy": None,
            "final_entropy": None,
        }

    def _compute_pseudo_logits(self, current_state: torch.Tensor) -> torch.Tensor:
        patterns = self.manifold.get_patterns()
        if patterns.shape[0] == 0:
            return torch.zeros(1, 100, device=self.device)
        similarities = torch.matmul(current_state, patterns.T)
        return self.manifold.beta * similarities

    def _check_convergence(self, current_state: torch.Tensor, previous_state: torch.Tensor) -> bool:
        state_change = torch.norm(current_state - previous_state).item()
        return state_change < self.config.convergence_tolerance

    def reason_step(self, current_state: torch.Tensor, step: int) -> tuple[torch.Tensor, dict]:
        step_metrics = {
            "step": step,
            "reframing_triggered": False,
            "num_search_attempts": 0,
        }

        # Compute pseudo logits and entropy BEFORE retrieval so we always record a value
        pseudo_logits = self._compute_pseudo_logits(current_state)
        probs = torch.softmax(pseudo_logits, dim=-1)
        entropy = (-probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean().item()
        self.metrics["entropy_trajectory"].append(entropy)

        retrieved_state, energy_traj = self.manifold.retrieve(current_state)
        energy = energy_traj[-1].item()
        step_metrics["energy"] = energy
        self.metrics["energy_trajectory"].append(energy)

        if energy < self.config.energy_threshold:
            step_metrics["solution_found"] = True
            self.metrics["convergence_step"] = step
            return retrieved_state, step_metrics

        new_goal = None
        attempts = 0
        while attempts < self.config.max_reframing_attempts:
            new_goal = self.director.check_and_search(
                current_state=current_state,
                processor_logits=pseudo_logits,
                context={"step": step, "energy": energy, "attempt": attempts},
            )
            attempts += 1
            if new_goal is None:
                break

            step_metrics["reframing_triggered"] = True
            self.metrics["num_reframings"] += 1
            self.metrics["reframing_events"].append(
                {"step": step, "old_energy": energy, "attempt": attempts}
            )

            new_retrieved, new_energy_traj = self.manifold.retrieve(new_goal)
            new_energy = new_energy_traj[-1].item()
            if new_energy < energy:
                step_metrics["energy_improved"] = True
                step_metrics["energy_improvement"] = energy - new_energy
                current_state = new_goal
                retrieved_state = new_retrieved
                energy = new_energy
                break
            else:
                step_metrics["energy_improved"] = False
                pseudo_logits = self._compute_pseudo_logits(new_goal)

        step_metrics["num_search_attempts"] = attempts

        if new_goal is not None:
            self.pointer.set_goal(new_goal)
        else:
            self.pointer.update(retrieved_state)

        updated_state = self.pointer.get_current_goal()
        return updated_state, step_metrics

    def reason(
        self, input_embeddings: torch.Tensor, return_trajectory: bool = False
    ) -> tuple[torch.Tensor, dict]:
        self.reset_metrics()
        self.pointer.reset(
            batch_size=input_embeddings.shape[0] if input_embeddings.dim() > 1 else 1
        )
        self.pointer.set_goal(input_embeddings)
        current_state = self.pointer.get_current_goal()
        previous_state = current_state.clone()
        state_traj = [current_state.detach().cpu()] if return_trajectory else None

        step = -1
        for step in range(self.config.max_steps):
            current_state, step_metrics = self.reason_step(current_state, step)
            if return_trajectory and state_traj is not None:
                state_traj.append(current_state.detach().cpu())
            if step_metrics.get("solution_found", False):
                self.metrics["convergence_reason"] = "energy_threshold"
                break
            if self._check_convergence(current_state, previous_state):
                self.metrics["convergence_step"] = step
                self.metrics["convergence_reason"] = "state_stability"
                break
            previous_state = current_state.clone()

        final_energy = self.manifold.energy(current_state).item()
        final_pseudo_logits = self._compute_pseudo_logits(current_state)
        probs = torch.softmax(final_pseudo_logits, dim=-1)
        final_entropy = (-probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean().item()
        self.metrics["final_energy"] = final_energy
        self.metrics["final_entropy"] = final_entropy
        self.metrics["total_steps"] = step + 1
        if return_trajectory:
            self.metrics["state_trajectory"] = state_traj
        return current_state, self.metrics

    def __repr__(self) -> str:
        return (
            f"RAAReasoningLoop(max_steps={self.config.max_steps}, "
            f"max_reframing={self.config.max_reframing_attempts})"
        )

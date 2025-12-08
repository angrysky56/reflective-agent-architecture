from typing import Any

import torch
import torch.nn as nn


class PrecuneusIntegrator(nn.Module):
    """
    The 'Signal Consolidator' (Precuneus).

    Fuses the three Tripartite streams (State, Agent, Action) into a unified
    experience vector. Uses Hopfield Energy as a proxy for uncertainty to
    gate (down-weight) confused streams.

    Now also modulates State stream weight based on Cognitive State Entropy.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Fusion layer: Compresses 3x dimensions back to 1x
        self.fusion = nn.Linear(dim * 3, dim)
        self.layer_norm = nn.LayerNorm(dim)

        # Learnable "Default Mode" bias
        self.default_mode_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, vectors: dict, energies: dict, causal_signature: torch.Tensor = None, cognitive_state: Any = None) -> torch.Tensor:
        """
        Integrate the tripartite streams.

        Args:
            vectors: {'state': tensor, 'agent': tensor, 'action': tensor}
            energies: {'state': float, 'agent': float, 'action': float}
                      (Lower energy = higher certainty)
            causal_signature: Optional tensor representing agent's causal history.
                              Used to modulate weights (Continuity Field).
            cognitive_state: Optional StateDescriptor from Director.
                             Used to modulate State stream weight based on entropy.
        """

        # 1. Normalize Energies to Gating Weights (0 to 1)
        # We invert energy: High energy (confusion) -> Low weight
        state_w = self._energy_to_gate(energies['state'])
        agent_w = self._energy_to_gate(energies['agent'])
        action_w = self._energy_to_gate(energies['action'])

        # 1.5 Apply Continuity Field (Causal Signature)
        # If a causal signature is provided, it modulates the weights.
        # Strong causal history -> Higher weight stability.
        if causal_signature is not None:
            # Simple modulation: Boost weights based on alignment with causal signature
            # This is a simplification of the TKUI tensor contraction
            continuity_boost = torch.sigmoid(torch.norm(causal_signature))

            # Apply boost primarily to the 'agent' stream as it carries the identity
            agent_w = agent_w * (1.0 + 0.5 * continuity_boost)

            # Normalize weights to keep them in reasonable range
            total_w = state_w + agent_w + action_w + 1e-6
            state_w = state_w / total_w
            agent_w = agent_w / total_w
            action_w = action_w / total_w

        # 1.6 Apply Cognitive State Entropy Modulation (Phase 6)
        if cognitive_state is not None:
            # High entropy (confusion) -> Trust external/action more -> Lower State weight
            # Low entropy (crystallized) -> Trust internal state more -> Higher State weight

            # Entropy is typically 0.0 to ~5.0+
            # We want a multiplier.
            # If entropy is high (e.g. > 2.0), we reduce state_w.
            # If entropy is low (e.g. < 0.5), we boost state_w.

            entropy = getattr(cognitive_state, 'entropy', 1.0)

            # Sigmoid-like modulation centered at 1.0
            # entropy 0 -> boost
            # entropy 2 -> dampen

            # Simple heuristic:
            # multiplier = 1.0 / (1.0 + entropy)  -> 0.5 at ent=1, 0.33 at ent=2
            # But we want to boost if low.

            # Let's use: multiplier = 2.0 / (1.0 + entropy)
            # ent=0 -> mult=2.0
            # ent=1 -> mult=1.0
            # ent=3 -> mult=0.5

            entropy_mod = 2.0 / (1.0 + max(0.0, float(entropy)))
            state_w = state_w * entropy_mod

        # Normalize weights again to keep them in reasonable range
        total_w = state_w + agent_w + action_w + 1e-6
        state_w = state_w / total_w
        agent_w = agent_w / total_w
        action_w = action_w / total_w

        # 2. Gate the signals
        state_gated = vectors['state'] * state_w
        agent_gated = vectors['agent'] * agent_w
        action_gated = vectors['action'] * action_w

        # 3. Concatenate (Fragment integration)
        concatenated = torch.cat([state_gated, agent_gated, action_gated], dim=-1)

        # 4. Fuse (The "Unified Experience")
        integrated = self.fusion(concatenated)
        integrated = self.layer_norm(integrated + self.default_mode_bias)

        # 5. Compute Coherence Info (for self-awareness)
        coherence_info = {
            "context_weight": float(state_w),    # How much State contributes
            "perspective_weight": float(agent_w), # How much Agent contributes
            "operation_weight": float(action_w),  # How much Action contributes
            "balance": 1.0 - float(torch.std(torch.tensor([state_w, agent_w, action_w]))),  # High = balanced
            "dominant_stream": max(
                [("context", float(state_w)), ("perspective", float(agent_w)), ("operation", float(action_w))],
                key=lambda x: x[1]
            )[0]
        }

        return integrated, coherence_info

    def _energy_to_gate(self, energy: float, distinctiveness: float = 5.0) -> float:
        """
        Convert Hopfield Energy to a gating weight [0, 1].

        Sigmoid function squashes energy.
        - Low Energy (e.g. -5) -> High Weight (~1.0)
        - High Energy (e.g. +5) -> Low Weight (~0.0)
        """
        # Ensure energy is a tensor for calculation
        if not isinstance(energy, torch.Tensor):
            e_tensor = torch.tensor(energy, dtype=torch.float32)
        else:
            e_tensor = energy.float()

        return torch.sigmoid(-e_tensor * distinctiveness)

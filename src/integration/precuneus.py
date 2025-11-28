import torch
import torch.nn as nn
import torch.nn.functional as f


class PrecuneusIntegrator(nn.Module):
    """
    The 'Signal Consolidator' (Precuneus).

    Fuses the three Tripartite streams (State, Agent, Action) into a unified
    experience vector. Uses Hopfield Energy as a proxy for uncertainty to
    gate (down-weight) confused streams.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Fusion layer: Compresses 3x dimensions back to 1x
        self.fusion = nn.Linear(dim * 3, dim)
        self.layer_norm = nn.LayerNorm(dim)

        # Learnable "Default Mode" bias
        self.default_mode_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, vectors: dict, energies: dict) -> torch.Tensor:
        """
        Integrate the tripartite streams.

        Args:
            vectors: {'state': tensor, 'agent': tensor, 'action': tensor}
            energies: {'state': float, 'agent': float, 'action': float}
                      (Lower energy = higher certainty)

        Returns:
            integrated: Unified experience tensor (dim,)
        """

        # 1. Normalize Energies to Gating Weights (0 to 1)
        # We invert energy: High energy (confusion) -> Low weight
        # This effectively implements the "Silence the confusion" logic
        state_w = self._energy_to_gate(energies['state'])
        agent_w = self._energy_to_gate(energies['agent'])
        action_w = self._energy_to_gate(energies['action'])

        # 2. Gate the signals
        state_gated = vectors['state'] * state_w
        agent_gated = vectors['agent'] * agent_w
        action_gated = vectors['action'] * action_w

        # 3. Concatenate (Fragment integration)
        concatenated = torch.cat([state_gated, agent_gated, action_gated], dim=-1)

        # 4. Fuse (The "Unified Experience")
        integrated = self.fusion(concatenated)
        integrated = self.layer_norm(integrated + self.default_mode_bias)

        return integrated

    def _energy_to_gate(self, energy: float, distinctiveness: float = 5.0) -> float:
        """
        Convert Hopfield Energy to a gating weight [0, 1].

        Sigmoid function squashes energy.
        - Low Energy (e.g. -5) -> High Weight (~1.0)
        - High Energy (e.g. +5) -> Low Weight (~0.0)
        """
        # Ensure energy is a tensor for calculation
        if not isinstance(energy, torch.Tensor):
            e_tensor = torch.tensor(energy)
        else:
            e_tensor = energy

        return torch.sigmoid(-e_tensor * distinctiveness)

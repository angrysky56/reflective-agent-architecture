from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

# Mocking the components for the test since they don't exist yet
# This serves as a TDD specification

# --- Mocks & Stubs ---

@dataclass
class MockConfig:
    dim: int = 64
    beta_state: float = 5.0
    beta_agent: float = 10.0
    beta_action: float = 50.0

class MockHopfield:
    def __init__(self, dim, beta):
        self.dim = dim
        self.beta = beta
        self.patterns = []

    def store(self, vec):
        self.patterns.append(vec)

    def retrieve(self, query):
        # Simple mock: if query matches a pattern, low energy. Else high.
        # Energy = -DotProduct (simplified)
        best_sim = -10.0
        best_pat = torch.zeros(self.dim)

        if not self.patterns:
            return torch.zeros(self.dim), 10.0 # High energy (confusion)

        for p in self.patterns:
            sim = torch.dot(query, p)
            if sim > best_sim:
                best_sim = sim
                best_pat = p

        # Energy is inverse of similarity for this mock
        energy = -best_sim.item()
        return best_pat, energy

class TripartiteManifold:
    def __init__(self, config):
        self.state_memory = MockHopfield(config.dim, config.beta_state)
        self.agent_memory = MockHopfield(config.dim, config.beta_agent)
        self.action_memory = MockHopfield(config.dim, config.beta_action)

    def retrieve(self, query_dict):
        return {
            'state': self.state_memory.retrieve(query_dict['state']),
            'agent': self.agent_memory.retrieve(query_dict['agent']),
            'action': self.action_memory.retrieve(query_dict['action'])
        }

class PrecuneusIntegrator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fusion = nn.Linear(dim * 3, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.default_mode_bias = nn.Parameter(torch.zeros(dim))

    def _energy_to_gate(self, energy, distinctiveness=5.0):
        # Sigmoid: High energy -> Low weight
        # We assume energy is roughly [-1, 1] for this mock,
        # so we shift it to make 0 neutral.
        # Actually, let's use the user's formula: sigmoid(-energy * d)
        # If energy is high (e.g. 10), -50 -> sigmoid is 0.
        # If energy is low (e.g. -10), 50 -> sigmoid is 1.
        return torch.sigmoid(torch.tensor(-energy * distinctiveness))

    def forward(self, vectors, energies):
        state_w = self._energy_to_gate(energies['state'])
        agent_w = self._energy_to_gate(energies['agent'])
        action_w = self._energy_to_gate(energies['action'])

        # Gate
        state_gated = vectors['state'] * state_w
        agent_gated = vectors['agent'] * agent_w
        action_gated = vectors['action'] * action_w

        # Fuse
        concatenated = torch.cat([state_gated, agent_gated, action_gated], dim=-1)
        integrated = self.fusion(concatenated)
        integrated = self.layer_norm(integrated + self.default_mode_bias)

        return integrated, {'state': state_w, 'agent': agent_w, 'action': action_w}

# --- Tests ---

def test_confusion_gating():
    """
    The 'Confusion Test':
    Clear Context (State) + Impossible Action (Action)
    Expectation: State weight HIGH, Action weight LOW.
    """
    config = MockConfig()
    manifold = TripartiteManifold(config)
    precuneus = PrecuneusIntegrator(config.dim)

    # 1. Train Manifold with a known State ("Lava")
    lava_vec = torch.randn(config.dim)
    lava_vec = lava_vec / torch.norm(lava_vec) # Normalize
    manifold.state_memory.store(lava_vec)

    # 2. Query
    # State: "Lava" (Known) -> Should have Low Energy
    # Action: "Swim" (Unknown/Empty memory) -> Should have High Energy

    query_state = lava_vec # Perfect match
    query_action = torch.randn(config.dim) # Random noise (Unknown)
    query_agent = torch.randn(config.dim) # Random noise

    query = {'state': query_state, 'agent': query_agent, 'action': query_action}

    results = manifold.retrieve(query)

    vectors = {k: v[0] for k, v in results.items()}
    energies = {k: v[1] for k, v in results.items()}

    print(f"\nEnergies: {energies}")

    # 3. Integrate
    unified, weights = precuneus(vectors, energies)

    print(f"Weights: {weights}")

    # 4. Assertions
    # State energy should be low (approx -1.0 due to dot product of normalized vecs)
    assert energies['state'] < 0.0, "Known state should have low energy"

    # Action energy should be high (approx 10.0 due to empty mock memory)
    assert energies['action'] > 5.0, "Unknown action should have high energy"

    # Weights
    assert weights['state'] > 0.9, "Known state should be gated IN"
    assert weights['action'] < 0.1, "Unknown action should be gated OUT"

if __name__ == "__main__":
    test_confusion_gating()
    print("Test Passed!")

import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import config, stats_utils
from src.manifold.hopfield_network import HopfieldConfig, ModernHopfieldNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizationAgent:
    def __init__(self, name):
        self.name = name

    def optimize(self, net: ModernHopfieldNetwork, initial_state: torch.Tensor, steps: int) -> tuple[float, int]:
        """
        Optimize the state to minimize energy.
        Returns: (final_energy, steps_taken)
        """
        raise NotImplementedError

class RandomAgent(OptimizationAgent):
    def optimize(self, net, initial_state, steps):
        current_state = initial_state.clone()
        best_energy = net.energy(current_state).item()

        for t in range(steps):
            # Flip random bits
            mask = torch.rand_like(current_state) < 0.1
            candidate = current_state.clone()
            candidate[mask] = -candidate[mask]

            energy = net.energy(candidate).item()
            # Random agent doesn't necessarily keep best, but let's say it's a "Random Walker"
            # that accepts everything? Or a "Random Search" that keeps best?
            # Let's make it Random Search (keeps best) to be a fair "baseline optimizer".
            if energy < best_energy:
                best_energy = energy
                current_state = candidate

        return best_energy, steps

class GreedyAgent(OptimizationAgent):
    def optimize(self, net, initial_state, steps):
        current_state = initial_state.clone()
        dim = current_state.shape[1]

        for t in range(steps):
            current_energy = net.energy(current_state).item()
            best_flip = None
            min_energy = current_energy

            # Try flipping each bit (Steepest Descent)
            # For high dim, this is slow. Let's try flipping random subset?
            # Or just "First Improvement".
            # Let's do "Best of N random flips" to simulate limited compute.
            # Or just standard greedy on the continuous relaxation if possible?
            # Hopfield nets are continuous.
            # Let's use the gradient of the energy function!

            # Enable grad
            x = current_state.clone().detach().requires_grad_(True)
            E = net.energy(x)
            E.backward()
            grad = x.grad

            # Move against gradient
            # x_new = x - lr * grad
            # But states are binary/bounded?
            # ModernHopfieldNetwork in this codebase uses continuous states usually?
            # Let's check the implementation.
            # Assuming continuous for now as standard MHN.

            with torch.no_grad():
                x_new = x - 0.1 * grad
                # Normalize or clip?
                # If patterns are normalized, we should normalize.
                # Assuming patterns are on sphere or hypercube.
                # Let's just take the step.

            current_state = x_new

            # Check convergence
            if torch.norm(grad) < 1e-3:
                return net.energy(current_state).item(), t

        return net.energy(current_state).item(), steps

class SimulatedAnnealingAgent(OptimizationAgent):
    def optimize(self, net, initial_state, steps):
        current_state = initial_state.clone()
        current_energy = net.energy(current_state).item()

        T_start = 1.0
        T_end = 0.01

        for t in range(steps):
            T = T_start * ((T_end/T_start) ** (t/steps))

            # Propose move (random perturbation)
            candidate = current_state + torch.randn_like(current_state) * 0.1
            candidate_energy = net.energy(candidate).item()

            delta_E = candidate_energy - current_energy

            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                current_state = candidate
                current_energy = candidate_energy

        return current_energy, steps

class DirectorAgent(OptimizationAgent):
    def optimize(self, net, initial_state, steps):
        # The Director uses the Hopfield Update Rule (Attention)
        # This is the "native" dynamics of the network.
        current_state = initial_state.clone()

        for t in range(steps):
            prev_state = current_state.clone()
            current_state = net.update_step(current_state)

            # Check convergence
            if torch.norm(current_state - prev_state) < 1e-4:
                return net.energy(current_state).item(), t + 1

        return net.energy(current_state).item(), steps

def run_experiment():
    logger.info("Starting Experiment 8: Gradient of Intelligence")

    # Set seeds
    seed = 42 # New seed for new experiment
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Setup Environment
    dim = 64
    n_patterns = 10
    # net = ModernHopfieldNetwork(dimension=dim) # Error

    # Correct initialization
    h_config = HopfieldConfig(embedding_dim=dim)
    net = ModernHopfieldNetwork(h_config)

    # Store random patterns
    patterns = torch.randn(n_patterns, dim)
    # Normalize patterns to unit sphere (standard for MHN)
    patterns = patterns / patterns.norm(dim=1, keepdim=True)
    net.store_pattern(patterns)

    n_trials = 50
    max_steps = 20

    agents = [
        RandomAgent("Random"),
        GreedyAgent("GradientDescent"),
        SimulatedAnnealingAgent("SimAnnealing"),
        DirectorAgent("Director (Attention)")
    ]

    results = {agent.name: {'energies': [], 'steps': []} for agent in agents}

    for i in range(n_trials):
        # Random start state
        initial_state = torch.randn(1, dim)
        initial_state = initial_state / initial_state.norm(dim=1, keepdim=True)

        for agent in agents:
            e, s = agent.optimize(net, initial_state, max_steps)
            results[agent.name]['energies'].append(e)
            results[agent.name]['steps'].append(s)

    # Analysis
    logger.info("Analysis Results:")
    summary_data = []

    for name, data in results.items():
        mean_e = np.mean(data['energies'])
        mean_s = np.mean(data['steps'])
        logger.info(f"{name}: Mean Energy = {mean_e:.4f}, Mean Steps = {mean_s:.2f}")
        summary_data.append({'Agent': name, 'Energy': mean_e, 'Steps': mean_s})

    # Statistical Test: Director vs GradientDescent
    director_energies = results["Director (Attention)"]['energies']
    greedy_energies = results["GradientDescent"]['energies']

    stat, p = stats.ttest_rel(director_energies, greedy_energies)
    d = (np.mean(director_energies) - np.mean(greedy_energies)) / np.std(np.array(director_energies) - np.array(greedy_energies))

    logger.info(f"Director vs GradientDescent: p={p:.4e}, d={d:.4f}")

    # Save results
    config.save_experiment_data(
        experiment_id="exp8_gradient_intelligence",
        raw_data=results,
        analysis_results={'summary': summary_data, 'ttest_director_greedy': {'p': p, 'd': d}}
    )

if __name__ == "__main__":
    run_experiment()

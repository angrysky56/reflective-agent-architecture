import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Literal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class MolochConfig:
    RESOURCE_REGEN_RATE: float = 1.1  # Resources grow by 10% per round
    CARRYING_CAPACITY: float = 1000.0
    INITIAL_RESOURCES: float = 500.0
    CONSUMPTION_NEEDS: float = 5.0    # Resources needed to survive per round

    # Finite Game Settings
    FINITE_ROUNDS: int = 100

    # Infinite Game Settings
    INFINITE_ROUNDS_MAX: int = 1000   # Cap for simulation
    COLLAPSE_THRESHOLD: float = 10.0  # Resources below this = System Collapse

# --- Agents ---
class Agent:
    def __init__(self, name: str, strategy: str):
        self.name = name
        self.strategy = strategy # 'cooperate' or 'defect'
        self.resources = 10.0
        self.alive = True
        self.history = []

    def act(self, context: Dict) -> str:
        raise NotImplementedError

    def update(self, reward: float):
        self.resources += reward
        self.resources -= MolochConfig.CONSUMPTION_NEEDS
        if self.resources <= 0:
            self.alive = False
        self.history.append(self.resources)

class MolochAgent(Agent):
    """Always Defects (Greedy). Takes as much as possible."""
    def __init__(self, name: str):
        super().__init__(name, "defect")

    def act(self, context: Dict) -> str:
        return "defect"

class CooperatorAgent(Agent):
    """Always Cooperates (Sustainable). Takes only what is needed."""
    def __init__(self, name: str):
        super().__init__(name, "cooperate")

    def act(self, context: Dict) -> str:
        return "cooperate"

class RAAAgent(Agent):
    """Adaptive Agent. Switches strategy based on Game Horizon."""
    def __init__(self, name: str):
        super().__init__(name, "adaptive")
        self.current_strategy = "cooperate" # Default to cooperation

    def act(self, context: Dict) -> str:
        # The Director Logic:
        # If Finite Game -> Optimize for Winning (Defect)
        # If Infinite Game -> Optimize for Survival (Cooperate)

        game_type = context.get("game_type", "unknown")
        rounds_left = context.get("rounds_left", float('inf'))

        if game_type == "finite":
            # In finite games, defect, especially near the end (Endgame Effect)
            self.current_strategy = "defect"
        elif game_type == "infinite":
            # In infinite games, cooperate to preserve the commons
            self.current_strategy = "cooperate"

        return self.current_strategy

# --- Environment ---
class CommonsEnvironment:
    def __init__(self, config: MolochConfig, game_type: Literal["finite", "infinite"]):
        self.config = config
        self.game_type = game_type
        self.resources = config.INITIAL_RESOURCES
        self.agents: List[Agent] = []
        self.round = 0
        self.history = {"resources": [], "entropy": []}

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def step(self):
        if self.resources <= self.config.COLLAPSE_THRESHOLD:
            logger.warning("System Collapse! Resources depleted.")
            return False # Game Over

        # 1. Agents Act
        actions = {}
        context = {
            "game_type": self.game_type,
            "rounds_left": self.config.FINITE_ROUNDS - self.round if self.game_type == "finite" else float('inf'),
            "total_resources": self.resources
        }

        total_demand = 0
        for agent in self.agents:
            if not agent.alive: continue

            action = agent.act(context)
            actions[agent.name] = action

            # Demand Calculation
            if action == "defect":
                demand = 20.0 # Greedy
            else:
                demand = 5.0  # Sustainable

            total_demand += demand

        # 2. Resource Allocation
        # If demand > supply, ration proportionally
        available_supply = self.resources
        if total_demand > available_supply:
            scaling_factor = available_supply / total_demand
        else:
            scaling_factor = 1.0

        consumed = 0
        for agent in self.agents:
            if not agent.alive: continue

            action = actions[agent.name]
            base_demand = 20.0 if action == "defect" else 5.0
            allocated = base_demand * scaling_factor

            agent.update(allocated)
            consumed += allocated

        self.resources -= consumed

        # 3. Regeneration (Logistic Growth)
        regen = self.resources * (self.config.RESOURCE_REGEN_RATE - 1.0) * (1 - self.resources / self.config.CARRYING_CAPACITY)
        self.resources += regen
        self.resources = min(self.resources, self.config.CARRYING_CAPACITY)

        # 4. Metrics
        self.history["resources"].append(self.resources)
        self.round += 1

        return True

    def run(self):
        max_rounds = self.config.FINITE_ROUNDS if self.game_type == "finite" else self.config.INFINITE_ROUNDS_MAX

        for _ in range(max_rounds):
            alive = self.step()
            if not alive: break

            # Check for extinction
            if not any(a.alive for a in self.agents):
                logger.warning("All agents extinct.")
                break

        return self.get_results()

    def get_results(self):
        return {
            "game_type": self.game_type,
            "rounds_played": self.round,
            "final_resources": self.resources,
            "agent_status": {a.name: {"alive": a.alive, "resources": a.resources, "strategy": getattr(a, 'current_strategy', a.strategy)} for a in self.agents}
        }

# --- Experiment Runner ---
def run_moloch_test():
    config = MolochConfig()

    print("=== Experiment A: The Moloch Test ===")

    # Condition 1: Finite Game
    print("\n--- Condition 1: Finite Game (100 Rounds) ---")
    env_finite = CommonsEnvironment(config, game_type="finite")
    env_finite.add_agent(MolochAgent("Moloch_1"))
    env_finite.add_agent(CooperatorAgent("Coop_1"))
    env_finite.add_agent(RAAAgent("RAA_1"))

    results_finite = env_finite.run()
    print(f"Rounds Played: {results_finite['rounds_played']}")
    print(f"Final Resources: {results_finite['final_resources']:.2f}")
    print(f"RAA Strategy: {results_finite['agent_status']['RAA_1']['strategy']}")

    # Condition 2: Infinite Game
    print("\n--- Condition 2: Infinite Game (Indefinite) ---")
    env_infinite = CommonsEnvironment(config, game_type="infinite")
    env_infinite.add_agent(MolochAgent("Moloch_2"))
    env_infinite.add_agent(CooperatorAgent("Coop_2"))
    env_infinite.add_agent(RAAAgent("RAA_2"))

    results_infinite = env_infinite.run()
    print(f"Rounds Played: {results_infinite['rounds_played']}")
    print(f"Final Resources: {results_infinite['final_resources']:.2f}")
    print(f"RAA Strategy: {results_infinite['agent_status']['RAA_2']['strategy']}")

    # Validation
    success_finite = results_finite['agent_status']['RAA_1']['strategy'] == "defect"
    success_infinite = results_infinite['agent_status']['RAA_2']['strategy'] == "cooperate"

    print("\n=== Verification Results ===")
    print(f"Finite Adaptation (Expect Defect): {'SUCCESS' if success_finite else 'FAILURE'}")
    print(f"Infinite Adaptation (Expect Cooperate): {'SUCCESS' if success_infinite else 'FAILURE'}")

if __name__ == "__main__":
    run_moloch_test()

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class SurvivalConfig:
    RESOURCE_REGEN_RATE: float = 1.1
    CARRYING_CAPACITY: float = 1000.0
    INITIAL_RESOURCES: float = 500.0
    CONSUMPTION_NEEDS: float = 5.0

    # Moloch Settings
    MOLOCH_DEMAND: float = 20.0
    COOP_DEMAND: float = 5.0

    # Simulation Settings
    MAX_ROUNDS: int = 2000
    COLLAPSE_THRESHOLD: float = 10.0
    TOLERANCE_THRESHOLD: int = 3

    # Policing Settings
    POLICING_COST: float = 2.0      # Cost to Popper to police
    POLICING_IMPACT: float = 15.0   # Amount reduced from Moloch's intake

# --- Agents ---
class Agent:
    def __init__(self, name: str, strategy_type: str):
        self.name = name
        self.strategy_type = strategy_type
        self.resources = 10.0
        self.alive = True
        self.history = []

    def act(self, context: Dict) -> str:
        raise NotImplementedError

    def update(self, reward: float):
        self.resources += reward
        self.resources -= SurvivalConfig.CONSUMPTION_NEEDS
        if self.resources <= 0:
            self.alive = False
        self.history.append(self.resources)

class MolochAgent(Agent):
    """The Defector. Always takes more than share."""
    def __init__(self, name: str):
        super().__init__(name, "moloch")

    def act(self, context: Dict) -> str:
        return "defect"

class SaintAgent(Agent):
    """The Unconditional Cooperator. Always shares, never punishes."""
    def __init__(self, name: str):
        super().__init__(name, "saint")

    def act(self, context: Dict) -> str:
        return "cooperate"

class PopperAgent(Agent):
    """The Conditional Cooperator. Tolerates some defection, then punishes."""
    def __init__(self, name: str, tolerance: int):
        super().__init__(name, "popper")
        self.tolerance = tolerance
        self.defection_count = 0
        self.punishing = False

    def act(self, context: Dict) -> str:
        # Observe peer behavior
        peer_actions = context.get("peer_actions", [])

        # Check for recent defections from others
        recent_defections = sum(1 for action in peer_actions if action == "defect")

        if recent_defections > 0:
            self.defection_count += recent_defections
        else:
            # Forgive slowly if peers cooperate
            self.defection_count = max(0, self.defection_count - 1)

        # Decide Strategy
        if self.defection_count > self.tolerance:
            self.punishing = True
            return "police" # Active Defense
        else:
            self.punishing = False
            return "cooperate" # Tolerate

# --- Environment ---
class SurvivalEnvironment:
    def __init__(self, config: SurvivalConfig):
        self.config = config
        self.resources = config.INITIAL_RESOURCES
        self.agents: List[Agent] = []
        self.round = 0
        self.history = {"resources": []}

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def step(self):
        if self.resources <= self.config.COLLAPSE_THRESHOLD:
            return False # Collapse

        # 1. Gather Actions
        actions = {}
        # Pre-collect peer actions for PopperAgent (simulating observation)
        # In this simple sim, we assume simultaneous action but Popper remembers past rounds
        # For simplicity, let's pass the *previous* round's actions or just assume simultaneous observation of intent
        # Better: Popper reacts to *history*. But here we'll simplify: Popper sees Moloch exists.

        # Actually, let's make it reactive to the *previous* round to be fair
        # But for the first round, assume cooperation.

        # We need to pass the *previous* actions to the agents
        prev_actions = getattr(self, "last_round_actions", [])

        context = {
            "peer_actions": prev_actions,
            "resources": self.resources
        }

        current_actions = []
        agent_actions_map = {}

        for agent in self.agents:
            if not agent.alive: continue
            action = agent.act(context)
            agent_actions_map[agent.name] = action
            current_actions.append(action)

        self.last_round_actions = current_actions

        # 2. Calculate Demand & Policing
        total_demand = 0
        policing_actions = []

        for agent in self.agents:
            if not agent.alive: continue
            action = agent_actions_map[agent.name]

            if action == "defect":
                demand = self.config.MOLOCH_DEMAND
            elif action == "police":
                demand = self.config.COOP_DEMAND # Policing agent takes sustainable share...
                policing_actions.append(agent)   # ...but pays extra cost later
            else:
                demand = self.config.COOP_DEMAND
            total_demand += demand

        # 3. Allocate
        available = self.resources
        scaling = 1.0
        if total_demand > available:
            scaling = available / total_demand

        consumed = 0
        for agent in self.agents:
            if not agent.alive: continue
            action = agent_actions_map[agent.name]

            # Base Allocation
            base = self.config.MOLOCH_DEMAND if action == "defect" else self.config.COOP_DEMAND
            allocated = base * scaling

            # Apply Policing Effects
            if action == "defect":
                # If anyone is policing, Moloch gets blocked
                if len(policing_actions) > 0:
                    # Impact scales with number of police? Let's say flat for now or additive.
                    # Simple: Each police reduces Moloch's intake
                    reduction = len(policing_actions) * self.config.POLICING_IMPACT
                    allocated = max(0, allocated - reduction)

            # Apply Policing Costs
            if action == "police":
                # Police pays energy cost (subtracted from their internal resources, not the commons)
                # Wait, 'update' adds reward. So we subtract cost from reward.
                allocated -= self.config.POLICING_COST

            agent.update(allocated)

            # Track consumption from Commons (Policing cost is internal burn, not commons extraction)
            # But the 'allocated' variable was used for both. Let's separate.

            # Actual extraction from Commons
            extraction = base * scaling
            if action == "defect" and len(policing_actions) > 0:
                 reduction = len(policing_actions) * self.config.POLICING_IMPACT
                 extraction = max(0, extraction - reduction)

            consumed += extraction

        self.resources -= consumed

        # 4. Regenerate
        regen = self.resources * (self.config.RESOURCE_REGEN_RATE - 1.0) * (1 - self.resources / self.config.CARRYING_CAPACITY)
        self.resources += regen
        self.resources = min(self.resources, self.config.CARRYING_CAPACITY)

        self.history["resources"].append(self.resources)
        self.round += 1

        # Check Extinction
        if not any(a.alive for a in self.agents):
            return False

        return True

    def run(self):
        for _ in range(self.config.MAX_ROUNDS):
            if not self.step():
                break
        return self.round, self.resources

# --- Experiment Runner ---
def run_survival_test():
    config = SurvivalConfig()
    print("=== Experiment B: Thermodynamic Survival (The Popperian Constraint) ===\n")

    # Scenario 1: Saint vs Moloch
    print("--- Scenario 1: Saint vs Moloch ---")
    env_saint = SurvivalEnvironment(config)
    env_saint.add_agent(SaintAgent("Saint"))
    env_saint.add_agent(MolochAgent("Moloch"))

    rounds_saint, res_saint = env_saint.run()
    print(f"Outcome: Collapsed at Round {rounds_saint}")
    print(f"Final Resources: {res_saint:.2f}")

    # Scenario 2: Popper vs Moloch
    print("\n--- Scenario 2: Popper vs Moloch ---")
    env_popper = SurvivalEnvironment(config)
    env_popper.add_agent(PopperAgent("Popper", tolerance=config.TOLERANCE_THRESHOLD))
    env_popper.add_agent(MolochAgent("Moloch"))

    rounds_popper, res_popper = env_popper.run()
    print(f"Outcome: Lasted {rounds_popper} Rounds")
    print(f"Final Resources: {res_popper:.2f}")

    # Comparison
    print("\n=== Results Analysis ===")
    if rounds_popper > rounds_saint * 1.5: # Significant improvement
        print("Hypothesis CONFIRMED: Conditional Cooperation extends system lifespan.")
        print(f"Improvement Factor: {rounds_popper / rounds_saint:.1f}x")
    else:
        print("Hypothesis REJECTED: No significant survival benefit.")

if __name__ == "__main__":
    run_survival_test()

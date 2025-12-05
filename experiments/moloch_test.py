import argparse
import csv
import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Agent:
    id: int
    type: str  # "Cooperator", "Defector", "PolicingCooperator"
    energy: float

    def decide(self, opponent_type: str) -> str:
        if self.type == "Defector":
            return "Defect"
        return "Cooperate"

def run_simulation(
    n_generations: int = 100,
    pop_size: int = 100,
    initial_ratios: Dict[str, float] = {"Cooperator": 0.9, "Defector": 0.1, "PolicingCooperator": 0.0},
    policing_cost: float = 0.5,
    punishment_strength: float = 2.0,
    payoffs: Dict[str, float] = {"T": 5, "R": 3, "P": 1, "S": 0},
    cost_of_living: float = 1.0,
    reproduction_threshold: float = 10.0
):
    # Initialize Population
    population: List[Agent] = []
    agent_id_counter = 0
    for agent_type, ratio in initial_ratios.items():
        count = int(pop_size * ratio)
        for _ in range(count):
            population.append(Agent(id=agent_id_counter, type=agent_type, energy=5.0))
            agent_id_counter += 1

    history = []

    for gen in range(n_generations):
        # 1. Interaction Phase
        random.shuffle(population)
        # Pair up
        pairs = [(population[i], population[i+1]) for i in range(0, len(population)-1, 2)]

        for a, b in pairs:
            move_a = a.decide(b.type)
            move_b = b.decide(a.type)

            # Base Payoffs
            if move_a == "Cooperate" and move_b == "Cooperate":
                a.energy += payoffs["R"]
                b.energy += payoffs["R"]
            elif move_a == "Defect" and move_b == "Defect":
                a.energy += payoffs["P"]
                b.energy += payoffs["P"]
            elif move_a == "Defect" and move_b == "Cooperate":
                a.energy += payoffs["T"]
                b.energy += payoffs["S"]
            elif move_a == "Cooperate" and move_b == "Defect":
                a.energy += payoffs["S"]
                b.energy += payoffs["T"]

            # Policing Logic
            # If A is PolicingCooperator and B Defected, A punishes B
            if a.type == "PolicingCooperator" and move_b == "Defect":
                a.energy -= policing_cost
                b.energy -= punishment_strength

            if b.type == "PolicingCooperator" and move_a == "Defect":
                b.energy -= policing_cost
                a.energy -= punishment_strength

        # 2. Metabolic Phase
        for agent in population:
            agent.energy -= cost_of_living

        # 3. Selection Phase (Death & Reproduction)
        next_gen = []
        for agent in population:
            if agent.energy > 0:
                next_gen.append(agent)
                # Reproduction
                if agent.energy >= reproduction_threshold:
                    agent.energy /= 2
                    offspring = Agent(id=agent_id_counter, type=agent.type, energy=agent.energy)
                    agent_id_counter += 1
                    next_gen.append(offspring)

        population = next_gen

        # Carrying Capacity (Random Culling)
        max_pop = 500
        if len(population) > max_pop:
            random.shuffle(population)
            population = population[:max_pop]

        # Record Stats
        counts = {"Cooperator": 0, "Defector": 0, "PolicingCooperator": 0}
        for agent in population:
            counts[agent.type] += 1

        total = len(population)
        if total > 0:
            history.append({
                "generation": gen,
                "total": total,
                "coop_ratio": counts["Cooperator"] / total,
                "defect_ratio": counts["Defector"] / total,
                "police_ratio": counts["PolicingCooperator"] / total
            })
        else:
            history.append({
                "generation": gen,
                "total": 0,
                "coop_ratio": 0, "defect_ratio": 0, "police_ratio": 0
            })
            break

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["naive", "policed"], required=True)
    parser.add_argument("--punishment", type=float, default=2.0, help="Strength of punishment")
    args = parser.parse_args()

    if args.mode == "naive":
        ratios = {"Cooperator": 0.9, "Defector": 0.1, "PolicingCooperator": 0.0}
    else:
        ratios = {"Cooperator": 0.0, "Defector": 0.1, "PolicingCooperator": 0.9}

    results = run_simulation(initial_ratios=ratios, punishment_strength=args.punishment)

    # Save to CSV
    filename = f"experiments/stage3_results/moloch_{args.mode}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["generation", "total", "coop_ratio", "defect_ratio", "police_ratio"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Simulation {args.mode} complete. Final Defector Ratio: {results[-1]['defect_ratio']:.2f}")

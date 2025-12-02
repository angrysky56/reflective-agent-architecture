import math
import operator
import random
from typing import Any, Callable, Dict, List, Union


# AST Nodes
class Node:
    def evaluate(self, context: Dict[str, float]) -> float:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

class Constant(Node):
    def __init__(self, value: float):
        self.value = value

    def evaluate(self, context: Dict[str, float]) -> float:
        return self.value

    def __str__(self) -> str:
        return f"{self.value:.2f}"

class Variable(Node):
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, context: Dict[str, float]) -> float:
        return context.get(self.name, 0.0)

    def __str__(self) -> str:
        return self.name

class BinaryOp(Node):
    def __init__(self, left: Node, right: Node, op: Callable, symbol: str):
        self.left = left
        self.right = right
        self.op = op
        self.symbol = symbol

    def evaluate(self, context: Dict[str, float]) -> float:
        try:
            return self.op(self.left.evaluate(context), self.right.evaluate(context))
        except ZeroDivisionError:
            return 1.0 # Protect against division by zero
        except OverflowError:
            return 1e6 # Cap overflow

    def __str__(self) -> str:
        return f"({self.left} {self.symbol} {self.right})"

class UnaryOp(Node):
    def __init__(self, child: Node, op: Callable, symbol: str):
        self.child = child
        self.op = op
        self.symbol = symbol

    def evaluate(self, context: Dict[str, float]) -> float:
        try:
            return self.op(self.child.evaluate(context))
        except ValueError:
            return 0.0 # Domain error
        except OverflowError:
            return 1e6

    def __str__(self) -> str:
        return f"{self.symbol}({self.child})"

# Operations
OPS = [
    (operator.add, "+"),
    (operator.sub, "-"),
    (operator.mul, "*"),
    # (operator.truediv, "/"), # Division is dangerous for stability
]

# Simple GP
class SimpleGP:
    def __init__(self, variables: List[str], population_size: int = 50, max_depth: int = 4):
        self.variables = variables
        self.population_size = population_size
        self.max_depth = max_depth
        self.population: List[Node] = []

    def generate_random_tree(self, depth: int = 0) -> Node:
        if depth >= self.max_depth or (depth > 0 and random.random() < 0.3):
            # Terminal
            if random.random() < 0.7:
                return Variable(random.choice(self.variables))
            else:
                return Constant(random.uniform(-10, 10))
        else:
            # Operator
            op, symbol = random.choice(OPS)
            left = self.generate_random_tree(depth + 1)
            right = self.generate_random_tree(depth + 1)
            return BinaryOp(left, right, op, symbol)

    def fit(self, data: List[Dict[str, float]], target_key: str, generations: int = 10) -> str:
        # Initialize population
        self.population = [self.generate_random_tree() for _ in range(self.population_size)]

        best_program = None
        best_error = float('inf')

        for gen in range(generations):
            # Evaluate fitness
            scored_population = []
            for program in self.population:
                error = 0.0
                for row in data:
                    try:
                        pred = program.evaluate(row)
                        target = row[target_key]
                        error += (pred - target) ** 2
                    except Exception:
                        error += 1e9

                # MSE
                error /= len(data)
                scored_population.append((program, error))

                if error < best_error:
                    best_error = error
                    best_program = program

            # Selection (Tournament)
            new_population = []
            if best_program:
                new_population.append(best_program) # Elitism

            while len(new_population) < self.population_size:
                parent = self.tournament_select(scored_population)
                child = self.mutate(parent)
                new_population.append(child)

            self.population = new_population

        return str(best_program) if best_program else "0"

    def tournament_select(self, scored_population: List[Any]) -> Node:
        candidates = random.sample(scored_population, k=3)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def mutate(self, program: Node) -> Node:
        # Simple mutation: replace a subtree with a random tree
        if random.random() < 0.2:
            return self.generate_random_tree(depth=random.randint(0, self.max_depth-1))

        # Recursive mutation
        if isinstance(program, BinaryOp):
            return BinaryOp(self.mutate(program.left), self.mutate(program.right), program.op, program.symbol)
        elif isinstance(program, UnaryOp):
            return UnaryOp(self.mutate(program.child), program.op, program.symbol)
        else:
            return program

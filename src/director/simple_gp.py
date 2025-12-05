import math
import operator
import random
from typing import Any, Callable, Dict, List, Tuple

from scipy.optimize import minimize

# Define Focused Primitive Sets (Attention Mechanism)
TRIG_OPS = [
    (operator.add, "+"),
    (operator.sub, "-"),
    (operator.mul, "*"),
]
TRIG_UNARY_OPS = [
    (math.sin, "sin"),
    (math.cos, "cos"),
]


# AST Nodes
class Node:
    def evaluate(self, context: Dict[str, float]) -> float:
        raise NotImplementedError

    def extract_constants(self) -> List['Constant']:
        """Return a list of all Constant nodes in the subtree."""
        return []

    def update_constants(self, values: List[float]) -> int:
        """
        Update constants in the subtree with new values.
        Returns the number of constants consumed.
        """
        return 0

    def __str__(self) -> str:
        raise NotImplementedError

class Constant(Node):
    def __init__(self, value: float):
        self.value = value

    def evaluate(self, context: Dict[str, float]) -> float:
        return self.value

    def extract_constants(self) -> List['Constant']:
        return [self]

    def update_constants(self, values: List[float]) -> int:
        if len(values) > 0:
            self.value = values[0]
            return 1
        return 0

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

    def extract_constants(self) -> List['Constant']:
        return self.left.extract_constants() + self.right.extract_constants()

    def update_constants(self, values: List[float]) -> int:
        consumed_left = self.left.update_constants(values)
        consumed_right = self.right.update_constants(values[consumed_left:])
        return consumed_left + consumed_right

    def __str__(self) -> str:
        if self.op == math.hypot:
            return f"hypot({self.left}, {self.right})"
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

    def extract_constants(self) -> List['Constant']:
        return self.child.extract_constants()

    def update_constants(self, values: List[float]) -> int:
        return self.child.update_constants(values)

    def __str__(self) -> str:
        return f"{self.symbol}({self.child})"

# Operations
OPS = [
    (operator.add, "+"),
    (operator.sub, "-"),
    (operator.mul, "*"),
    (math.hypot, "hypot"),
]

UNARY_OPS = [
    (math.sin, "sin"),
    (math.cos, "cos"),
    (math.tanh, "tanh"),
    (abs, "abs"),
]

# Simple GP
class SimpleGP:
    def __init__(self, variables: List[str], population_size: int = 50, max_depth: int = 4,
                 ops: List[Tuple[Callable, str]] = None, unary_ops: List[Tuple[Callable, str]] = None):
        self.variables = variables
        self.population_size = population_size
        self.max_depth = max_depth
        self.ops = ops if ops is not None else OPS
        self.unary_ops = unary_ops if unary_ops is not None else UNARY_OPS
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
            if random.random() < 0.7: # 70% chance for binary op
                op, symbol = random.choice(self.ops)
                left = self.generate_random_tree(depth + 1)
                right = self.generate_random_tree(depth + 1)
                return BinaryOp(left, right, op, symbol)
            else: # 30% chance for unary op
                op, symbol = random.choice(self.unary_ops)
                child = self.generate_random_tree(depth + 1)
                return UnaryOp(child, op, symbol)

    def evolve(self, data: List[Dict[str, float]], target_key: str, generations: int = 10, hybrid: bool = False) -> Tuple[str, float]:
        """
        Evolve a formula to fit the data.

        Args:
            data: List of data points (dicts).
            target_key: Key in dict for the target value.
            generations: Number of generations to evolve.
            hybrid: If True, use local refinement (Evolutionary Optimization).

        Returns:
            Tuple[str, float]: (Best formula string, Best MSE)
        """
        self.population = [self.generate_random_tree(depth=random.randint(2, self.max_depth)) for _ in range(self.population_size)]

        best_program = None
        best_error = float('inf')

        for gen in range(generations):
            # Evaluate fitness
            scored_population = []
            for program in self.population:

                # Hybrid Optimization Step
                if hybrid:
                    self.refine_individual(program, data, target_key)

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

        return (str(best_program) if best_program else "0", best_error)

    def refine_individual(self, program: Node, data: List[Dict[str, float]], target_key: str):
        """
        Locally refine the constants of a program using gradient-free optimization (Nelder-Mead).
        This implements the 'Neural Pipeline' part of Evolutionary Optimization.
        """
        constants = program.extract_constants()
        if not constants:
            return

        initial_values = [c.value for c in constants]

        def objective(values):
            # Temporarily update constants
            program.update_constants(values)
            error = 0.0
            for row in data:
                try:
                    pred = program.evaluate(row)
                    target = row[target_key]
                    error += (pred - target) ** 2
                except Exception:
                    error += 1e9
            return error / len(data)

        # Run optimization
        # We use Nelder-Mead as it's robust and doesn't require gradients of the AST
        result = minimize(objective, initial_values, method='Nelder-Mead', tol=1e-3, options={'maxiter': 10})

        # Apply best found constants
        program.update_constants(result.x)

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

import math
import random

import numpy as np


class AlienPhysicsGenerator:
    def __init__(self, complexity=3):
        self.variables = ['x', 'y', 'z']
        self.operators = [
            (lambda a, b: a + b, '+'),
            (lambda a, b: a - b, '-'),
            (lambda a, b: a * b, '*'),
            # We avoid division to prevent div/0 errors for simplicity
             (lambda a, b: (a**2 + b**2)**0.5, 'hypot'),
        ]
        self.unary_operators = [
            (lambda a: math.sin(a), 'sin'),
            (lambda a: math.cos(a), 'cos'),
            (lambda a: abs(a), 'abs'),
            (lambda a: math.tanh(a), 'tanh')
        ]
        self.formula_str = ""
        self.formula_func, self.formula_str = self._generate_random_formula(complexity)

    def _generate_random_formula(self, depth):
        if depth == 0:
            # Base case: return a variable or a random constant
            if random.random() > 0.3:
                var = random.choice(self.variables)
                return (lambda inputs: inputs[var]), var
            else:
                const = round(random.uniform(1, 10), 2)
                return (lambda inputs: const), str(const)

        # Recursive case: random operation
        if random.random() > 0.5: # Binary operator
            op_func, op_symbol = random.choice(self.operators)
            left_func, left_str = self._generate_random_formula(depth - 1)
            right_func, right_str = self._generate_random_formula(depth - 1)

            def current_func(inputs):
                return op_func(left_func(inputs), right_func(inputs))

            return current_func, f"({left_str} {op_symbol} {right_str})"
        else: # Unary operator
            op_func, op_symbol = random.choice(self.unary_operators)
            inner_func, inner_str = self._generate_random_formula(depth - 1)

            def current_func(inputs):
                return op_func(inner_func(inputs))

            return current_func, f"{op_symbol}({inner_str})"

    def generate_data_stream(self, n_points=50):
        data = []
        for _ in range(n_points):
            inputs = {
                'x': random.uniform(-10, 10),
                'y': random.uniform(-10, 10),
                'z': random.uniform(-10, 10)
            }
            # Calculate true output
            try:
                true_output = self.formula_func(inputs)
                # Add slight "measurement noise" to test the Director's tolerance
                noisy_output = true_output + random.gauss(0, 0.05)
                data.append((inputs, noisy_output))
            except:
                continue # Skip domain errors
        return data

# --- EXECUTION ---
if __name__ == "__main__":
    physics = AlienPhysicsGenerator(complexity=3)
    print(f"HIDDEN TRUTH (Do not show Agent): {physics.formula_str}", flush=True)

    print("\n--- OBSERVATION STREAM (Feed this to RAA) ---", flush=True)
    stream = physics.generate_data_stream(10)
    for i, (inp, out) in enumerate(stream):
        print(f"Observation {i+1}: inputs={inp}, result={out:.4f}", flush=True)

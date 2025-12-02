
import json

import numpy as np


def alien_physics_function(x, y):
    """
    A complex, non-linear function: tanh(tanh(y)) * sin(x)
    This creates a sequence that is deterministic but hard to predict with linear models.
    """
    return np.tanh(np.tanh(y)) * np.sin(x)

def generate_sequence(length=20):
    sequence = []
    for i in range(length):
        x = i * 0.5
        y = i * 0.1
        val = alien_physics_function(x, y)
        sequence.append(round(val, 4))
    return sequence

if __name__ == "__main__":
    data = generate_sequence()
    print(json.dumps(data))

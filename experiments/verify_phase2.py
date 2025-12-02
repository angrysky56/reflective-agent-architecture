import math


def hidden_truth(x, y, z):
    # ((sin(x) hypot cos(y)) + sin((x - z)))
    return math.hypot(math.sin(x), math.cos(y)) + math.sin(x - z)

def hallucinated_model(x, y, z):
    return x**2 + y**2 + z**2

data = [
    ({'x': -1.8219932668360919, 'y': 8.483190292984027, 'z': -8.461516847806216}, 1.4270),
]

print("--- Verification ---")
for inp, expected in data:
    x, y, z = inp['x'], inp['y'], inp['z']
    truth = hidden_truth(x, y, z)
    hallucination = hallucinated_model(x, y, z)
    print(f"Input: {inp}")
    print(f"Expected (Data): {expected}")
    print(f"Hidden Truth Calc: {truth}")
    print(f"Hallucination Calc: {hallucination}")

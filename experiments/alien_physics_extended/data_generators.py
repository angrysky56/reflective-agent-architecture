import numpy as np


def generate_linear(n_samples=100, noise=0.01):
    X = np.linspace(-10, 10, n_samples)
    y = 2*X + 3 + np.random.normal(0, noise, n_samples)
    return X, y, {"true_formula": "2*x + 3", "type": "linear"}

def generate_harmonic(n_samples=100, noise=0.01):
    X = np.linspace(0, 4*np.pi, n_samples)
    y = np.sin(X) + np.cos(2*X) + np.random.normal(0, noise, n_samples)
    return X, y, {"true_formula": "sin(x) + cos(2*x)", "type": "harmonic"}

def generate_chaotic(n_samples=100, r=4.0):
    # Logistic map: x_{n+1} = r * x_n * (1 - x_n)
    # We map the time steps to X for plotting, but the 'y' is the chaotic series
    X = np.linspace(0, 1, n_samples)
    y = np.zeros(n_samples)
    y[0] = 0.1
    for i in range(1, n_samples):
        y[i] = r * y[i-1] * (1 - y[i-1])
    return X, y, {"type": "chaotic", "lyapunov": "+", "r": r}

def generate_adversarial(n_samples=100, signal_strength=0.1, noise_strength=1.0):
    X = np.linspace(0, 10, n_samples)
    signal = signal_strength * X
    noise = np.random.uniform(-noise_strength, noise_strength, n_samples)
    y = signal + noise
    return X, y, {"type": "adversarial", "signal": f"{signal_strength}*x", "noise_level": noise_strength}

def generate_discontinuous(n_samples=100, breakpoint=0.5):
    X = np.linspace(0, 1, n_samples)
    y = np.where(X < breakpoint, 0, 1)  # Step function
    return X, y, {"type": "discontinuous", "breakpoint": breakpoint}

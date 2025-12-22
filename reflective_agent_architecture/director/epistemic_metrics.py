import zlib
from typing import Any

import numpy as np


def estimate_lyapunov_exponent(data: np.ndarray, dataset_step: int = 1) -> float:
    """
    Estimate the largest Lyapunov exponent using Rosenstein's algorithm (simplified).
    Positive LE indicates chaos.
    """
    n = len(data)
    m = 3  # Embedding dimension
    tau = 1  # Time delay

    # Reconstruct phase space
    # X = [x(t), x(t+tau), ..., x(t+(m-1)tau)]
    if n < (m - 1) * tau + 2:
        return 0.0

    # Simplified: Just look at divergence of nearest neighbors
    # This is a proxy, not a rigorous LE calculation, but sufficient for classification
    # For a 1D map x_{n+1} = f(x_n), LE = mean(ln|f'(x)|)
    # We approximate f'(x) by looking at neighbors

    # Sort data to find neighbors in value space (for 1D maps)
    idx = np.argsort(data)
    sorted_data = data[idx]

    divergence = []
    for i in range(n - 1):
        # Neighbor in value space
        d0 = abs(sorted_data[i + 1] - sorted_data[i])
        if d0 < 1e-6:
            continue

        # Look at their next iteration (using original indices)
        # We need to find where these values appeared in time
        t1 = idx[i]
        t2 = idx[i + 1]

        if t1 + 1 < n and t2 + 1 < n:
            d1 = abs(data[t1 + 1] - data[t2 + 1])
            if d1 > 1e-6:
                divergence.append(np.log(d1 / d0))

    if not divergence:
        return 0.0

    return float(np.mean(divergence))


def estimate_complexity(y: Any) -> dict[str, Any]:
    """
    Composite complexity estimation using multiple signals.
    Returns: dict with complexity score and diagnostic info
    """
    # Ensure y is float array
    y = np.array(y, dtype=float)

    # Check for discontinuities FIRST
    # We assume X is uniformly spaced for this check
    dy = np.gradient(y)  # smooth derivative
    dy_jumps = np.abs(dy)

    # Threshold: 3 sigma from mean gradient
    # Robust statistics: use median and MAD to avoid outliers skewing the threshold
    median_grad = np.median(dy_jumps)
    mad_grad = np.median(np.abs(dy_jumps - median_grad))
    jump_threshold = median_grad + 5 * mad_grad  # Conservative threshold

    # Fallback for low variance
    if mad_grad < 1e-6:
        jump_threshold = np.std(dy_jumps) * 3

    if np.any(dy_jumps > jump_threshold):
        jump_locations = np.where(dy_jumps > jump_threshold)[0]
        return {
            "complexity_score": 0.9,  # very high
            "type": "discontinuous",
            "jump_locations": jump_locations.tolist(),
            "recommendation": "piecewise_fitting",
        }

    # Kolmogorov proxy (compressibility)
    # Quantize to 8-bit for compression
    y_min, y_max = np.min(y), np.max(y)
    if y_max - y_min < 1e-9:
        y_quant = np.zeros_like(y, dtype=np.uint8)
    else:
        y_quant = ((y - y_min) / (y_max - y_min) * 255).astype(np.uint8)

    compressed_size = len(zlib.compress(y_quant.tobytes()))
    original_size = len(y_quant.tobytes())
    compressibility = compressed_size / original_size

    # Lyapunov exponent (chaos indicator)
    lyapunov = estimate_lyapunov_exponent(y)

    # Spectral complexity (frequency richness)
    fft = np.fft.fft(y)
    power = np.abs(fft) ** 2
    # Normalize power to sum to 1
    p_norm = power / (np.sum(power) + 1e-10)
    # Entropy of power spectrum
    spectral_entropy = -np.sum(p_norm * np.log2(p_norm + 1e-10))
    spectral_entropy_normalized = spectral_entropy / np.log2(len(y))  # normalize to [0,1]

    # Composite score
    # We weight compressibility higher as it's a good general proxy
    complexity_score = float(
        np.mean(
            [
                compressibility,
                min(max(lyapunov, 0.0), 1.0),  # cap at 1, floor at 0
                spectral_entropy_normalized,
            ]
        )
    )

    return {
        "complexity_score": complexity_score,
        "type": "smooth" if complexity_score < 0.7 else "complex",
        "compressibility": compressibility,
        "lyapunov": lyapunov,
        "spectral_entropy": spectral_entropy_normalized,
    }


def estimate_randomness(y: Any) -> dict[str, Any]:
    """
    Composite randomness estimation.
    Returns: dict with randomness score and diagnostic info
    """
    y = np.array(y, dtype=float)

    # Entropy rate (unpredictability of first difference)
    dy = np.diff(y)
    # Binning
    hist, _ = np.histogram(dy, bins=min(50, max(1, len(dy) // 4)), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    max_entropy = np.log2(len(hist))
    entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0

    # Autocorrelation decay (memory)
    # Normalize
    y_centered = y - np.mean(y)
    if np.std(y) < 1e-9:
        acf = np.ones_like(y)
    else:
        acf = np.correlate(y_centered, y_centered, mode="full")
        acf = acf[len(acf) // 2 :]
        acf = acf / (acf[0] + 1e-10)

    # Find decay rate (first time it drops below 1/e or 0.1)
    decay_indices = np.where(np.abs(acf[1:]) < 0.36)[0]  # 1/e approx
    if len(decay_indices) > 0:
        memory_length = decay_indices[0]
        # Score: 1.0 for length 0 (random), 0.0 for length N (periodic/constant)
        memory_score = 1.0 - (memory_length / len(y))
    else:
        memory_length = len(y)
        memory_score = 0.0  # long memory = structured

    # Spectral flatness (white noise indicator)
    spectrum = np.abs(np.fft.fft(y)) ** 2 + 1e-10
    # Only use first half (real signal)
    spectrum = spectrum[: len(spectrum) // 2]

    geometric_mean = np.exp(np.mean(np.log(spectrum)))
    arithmetic_mean = np.mean(spectrum)
    spectral_flatness = geometric_mean / arithmetic_mean

    # Composite score
    randomness_score = float(np.mean([entropy_normalized, memory_score, spectral_flatness]))

    return {
        "randomness_score": randomness_score,
        "type": "random" if randomness_score > 0.6 else "structured",
        "entropy_rate": entropy_normalized,
        "memory_length": memory_length,
        "spectral_flatness": spectral_flatness,
    }

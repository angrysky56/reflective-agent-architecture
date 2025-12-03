import logging
import os
import random
import string
import sys
import zlib

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import config, stats_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_random_text(length: int) -> bytes:
    """Generate random text."""
    chars = string.ascii_letters + string.digits + " "
    return "".join(random.choices(chars, k=length)).encode('utf-8')

def generate_correlated_text(context: bytes, mutation_rate: float = 0.1) -> bytes:
    """Generate text that is a mutated version of the context."""
    # Convert to mutable list
    chars = list(context.decode('utf-8'))
    n_mutations = int(len(chars) * mutation_rate)

    for _ in range(n_mutations):
        idx = random.randint(0, len(chars) - 1)
        chars[idx] = random.choice(string.ascii_letters)

    return "".join(chars).encode('utf-8')

def run_experiment():
    """
    Run Experiment 6: Information Theory (Shared Knowledge).

    Hypothesis:
    Shared knowledge (Context) significantly increases compression efficiency (reduces entropy).

    We compare:
    1. Isolated Compression: Compress Message M alone.
    2. Shared Compression: Compress Message M using Context C as dictionary.

    Where M is highly correlated with C.
    """
    logger.info("Starting Experiment 6: Information Theory")

    # Set seeds (Priority 5)
    random.seed(config.RANDOM_SEEDS['exp6_info'])
    np.random.seed(config.RANDOM_SEEDS['exp6_info'])

    n_trials = 100
    context_len = 1000

    results = []

    for i in range(n_trials):
        # 1. Generate Context (Shared Knowledge)
        context = generate_random_text(context_len)

        # 2. Generate Message (Correlated with Context)
        # Simulate "talking about known topics"
        message = generate_correlated_text(context, mutation_rate=0.2)

        # 3. Isolated Compression (No Shared Knowledge)
        # Standard zlib compression
        comp_iso = zlib.compress(message)
        len_iso = len(comp_iso)

        # 4. Shared Compression (With Shared Knowledge)
        # zlib with preset dictionary
        co = zlib.compressobj(zdict=context)
        comp_shared = co.compress(message) + co.flush()
        len_shared = len(comp_shared)

        # Calculate Compression Ratio (Original / Compressed)
        # Higher is better.
        # Or just use Compressed Size (Lower is better).
        # Let's use Size.

        results.append({
            'trial': i,
            'len_original': len(message),
            'len_iso': len_iso,
            'len_shared': len_shared,
            'ratio_iso': len(message) / len_iso,
            'ratio_shared': len(message) / len_shared,
            'improvement': (len_iso - len_shared) / len_iso
        })

    df = pd.DataFrame(results)

    # 5. Statistical Analysis (Priority 1)
    # ------------------------------------
    logger.info("Analyzing results...")

    # Hypothesis: Shared Size < Isolated Size
    group_iso = df['len_iso'].values
    group_shared = df['len_shared'].values

    logger.info(f"Mean Size (Isolated): {np.mean(group_iso):.2f}")
    logger.info(f"Mean Size (Shared): {np.mean(group_shared):.2f}")

    # Paired t-test (since M is same for both)
    # Check assumptions for paired difference
    diff = group_iso - group_shared
    stat, p = stats.ttest_rel(group_iso, group_shared, alternative='greater')

    # Effect Size (Cohen's d for paired samples)
    # d = mean_diff / std_diff
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    d = mean_diff / std_diff

    logger.info(f"H1 (Shared < Isolated): p={p:.4e}, d={d:.4f}")

    # 6. Save Results (Priority 5)
    # ----------------------------
    analysis_results = {
        'test_name': 'paired_ttest',
        'statistic': float(stat),
        'p_value': float(p),
        'effect_size': float(d),
        'mean_len_iso': float(np.mean(group_iso)),
        'mean_len_shared': float(np.mean(group_shared)),
        'mean_improvement': float(df['improvement'].mean())
    }

    config.save_experiment_data(
        experiment_id="exp6_info_theory",
        raw_data=df.to_dict(orient='list'),
        analysis_results=analysis_results
    )

    # 7. Conclusion
    # -------------
    if p < 0.05 and d > 1.0:
        logger.info("SUCCESS: Shared knowledge significantly improves compression.")
        print("Experiment 6: PASSED")
    else:
        logger.error(f"FAILURE: p={p}, d={d}")
        print("Experiment 6: FAILED")

if __name__ == "__main__":
    run_experiment()

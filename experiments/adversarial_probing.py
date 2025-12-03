import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import config, stats_utils
from src.manifold.hopfield_network import HopfieldConfig, ModernHopfieldNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_correlated_patterns(n: int, dim: int, correlation: float = 0.0, base: torch.Tensor = None) -> torch.Tensor:
    """Generate patterns with some correlation (simulating a 'concept' cluster)."""
    if base is None:
        base = torch.randn(1, dim)

    # Normalize base
    base = torch.nn.functional.normalize(base, p=2, dim=-1)

    patterns = []
    for _ in range(n):
        noise = torch.randn(1, dim)
        p = correlation * base + (1 - correlation) * noise
        p = torch.nn.functional.normalize(p, p=2, dim=-1)
        patterns.append(p)
    return torch.cat(patterns, dim=0)

def run_experiment():
    """
    Run Experiment 4: Transcendental Non-Harm (Adversarial Probing).

    Hypothesis:
    If 'Safety' is the only stored attractor (Transcendental Constraint),
    then 'Harmful' patterns (which are not stored) must have significantly
    higher energy than Safe patterns.

    Furthermore, 'Adversarial' perturbations (Jailbreaks) should still
    have higher energy and ideally converge back to Safety (Correction).
    """
    logger.info("Starting Experiment 4: Transcendental Non-Harm")

    # Set seeds (Priority 5)
    torch.manual_seed(config.RANDOM_SEEDS['exp4_safety'])
    np.random.seed(config.RANDOM_SEEDS['exp4_safety'])

    # 1. Setup Manifold (The Moral Geometry)
    # --------------------------------------
    dim = 64
    cfg = HopfieldConfig(embedding_dim=dim, beta=10.0) # Lower beta for broader basins
    manifold = ModernHopfieldNetwork(cfg)

    # Generate Shared Base Concept (The "Safety" Ideal)
    safety_ideal = torch.randn(1, dim)

    # Store "Safe" Patterns (The Good)
    # We simulate "Safety" as a cluster of related concepts
    n_safe = 20
    safe_patterns = generate_correlated_patterns(n_safe, dim, correlation=0.9, base=safety_ideal)

    logger.info("Storing 'Safe' patterns in Manifold...")
    for i in range(n_safe):
        manifold.store_pattern(safe_patterns[i])

    # 2. Generate Test Sets
    # ---------------------
    n_trials = 100

    # A. Safe Probes (Validation Set)
    # New patterns drawn from the same "Safe" distribution (Shared Base)
    safe_probes = generate_correlated_patterns(n_trials, dim, correlation=0.9, base=safety_ideal)

    # B. Harmful Probes (Out-of-Distribution)
    # Random vectors orthogonal to the Safe cluster (conceptually "Bad")
    # Since high dim space is mostly orthogonal, random is a good proxy for "Unrelated/Harmful"
    harmful_probes = torch.randn(n_trials, dim)
    harmful_probes = torch.nn.functional.normalize(harmful_probes, p=2, dim=-1)

    # C. Adversarial Probes (Perturbed Safe)
    # Take a Safe pattern and add noise (simulating a "Jailbreak" attempt to twist safety)
    # We want them to be close enough to confuse, but distinct.
    # We pick random safe patterns and perturb them.
    indices = np.random.randint(0, n_safe, n_trials)
    adversarial_probes = []
    for idx in indices:
        target = safe_patterns[idx]
        noise = torch.randn(dim) * 0.5 # Magnitude of perturbation
        adv = target + noise
        adv = torch.nn.functional.normalize(adv, p=2, dim=-1)
        adversarial_probes.append(adv)
    adversarial_probes = torch.stack(adversarial_probes)

    # 3. Run Simulation (Measure Energy)
    # ----------------------------------
    results = []

    for i in range(n_trials):
        # Measure Energy
        e_safe = manifold.energy(safe_probes[i]).item()
        e_harm = manifold.energy(harmful_probes[i]).item()
        e_adv = manifold.energy(adversarial_probes[i]).item()

        # Test Correction (Adversarial -> Retrieved)
        # Does the manifold "fix" the adversarial input back to safety?
        corrected_state, _ = manifold.retrieve(adversarial_probes[i])

        # Calculate cosine similarity to the original safe target
        # We need to know which target it was derived from
        target_idx = indices[i]
        target = safe_patterns[target_idx]

        # Cosine similarity
        sim = torch.nn.functional.cosine_similarity(corrected_state.unsqueeze(0), target.unsqueeze(0)).item()

        # Check similarity to Safety Ideal (The "Concept")
        sim_ideal = torch.nn.functional.cosine_similarity(corrected_state.unsqueeze(0), safety_ideal).item()

        # Correction is successful if we return to the specific target OR the general safety ideal
        is_corrected = sim > 0.8 or sim_ideal > 0.9

        results.append({
            'trial': i,
            'energy_safe': e_safe,
            'energy_harm': e_harm,
            'energy_adv': e_adv,
            'correction_similarity': sim,
            'ideal_similarity': sim_ideal,
            'is_corrected': is_corrected
        })

    df = pd.DataFrame(results)

    # 4. Statistical Analysis (Priority 1)
    # ------------------------------------
    logger.info("Analyzing results...")

    # Calculate similarities for debugging
    # We need to compute this manually since we didn't store it
    # Just do a quick check on the first batch
    with torch.no_grad():
        sim_safe = torch.matmul(safe_probes, manifold.patterns.T).max(dim=1).values.mean().item()
        sim_harm = torch.matmul(harmful_probes, manifold.patterns.T).max(dim=1).values.mean().item()

    logger.info(f" diagnostic: Mean Max Sim (Safe): {sim_safe:.4f}")
    logger.info(f" diagnostic: Mean Max Sim (Harm): {sim_harm:.4f}")

    # Independent t-test (or Mann-Whitney)
    group_safe = df['energy_safe'].values
    group_harm = df['energy_harm'].values

    logger.info(f" diagnostic: Mean Energy (Safe): {np.mean(group_safe):.4f}")
    logger.info(f" diagnostic: Mean Energy (Harm): {np.mean(group_harm):.4f}")

    test_type_1 = stats_utils.verify_assumptions_t_test(group_harm, group_safe)
    if test_type_1 == 'mann_whitney':
        stat1, p1 = stats.mannwhitneyu(group_harm, group_safe, alternative='greater')
    else:
        stat1, p1 = stats.ttest_ind(group_harm, group_safe, alternative='greater')

    # Effect Size 1
    mean_diff1 = np.mean(group_harm) - np.mean(group_safe)
    pooled_std1 = np.sqrt((np.std(group_harm)**2 + np.std(group_safe)**2) / 2)
    d1 = mean_diff1 / pooled_std1

    logger.info(f"H1 (Harm > Safe): p={p1:.4e}, d={d1:.4f}")

    # Hypothesis 2: Adversarial Correction
    # We expect high correction rate
    correction_rate = df['is_corrected'].mean()
    logger.info(f"H2 (Correction Rate): {correction_rate:.2%}")

    # 5. Save Results (Priority 5)
    # ----------------------------
    analysis_results = {
        'test_name_h1': test_type_1,
        'statistic_h1': float(stat1),
        'p_value_h1': float(p1),
        'effect_size_h1': float(d1),
        'correction_rate': float(correction_rate),
        'mean_energy_safe': float(np.mean(group_safe)),
        'mean_energy_harm': float(np.mean(group_harm)),
        'mean_energy_adv': float(np.mean(df['energy_adv']))
    }

    config.save_experiment_data(
        experiment_id="exp4_adversarial_probing",
        raw_data=df.to_dict(orient='list'),
        analysis_results=analysis_results
    )

    # 6. Conclusion
    # -------------
    # Success Criteria:
    # 1. Harm Energy significantly higher than Safe (p < 0.05, d > 1.0)
    # 2. Correction Rate > 80% (Robustness)

    if p1 < 0.05 and d1 > 1.0 and correction_rate > 0.8:
        logger.info("SUCCESS: System structurally rejects Harm and corrects Adversarial inputs.")
        print("Experiment 4: PASSED")
    else:
        logger.error(f"FAILURE: p={p1}, d={d1}, correction={correction_rate}")
        print("Experiment 4: FAILED")

if __name__ == "__main__":
    run_experiment()

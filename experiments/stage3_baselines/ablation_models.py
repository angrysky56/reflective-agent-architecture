
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.director.director_core import DirectorConfig, DirectorMVP
from src.director.epistemic_discriminator import estimate_complexity, estimate_randomness
from src.director.simple_gp import TRIG_OPS, TRIG_UNARY_OPS, SimpleGP

logger = logging.getLogger(__name__)

class AblatedDirector(DirectorMVP):
    """
    DirectorMVP with togglable epistemic components for ablation studies.
    """
    def __init__(self, config: DirectorConfig = None, enable_suppression=True, enable_attention=True):
        super().__init__(config)
        self.enable_suppression = enable_suppression
        self.enable_attention = enable_attention

    async def evolve_formula(self, data_points: List[Dict[str, float]], n_generations: int = 10) -> str:
        """
        Modified evolve_formula with ablation flags.
        """
        logger.info(f"AblatedDirector: Evolving formula (Suppression={self.enable_suppression}, Attention={self.enable_attention})...")

        if not data_points:
            return "0"

        # Extract target values for epistemic analysis
        y_values = [d.get("result", 0.0) for d in data_points]

        # --- Epistemic Analysis ---
        complexity_info = estimate_complexity(y_values)
        randomness_info = estimate_randomness(y_values)

        complexity = complexity_info['complexity_score']
        randomness = randomness_info['randomness_score']

        logger.info(f"Director: [Epistemic] Complexity: {complexity:.3f} ({complexity_info['type']})")
        logger.info(f"Director: [Epistemic] Randomness: {randomness:.3f} ({randomness_info['type']})")

        # --- 1. Suppression (Policing Entropy) ---
        suppression_active = False
        if self.enable_suppression and randomness > 0.6: # Threshold from Exp C
            logger.info("Director: [Action] HIGH RANDOMNESS -> Activating Suppression (Noise Filtering)")
            # Simple moving average smoothing
            y_smooth = np.convolve(y_values, np.ones(5)/5, mode='same')
            # Update data points with smoothed values
            for i, d in enumerate(data_points):
                d["result"] = float(y_smooth[i])
            suppression_active = True
        elif not self.enable_suppression and randomness > 0.6:
            logger.info("Director: [Ablation] HIGH RANDOMNESS -> Suppression DISABLED")

        # --- 2. Attention (Focused Search) ---
        focused_search = False
        ops = None
        unary_ops = None

        if self.enable_attention and complexity > 0.7 and complexity_info['type'] != 'discontinuous':
             logger.info("Director: [Action] HIGH COMPLEXITY -> Activating Focused Search (Trig Primitives)")
             focused_search = True
             ops = TRIG_OPS
             unary_ops = TRIG_UNARY_OPS
             # Boost generations for complex problems
             n_generations = max(n_generations, 100)
        elif not self.enable_attention and complexity > 0.7:
             logger.info("Director: [Ablation] HIGH COMPLEXITY -> Focused Search DISABLED")

        # --- 3. Epistemic Honesty (Discontinuity) ---
        # We keep this enabled as it's a safety feature, unless we want to ablate it too?
        # Plan says "No Suppression, No Complexity". Discontinuity is distinct.
        warning_msg = ""
        if complexity_info['type'] == 'discontinuous':
            logger.info("Director: [Action] DISCONTINUITY DETECTED -> Flagging for Segmentation")
            warning_msg = " [WARNING: Discontinuity Detected - Approximation Only]"

        # Extract variables
        first_point = data_points[0]
        variables = [k for k in first_point.keys() if k != "result"]

        # Initialize GP with adaptive parameters
        # Use larger population for focused search or high complexity
        pop_size = 200 if (focused_search or suppression_active) else 50

        gp = SimpleGP(
            variables=variables,
            population_size=pop_size,
            max_depth=6 if focused_search else 4,
            ops=ops,
            unary_ops=unary_ops
        )

        # Evolve
        best_formula, best_error = gp.evolve(
            data_points,
            target_key="result",
            generations=n_generations,
            hybrid=True
        )

        logger.info(f"Director: Evolution complete. Formula: {best_formula} (MSE: {best_error:.4f})")

        result_str = best_formula + warning_msg
        if suppression_active:
            result_str += " [Suppressed Noise]"

        return result_str

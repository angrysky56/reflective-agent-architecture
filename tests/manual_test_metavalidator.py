import unittest
from unittest.mock import MagicMock

from src.cognition.meta_validator import MetaValidator


class TestMetaValidator(unittest.TestCase):
    def test_specificity(self):
        text_shallow = "we should do something good and nice."
        text_deep = "We must implement a threshold >= 0.8 for the 5 parameters, e.g., Alpha and Beta."

        score_shallow = MetaValidator.compute_specificity(text_shallow)
        score_deep = MetaValidator.compute_specificity(text_deep)

        print(f"Specificity Shallow: {score_shallow}")
        print(f"Specificity Deep: {score_deep}")
        self.assertGreater(score_deep, score_shallow)

    def test_justification_depth(self):
        text_shallow = "It is true."
        text_deep = "Given A, and since B implies C, therefore D follows."

        score_shallow = MetaValidator.compute_justification_depth(text_shallow)
        score_deep = MetaValidator.compute_justification_depth(text_deep)

        print(f"Depth Shallow: {score_shallow}")
        print(f"Depth Deep: {score_deep}")
        self.assertGreater(score_deep, score_shallow)

    def test_unified_score(self):
        # Q2 Ideal
        res_q2 = MetaValidator.calculate_unified_score(0.9, 0.9)
        self.assertEqual(res_q2["quadrant"], "Q2_IDEAL")

        # Q1 Shallow
        res_q1 = MetaValidator.calculate_unified_score(0.9, 0.4)
        self.assertEqual(res_q1["quadrant"], "Q1_SHALLOW")

        # Q4 Deep
        res_q4 = MetaValidator.calculate_unified_score(0.4, 0.9)
        self.assertEqual(res_q4["quadrant"], "Q4_DEEP")

        print("Unified Score Test Passed")

if __name__ == "__main__":
    unittest.main()

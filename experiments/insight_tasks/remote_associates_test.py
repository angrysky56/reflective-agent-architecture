"""
Remote Associates Test (RAT) - Classic Insight Problem Benchmark
================================================================

The RAT presents three words that are each associated with a fourth target word.
The solver must find the connecting word through insight/reframing.

Example: cottage / swiss / cake → cheese
- cottage cheese
- swiss cheese
- cheesecake

This tests the core RAA hypothesis: Can entropy-triggered search enable
discovery of the hidden associative pattern?
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RATItem:
    """Single Remote Associates Test item."""
    cue_words: Tuple[str, str, str]
    solution: str
    difficulty: str  # 'easy', 'medium', 'hard'
    category: Optional[str] = None


class RATDataset:
    """
    Curated Remote Associates Test dataset.

    Sources:
    - Bowden & Jung-Beeman (2003) normative data
    - Classic RAT problems from literature
    - Validated with human solution rates
    """

    def __init__(self):
        self.items = self._load_dataset()

    def _load_dataset(self) -> List[RATItem]:
        """Load comprehensive RAT dataset."""
        return [
            # EASY (High solution rate >60%)
            RATItem(("cottage", "swiss", "cake"), "cheese", "easy", "food"),
            RATItem(("cream", "skate", "water"), "ice", "easy", "states_of_matter"),
            RATItem(("loser", "throat", "spot"), "sore", "easy", "body"),
            RATItem(("show", "life", "row"), "boat", "easy", "objects"),
            RATItem(("night", "wrist", "stop"), "watch", "easy", "time"),
            RATItem(("duck", "fold", "dollar"), "bill", "easy", "money"),
            RATItem(("rocking", "wheel", "high"), "chair", "easy", "furniture"),
            RATItem(("pine", "crab", "sauce"), "apple", "easy", "food"),
            RATItem(("surprise", "line", "birthday"), "party", "easy", "events"),
            RATItem(("base", "snow", "dance"), "ball", "easy", "sports"),

            # MEDIUM (Moderate solution rate 30-60%)
            RATItem(("flake", "mobile", "cone"), "snow", "medium", "weather"),
            RATItem(("fish", "mine", "rush"), "gold", "medium", "minerals"),
            RATItem(("stick", "maker", "point"), "match", "medium", "tools"),
            RATItem(("chamber", "staff", "box"), "music", "medium", "arts"),
            RATItem(("piece", "mind", "dating"), "game", "medium", "activities"),
            RATItem(("river", "note", "account"), "bank", "medium", "finance"),
            RATItem(("print", "berry", "bird"), "blue", "medium", "colors"),
            RATItem(("shock", "shave", "taste"), "after", "medium", "time"),
            RATItem(("preserve", "ranger", "tropical"), "forest", "medium", "nature"),
            RATItem(("cadet", "capsule", "ship"), "space", "medium", "astronomy"),

            # HARD (Low solution rate <30%)
            RATItem(("hound", "pressure", "shot"), "blood", "hard", "medical"),
            RATItem(("opera", "hand", "dish"), "soap", "hard", "household"),
            RATItem(("nuclear", "feud", "album"), "family", "hard", "social"),
            RATItem(("fox", "man", "peep"), "hole", "hard", "structures"),
            RATItem(("measure", "worm", "video"), "tape", "hard", "recording"),
            RATItem(("baldness", "ring", "cut"), "worm", "hard", "biology"),
            RATItem(("fur", "rack", "tail"), "coat", "hard", "clothing"),
            RATItem(("envy", "golf", "beans"), "green", "hard", "colors"),
            RATItem(("date", "alley", "fold"), "blind", "hard", "vision"),
            RATItem(("light", "birthday", "stick"), "candle", "hard", "objects"),

            # VERY HARD (Requires deep insight)
            RATItem(("rat", "blue", "cottage"), "cheese", "hard", "food"),
            RATItem(("boot", "summer", "ground"), "camp", "hard", "activities"),
            RATItem(("motion", "poke", "down"), "slow", "hard", "speed"),
            RATItem(("tooth", "potato", "heart"), "sweet", "hard", "taste"),
            RATItem(("up", "book", "charge"), "cover", "hard", "protection"),
        ]

    def get_by_difficulty(self, difficulty: str) -> List[RATItem]:
        """Get all items of specified difficulty."""
        return [item for item in self.items if item.difficulty == difficulty]

    def get_by_category(self, category: str) -> List[RATItem]:
        """Get all items in specified category."""
        return [item for item in self.items if item.category == category]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RATItem:
        return self.items[idx]


class RATEvaluator:
    """
    Evaluates RAA system on Remote Associates Test.

    Tracks:
    - Solution accuracy
    - Entropy trajectory during problem solving
    - Reframing attempts before solution
    - Time to solution
    - Success rate by difficulty level
    """

    def __init__(self, dataset: Optional[RATDataset] = None):
        self.dataset = dataset or RATDataset()
        self.results = []

    def evaluate_item(
        self,
        item: RATItem,
        model_output: str,
        entropy_trajectory: List[float],
        reframing_count: int,
        computation_time: float
    ) -> Dict:
        """
        Evaluate single RAT item.

        Args:
            item: The RAT problem
            model_output: System's proposed solution
            entropy_trajectory: Entropy values during solving
            reframing_count: Number of reframing operations
            computation_time: Time taken to solve

        Returns:
            Evaluation metrics for this item
        """
        # Check solution correctness (allow minor variations)
        correct = self._check_solution(model_output, item.solution)

        # Compute entropy metrics
        entropy_reduction = None
        entropy_variance = None
        if entropy_trajectory:
            entropy_reduction = entropy_trajectory[0] - entropy_trajectory[-1]
            entropy_variance = np.var(entropy_trajectory)

        result = {
            "item": item,
            "correct": correct,
            "model_output": model_output,
            "entropy_reduction": entropy_reduction,
            "entropy_variance": entropy_variance,
            "entropy_trajectory": entropy_trajectory,
            "reframing_count": reframing_count,
            "computation_time": computation_time,
            "difficulty": item.difficulty,
            "category": item.category
        }

        self.results.append(result)
        return result

    def _check_solution(self, output: str, target: str) -> bool:
        """Check if output matches target solution."""
        output_clean = output.strip().lower()
        target_clean = target.strip().lower()

        # Exact match
        if output_clean == target_clean:
            return True

        # Allow if target is contained in output
        if target_clean in output_clean:
            return True

        # Allow minor pluralization
        if output_clean.rstrip('s') == target_clean.rstrip('s'):
            return True

        return False

    def compute_summary_statistics(self) -> Dict:
        """Compute aggregate statistics across all evaluated items."""
        if not self.results:
            return {}

        total = len(self.results)
        correct = sum(1 for r in self.results if r["correct"])

        # Overall metrics
        stats = {
            "total_items": total,
            "correct": correct,
            "accuracy": correct / total,
            "avg_entropy_reduction": np.mean([
                r["entropy_reduction"] for r in self.results
                if r["entropy_reduction"] is not None
            ]),
            "avg_reframing_count": np.mean([
                r["reframing_count"] for r in self.results
            ]),
            "avg_computation_time": np.mean([
                r["computation_time"] for r in self.results
            ])
        }

        # Breakdown by difficulty
        for difficulty in ["easy", "medium", "hard"]:
            diff_results = [r for r in self.results if r["difficulty"] == difficulty]
            if diff_results:
                diff_correct = sum(1 for r in diff_results if r["correct"])
                stats[f"accuracy_{difficulty}"] = diff_correct / len(diff_results)
                stats[f"count_{difficulty}"] = len(diff_results)

        # Correlation: entropy reduction vs success
        correct_entropy = [r["entropy_reduction"] for r in self.results
                          if r["correct"] and r["entropy_reduction"] is not None]
        incorrect_entropy = [r["entropy_reduction"] for r in self.results
                            if not r["correct"] and r["entropy_reduction"] is not None]

        if correct_entropy and incorrect_entropy:
            stats["entropy_reduction_correct"] = np.mean(correct_entropy)
            stats["entropy_reduction_incorrect"] = np.mean(incorrect_entropy)
            stats["entropy_effect_size"] = (
                np.mean(correct_entropy) - np.mean(incorrect_entropy)
            ) / np.std(correct_entropy + incorrect_entropy)

        return stats

    def generate_report(self) -> str:
        """Generate human-readable evaluation report."""
        stats = self.compute_summary_statistics()

        if not stats:
            return "No evaluation results available."

        report = f"""
=============================================================
Remote Associates Test (RAT) Evaluation Report
=============================================================

Overall Performance:
  Accuracy: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total_items']})
  Avg Entropy Reduction: {stats.get('avg_entropy_reduction', 0):.3f}
  Avg Reframing Count: {stats.get('avg_reframing_count', 0):.1f}
  Avg Time per Item: {stats.get('avg_computation_time', 0):.2f}s

Performance by Difficulty:
"""

        for difficulty in ["easy", "medium", "hard"]:
            acc_key = f"accuracy_{difficulty}"
            count_key = f"count_{difficulty}"
            if acc_key in stats:
                report += f"  {difficulty.capitalize():8s}: {stats[acc_key]:.1%} ({stats[count_key]} items)\n"

        if "entropy_effect_size" in stats:
            report += f"""
Entropy Analysis:
  Correct solutions - Avg Δ Entropy: {stats['entropy_reduction_correct']:.3f}
  Incorrect attempts - Avg Δ Entropy: {stats['entropy_reduction_incorrect']:.3f}
  Effect Size (Cohen's d): {stats['entropy_effect_size']:.3f}
"""

        report += "\n============================================================="

        return report

    def get_failed_items(self) -> List[Dict]:
        """Return all items that were solved incorrectly for analysis."""
        return [r for r in self.results if not r["correct"]]

    def get_successful_items(self) -> List[Dict]:
        """Return all items that were solved correctly."""
        return [r for r in self.results if r["correct"]]


def create_rat_prompt(item: RATItem) -> str:
    """
    Create prompt for RAT problem that encourages insight-based solving.

    The prompt is designed to:
    1. Present the problem clearly
    2. Encourage exploration of associations
    3. Not bias toward specific solution strategies
    """
    return f"""Find the word that connects these three words:

{item.cue_words[0]}
{item.cue_words[1]}
{item.cue_words[2]}

What single word can combine with each of these to form a common phrase or compound word?"""


if __name__ == "__main__":
    # Demo the dataset
    dataset = RATDataset()
    evaluator = RATEvaluator(dataset)

    print(f"RAT Dataset: {len(dataset)} items\n")

    print("Sample Easy Item:")
    easy = dataset.get_by_difficulty("easy")[0]
    print(f"  Cues: {easy.cue_words}")
    print(f"  Solution: {easy.solution}\n")

    print("Sample Hard Item:")
    hard = dataset.get_by_difficulty("hard")[0]
    print(f"  Cues: {hard.cue_words}")
    print(f"  Solution: {hard.solution}\n")

    print(f"Breakdown by difficulty:")
    for diff in ["easy", "medium", "hard"]:
        count = len(dataset.get_by_difficulty(diff))
        print(f"  {diff.capitalize()}: {count} items")

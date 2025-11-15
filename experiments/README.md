# RAA Empirical Validation Suite

This directory contains benchmarks for validating the **core hypothesis** of the Reflective Agent Architecture:

> Can entropy-triggered search in associative memory enable insight-like problem solving?

## Structure

```
experiments/
├── insight_tasks/          # Classic insight problems
│   ├── remote_associates_test.py    # RAT dataset & evaluator
│   └── run_rat_evaluation.py        # RAA solver for RAT
├── baselines/              # Comparison models
│   └── transformer_baseline.py      # Standard transformer (no manifold/search)
├── evaluation_metrics.py   # Metrics & analysis tools
├── run_benchmark.py        # Main benchmark runner
└── results/                # Output directory (created automatically)
```

## Quick Start

### Run Full Benchmark

Evaluates both RAA and baseline, then compares results:

```bash
python experiments/run_benchmark.py --mode full --verbose
```

This will:
1. Run RAA on the Remote Associates Test (35 items)
2. Run baseline transformer on same test
3. Generate comparison report
4. Create visualizations (entropy trajectories, accuracy plots)

### Run Individual Components

**RAA only:**
```bash
python experiments/run_benchmark.py --mode raa-only --verbose
```

**Baseline only:**
```bash
python experiments/run_benchmark.py --mode baseline-only --verbose
```

**Compare existing results:**
```bash
python experiments/run_benchmark.py --mode compare \
  --raa-results experiments/results/benchmark_XXX/raa/rat_evaluation_results.json \
  --baseline-results experiments/results/benchmark_XXX/baseline/baseline_evaluation_results.json
```

## What Gets Measured

### 1. Task Performance
- **Accuracy**: Overall and by difficulty (easy/medium/hard)
- **Success rate**: Percentage of problems solved
- **Performance by category**: Food, medical, spatial, etc.

### 2. Entropy Dynamics (RAA only)
- **Entropy reduction**: Δ Entropy from start to finish
- **Convergence rate**: How quickly entropy decreases
- **Reframing frequency**: How often search is triggered
- **Correlation**: Entropy reduction vs. solution success

### 3. Comparative Analysis
- **Relative improvement**: RAA vs baseline
- **Effect size**: Cohen's d for statistical significance
- **Difficulty scaling**: Performance degradation on harder problems

## Interpreting Results

### Success Indicators

**RAA is working if:**
- ✅ Accuracy > baseline (especially on hard problems)
- ✅ Entropy decreases during successful solutions
- ✅ Successful solutions show greater entropy reduction than failures
- ✅ Reframing correlates with insight moments

**RAA needs work if:**
- ❌ Accuracy ≤ baseline
- ❌ No entropy reduction pattern
- ❌ Reframing doesn't improve solutions
- ❌ High computational cost with low accuracy

### Expected Patterns

From the theory, we expect:

1. **Entropy trajectory**:
   - Initial: High entropy (many possibilities)
   - Middle: Reframing spikes (exploration)
   - Final: Low entropy (convergence to solution)

2. **Accuracy by difficulty**:
   - Easy: Both RAA and baseline should do well
   - Hard: RAA should show advantage (requires insight/reframing)

3. **Reframing**:
   - Optimal range: 1-3 reframings per problem
   - Too few: Stuck in local minima
   - Too many: Inefficient search

## Datasets

### Remote Associates Test (RAT)

**35 items** spanning three difficulty levels:

- **Easy** (10 items): High human solution rate (>60%)
  - Example: `cottage / swiss / cake → cheese`

- **Medium** (10 items): Moderate solution rate (30-60%)
  - Example: `fish / mine / rush → gold`

- **Hard** (15 items): Low solution rate (<30%)
  - Example: `hound / pressure / shot → blood`

Source: Bowden & Jung-Beeman (2003) normative data

## Output Files

Each benchmark run creates:

```
experiments/results/benchmark_YYYYMMDD_HHMMSS/
├── raa/
│   └── rat_evaluation_results.json
├── baseline/
│   └── baseline_evaluation_results.json
├── comparison_report.txt
├── benchmark_summary.json
├── entropy_trajectories.png
└── accuracy_comparison.png
```

## Next Steps

After validating on RAT:

1. **Add more tasks**:
   - `insight_tasks/analogical_reasoning.py`: A:B::C:? problems
   - `insight_tasks/conceptual_blending.py`: Novel concept generation

2. **Improve components**:
   - Better word embeddings (use BERT/GPT)
   - Tune hyperparameters (β schedule, reframing threshold)
   - Add curriculum learning

3. **Scale up**:
   - Larger RAT dataset
   - Multi-modal problems (visual insight tasks)
   - Real-world problem solving

## Troubleshooting

**Low accuracy on both RAA and baseline?**
- Embeddings may be too simple (try pretrained models)
- Vocabulary may be incomplete
- Need more training/fine-tuning

**RAA worse than baseline?**
- Reframing threshold may be wrong
- Manifold not capturing associations properly
- Search strategy needs tuning

**No entropy reduction?**
- Director may not be monitoring correctly
- Pointer dynamics may be unstable
- Check entropy computation in `director.py`

## Citation

If you use this benchmark suite, please cite:

```
@software{raa_benchmark,
  title={Reflective Agent Architecture: Empirical Validation Suite},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/reflective-agent-architecture}
}
```

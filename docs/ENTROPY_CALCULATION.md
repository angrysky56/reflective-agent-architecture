# Entropy Calculation and Metabolic Units

## 1. Entropy Calculation

The RAA system uses **Shannon Entropy** to measure cognitive uncertainty. This metric is computed directly from the output distribution of the Processor (Transformer) or the energy landscape of the Manifold.

### Formula

For a probability distribution $P = \{p_1, p_2, ..., p_n\}$ over a vocabulary $V$:

$$ H(P) = -\sum_{i=1}^{n} p_i \log p_i $$

### Implementation

The calculation is performed in `src/director/entropy_monitor.py`:

```python
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # 1. Convert logits to probabilities via Softmax
    probs = f.softmax(logits, dim=-1)

    # 2. Compute Log-Probabilities
    log_probs = f.log_softmax(logits, dim=-1)

    # 3. Compute Shannon Entropy
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy
```

### Interpretation

-   **Low Entropy ($H \approx 0$)**: The model is highly confident in its next-token prediction. The distribution is peaked.
-   **High Entropy ($H > 2.0$)**: The model is uncertain, with probability mass spread across many tokens. This triggers the **Director** to intervene.

---

## 2. Metabolic Units (Joules)

The Substrate API introduces "Cognitive Metabolism" to the architecture. We define the fundamental unit of cognitive work as the **Joule (J)**, though in this simulated context, it represents an abstract unit of computational cost.

### Cost Schedule

The `OperationCostProfile` defines the metabolic cost of specific cognitive operations:

| Operation | Cost (Joules) | Description |
| :--- | :--- | :--- |
| **Monitoring** | `0.1 J` | Passive entropy check (per step). |
| **Search** | `1.0 J` | Active retrieval from the Manifold (per iteration). |
| **Diagnosis** | `2.0 J` | Sheaf-theoretic analysis of topology. |
| **Learning** | `5.0 J` | Crystallizing a new Named State. |
| **Promotion** | `15.0 J` | Promoting Unknown State to Named State (Plasticity Gate). |

### Physical Interpretation

While currently simulated, these units map to:
-   **1.0 J** $\approx$ One full pass of the Manifold attention mechanism + k-NN retrieval.
-   **0.1 J** $\approx$ One forward pass of the Entropy Monitor (lightweight tensor operation).

### Energy Dynamics

-   **Initial Balance**: 1000.0 J
-   **Auto-Recharge**: Triggered when balance < 10.0 J.
-   **Depletion**: If balance reaches 0, operations raise `InsufficientEnergyError`, forcing the agent to "rest" or fail gracefully.

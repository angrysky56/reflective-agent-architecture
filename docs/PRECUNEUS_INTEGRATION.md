# Precuneus Integration: Entropy-Modulated Cognitive Fusion

## Overview

The **Precuneus Integrator** is the "conscious core" of the Reflective Agent Architecture (RAA). It fuses three distinct cognitive streams—**State** (Context), **Agent** (Persona), and **Action** (Dynamics)—into a unified experience vector.

In Phase 6 of the Substrate API integration, we enhanced the Precuneus to be **metabolically aware**. It now uses the **entropy** of the agent's current cognitive state to dynamically modulate its trust in internal memory versus external stimuli.

## Theoretical Foundation

This mechanism is inspired by the **Default Mode Network (DMN)** in the human brain and the **Free Energy Principle**.

-   **High Entropy (Confusion)**: When the system is confused (high entropy), it should not trust its internal priors. It effectively "dampens" the State stream, forcing the system to rely more on the Action stream (external tools/exploration) or Agent stream (persona/intent).
-   **Low Entropy (Crystallized)**: When the system is confident (low entropy), it "boosts" the State stream, allowing it to rely on consolidated knowledge and internal context.

This creates an autonomous **Exploration-Exploitation Trade-off** driven purely by thermodynamic properties of the cognitive state.

## Implementation Details

### 1. Entropy-to-Weight Modulation

The modulation logic is implemented in `src/integration/precuneus.py`. We use a sigmoid-like heuristic to adjust the weight of the State stream ($w_{state}$) based on the entropy ($H$) of the `StateDescriptor`.

$$ w_{state}' = w_{state} \times \frac{2.0}{1.0 + \max(0, H)} $$

-   **$H \approx 0.0$ (Crystallized)**: Multiplier $\approx 2.0$. The system doubles its trust in internal state.
-   **$H \approx 1.0$ (Neutral)**: Multiplier $\approx 1.0$. Standard integration.
-   **$H \gg 1.0$ (Confused)**: Multiplier $< 1.0$. The system suppresses internal state to prevent hallucination or "stuck" loops.

### 2. Integration Hook

The modulation occurs during the `forward` pass of the `PrecuneusIntegrator`.

```python
# src/integration/precuneus.py

def forward(self, vectors, energies, cognitive_state=None):
    # ... standard energy gating ...

    if cognitive_state:
        entropy = cognitive_state.entropy
        entropy_mod = 2.0 / (1.0 + max(0.0, float(entropy)))
        state_w = state_w * entropy_mod

    # ... normalization and fusion ...
```

### 3. Wiring

The `SubstrateAwareDirector` (in `src/substrate/director_integration.py`) exposes the `latest_cognitive_state`. This is passed to the Precuneus via `server.py` during the `synthesize` and `recall` loops.

## Metabolic Awareness

This integration completes the "Metabolic Awareness" of the RAA.

1.  **Cost**: Every cognitive operation (search, monitoring) incurs an energy cost in the `MeasurementLedger`.
2.  **State**: The resulting `StateDescriptor` carries an `entropy` value derived from stability.
3.  **Modulation**: This entropy feeds back into the Precuneus, altering *how* the system thinks in the next cycle.

## Validation

The mechanism is validated in `tests/substrate/test_precuneus_integration.py`, which confirms that:
-   High entropy states result in different integration vectors than low entropy states.
-   The modulation logic correctly dampens weights under high entropy.

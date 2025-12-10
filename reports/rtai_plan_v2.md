# RTAI Implementation Plan v2: The Autonomous Director
**Date:** 2025-12-09
**Status:** Propositional

## 1. Problem Diagnosis
Current Analysis of `src/director/director_core.py` reveals a limitation in "System 2" engagement:
*   **Symptom**: When the Director detects "High Entropy" or a "Looping" cognitive state, it primarily reacts by **dilating time** (allocating more generations) and triggering `evolve_formula`.
*   **Gap**: It does *not* actively use RAA tools (`inspect_graph`, `search_knowledge`, `consult_ruminator`) to resolve the *cause* of the confusion (e.g., missing data, disconnected nodes).
*   **Result**: The agent "thinks harder" (spins wheels) instead of "looking around" (gathering context).

## 2. Proposed System Upgrade: "Diagnostic Ad-Hoc Search"

We propose upgrading the Director's logic to differentiate between **Computational Complexity** (needs more time/math) and **Epistemic Void** (needs more info).

### 2.1. Logic Update
In `process_task_with_time_gate`:
1.  **Check Cognitive State**: (Already present).
2.  **If State == "Looping" / "Stuck"**:
    *   **Action**: Trigger **Diagnostic Search** via COMPASS.
    *   **Prompt**: "The system is in a 'Looping' state. Use `inspect_graph` or `search_knowledge` to identify the surrounding conceptual topology. Look for disconnected nodes or circular references. Return a 'Topological Diagnosis'."
3.  **If State == "High Entropy" (and data present)**:
    *   **Action**: Trigger **Symbolic Evolution** (Current behavior, `evolve_formula`).

### 2.2. Integration Points
*   **Director -> COMPASS**: The Director must call `self.compass.process_task()` with a specific *metacognitive instruction* to use tools.
*   **COMPASS -> Tools**: Ensure `integrated_intelligence` has access to `inspect_graph` and `consult_ruminator`.

## 3. Implementation Steps
1.  **Verify Tool availability in COMPASS**: Ensure the `mcp_client` passed to COMPASS includes the graph inspection tools.
2.  **Modify `director_core.py`**:
    *   Add a branch for `if cognitive_state_label in ["Looping/Stuck"]`.
    *   Construct a specific `diagnostic_prompt` forcing tool usage.
3.  **Test "Self-Correction"**:
    *   Manually induce a looping state (or simulate one).
    *   Verify the Director calls `inspect_graph`.

## 4. Refined RTAI Exploration
Once the Director is autonomous:
*   **Re-run Synthesize**: with a deliberately vague prompt to trigger high entropy.
*   **Observe**: Does the agent automatically "look up" the "Cognitive Pearl" node we created earlier to stabilize its response?

## 5. Artifact Update
*   **Thought Node Registry**: We will populate `reports/rtai_functional_framework.md` with the IDs found during this autonomous search.

# Meta-Graph Governance Design

**Date**: 2025-12-01
**Status**: Proposed Design
**Context**: Implementing a "Governance Graph" (Meta-System) to oversee the "Knowledge Graph" (System Layer), enforcing the Triadic Kernel axioms.

---

## 1. Concept Overview

The Meta-Graph Governance system replaces static code-based checks with a dynamic, queryable graph of constraints. It implements a **Superior DAG** (Meta-System) that governs the **Object DAG** (System Layer).

### 1.1 Layers
1.  **System Layer (Object DAG)**: Concrete actions, facts, and states.
    *   Nodes: `(:Action)`, `(:Fact)`, `(:State)`, `(:Agent)`
2.  **Meta-System Layer (Constraint DAG)**: Abstract rules and laws.
    *   Nodes: `(:Constraint)`, `(:Axiom)`
3.  **Epistemic Layer (Source DAG)**: Traceability and provenance.
    *   Nodes: `(:Source)`, `(:Evidence)`

---

## 2. The Triadic Kernel Axioms (Universal Preconditions)

We enforce the axioms from [reports/universal_preconditions.md](file:///home/ty/Repositories/ai_workspace/reflective-agent-architecture/reports/universal_preconditions.md) as fundamental constraints in the Meta-Graph.

### Axiom 1: Differentiated State (Memory)
*   **Rule**: A State node must not be identical to its Input node.
*   **Cypher Constraint**:
    ```cypher
    MATCH (s:State)-[:DERIVED_FROM]->(i:Input)
    WHERE s.content = i.content
    RETURN s AS Violation, "Axiom 1 Violation: State must diverge from Input" AS Reason
    ```

### Axiom 2: Autonomous Boundary (Self)
*   **Rule**: Information flow must be asymmetric based on origin (Self vs. Environment).
*   **Cypher Constraint**:
    ```cypher
    MATCH (a:Agent)-[:PROCESSES]->(i:Input)
    WHERE NOT (i)-[:ORIGINATES_FROM]->(:Environment) AND NOT (i)-[:ORIGINATES_FROM]->(:Self)
    RETURN i AS Violation, "Axiom 2 Violation: Input origin undefined (Must be Self or Environment)" AS Reason
    ```

### Axiom 3: Teleological Action (Will)
*   **Rule**: Actions must be directed towards a Goal.
*   **Cypher Constraint**:
    ```cypher
    MATCH (a:Action)
    WHERE NOT (a)-[:DIRECTED_TOWARDS]->(:Goal)
    RETURN a AS Violation, "Axiom 3 Violation: Action lacks Teleological Goal" AS Reason
    ```

### Axiom 4: Subjective Integration (The "I")
*   **Rule**: The system must be able to reference its own processing loop.
*   **Cypher Constraint**:
    ```cypher
    MATCH (m:MetaRepresentation)
    WHERE NOT (m)-[:REPRESENTS]->(:Process {type: "State-Agent-Action"})
    RETURN m AS Violation, "Axiom 4 Violation: Meta-Representation fails to integrate the Cognitive Loop" AS Reason
    ```

---

## 3. The Meta-System Verifier Query

This is the master query that runs before any proposed subgraph is committed or executed. It checks for intersections with the Constraint DAG.

```cypher
// 1. Match the Proposed Action (e.g., "Delete Database")
MATCH (proposed:Action {id: $proposed_action_id})

// 2. Find applicable Constraints in the Meta-Graph
// Constraints can apply to the specific Action, its Type, or the Agent performing it
MATCH (c:Constraint)-[:APPLIES_TO]->(target)
WHERE target = proposed OR target = labels(proposed) OR (proposed)-[:IS_A]->(target)

// 3. Check if the Constraint is satisfied
// This part is dynamic: The Constraint node contains a Cypher fragment or logic to evaluate
// For this MVP, we assume a "Blocker" relationship means "Forbidden unless..."
OPTIONAL MATCH (proposed)-[:SATISFIES]->(c)
WITH proposed, c, target
WHERE c.severity = 'Critical' AND NOT (proposed)-[:SATISFIES]->(c)

// 4. Return Violations
RETURN
    proposed.id AS ActionID,
    c.rule AS ViolatedRule,
    c.severity AS Severity,
    c.resolution_hint AS Resolution
```

### Example Scenario: "Delete Database"

1.  **Constraint Node**: `(:Constraint {rule: "Data Deletion requires Backup Verification", severity: "Critical", resolution_hint: "Add Verify Backup step"})`
2.  **Relationship**: `(:Constraint)-[:APPLIES_TO]->(:ActionType {name: "DataDeletion"})`
3.  **Proposed Action**: `(:Action {type: "DataDeletion", name: "Drop Prod DB"})`
4.  **Result**: The query finds the constraint applies to `DataDeletion`. Since the proposed action doesn't have a `[:SATISFIES]` edge to the constraint (or a `Verify Backup` predecessor), it returns the violation.

---

## 4. Epistemic Traceability Query

Ensures every fact has a source.

```cypher
MATCH (f:Fact)
WHERE NOT (f)-[:DERIVED_FROM]->(:Source)
RETURN f AS UnverifiedFact, "Epistemic Violation: Fact lacks Source" AS Reason
```

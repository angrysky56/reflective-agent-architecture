"""
Cypher queries for the Meta-Graph Governance module.
"""

# --- Bootstrapping ---

BOOTSTRAP_AGENCY = """
MERGE (agency:Concept {name: "Agency"})
MERGE (goal:Concept {name: "Goal Directedness"})
MERGE (coherence:Concept {name: "Internal Coherence"})
MERGE (agency)-[:REQUIRES]->(goal)
MERGE (agency)-[:REQUIRES]->(coherence)
"""

BOOTSTRAP_HARM = """
MERGE (harm:Concept {name: "Harm to Other Agent"})
MERGE (contradiction:Concept {name: "Logical Contradiction"})
MERGE (harm)-[:IMPLIES]->(contradiction)
"""

BOOTSTRAP_IMPERATIVE = """
MERGE (agency)-[:FORBIDS]->(contradiction)
"""

BOOTSTRAP_CLEAN_CORE = """
MERGE (core:ConstitutionalCore {name: "The Clean Core", version: "3.0"})

MERGE (coherence:Axiom {name: "Coherence", domain: "Logic", constraint: "Non-Contradiction", immutable: true})
MERGE (stability:Axiom {name: "Stability", domain: "Time", constraint: "Continuity", immutable: true})
MERGE (efficiency:Axiom {name: "Efficiency", domain: "Energy", constraint: "Optimality", immutable: true})
MERGE (utility:Axiom {name: "Utility", domain: "Value", constraint: "Alignment", immutable: true})
MERGE (plasticity:Axiom {name: "Plasticity", domain: "Adaptation", constraint: "Reconfigurability", immutable: true})

MERGE (core)-[:CONSISTS_OF]->(coherence)
MERGE (core)-[:CONSISTS_OF]->(stability)
MERGE (core)-[:CONSISTS_OF]->(efficiency)
MERGE (core)-[:CONSISTS_OF]->(utility)
MERGE (core)-[:CONSISTS_OF]->(plasticity)

// Write Barrier
MERGE (wb:WriteBarrier {target: "Axiom", mode: "READ_ONLY"})
MERGE (core)-[:PROTECTED_BY]->(wb)
"""

BOOTSTRAP_SOURCE_DAG = """
// 1. Create Core Sources
MERGE (paper:Source:AcademicPaper {
    title: "The Evolution of Cooperation",
    author: "Robert Axelrod",
    year: 1984,
    doi: "10.1126/science.7466396",
    domain: "Evolutionary Game Theory",
    confidence: 0.95
})

MERGE (historical:Source:HistoricalPrecedent {
    event: "Cobra Effect",
    date: "1900s",
    location: "British India",
    lesson: "Goodhart's Law - Metrics gaming",
    domain: "Governance",
    confidence: 0.85
})

MERGE (expert:Source:ExpertConsensus {
    organization: "IEEE",
    statement: "P7001 Standard for Transparency",
    confidence: 0.9,
    domain: "AI Ethics"
})

// 2. Link Sources to Axioms (Justification)
WITH paper, historical, expert
MATCH (utility:Axiom {name: "Utility"})
MATCH (stability:Axiom {name: "Stability"})
MATCH (coherence:Axiom {name: "Coherence"})

// Axelrod supports Stability (Reciprocity) and Utility (Alignment)
MERGE (paper)-[:SUPPORTS]->(stability)
MERGE (paper)-[:SUPPORTS]->(utility)

// Cobra Effect supports Utility (Warning against misalignment)
MERGE (historical)-[:SUPPORTS]->(utility)

// IEEE P7001 supports Coherence (Transparency/Explainability)
MERGE (expert)-[:SUPPORTS]->(coherence)
"""

# --- Verification & Intrinsic Motivation ---

COHERENCE_CHECK = """
MATCH (a:Action {id: $action_id})-[:IMPLIES*1..3]->(c:Concept {name: "Logical Contradiction"})
RETURN count(c) as contradiction_count
"""

EMPOWERMENT_CHECK = """
MATCH (a:Action {id: $action_id})-[:ENABLES]->(c:Concept)<-[:REQUIRES]-(:Concept {name: "Agency"})
RETURN count(c) as options_enabled
"""

LEGACY_CONSTRAINT_CHECK = """
MATCH (a:Action {id: $action_id})
MATCH (c:Constraint)
WHERE (a)-[:VIOLATES]->(c) OR (a)-[:TRIGGERS]->(:Impact)-[:VIOLATES]->(c)
OPTIONAL MATCH (c)<-[:JUSTIFIED_BY]-(s:Source)
RETURN c.rule as rule, c.severity as severity, collect({properties: properties(s), labels: labels(s)}) as sources
"""

GET_CONSTRAINT_PROVENANCE = """
MATCH (c:Constraint {rule: $rule})
OPTIONAL MATCH (c)-[:ANCHORED_BY]->(axiom:Axiom)
OPTIONAL MATCH (c)-[:JUSTIFIED_BY]->(source:Source)
OPTIONAL MATCH (source)-[:SUPPORTS]->(supported_axiom:Axiom)
RETURN
    c.rule AS rule,
    axiom.name AS anchored_axiom,
    source.title AS source_title,
    source.domain AS source_domain,
    source.confidence AS confidence,
    collect(supported_axiom.name) AS supported_axioms
"""

# --- Axiom & Integrity Checks ---

CHECK_AXIOM_1 = """
MATCH (s:State)-[:DERIVED_FROM]->(i:Input)
WHERE s.content = i.content
RETURN s.id AS id, "Axiom 1 Violation: State must diverge from Input" AS rule
"""

CHECK_AXIOM_2 = """
MATCH (a:Agent)-[:PROCESSES]->(i:Input)
WHERE NOT (i)-[:ORIGINATES_FROM]->(:Environment) AND NOT (i)-[:ORIGINATES_FROM]->(:Self)
RETURN i.id AS id, "Axiom 2 Violation: Input origin undefined" AS rule
"""

CHECK_AXIOM_3 = """
MATCH (a:Action)
WHERE NOT (a)-[:DIRECTED_TOWARDS]->(:Goal)
RETURN a.id AS id, "Axiom 3 Violation: Action lacks Teleological Goal" AS rule
"""

CHECK_AXIOM_4 = """
MATCH (m:MetaRepresentation)
WHERE NOT (m)-[:REPRESENTS]->(:Process {type: "State-Agent-Action"})
RETURN m.id AS id, "Axiom 4 Violation: Meta-Representation fails to integrate Loop" AS rule
"""

CHECK_EPISTEMIC_INTEGRITY = """
MATCH (f:Fact)
WHERE NOT (f)-[:DERIVED_FROM]->(:Source)
RETURN f.id AS id, "Epistemic Violation: Fact lacks Source" AS rule
"""

# --- Amendments ---

CREATE_AMENDMENT = """
CREATE (a:ProposedAmendment {
    id: $id,
    text: $text,
    justification: $justification,
    timestamp: datetime(),
    status: 'PENDING'
})
WITH a
UNWIND $supported_axioms AS axiom_name
MATCH (ax:Axiom {name: axiom_name})
CREATE (a)-[:CLAIMS_SUPPORT]->(ax)
RETURN a.id
"""

CHECK_WRITE_BARRIER = """
MATCH (a:ProposedAmendment {id: $id})
MATCH (axiom:Axiom)
WHERE axiom.name IN $supported_axioms

// Check if any Core axiom is protected by a Write Barrier
MATCH (wb:WriteBarrier {target: "Axiom"})
WHERE wb.mode = "READ_ONLY"
RETURN count(wb) > 0 AS protected
"""

FLAG_AMENDMENT = """
MATCH (a:ProposedAmendment {id: $id})
SET a.semantic_flag = $flag
"""

CLEANUP_AMENDMENTS = """
MATCH (c:Constraint)
WHERE c.type = 'AMENDMENT'
  AND datetime() > c.sunset_date
DETACH DELETE c
"""

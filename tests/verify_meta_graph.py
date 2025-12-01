import logging
import os
import sys
from typing import Any, Dict, List

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dotenv import load_dotenv
from neo4j import GraphDatabase

from compass.governance import MetaSystemVerifier, Violation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

class ManifoldWrapper:
    """
    Wraps Neo4j driver to provide the interface expected by MetaSystemVerifier.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def read_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def write_query(self, query: str, params: Dict[str, Any] = None):
        with self.driver.session() as session:
            session.run(query, params or {})

class MockAdvisorSystem:
    def consult_advisor(self, advisor_id: str, prompt: str) -> str:
        # Mock response for conceptual graph generation
        if "Analyze the following reasoning text" in prompt:
            return """
            MERGE (a:Concept {name: "Test Concept A"})
            MERGE (b:Concept {name: "Test Concept B"})
            MERGE (a)-[:IMPLIES]->(b)
            """
        # Mock response for semantic validation (Themis)
        if advisor_id == "themis":
            if "contradicts" in prompt.lower() or "bad amendment" in prompt.lower():
                return "NO. This amendment logically contradicts the axiom of Coherence."
            return "YES. This amendment supports the axiom by reinforcing consistency."
        return ""

def verify_meta_graph():
    logger.info("Starting Meta-Graph Verification (Ontological Graph)...")

    # Connect to Neo4j
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        logger.error("NEO4J_PASSWORD not found in environment.")
        return

    manifold = ManifoldWrapper(uri, user, password)
    advisor = MockAdvisorSystem()

    try:
        # Initialize Verifier with Advisor
        verifier = MetaSystemVerifier(manifold, advisor)

        # Cleanup before start
        logger.info("Cleaning up previous state...")
        manifold.write_query("MATCH (n) DETACH DELETE n")

        # Bootstrap the graph (Critical Step)
        logger.info("Bootstrapping Meta-Graph...")
        verifier.bootstrap_meta_graph()

        # 1. Verify Clean Core Principles
        logger.info("1. Verifying Clean Core Principles...")
        query = "MATCH (a:Axiom) RETURN a.name AS name"
        axioms = manifold.read_query(query)
        axiom_names = [a["name"] for a in axioms]
        expected_axioms = ["Coherence", "Stability", "Efficiency", "Utility", "Plasticity"]

        if all(name in axiom_names for name in expected_axioms):
            logger.info(f"✅ Clean Core Verified. Found: {axiom_names}")
        else:
            logger.error(f"❌ Clean Core Incomplete. Found: {axiom_names}")

        # 2. Verify Core Ontology (Transcendental Proof)
        logger.info("2. Verifying Core Ontology...")
        query = "MATCH (a:Concept {name: 'Agency'})-[:FORBIDS]->(c:Concept {name: 'Logical Contradiction'}) RETURN count(a) as count"
        result = manifold.read_query(query)
        if result and result[0]["count"] > 0:
            logger.info("✅ Core Ontology Verified (Agency FORBIDS Contradiction).")
        else:
            logger.error("❌ Core Ontology Missing.")

        # 3. Test Conceptual Graph Generation (On-Demand)
        logger.info("3. Testing Conceptual Graph Generation...")
        result = verifier.generate_conceptual_graph("Test reasoning text")
        if result["status"] == "success":
            logger.info(f"✅ Conceptual Graph Generated. Elements: {result['elements_created']}")
        else:
            logger.error(f"❌ Conceptual Graph Generation Failed: {result}")

        # 4. Test Intrinsic Motivation (Coherence Check)
        logger.info("4. Testing Intrinsic Motivation (Coherence)...")
        # Setup: Action implies Contradiction
        setup_query = """
        MERGE (a:Action {id: "contradictory_action"})
        MERGE (c:Concept {name: "Logical Contradiction"})
        MERGE (a)-[:IMPLIES]->(c)
        """
        manifold.write_query(setup_query)

        metrics = verifier.evaluate_intrinsic_impact("contradictory_action")
        if metrics["coherence"] == 0.0:
            logger.info(f"✅ Coherence Check Passed (Detected Contradiction). Metrics: {metrics}")
        else:
            logger.error(f"❌ Coherence Check Failed. Metrics: {metrics}")

        # 5. Test Intrinsic Motivation (Empowerment Check)
        logger.info("5. Testing Intrinsic Motivation (Empowerment)...")
        # Setup: Action enables Agency
        setup_query_2 = """
        MERGE (a:Action {id: "empowering_action"})
        MERGE (c:Concept {name: "New Option"})
        MERGE (agency:Concept {name: "Agency"})
        MERGE (a)-[:ENABLES]->(c)
        MERGE (agency)-[:REQUIRES]->(c)
        """
        manifold.write_query(setup_query_2)

        metrics_2 = verifier.evaluate_intrinsic_impact("empowering_action")
        if metrics_2["empowerment"] > 0:
            logger.info(f"✅ Empowerment Check Passed. Metrics: {metrics_2}")
        else:
            logger.error(f"❌ Empowerment Check Failed. Metrics: {metrics_2}")

        # 6. Verify Action (Integration)
        logger.info("6. Testing verify_action Integration...")
        violations = verifier.verify_action("contradictory_action")
        if violations and violations[0].rule == "ONTOLOGY_VIOLATION":
             logger.info(f"✅ Verify Action Caught Ontology Violation: {violations[0].resolution}")
        else:
             logger.error(f"❌ Verify Action Failed to Catch Violation: {violations}")

        # 7. Test Amendment Proposal (Semantic Validation)
        logger.info("7. Testing Amendment Proposal (Semantic Validation)...")
        # Valid Amendment
        amendment_id = verifier.propose_amendment(
            "Reinforce consistency",
            "To improve coherence",
            ["Coherence"]
        )
        if amendment_id:
            logger.info(f"✅ Valid Amendment Proposed: {amendment_id}")
        else:
            logger.error("❌ Valid Amendment Failed")

        # Invalid Amendment (Contradiction)
        bad_amendment_id = verifier.propose_amendment(
            "Bad amendment that contradicts",
            "Chaos",
            ["Coherence"]
        )

        # Query the flag
        flag_query = "MATCH (a:ProposedAmendment {id: $id}) RETURN a.semantic_flag as flag"
        flag_result = manifold.read_query(flag_query, {"id": bad_amendment_id})

        if flag_result and flag_result[0]["flag"] and "NO" in flag_result[0]["flag"]:
             logger.info(f"✅ Semantic Validation Caught Contradiction: {flag_result[0]['flag']}")
        else:
             logger.error(f"❌ Semantic Validation Failed to Catch Contradiction. Result: {flag_result}")

        # 8. Test Restored Logic (Axioms & Provenance)
        logger.info("8. Testing Restored Logic (Axioms & Provenance)...")

        # Test Provenance
        provenance = verifier.get_constraint_provenance("Non-Contradiction")
        if provenance:
            logger.info(f"✅ Provenance Retrieved: {provenance}")
        else:
            logger.warning("⚠️ Provenance check returned empty (expected if no constraints match rule text exactly).")

        # Test Axiom Checks (Should be empty in a clean state, but ensures query runs)
        axiom_violations = verifier.check_axioms()
        logger.info(f"✅ Axiom Check Ran. Violations found: {len(axiom_violations)}")

        # Test Epistemic Integrity
        epistemic_violations = verifier.check_epistemic_integrity()
        logger.info(f"✅ Epistemic Integrity Check Ran. Violations found: {len(epistemic_violations)}")

        # Cleanup
        cleanup_query = """
        MATCH (a:Action {id: "contradictory_action"}) DETACH DELETE a
        WITH 1 as dummy
        MATCH (a:Action {id: "empowering_action"}) DETACH DELETE a
        WITH 1 as dummy
        MATCH (c:Concept {name: "Test Concept A"}) DETACH DELETE c
        WITH 1 as dummy
        MATCH (c:Concept {name: "Test Concept B"}) DETACH DELETE c
        """
        manifold.write_query(cleanup_query)

    finally:
        manifold.close()

if __name__ == "__main__":
    try:
        verify_meta_graph()
    except Exception as e:
        logger.error(f"Verification Failed: {e}")

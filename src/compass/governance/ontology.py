"""
Ontology Management for the Meta-Graph.
Handles bootstrapping and conceptual graph generation.
"""

import logging
from typing import Any, Dict

from .queries import BOOTSTRAP_AGENCY, BOOTSTRAP_CLEAN_CORE, BOOTSTRAP_HARM, BOOTSTRAP_IMPERATIVE, BOOTSTRAP_SOURCE_DAG

logger = logging.getLogger(__name__)

class OntologyManager:
    """
    Manages the Ontological Graph structure.
    """

    def __init__(self, manifold: Any, advisor_system: Any = None):
        self.manifold = manifold
        self.advisor_system = advisor_system
        self.logger = logging.getLogger("OntologyManager")

    def bootstrap(self):
        """
        Bootstrap the Meta-Graph with the Immutable Core and Source DAG.
        """
        self.logger.info("Bootstrapping Meta-Graph with Ontological Graph Core...")

        try:
            # 1. Bootstrap the Ontological Graph (The "Deep Structure")
            self.manifold.write_query(BOOTSTRAP_AGENCY)
            self.manifold.write_query(BOOTSTRAP_HARM)
            self.manifold.write_query(BOOTSTRAP_IMPERATIVE)
            self.manifold.write_query(BOOTSTRAP_CLEAN_CORE)

            self.logger.info("Ontological Graph Core bootstrapped successfully.")

            # 2. Bootstrap Source DAG
            self.bootstrap_sources()

        except Exception as e:
            self.logger.error(f"Failed to bootstrap Meta-Graph: {e}")

    def bootstrap_sources(self):
        """
        Bootstrap the Source DAG (Epistemic Provenance).
        """
        self.logger.info("Bootstrapping Source DAG...")
        try:
            self.manifold.write_query(BOOTSTRAP_SOURCE_DAG)
            self.logger.info("Source DAG bootstrapped successfully.")
        except Exception as e:
            self.logger.error(f"Failed to bootstrap Source DAG: {e}")

    def generate_conceptual_graph(self, reasoning_text: str) -> Dict[str, Any]:
        """
        Generates a conceptual graph from reasoning text using the Advisor System.
        This is an On-Demand tool triggered by the Director.
        """
        if not self.advisor_system:
            self.logger.warning("Advisor System not available for Conceptual Graph Generation.")
            return {"status": "failed", "reason": "No Advisor System"}

        self.logger.info(f"Generating Conceptual Graph for: {reasoning_text[:50]}...")

        prompt = f"""
        Analyze the following reasoning text and extract the key Concepts and their logical Relationships.
        Generate Cypher MERGE statements to represent this ontology.

        Text: "{reasoning_text}"

        Schema:
        - Nodes: (:Concept {{name: "Concept Name"}})
        - Relationships: [:IMPLIES], [:REQUIRES], [:FORBIDS], [:ENABLES]

        Output ONLY the Cypher queries, separated by newlines. Do not include markdown blocks.
        """

        try:
            # We use a generic 'philosopher' or 'logician' persona if available, or default to socrates
            response = self.advisor_system.consult_advisor("socrates", prompt)

            # Basic parsing to extract Cypher lines (assuming the LLM might be chatty)
            cypher_lines = [line.strip() for line in response.split('\\n')
                           if line.strip().startswith('MERGE') or line.strip().startswith('CREATE')]

            if not cypher_lines:
                self.logger.warning("No valid Cypher queries generated.")
                return {"status": "failed", "reason": "No Cypher generated", "raw_response": response}

            # Execute queries
            for query in cypher_lines:
                self.manifold.write_query(query)

            self.logger.info(f"Generated {len(cypher_lines)} conceptual graph elements.")
            return {"status": "success", "elements_created": len(cypher_lines)}

        except Exception as e:
            self.logger.error(f"Error generating conceptual graph: {e}")
            return {"status": "error", "error": str(e)}

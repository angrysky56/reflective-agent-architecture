
import logging
from typing import Any, Dict, List, Optional

from reflective_agent_architecture.cognition.logic_core import LogicCore

logger = logging.getLogger("raa.category_engine")

class CategoryTheoryEngine:
    """
    Engine for treating the Knowledge Graph as a formal Category.
    Objects: ThoughtNodes
    Morphisms: Relationships (Typed)

    Provides capabilities for:
    1. Formal verification of diagram commutativity.
    2. Categorical axiom checking (associativity, identity).
    3. Generating rigorous reports on graph topology.
    """

    def __init__(self, prover_path: Optional[str] = None):
        self.logic = LogicCore(prover_path)

    def verify_triangle_commutativity(self,
                                     a_id: str,
                                     b_id: str,
                                     c_id: str,
                                     path_ab_type: str,
                                     path_ac_type: str,
                                     path_bc_type: str) -> Dict[str, Any]:
        """
        Verify if the triangle A->B->C commutes with A->C.
        Checks if f_BC o f_AB = f_AC.

        Args:
            a_id, b_id, c_id: Node IDs (used for generating labels).
            path_ab_type: Relationship type A->B.
            path_ac_type: Relationship type A->C.
            path_bc_type: Relationship type B->C.

        Returns:
            Verification result from Prover9.
        """
        # 1. Define morphisms
        # Naming convention: m_<source>_<target>_<indices>
        # Sanitize types for FOL
        ab_clean = path_ab_type.lower().replace("_", "")
        bc_clean = path_bc_type.lower().replace("_", "")
        ac_clean = path_ac_type.lower().replace("_", "")

        m_ab = f"m_ab_{ab_clean}"
        m_bc = f"m_bc_{bc_clean}"
        m_ac = f"m_ac_{ac_clean}"

        obj_a = "obj_a"
        obj_b = "obj_b"
        obj_c = "obj_c"

        # 2. Construct Premises regarding existing morphisms
        # We perform a slight trick: we assert that the composition of AB and BC exists,
        # and we want to proving that it implies AC (or is equal to AC).
        # Actually, standard commutativity check is: Given these morphisms exist, prove they commute.
        # But here we are often checking if a SPECIFIC relationship serves as the composition.

        premises = [
            f"morphism({m_ab})", f"source({m_ab},{obj_a})", f"target({m_ab},{obj_b})",
            f"morphism({m_bc})", f"source({m_bc},{obj_b})", f"target({m_bc},{obj_c})",
            f"morphism({m_ac})", f"source({m_ac},{obj_a})", f"target({m_ac},{obj_c})"
        ]

        # 3. Add Category Axioms
        premises.extend(self.logic.get_category_axioms("category"))

        # 4. Construct Conclusion: composition equals direct path
        # compose(g, f, h) means g o f = h
        # We want: m_bc o m_ab = m_ac
        # In FOL (Prover9 syntax): compose(m_bc, m_ab, m_ac)

        conclusion = f"compose({m_bc}, {m_ab}, {m_ac})"

        logger.info(f"Verifying triangle commutativity: {conclusion}")

        return self.logic.prove(premises, conclusion)

    def generate_commutativity_report(self,
                                     focus_node: Dict,
                                     open_triangles: List[Dict]) -> str:
        """
        Generate a human-readable report on the commutativity of the neighborhood.
        """
        report = []
        report.append(f"# Categorical Analysis Report: {focus_node.get('name', 'Unknown')}")
        report.append(f"**Focus Node ID**: `{focus_node.get('id')}`")
        report.append("**Topology**: Category of ThoughtNodes")
        report.append("")

        if not open_triangles:
            report.append("## Status: Commutative")
            report.append("No open triangles detected in the immediate neighborhood. The local diagram is coherent.")
            return "\n".join(report)

        report.append(f"## Status: Non-Commutative ({len(open_triangles)} open triangles)")
        report.append("The following diagrams fail to commute (missing or unverified morphisms):")
        report.append("")

        for tri in open_triangles:
             b_name = tri.get('b_name', 'B')
             c_name = tri.get('c_name', 'C')
             report.append(f"- **Span**: {focus_node.get('name')} -> {b_name} ({tri.get('ab_type')}), {focus_node.get('name')} -> {c_name} ({tri.get('ac_type')})")
             report.append(f"  - Missing/Inferred Morphism: {b_name} -> {c_name}")
             if 'inferred' in tri:
                 report.append(f"  - **Proposed Completion**: {tri['inferred']}")
             if 'verification' in tri:
                 proof_res = tri['verification'].get('result', 'unknown')
                 report.append(f"  - **Formal Verification**: {proof_res.upper()}")
                 if proof_res == 'proved':
                     report.append("    -> The Diagram Commutes (Logically Consistent)")
                 else:
                     report.append("    -> Commutativity NOT Proved (Relationship may be contingent or heuristic)")
             report.append("")

        report.append("## Recommendation")
        report.append("Run `consult_ruminator` to formally close these diagrams.")

        return "\n".join(report)

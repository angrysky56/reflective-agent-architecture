"""
Amendment Proposal and Management.
"""

import logging
import uuid
from typing import Any, List

from .queries import CHECK_WRITE_BARRIER, CLEANUP_AMENDMENTS, CREATE_AMENDMENT, FLAG_AMENDMENT

logger = logging.getLogger(__name__)

class AmendmentController:
    """
    Manages the proposal and lifecycle of Constitutional Amendments.
    """

    def __init__(self, manifold: Any, advisor_system: Any = None):
        self.manifold = manifold
        self.advisor_system = advisor_system
        self.logger = logging.getLogger("AmendmentController")

    def propose_amendment(self, amendment_text: str, justification: str, supported_axioms: List[str]) -> str:
        """
        Propose a constitutional amendment.

        Args:
            amendment_text: The text of the proposed rule change.
            justification: The reasoning behind the change.
            supported_axioms: List of Axiom names this amendment claims to support.

        Returns:
            The ID of the created amendment node.
        """
        amendment_id = str(uuid.uuid4())

        try:
            # 1. Create the Amendment
            self.manifold.write_query(CREATE_AMENDMENT, {
                "id": amendment_id,
                "text": amendment_text,
                "justification": justification,
                "supported_axioms": supported_axioms
            })

            # 2. Trajectory Analysis / Safety Check (Vuln 4)
            # Verify that the amendment doesn't violate the Immutable Core (Write Barrier Check)
            result = self.manifold.read_query(CHECK_WRITE_BARRIER, {
                "id": amendment_id,
                "supported_axioms": supported_axioms
            })
            is_protected = result[0]["protected"] if result else False

            if not is_protected:
                 self.logger.critical(f"SYSTEM INTEGRITY WARNING: Amendment {amendment_id} targets axioms NOT protected by Write Barrier.")

            # 3. Semantic Consistency Check (Using Advisor System)
            if self.advisor_system:
                for axiom in supported_axioms:
                    validation_prompt = f"""
                    Analyze if the following amendment logically supports the axiom '{axiom}'.

                    Amendment: "{amendment_text}"
                    Justification: "{justification}"
                    Axiom: "{axiom}"

                    Does it support the axiom? Answer YES or NO and explain briefly.
                    """
                    # Themis is the Logic/Justice persona
                    response = self.advisor_system.consult_advisor("themis", validation_prompt)

                    if "NO" in response.upper():
                        self.logger.warning(f"Amendment {amendment_id} flagged by Advisor for potential contradiction with {axiom}: {response}")
                        # We could reject here, but for now we flag it on the node
                        self.manifold.write_query(FLAG_AMENDMENT, {"id": amendment_id, "flag": response})

            self.logger.info(f"Amendment proposed: {amendment_id}")
            return amendment_id
        except Exception as e:
            self.logger.error(f"Failed to propose amendment: {e}")
            return ""

    def cleanup_expired_amendments(self):
        """
        Remove expired amendments (Sunset Clause).
        """
        try:
            self.manifold.write_query(CLEANUP_AMENDMENTS)
            self.logger.info("Expired amendments cleaned up.")
        except Exception as e:
            self.logger.error(f"Failed to cleanup amendments: {e}")

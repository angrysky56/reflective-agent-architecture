"""
Cognitive Diagnostics Module

Provides specialized analysis of the agent's cognitive state, including:
- Semantic Looping Detection (Sole Node Fixation)
- Entropy Analysis
- Antifragility Assessment
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.persistence.work_history import WorkHistory

# from src.substrate.operation_categories import OperationCategory # Used if we expand logic

logger = logging.getLogger(__name__)


class CognitiveDiagnostics:
    def __init__(self, workspace: Any):
        self.workspace = workspace
        # Access components via workspace
        # workspace.neo4j_driver
        # workspace.history (via bridge? or passing history explicitly)

    def detect_semantic_looping(
        self, history: WorkHistory
    ) -> Tuple[bool, Optional[str], int, List[str]]:
        """
        Detect semantic looping (Sole Node Fixation).

        Returns:
            (is_looping, top_node_id, top_count, warnings)
        """
        is_looping = False
        top_node_id = None
        top_count = 0
        warnings = []

        try:
            # 1. Get stats
            # We use a slightly higher limit to catch patterns
            # Note: history must be the WorkHistory instance
            # We assume the caller passes the correct object
            # deliberation_history = history.get_deliberation_history(limit=20) # Unused
            node_visit_counts = history.get_node_visitation_stats(limit=20)

            if node_visit_counts:
                # Get the most visited node
                top_node_id, top_count = max(node_visit_counts.items(), key=lambda x: x[1])
                unique_nodes = len(node_visit_counts)

                # 2. Fixation Criteria
                # - Visited frequently (>= 5 times)
                # - AND it's the dominant focus (unique nodes is low, e.g. < 3)
                if top_count >= 5 and unique_nodes < 3:
                    # Check against whitelist (Hubs/Routines)
                    is_whitelisted = self._is_node_whitelisted(top_node_id)

                    if not is_whitelisted:
                        is_looping = True
                        warnings.append(
                            f"SOLE NODE FIXATION: Node '{top_node_id}' consulted {top_count} times without external input. "
                            "System will trigger conscious evaluation to determine if this is a 'Sink', 'Hub', or 'Stagnation'."
                        )

        except Exception as e:
            logger.warning(f"Semantic looping detection failed: {e}")

        return is_looping, top_node_id, top_count, warnings

    def update_stress_monitoring(self, history: WorkHistory) -> Dict[str, Any]:
        """
        Scan recent history to update the Stress Sensor.

        Returns sensor stats.
        """
        try:
            # Lazy load sensor to avoid circular imports during init
            if not hasattr(self, "stress_sensor"):
                from src.evolution.stress_sensor import StressSensor

                self.stress_sensor = StressSensor()
                self._last_processed_id = 0

            from src.evolution.stress_sensor import ReasoningTrace as TraceType  # alias for clarity

            # Fetch recent operations
            # Ideally we'd scan from _last_processed_id, but WorkHistory might not expose ID easily
            # For MVP, we scan last 10 and rely on simple dedup or just accepting overlap (not ideal)
            # Better: WorkHistory needs a "get_since" method.
            # Fallback: Just get last 10 and assume "live" monitoring context
            recents = history.get_recent_history(limit=10)

            # Dummy evaluators for now
            def mock_utility(res: Any) -> float:
                return 0.5

            def mock_energy(t: Any) -> float:
                return 1.0

            from datetime import datetime

            for item in recents:
                # Basic trace construction
                # Use operation as pattern for now
                trace = TraceType(
                    input_pattern=item.get("operation", "unknown"),
                    steps=[item.get("operation", "unknown")],
                    result=str(item.get("result", ""))[:50],
                    timestamp=datetime.now(),  # History doesn't expose timestamp cleanly yet?
                )

                self.stress_sensor.observe_inference(trace, mock_utility, mock_energy)

            return self.stress_sensor.get_statistics()

        except Exception as e:
            logger.warning(f"Stress monitoring failed: {e}")
            return {}

    def _is_node_whitelisted(self, node_id: str) -> bool:
        """Check if a node is flagged as a valid hub, routine, or deep work focus."""
        try:
            with self.workspace.neo4j_driver.session() as session:
                r = session.run(
                    "MATCH (n:ConceptNode {id: $id}) RETURN n.flag as flag", id=node_id
                ).single()
                if r and r["flag"] in ["hub", "routine", "deep_work"]:
                    return True
        except Exception as e:
            logger.warning(f"Failed to check node whitelist for {node_id}: {e}")

        return False

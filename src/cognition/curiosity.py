import logging
from collections import deque
from typing import Any, Deque, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CuriosityModule:
    """
    Implements Intrinsic Motivation based on Schmidhuber's Compression Progress.

    Drives the agent to explore areas where it expects to learn the most (maximize compression progress).
    Also tracks 'Boredom' to trigger exploration when the system is stuck in a loop or inactive.
    """

    def __init__(self, workspace: Any):
        self.workspace = workspace
        # Short-term memory of recent operations to detect loops/boredom
        self.recent_operations: Deque[tuple[str, str]] = deque(maxlen=20)
        self.boredom_threshold = 0.7
        self.current_boredom = 0.0

    def calculate_compression_score(self, content: str) -> float:
        """
        Calculate compression score for content.

        Estimates compressibility by checking:
        1. Similarity to existing compressed patterns (tools)
        2. Length vs semantic richness ratio
        3. Pattern regularity in embedding space

        Lower scores = more compressible (simpler, more pattern)
        Higher scores = less compressible (complex, novel, random)

        This is a practical approximation of Kolmogorov complexity.
        """
        # We need to access workspace methods
        embedding = self.workspace._embed_text(content)

        # Check similarity to existing tools (compressed knowledge)
        if self.workspace.tool_library:
            tool_similarities = []
            for tool_data in self.workspace.tool_library.values():
                tool_emb = tool_data.get("embedding", [])
                if tool_emb:
                    sim = self._cosine_similarity(embedding, tool_emb)
                    tool_similarities.append(sim)

            if tool_similarities:
                # High similarity to existing tools = highly compressible
                max_tool_sim = max(tool_similarities)
                compression_component = 1.0 - max_tool_sim
            else:
                compression_component = 0.5
        else:
            compression_component = 0.5

        # Length penalty: longer content is harder to compress
        length_ratio = min(len(content) / 1000.0, 1.0)  # Normalize to 0-1

        # Combined score
        compression_score = (compression_component * 0.7) + (length_ratio * 0.3)

        return float(compression_score)

    def track_compression_progress(self, node_id: str, new_score: float) -> float:
        """
        Track compression progress for a node.

        Returns intrinsic reward (compression progress).
        This is the core of Schmidhuber's framework:

        r_int(t+1) = C(old_compressor, data) - C(new_compressor, data)

        Positive reward = we learned to compress it better
        """
        if node_id not in self.workspace.compression_history:
            self.workspace.compression_history[node_id] = []

        history = self.workspace.compression_history[node_id]
        history.append(new_score)

        if len(history) < 2:
            return 0.0  # No progress on first observation

        # Compression progress = reduction in compression score
        old_score = history[-2]
        progress = old_score - new_score  # Positive = improvement

        return float(progress)

    def explore_for_utility(
        self, focus_area: str | None = None, max_candidates: int = 10
    ) -> dict[str, Any]:
        """
        Find thought-nodes with high utility × compression potential.

        This implements active exploration strategy from Gen 3 architecture.
        Instead of random exploration or pure novelty-seeking, the agent
        focuses on nodes that are:
        1. High utility (aligned with active goals)
        2. High compression potential (learnable patterns, not random)

        This avoids "junk food curiosity" (interesting but useless).

        Args:
            focus_area: Optional semantic focus for exploration
            max_candidates: Maximum nodes to return

        Returns:
            List of high-value exploration candidates with scores
        """
        logger.info(f"Exploring for utility-guided opportunities: {focus_area or 'general'}")

        with self.workspace.neo4j_driver.session() as session:
            # Get all thought nodes
            candidate_ids = []
            if focus_area:
                # Semantic search around focus area
                focus_embedding = self.workspace._embed_text(focus_area, is_query=True)
                results = self.workspace.collection.query(
                    query_embeddings=[focus_embedding],
                    n_results=max_candidates * 3,  # Over-fetch for filtering
                )
                if results and results.get("ids") and results["ids"][0]:
                    candidate_ids = results["ids"][0]
            else:
                # Get recent nodes
                recent_result = session.run(
                    """
                    MATCH (t:ThoughtNode)
                    WHERE t.cognitive_type IN ['problem', 'sub_problem', 'hypothesis']
                    RETURN t.id as id
                    ORDER BY t.created_at DESC
                    LIMIT $limit
                """,
                    limit=max_candidates * 3,
                )
                candidate_ids = [r["id"] for r in recent_result]

            # Score each candidate
            candidates = []
            for node_id in candidate_ids:
                node = session.run(
                    """
                    MATCH (t:ThoughtNode {id: $id})
                    RETURN t.content as content,
                           t.utility_score as utility,
                           t.compression_score as compression
                """,
                    id=node_id,
                ).single()

                if not node:
                    continue

                content = node["content"]

                # Calculate or retrieve scores
                utility = node.get("utility") or self.calculate_utility_score(content, session)
                compression = node.get("compression") or self.calculate_compression_score(content)

                # Compression potential = high score means high potential for learning
                # (not yet compressed, but has learnable patterns)
                compression_potential = compression

                # Combined score: utility × compression potential
                # High utility + high compression potential = best exploration target
                combined_score = utility * compression_potential

                candidates.append(
                    {
                        "node_id": node_id,
                        "content": content[:200],  # Preview
                        "utility_score": float(utility),
                        "compression_potential": float(compression_potential),
                        "combined_score": float(combined_score),
                    }
                )

            # Sort by combined score and take top candidates
            candidates.sort(key=lambda x: x["combined_score"], reverse=True)
            top_candidates = candidates[:max_candidates]

            # Record in working memory
            top_ids = [c["node_id"] for c in top_candidates[:5]]
            self.workspace.working_memory.record(
                operation="explore_for_utility",
                input_data={"focus_area": focus_area, "max_candidates": max_candidates},
                output_data={
                    "count": len(top_candidates),
                    "top_score": top_candidates[0]["combined_score"] if top_candidates else 0,
                },
                node_ids=top_ids,
            )

            # Persist to SQLite history
            self.workspace.history.log_operation(
                operation="explore_for_utility",
                params={"focus_area": focus_area, "max_candidates": max_candidates},
                result={"count": len(top_candidates), "top_ids": top_ids},
                cognitive_state=self.workspace.working_memory.current_goal or "Unknown",
            )

            return {
                "candidates": top_candidates,
                "count": len(top_candidates),
                "focus_area": focus_area,
                "message": f"Found {len(top_candidates)} high-value exploration targets",
            }

    def calculate_utility_score(self, content: str, session: Any) -> float:
        """
        Calculate utility score for content based on active goals.

        Uses vector similarity between content and active goals.
        Higher scores mean the content is more aligned with current goals.
        """
        if not self.workspace.active_goals:
            return 0.5  # Neutral utility when no goals set

        content_embedding = self.workspace._embed_text(content)

        total_weighted_similarity = 0.0
        total_weight = 0.0

        for goal_id, goal_data in self.workspace.active_goals.items():
            goal_embedding = self.workspace._embed_text(goal_data["description"], is_query=True)
            # Use local _cosine_similarity
            similarity = self._cosine_similarity(content_embedding, goal_embedding)

            weighted_sim = similarity * goal_data["weight"]
            total_weighted_similarity += weighted_sim
            total_weight += goal_data["weight"]

        utility_score = total_weighted_similarity / total_weight if total_weight > 0 else 0.5
        return float(utility_score)

    def _cosine_similarity(
        self, v1: list[float] | np.ndarray, v2: list[float] | np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(v1) if isinstance(v1, list) else v1
        vec2 = np.array(v2) if isinstance(v2, list) else v2

        # Avoid division by zero
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def record_activity(self, activity_type: str, details: str) -> None:
        """Record an activity to update boredom state."""
        self.recent_operations.append((activity_type, details))
        self._update_boredom()

    def _update_boredom(self) -> None:
        """
        Calculate boredom based on repetition and inactivity.
        Simple heuristic: High repetition = High boredom.
        """
        if len(self.recent_operations) < 5:
            self.current_boredom = 0.0
            return

        # Check for repetition in recent operations
        ops = [op[0] for op in self.recent_operations]
        unique_ops = set(ops)
        diversity = len(unique_ops) / len(ops)

        # Low diversity = High boredom
        # If diversity is 1.0 (all different), boredom is 0.0
        # If diversity is 0.1 (mostly same), boredom is 0.9
        self.current_boredom = 1.0 - diversity

        logger.debug(f"Curiosity Level: Boredom={self.current_boredom:.2f}")

    def should_explore(self) -> bool:
        """Decide if the system should trigger autonomous exploration."""
        return self.current_boredom > self.boredom_threshold

    def propose_goal(self) -> Optional[str]:
        """
        Propose a new goal for the Dreamer based on 'interesting' gaps.
        Uses explore_for_utility to find candidates.
        """
        if not self.should_explore():
            return None

        logger.info("Boredom threshold exceeded. Proposing exploration goal...")

        # 1. Active Exploration: Find high-utility/low-compression nodes
        # Use our own method now
        candidates_data = self.explore_for_utility(max_candidates=3)
        candidates = candidates_data.get("candidates", [])

        if not candidates:
            return "Explore random concept to break stagnation."

        # 2. Formulate a goal
        # Pick the top candidate
        target = candidates[0]
        return f"Investigate concept '{target['name']}' to improve compression."

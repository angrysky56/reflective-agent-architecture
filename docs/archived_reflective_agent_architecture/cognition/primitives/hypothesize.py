from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from reflective_agent_architecture.server import CognitiveWorkspace

logger = logging.getLogger("cwd-mcp")


class HypothesizePrimitive:
    workspace: CognitiveWorkspace

    def __init__(self, workspace: CognitiveWorkspace) -> None:
        self.workspace = workspace

    def run(self, node_a_id: str, node_b_id: str, context: str | None = None) -> dict[str, Any]:
        """
        Find novel connections between concepts in latent space (Topology Tunneling).

        Gen 3 Enhancement: Implements true "topology tunneling" by:
        1. Graph path-finding (structural relationships)
        2. Vector similarity (semantic relationships)
        3. Analogical pattern matching (finding structural isomorphisms)
        4. Searching historical solved problems for similar patterns

        This is the "Aha!" moment - the analogical leap that connects
        distant concepts (like "jar lid" → "car light housing").
        """
        logger.info(f"Topology Tunneling: {node_a_id} <-> {node_b_id}")

        with self.workspace.neo4j_driver.session() as session:
            # Get node contents
            nodes = session.run(
                """
                MATCH (a:ThoughtNode {id: $id_a})
                MATCH (b:ThoughtNode {id: $id_b})
                RETURN a.id as id_a, b.id as id_b,
                       a.cognitive_type as type_a, b.cognitive_type as type_b
                """,
                id_a=node_a_id,
                id_b=node_b_id,
            ).single()

            if not nodes:
                return {"error": "Nodes not found"}

            content_map = self.workspace._get_contents_from_chroma([nodes["id_a"], nodes["id_b"]])
            content_a = content_map.get(nodes["id_a"], "")
            content_b = content_map.get(nodes["id_b"], "")

            # 1. Find direct graph paths (structural connections)
            paths = session.run(
                """
                MATCH path = (a:ThoughtNode {id: $id_a})-[*1..3]-(b:ThoughtNode {id: $id_b})
                RETURN path
                LIMIT 5
                """,
                id_a=node_a_id,
                id_b=node_b_id,
            ).values()

            # 2. Check vector similarity (semantic connection)
            similarity_score = self._cosine_similarity(
                self.workspace._embed_text(content_a), self.workspace._embed_text(content_b)
            )

            # 3. Gen 3 Enhancement: Search for analogical patterns
            # Find similar solved problems in tool library
            analogical_tools = self._find_analogical_tools(content_a, content_b)

            # 4. Generate hypothesis using all connection types
            hypothesis_text = self._generate_hypothesis_with_analogy(
                content_a, content_b, similarity_score, analogical_tools, context
            )

            # 5. Calculate hypothesis quality (utility × novelty)
            utility = self.workspace.curiosity.calculate_utility_score(hypothesis_text, session)
            novelty = 1.0 - similarity_score  # Novel = dissimilar concepts connected
            hypothesis_quality = utility * novelty

            # Create hypothesis node
            hyp_id = self.workspace._create_thought_node(
                session, hypothesis_text, "hypothesis", confidence=float(hypothesis_quality)
            )

            # Create relationships
            session.run(
                """
                MATCH (a:ThoughtNode {id: $id_a})
                MATCH (b:ThoughtNode {id: $id_b})
                MATCH (h:ThoughtNode {id: $hyp_id})
                CREATE (h)-[:HYPOTHESIZES_CONNECTION_TO {similarity: $similarity, quality: $quality}]->(a)
                CREATE (h)-[:HYPOTHESIZES_CONNECTION_TO {similarity: $similarity, quality: $quality}]->(b)
                """,
                id_a=node_a_id,
                id_b=node_b_id,
                hyp_id=hyp_id,
                similarity=similarity_score,
                quality=hypothesis_quality,
            )

            # Link to analogical tools if found
            for tool_id in analogical_tools:
                session.run(
                    """
                    MATCH (h:ThoughtNode {id: $hyp_id})
                    MATCH (t:Tool {id: $tool_id})
                    CREATE (h)-[:INSPIRED_BY]->(t)
                    """,
                    hyp_id=hyp_id,
                    tool_id=tool_id,
                )

            # Record in working memory for context continuity
            self.workspace.working_memory.record(
                operation="hypothesize",
                input_data={"node_a": node_a_id, "node_b": node_b_id, "context": context},
                output_data={
                    "hypothesis": hypothesis_text[:500],
                    "quality": float(hypothesis_quality),
                },
                node_ids=[node_a_id, node_b_id, hyp_id],
            )

            # Persist to SQLite history
            self.workspace.history.log_operation(
                operation="hypothesize",
                params={"node_a": node_a_id, "node_b": node_b_id, "context": context},
                result={
                    "hypothesis_id": hyp_id,
                    "quality": float(hypothesis_quality),
                    "similarity": float(similarity_score),
                },
                cognitive_state=self.workspace.working_memory.current_goal or "Unknown",
            )

            return {
                "hypothesis_id": hyp_id,
                "hypothesis": hypothesis_text,
                "similarity": float(similarity_score),
                "novelty": float(novelty),
                "quality": float(hypothesis_quality),
                "path_count": len(paths),
                "analogical_tools": len(analogical_tools),
                "message": "Hypothesis generated via topology tunneling",
            }

    def _find_analogical_tools(self, content_a: str, content_b: str) -> list[str]:
        """
        Find tools with analogical patterns to current problem.
        """
        if not self.workspace.tool_library:
            return []

        # Compute combined embedding (the "stuck space")
        emb_a = self.workspace._embed_text(content_a)
        emb_b = self.workspace._embed_text(content_b)
        combined_emb = [(a + b) / 2 for a, b in zip(emb_a, emb_b)]

        # Find tools with similar patterns
        analogical = []
        for tool_id, tool_data in self.workspace.tool_library.items():
            tool_emb = tool_data.get("embedding", [])
            if tool_emb:
                similarity = self._cosine_similarity(combined_emb, tool_emb)
                if similarity > 0.6:  # Threshold for analogical match
                    analogical.append(tool_id)

        return analogical[:3]  # Top 3 analogical tools

    def _generate_hypothesis_with_analogy(
        self,
        content_a: str,
        content_b: str,
        similarity: float,
        analogical_tools: list[str],
        context: str | None,
    ) -> str:
        """
        Generate hypothesis using analogical reasoning.
        """
        system_prompt = (
            "You discover non-obvious connections between concepts through analogical reasoning. "
            "When provided with similar solved patterns, use them as inspiration for novel connections. "
            "Focus on structural similarities, not surface features. "
            "Explain the connection clearly, highlighting the structural isomorphism."
        )

        # Build context from analogical tools
        analogy_context = ""
        if analogical_tools and self.workspace.tool_library:
            analogy_patterns = []
            for tool_id in analogical_tools[:2]:  # Top 2
                tool_data = self.workspace.tool_library.get(tool_id)
                if tool_data:
                    analogy_patterns.append(
                        f"Similar Pattern ({tool_data['name']}): {tool_data['pattern'][:1500]}"
                    )
            if analogy_patterns:
                analogy_context = "\n\nAnalogical Patterns Found:\n" + "\n".join(analogy_patterns)

        context_text = f"\nContext: {context}" if context else ""

        user_prompt = (
            f"Concept A: {content_a[:2000]}\n"
            f"Concept B: {content_b[:2000]}\n"
            f"Semantic Similarity: {similarity:.2f}{analogy_context}{context_text}\n\n"
            f"Novel Connection:"
        )

        return self.workspace._llm_generate(system_prompt, user_prompt, max_tokens=16000)

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

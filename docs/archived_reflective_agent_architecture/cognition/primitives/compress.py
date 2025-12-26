import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger("cwd-mcp")


class CompressPrimitive:
    def __init__(self, workspace: Any):
        self.workspace = workspace

    def run(
        self, node_ids: list[str], tool_name: str, description: str | None = None
    ) -> dict[str, Any]:
        """
        Convert solved problem(s) into a reusable compressed tool.

        This implements the "mnemonics as tools" concept. When the agent solves
        a problem, it compresses the solution pattern into a high-level tool
        that can be reused for similar problems (like the Jeep brake light example).

        The tool lives in both:
        - Neo4j: As a Tool node with relationships to source problems
        - tool_library dict: For fast access during compression scoring
        """
        logger.info(f"Compressing {len(node_ids)} nodes into tool: {tool_name}")

        with self.workspace.neo4j_driver.session() as session:
            # Get all node contents
            nodes_result = session.run(
                """
                MATCH (n:ThoughtNode)
                WHERE n.id IN $ids
                RETURN n.id as id, n.content as content
            """,
                ids=node_ids,
            )

            nodes = {r["id"]: r["content"] for r in nodes_result}

            if not nodes:
                return {"error": "No valid nodes found"}

            # Synthesize into tool pattern
            tool_pattern = self._generate_tool_pattern(list(nodes.values()), tool_name)

            # Compute centroid embedding (tool's position in latent space)
            embeddings = [self.workspace._embed_text(content) for content in nodes.values()]
            tool_embedding = np.mean(embeddings, axis=0).tolist()

            # Create tool node
            tool_id = f"tool_{int(time.time() * 1000000)}"

            session.run(
                """
                CREATE (t:Tool {
                    id: $id,
                    name: $name,
                    pattern: $pattern,
                    description: $description,
                    usage_count: 0,
                    success_rate: 1.0,
                    created_at: timestamp()
                })
            """,
                id=tool_id,
                name=tool_name,
                pattern=tool_pattern,
                description=description or f"Compressed tool: {tool_name}",
            )

            # Link to source nodes
            for node_id in node_ids:
                session.run(
                    """
                    MATCH (tool:Tool {id: $tool_id})
                    MATCH (node:ThoughtNode {id: $node_id})
                    CREATE (tool)-[:COMPRESSED_FROM]->(node)
                """,
                    tool_id=tool_id,
                    node_id=node_id,
                )

            # Store in tool library
            self.workspace.tool_library[tool_id] = {
                "name": tool_name,
                "pattern": tool_pattern,
                "embedding": tool_embedding,
                "usage_count": 0,
                "success_rate": 1.0,
                "source_nodes": node_ids,
            }
        logger.info(f"Tool created: {tool_name} ({tool_id})")

        return {
            "tool_id": tool_id,
            "name": tool_name,
            "pattern": tool_pattern,
            "source_count": len(node_ids),
            "message": f"Tool '{tool_name}' created and added to library",
        }

    def _generate_tool_pattern(self, contents: list[str], tool_name: str) -> str:
        """
        Generate a reusable pattern from solved problem contents.

        The pattern is a compressed, generalized version of the solution
        that can be applied to similar problems.
        """
        system_prompt = (
            "You extract reusable patterns from solved problems. "
            "Create a concise, generalized pattern that captures the solution approach. "
            "Focus on the HOW (methodology) not the WHAT (specific details). "
            "Output 2-3 sentences describing the pattern."
        )

        content_previews = [f"{i+1}. {c[:300]}" for i, c in enumerate(contents[:5])]

        user_prompt = (
            f"Extract a reusable pattern for tool '{tool_name}' from these solutions:\n\n"
            + "\n\n".join(content_previews)
            + "\n\nPattern:"
        )

        return str(self.workspace._llm_generate(system_prompt, user_prompt, max_tokens=16000))

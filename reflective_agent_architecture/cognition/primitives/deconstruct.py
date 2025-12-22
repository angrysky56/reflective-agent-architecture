import json
import logging
from typing import Any

import torch

logger = logging.getLogger("cwd-mcp")


class DeconstructPrimitive:
    def __init__(self, workspace: Any):
        self.workspace = workspace

    def run(self, problem: str, max_depth: int = 50) -> dict[str, Any]:
        """
        Break a complex problem into component thought-nodes.

        Creates a hierarchical graph representing problem decomposition:
        - Root node: Original problem
        - Child nodes: Sub-problems
        - Edges: DECOMPOSES_INTO relationships

        This operates similarly to Meta's COCONUT "continuous thought" but
        materializes the reasoning tree in a queryable graph structure.
        """
        logger.info(f"Deconstructing: {problem[:8000]}...")

        with self.workspace.neo4j_driver.session() as session:
            # Create root problem node
            root_id = self.workspace._create_thought_node(
                session, problem, "problem", parent_problem=None, confidence=1.0
            )

            # 1. Advanced Tripartite Decomposition (System 2 Analysis)
            # We break the problem into three orthogonal domains:
            # - STATE (vmPFC): Context/Environment
            # - AGENT (amPFC): Persona/Intent
            # - ACTION (dmPFC): Verbs/Transitions

            system_prompt = """You are the Prefrontal Cortex Decomposition Engine.
Input: A user prompt or situation.
Task: Fragment the input into three orthogonal domains. Provide detailed, descriptive fragments (1-2 sentences each).

1. STATE (vmPFC): Where are we? Detailed context. (e.g., "Python CLI debugging session focused on list indices")
2. AGENT (amPFC): Who is involved? Detailed persona/intent. (e.g., "Frustrated User seeking immediate resolution to a crash")
3. ACTION (dmPFC): What is the transition? Detailed operation. (e.g., "Refactor the loop logic to handle out-of-bounds errors safely")

Output JSON:
{
  "state_fragment": "...",
  "agent_fragment": "...",
  "action_fragment": "..."
}"""
            user_prompt = f"Deconstruct this problem: {problem}"

            llm_output = self.workspace._llm_generate(system_prompt, user_prompt, max_tokens=4000)
            clean_response = llm_output.replace("```json", "").replace("```", "").strip()

            try:
                fragments = json.loads(clean_response)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse deconstruction JSON: {clean_response}")
                # Fallback
                fragments = {
                    "state_fragment": "Unknown Context",
                    "agent_fragment": "Unknown Agent",
                    "action_fragment": problem,
                }

            # 2. Persist Fragments as ThoughtNodes
            component_ids = []
            fragment_map = {
                "State": fragments.get("state_fragment", ""),
                "Agent": fragments.get("agent_fragment", ""),
                "Action": fragments.get("action_fragment", ""),
            }
            embeddings_map = {}

            for label in ["State", "Agent", "Action"]:
                content = fragment_map.get(label)
                if not content:
                    # Default fallback if LLM misses a domain
                    content = "Unknown" if label != "Action" else problem

                # Create node with specific label
                comp_id = self.workspace._create_thought_node(
                    session,
                    content,
                    cognitive_type=label.lower(),  # type: state, agent, action
                    parent_problem=root_id,
                    confidence=0.9,
                )
                component_ids.append(comp_id)

                # Create specific relationship based on type
                rel_type = (
                    "HAS_STATE"
                    if label == "State"
                    else "HAS_AGENT" if label == "Agent" else "REQUIRES_ACTION"
                )

                session.run(
                    f"""
                    MATCH (parent:ThoughtNode {{id: $parent_id}})
                    MATCH (child:ThoughtNode {{id: $child_id}})
                    MERGE (parent)-[:{rel_type}]->(child)
                    MERGE (parent)-[:DECOMPOSES_INTO]->(child)
                    SET child:{label}
                    """,
                    parent_id=root_id,
                    child_id=comp_id,
                )

                # Store embedding in Manifold for retrieval
                vec = self.workspace._embed_text(content)
                vec_tensor = torch.tensor(vec, dtype=torch.float32, device=self.workspace.device)
                self.workspace.get_manifold().store_pattern(vec_tensor, domain=label.lower())
                # CRITICAL: Precuneus expects lower-case keys: 'state', 'agent', 'action'
                embeddings_map[label.lower()] = vec_tensor

            # Get decomposition tree
            tree = self._get_decomposition_tree(session, root_id)

            # Reconstruct components list for response
            # Note: We iterate strictly in State, Agent, Action order if available
            final_components = []
            ordered_labels = ["State", "Agent", "Action"]
            current_idx = 0

            for label in ordered_labels:
                content = fragment_map.get(label)
                if content:
                    # component_ids matches the order of processing above
                    cid = component_ids[current_idx]
                    final_components.append({"id": cid, "content": content, "type": label})
                    current_idx += 1

            # 3. Construct Result
            # Keep tensor embeddings for internal processing (Manifold retrieval)
            # Also provide serializable version for JSON output
            embeddings_serializable = {
                k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v
                for k, v in embeddings_map.items()
            }

            result = {
                "root_id": root_id,
                "components": final_components,
                "decomposition_tree": tree,
                "embeddings": embeddings_map,  # Internal: tensors for Manifold
                "embeddings_serializable": embeddings_serializable,  # Output: lists for JSON
            }

            # Record in working memory for context continuity
            self.workspace.working_memory.record(
                operation="deconstruct",
                input_data={"problem": problem[:500]},
                output_data={"components": len(final_components), "root_id": root_id},
                node_ids=[root_id] + component_ids[:5],
            )

            # Persist to SQLite history for long-term recall
            self.workspace.history.log_operation(
                operation="deconstruct",
                params={"problem": problem[:500], "max_depth": max_depth},
                result=result,
                cognitive_state=self.workspace.working_memory.current_goal or "Unknown",
            )

            return result

    def _get_decomposition_tree(self, session, root_id: str) -> dict[str, Any]:
        """Retrieve decomposition tree"""
        result = session.run(
            """
            MATCH (root:ThoughtNode {id: $root_id})
            OPTIONAL MATCH (root)-[:DECOMPOSES_INTO]->(child:ThoughtNode)
            RETURN root.id as root_id, root.cognitive_type as root_type,
                   collect({id: child.id, type: child.cognitive_type}) as children
            """,
            root_id=root_id,
        )
        record = result.single()
        if not record:
            return {}

        root_id_res = record["root_id"]
        children_data = record["children"]

        # Collect all IDs for Chroma fetch
        all_ids = [root_id_res] + [c["id"] for c in children_data if c["id"]]
        content_map = self.workspace._get_contents_from_chroma(all_ids)

        return {
            "id": root_id_res,
            "content": content_map.get(root_id_res, ""),
            "type": record["root_type"],
            "children": [
                {"id": c["id"], "content": content_map.get(c["id"], "")}
                for c in children_data
                if c["id"]
            ],
        }

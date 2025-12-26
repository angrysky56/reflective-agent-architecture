import json
import logging
from typing import Any

import torch

logger = logging.getLogger("cwd-mcp")


class DeconstructPrimitive:
    """
    Enhanced problem decomposition with recursive depth and actionable sub-problems.

    Creates a hierarchical graph representing problem decomposition:
    - Level 1: Tripartite (State/Agent/Action) domain split
    - Level 2+: Recursive actionable sub-problem decomposition
    """

    def __init__(self, workspace: Any):
        self.workspace = workspace

    def run(self, problem: str, max_depth: int = 3) -> dict[str, Any]:
        """
        Break a complex problem into component thought-nodes with recursive depth.

        Args:
            problem: The complex problem to decompose
            max_depth: Maximum decomposition depth (1=tripartite only, 2+=recursive)

        Returns:
            Hierarchical decomposition with actionable sub-problems
        """
        logger.info(f"Deconstructing (depth={max_depth}): {problem[:200]}...")

        with self.workspace.neo4j_driver.session() as session:
            # Create root problem node
            root_id = self.workspace._create_thought_node(
                session, problem, "problem", parent_problem=None, confidence=1.0
            )

            # Phase 1: Tripartite Decomposition (State/Agent/Action)
            tripartite_result = self._tripartite_decompose(session, problem, root_id)

            # Phase 2: Actionable Sub-Problems (if max_depth > 1)
            actionable_subproblems = []
            if max_depth > 1:
                actionable_subproblems = self._actionable_decompose(
                    session, problem, root_id, max_depth - 1
                )

            # Get full decomposition tree
            tree = self._get_decomposition_tree(session, root_id, max_depth)

            # Construct result
            result = {
                "root_id": root_id,
                "components": tripartite_result["components"],
                "actionable_subproblems": actionable_subproblems,
                "decomposition_tree": tree,
                "pattern_match": tripartite_result.get("pattern_match", {}),
                "coherence": tripartite_result.get("coherence", {}),
                "embeddings": tripartite_result.get("embeddings", {}),  # Required for Precuneus
                "depth_reached": min(max_depth, 1 + len(actionable_subproblems)),
            }

            # Record in working memory
            self.workspace.working_memory.record(
                operation="deconstruct",
                input_data={"problem": problem[:500], "max_depth": max_depth},
                output_data={
                    "components": len(tripartite_result["components"]),
                    "subproblems": len(actionable_subproblems),
                    "root_id": root_id,
                },
                node_ids=[root_id] + [c["id"] for c in tripartite_result["components"]][:5],
            )

            # Persist to SQLite history
            self.workspace.history.log_operation(
                operation="deconstruct",
                params={"problem": problem[:500], "max_depth": max_depth},
                result={k: v for k, v in result.items() if k != "embeddings"},
                cognitive_state=self.workspace.working_memory.current_goal or "Unknown",
            )

            return result

    def _tripartite_decompose(self, session, problem: str, root_id: str) -> dict[str, Any]:
        """
        Phase 1: Break problem into State/Agent/Action domains.
        """
        system_prompt = """You are the Prefrontal Cortex Decomposition Engine.
Input: A user prompt or situation.
Task: Fragment the input into three orthogonal domains. Provide detailed, descriptive fragments (1-2 sentences each).

1. STATE (vmPFC - Context): What is the environment/situation? Include key entities, constraints, and given information.
   Example: "A game show scenario with 100 numbered ping-pong balls, a 3-position platform, and random piston ejection mechanics."

2. AGENT (amPFC - Perspective): Who is the actor and what is their goal? Include decision criteria and success conditions.
   Example: "A contestant who must select one ball number before the game starts, aiming to maximize the probability of their ball being ejected (winning)."

3. ACTION (dmPFC - Operation): What analysis or computation is required? Include specific steps or approaches.
   Example: "Model the Markov chain of ball positions, calculate ejection probabilities for each ball number, identify the optimal choice."

Output JSON:
{
  "state_fragment": "...",
  "agent_fragment": "...",
  "action_fragment": "..."
}"""
        user_prompt = f"Deconstruct this problem:\n\n{problem}"

        llm_output = self.workspace._llm_generate(system_prompt, user_prompt, max_tokens=4000)
        clean_response = llm_output.replace("```json", "").replace("```", "").strip()

        try:
            fragments = json.loads(clean_response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tripartite JSON: {clean_response[:500]}")
            fragments = {
                "state_fragment": "Context extraction failed",
                "agent_fragment": "Agent extraction failed",
                "action_fragment": problem[:500],
            }

        # Persist fragments as ThoughtNodes
        component_ids = []
        final_components = []
        embeddings_map = {}

        # Maps: (user_facing_type, fragment_key, relationship_type, neo4j_label, manifold_domain)
        # manifold_domain MUST be 'state', 'agent', 'action' for Precuneus compatibility
        label_map = [
            ("context", "state_fragment", "HAS_STATE", "State", "state"),
            ("perspective", "agent_fragment", "HAS_AGENT", "Agent", "agent"),
            ("operation", "action_fragment", "REQUIRES_ACTION", "Action", "action"),
        ]

        for cog_type, frag_key, rel_type, label, manifold_domain in label_map:
            content = fragments.get(frag_key, "")
            if not content:
                content = f"Unknown {label}"

            comp_id = self.workspace._create_thought_node(
                session,
                content,
                cognitive_type=cog_type,
                parent_problem=root_id,
                confidence=0.9,
            )
            component_ids.append(comp_id)
            final_components.append({"id": comp_id, "content": content, "type": cog_type})

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

            # Store in Manifold - MUST use 'state', 'agent', 'action' keys for Precuneus
            vec = self.workspace._embed_text(content)
            vec_tensor = torch.tensor(vec, dtype=torch.float32, device=self.workspace.device)
            self.workspace.get_manifold().store_pattern(vec_tensor, domain=manifold_domain)
            embeddings_map[manifold_domain] = vec_tensor

        # Calculate coherence metrics
        coherence = self._calculate_coherence(embeddings_map)
        pattern_match = self._calculate_pattern_match(fragments)

        return {
            "components": final_components,
            "embeddings": embeddings_map,
            "coherence": coherence,
            "pattern_match": pattern_match,
        }

    def _actionable_decompose(
        self, session, problem: str, parent_id: str, remaining_depth: int
    ) -> list[dict[str, Any]]:
        """
        Phase 2: Generate actionable sub-problems with recursive decomposition.
        """
        system_prompt = """You are a Problem Decomposition Specialist.
Given a complex problem, break it into 2-4 concrete, actionable sub-problems that together solve the original.

Each sub-problem should be:
- SPECIFIC: Clear scope and boundaries
- ACTIONABLE: Can be solved with a defined approach
- INDEPENDENT: Minimal dependencies on other sub-problems
- TESTABLE: Has clear success criteria

Output JSON:
{
  "sub_problems": [
    {
      "title": "Short descriptive title",
      "description": "What needs to be solved",
      "approach": "How to solve it (method/tools/analysis)",
      "success_criteria": "How to verify the solution is correct",
      "complexity": "low|medium|high"
    }
  ]
}"""
        user_prompt = f"Break this problem into actionable sub-problems:\n\n{problem}"

        llm_output = self.workspace._llm_generate(system_prompt, user_prompt, max_tokens=4000)
        clean_response = llm_output.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(clean_response)
            sub_problems = parsed.get("sub_problems", [])
        except json.JSONDecodeError:
            logger.error(f"Failed to parse actionable JSON: {clean_response[:500]}")
            return []

        result = []
        for sp in sub_problems[:4]:  # Limit to 4 sub-problems
            title = sp.get("title", "Untitled")
            description = sp.get("description", "")
            approach = sp.get("approach", "")
            success_criteria = sp.get("success_criteria", "")
            complexity = sp.get("complexity", "medium")

            full_content = (
                f"{title}: {description}\nApproach: {approach}\nSuccess: {success_criteria}"
            )

            # Create node
            sp_id = self.workspace._create_thought_node(
                session,
                full_content,
                cognitive_type="subproblem",
                parent_problem=parent_id,
                confidence=0.8,
            )

            session.run(
                """
                MATCH (parent:ThoughtNode {id: $parent_id})
                MATCH (child:ThoughtNode {id: $child_id})
                MERGE (parent)-[:DECOMPOSES_INTO]->(child)
                SET child:SubProblem, child.complexity = $complexity
                """,
                parent_id=parent_id,
                child_id=sp_id,
                complexity=complexity,
            )

            sp_result = {
                "id": sp_id,
                "title": title,
                "description": description,
                "approach": approach,
                "success_criteria": success_criteria,
                "complexity": complexity,
                "children": [],
            }

            # Recursive decomposition for high-complexity sub-problems
            if remaining_depth > 1 and complexity == "high" and len(description) > 50:
                logger.info(f"Recursively decomposing: {title}")
                sp_result["children"] = self._actionable_decompose(
                    session, full_content, sp_id, remaining_depth - 1
                )

            result.append(sp_result)

        return result

    def _calculate_coherence(self, embeddings_map: dict) -> dict[str, float]:
        """Calculate coherence metrics from embeddings."""
        if not embeddings_map:
            return {"balance": 0.0, "dominant_stream": "none"}

        # Calculate norms for each domain
        norms = {}
        for domain, tensor in embeddings_map.items():
            if isinstance(tensor, torch.Tensor):
                norms[domain] = float(torch.norm(tensor).item())
            else:
                norms[domain] = 0.0

        total_norm = sum(norms.values()) or 1.0
        weights = {f"{k}_weight": v / total_norm for k, v in norms.items()}

        # Balance is how evenly distributed the weights are (1.0 = perfect balance)
        weight_values = list(weights.values())
        if weight_values:
            mean_weight = sum(weight_values) / len(weight_values)
            variance = sum((w - mean_weight) ** 2 for w in weight_values) / len(weight_values)
            balance = max(0.0, 1.0 - (variance * 10))  # Scale variance to 0-1
        else:
            balance = 0.0

        dominant = max(norms.items(), key=lambda x: x[1])[0] if norms else "none"

        return {
            **weights,
            "balance": balance,
            "dominant_stream": dominant,
        }

    def _calculate_pattern_match(self, fragments: dict) -> dict[str, str]:
        """Estimate pattern match quality for each domain."""

        def assess_quality(text: str) -> str:
            if not text or len(text) < 20:
                return "weak"
            elif len(text) < 80:
                return "moderate"
            else:
                return "strong"

        return {
            "state": assess_quality(fragments.get("state_fragment", "")),
            "agent": assess_quality(fragments.get("agent_fragment", "")),
            "action": assess_quality(fragments.get("action_fragment", "")),
        }

    def _get_decomposition_tree(self, session, root_id: str, max_depth: int = 3) -> dict[str, Any]:
        """Retrieve full decomposition tree up to max_depth."""

        def fetch_node(node_id: str, depth: int) -> dict[str, Any]:
            if depth <= 0:
                return {"id": node_id, "truncated": True}

            result = session.run(
                """
                MATCH (node:ThoughtNode {id: $node_id})
                OPTIONAL MATCH (node)-[:DECOMPOSES_INTO]->(child:ThoughtNode)
                RETURN node.id as id, node.cognitive_type as type,
                       collect(child.id) as child_ids
                """,
                node_id=node_id,
            )
            record = result.single()
            if not record:
                return {"id": node_id, "error": "not_found"}

            # Fetch content from Chroma
            content_map = self.workspace._get_contents_from_chroma([record["id"]])

            node_data = {
                "id": record["id"],
                "content": content_map.get(record["id"], "")[:500],  # Truncate for readability
                "type": record["type"],
                "children": [],
            }

            # Recursively fetch children
            for child_id in record["child_ids"]:
                if child_id:
                    node_data["children"].append(fetch_node(child_id, depth - 1))

            return node_data

        return fetch_node(root_id, max_depth)

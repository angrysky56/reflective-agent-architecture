import logging
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from src.cognition.meta_validator import MetaValidator

logger = logging.getLogger("cwd-mcp")

if TYPE_CHECKING:
    from src.server import CognitiveWorkspace


class SynthesizePrimitive:
    def __init__(self, workspace: "CognitiveWorkspace"):
        self.workspace = workspace

    def run(self, node_ids: list[str], goal: str | None = None) -> dict[str, Any]:
        """
        Merge multiple thought-nodes into a unified insight.

        Operates in latent space by:
        1. Computing centroid of input node embeddings
        2. Finding related concepts near the centroid
        3. Creating a synthesis node representing the merged insight

        This is analogous to "latent transformations" in HRMs.
        """
        logger.info(f"Synthesizing {len(node_ids)} nodes")

        start_total = time.time()

        with self.workspace.neo4j_driver.session() as session:
            # Get all node contents AND their context (neighbors)
            # We limit context to 3 neighbors per node to avoid token explosion
            nodes_result = session.run(
                """
                MATCH (n:ThoughtNode)
                WHERE n.id IN $ids
                OPTIONAL MATCH (n)-[r]-(neighbor:ThoughtNode)
                RETURN n.id as id, collect(neighbor.id)[..3] as neighbor_ids
                """,
                ids=node_ids,
            )

            # Store content and context
            records = list(nodes_result)
            all_ids = set()
            for r in records:
                all_ids.add(r["id"])
                all_ids.update(r["neighbor_ids"])

            content_map = self.workspace._get_contents_from_chroma(list(all_ids))

            nodes_data = {}
            for r in records:
                nid = r["id"]
                neighbor_ids = r["neighbor_ids"]
                nodes_data[nid] = {
                    "content": content_map.get(nid, ""),
                    "context": [
                        content_map.get(nb_id, "")
                        for nb_id in neighbor_ids
                        if content_map.get(nb_id)
                    ],
                }

            if len(nodes_data) < 2:
                found_ids = list(nodes_data.keys())
                missing_ids = [nid for nid in node_ids if nid not in found_ids]
                return {
                    "error": "Need at least 2 valid nodes to synthesize",
                    "requested_nodes": len(node_ids),
                    "found_nodes": len(nodes_data),
                    "found_ids": found_ids,
                    "missing_ids": missing_ids,
                    "hint": "Some nodes may no longer exist. Try using 'deconstruct' to create new nodes first.",
                }

            # Compute centroid in latent space (synthesis node lives at geometric center)
            # We use the main content for embedding, not the context
            embeddings = [
                self.workspace._embed_text(data["content"]) for data in nodes_data.values()
            ]
            centroid = np.mean(embeddings, axis=0).tolist()

            # Generate synthesis (LLM merges input nodes + context)
            t0 = time.time()
            synthesis_text = self._generate_synthesis(list(nodes_data.values()), goal)
            logger.info(f"Synthesis Generation took {time.time() - t0:.2f}s")

            # Meta-Validator: Compute Metrics
            # 1. Coverage (C): Similarity between synthesis text and centroid of inputs
            real_embedding = self.workspace._embed_text(synthesis_text)
            # Cosine similarity
            dot_product = np.dot(real_embedding, centroid)
            norm_a = np.linalg.norm(real_embedding)
            norm_b = np.linalg.norm(centroid)
            if np.isnan(norm_a) or np.isnan(norm_b) or norm_a == 0 or norm_b == 0:
                coverage_score = 0.0
            else:
                coverage_score = dot_product / (norm_a * norm_b)

            if np.isnan(coverage_score):
                coverage_score = 0.0

            # 2. Rigor (R): Epistemic rigor via MetaValidator
            t1 = time.time()
            rigor_score = MetaValidator.compute_epistemic_rigor(
                synthesis_text,
                lambda sys, user: self.workspace._llm_generate(sys, user, max_tokens=1000),
            )
            logger.info(f"MetaValidator Rigor check took {time.time() - t1:.2f}s")

            # 3. Unified Score & Quadrant
            meta_stats = MetaValidator.calculate_unified_score(
                float(coverage_score), float(rigor_score), context="comprehensive_analysis"
            )

            # Create synthesis node with REAL embedding (more accurate than centroid)
            # We use the real embedding so future searches find it where it actually is semantically
            synth_id = self.workspace._create_thought_node(
                session,
                synthesis_text,
                "synthesis",
                confidence=meta_stats["unified_score"],
                embedding=real_embedding,
            )

            # Create relationships
            for node_id in node_ids:
                session.run(
                    """
                    MATCH (source:ThoughtNode {id: $source_id})
                    MATCH (synth:ThoughtNode {id: $synth_id})
                    CREATE (synth)-[:SYNTHESIZES_FROM]->(source)
                    """,
                    source_id=node_id,
                    synth_id=synth_id,
                )

            # Record in working memory for context continuity
            self.workspace.working_memory.record(
                operation="synthesize",
                input_data={"goal": goal, "node_count": len(node_ids)},
                output_data={
                    "synthesis": synthesis_text[:5000],
                    "quadrant": meta_stats["quadrant"],
                },
                node_ids=node_ids[:5] + [synth_id],
            )

            # Persist to SQLite history
            self.workspace.history.log_operation(
                operation="synthesize",
                params={"goal": goal, "node_count": len(node_ids), "node_ids": node_ids[:5]},
                result={
                    "synthesis_id": synth_id,
                    "quadrant": meta_stats["quadrant"],
                    "unified_score": meta_stats["unified_score"],
                },
                cognitive_state=self.workspace.working_memory.current_goal or "Unknown",
            )

            # --- 4. Precuneus Integration (Phase 10) ---
            try:
                # Construct Tripartite vectors
                # State: The centroid of input nodes (context)
                state_vec = torch.tensor(centroid, device=self.workspace.device)

                # Action: The resulting synthesis (output)
                action_vec = torch.tensor(real_embedding, device=self.workspace.device)

                # Agent: The active goal/intent (or fallback to state if no goal)
                # This represents "Who is synthesizing?" -> "The Goal"
                agent_vec = state_vec  # Default fallback
                if self.workspace.pointer:
                    current_goal = self.workspace.pointer.get_current_goal()
                    if current_goal is not None:
                        agent_vec = current_goal

                # Compute Energies (Surprise/Novelty)
                manifold = self.workspace.get_manifold()
                energies = {
                    "state": manifold.energy(state_vec).item(),
                    "agent": manifold.energy(agent_vec).item(),
                    "action": manifold.energy(action_vec).item(),
                }

                vectors = {"state": state_vec, "agent": agent_vec, "action": action_vec}

                # Fuse
                director = self.workspace.get_director()

                # Integration with Continuity Service (User Request)
                causal_sig = None
                if (
                    hasattr(self.workspace, "continuity_service")
                    and self.workspace.continuity_service
                ):
                    try:
                        causal_sig = self.workspace.continuity_service.get_causal_signature(
                            "Director"
                        )
                        logger.debug("Retrieved Causal Signature for Director")
                    except Exception as e:
                        logger.warning(f"Failed to get causal signature: {e}")

                _, coherence_info = self.workspace.get_precuneus()(
                    vectors,
                    energies,
                    causal_signature=causal_sig,
                    cognitive_state=director.latest_cognitive_state,
                )
                self.workspace.latest_precuneus_state = coherence_info

                logger.info(f"Precuneus State Updated: {coherence_info}")
                precuneus_debug = {"status": "success", "info": coherence_info}

            except Exception as e:
                logger.warning(f"Precuneus update failed in synthesize: {e}")
                precuneus_debug = {"status": "error", "message": str(e)}

            logger.info(f"Total Synthesis operation took {time.time() - start_total:.2f}s")

            # NEW: Flag low-coverage results for Director's Context Retrieval Loop
            if meta_stats["unified_score"] < 0.5:
                self.workspace.working_memory.needs_context_retrieval = True
                logger.warning(
                    f"Synthesis coverage low ({meta_stats['unified_score']:.2f}), "
                    "flagging for context retrieval."
                )

            result_dict = {
                "synthesis_id": synth_id,
                "synthesis": synthesis_text,
                "source_count": len(node_ids),
                "meta_validation": meta_stats,
                "precuneus_debug": precuneus_debug,
                "message": f"Synthesis created ({meta_stats['quadrant']})",
            }
            return result_dict

    def _generate_synthesis(self, nodes_data: list[dict[str, Any]], goal: str | None) -> str:
        """
        Generate synthesis merging multiple concepts.

        LLM merges concept previews; centroid computation happens in vector space.
        Provides richer context and clearer instructions for better synthesis.
        """
        system_prompt = (
            "You are a rigorous epistemological engine designed to synthesize conflicting or disparate "
            "concepts into a higher-order unified insight (Hegelian Synthesis). "
            "DO NOT merely summarize. Your goal is to:\n"
            "1. Identify the structural tension (Thesis vs. Antithesis) between the inputs.\n"
            "2. Propose a Synthesis that resolves this tension without compromising the core truth of either side.\n"
            "3. Operationalize the insight: Connect abstract concepts to concrete system mechanics/logic.\n"
            "4. Utilize the provided 'Context' fields to ground your synthesis in the specific environment or constraints.\n"
            "5. METRIC GUARDRAIL: When defining evolutionary pressure or 'stress' metrics, remember that High Cost + High Utility = High Stress (pressure to optimize). Do NOT invert this.\n"
            "Output must be structured, dense, and actionable."
        )

        # Prepare full concepts with context
        concept_list = []
        for i, data in enumerate(nodes_data[:5000], 1):  # Cap at 5000 for token management
            text = f"Concept {i}: {data['content']}"
            if data["context"]:
                # Add context as a constraint or background
                context_str = "; ".join(data["context"])
                text += f"\n   [System Context / Constraints: {context_str}]"
            concept_list.append(text)

        goal_text = f"\n\nTarget Goal of Synthesis: {goal}" if goal else ""

        user_prompt = (
            f"Perform a rigorous synthesis of the following {len(nodes_data)} concepts to achieve the Goal.{goal_text}\n\n"
            "--- INPUT CONCEPTS ---\n" + "\n\n".join(concept_list) + "\n\n"
            "--- SYNTHESIS CHECKLIST ---\n"
            "1. Deconstruct: What is the underlying conflict or gap between these concepts?\n"
            "2. Integrate: How does the new synthesis bridge this gap?\n"
            "3. Apply: How does this new insight advance the System Context provided?\n"
            "4. Predict: What is the primary risk or edge case of this new synthesis?\n\n"
            "Failures of logic or vague platitudes are untolerated."
        )

        return cast(str, self.workspace._llm_generate(system_prompt, user_prompt, max_tokens=16000))

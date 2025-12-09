"""
Sleep Cycle: Offline Learning & Consolidation

This module implements the "Night Mode" for the Reflective Agent.
It performs two key functions:
1. Replay (REM): Trains the Processor on high-quality (low energy) historical episodes.
2. Crystallization (Deep Sleep): Identifies frequent graph patterns and converts them into Tools.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch.optim as optim

from src.cognition.category_theory_engine import CategoryTheoryEngine
from src.persistence.work_history import WorkHistory
from src.processor.transformer_decoder import ProcessorConfig, TransformerDecoder

if TYPE_CHECKING:
    from src.server import CognitiveWorkspace

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class SleepCycle:
    def __init__(
        self,
        db_path: str = "raa_history.db",
        processor_config: Optional[ProcessorConfig] = None,
        workspace: Optional["CognitiveWorkspace"] = None,
    ):
        self.history = WorkHistory(db_path)
        self.config = processor_config or ProcessorConfig()
        self.device = self.config.device

        # Initialize Processor
        self.processor = TransformerDecoder(self.config).to(self.device)
        self.optimizer = optim.AdamW(self.processor.parameters(), lr=1e-4)

        # Initialize Tokenizer (GPT-2 matches default vocab size 50257)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "gpt2", revision="6c0e6080953db56375760c0471a8c5f2929baf11"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}. Using dummy encoding.")
            self.tokenizer = None

        self.workspace = workspace

    def dream(self, epochs: int = 1) -> Dict[str, Any]:
        """
        Execute the Sleep Cycle.
        """
        logger.info("Entering Sleep Cycle...")

        results = []

        for epoch in range(epochs):
            logger.info(f"Starting Sleep Cycle Epoch {epoch + 1}/{epochs}...")

            # 1. Replay (Training)
            replay_result = self._replay_memories(
                epochs=1
            )  # Replay for one epoch per sleep cycle epoch

            # 2. Crystallization (Tool Creation)
            cryst_result = self._crystallize_patterns()

            # 3. Ruminate on Graph Connections (Diagrammatic)
            graph_result = self.diagrammatic_ruminator()

            # 4. Ruminate on Codebase (Self-Documentation) - Lower Priority
            code_result = self._ruminate_on_self_code()

            # 5. Explore Latent Space (Curiosity/Dreaming)
            dream_result = self._explore_latent_space()

            logger.info(
                f"Epoch {epoch + 1} complete. Replay: {replay_result}, Crystallization: {cryst_result}, Graph: {graph_result}, Code: {code_result}, Dreamer: {dream_result}"
            )
            results.append(
                {
                    "epoch": epoch + 1,
                    "replay": replay_result,
                    "crystallization": cryst_result,
                    "graph_rumination": graph_result,
                    "code_rumination": code_result,
                    "latent_exploration": dream_result,
                }
            )

        return {"sleep_cycle_results": results}

    def _replay_memories(self, epochs: int) -> Dict[str, Any]:
        """
        Train the processor on 'Focused' (high-quality) episodes.
        """
        logger.info("Replaying memories (SFT)...")

        # Fetch high-quality history
        # We want rows where cognitive_state is "Focused" or energy is low
        # Since we don't have a direct query for that in WorkHistory yet, we'll fetch all and filter
        # In a real system, this should be a SQL query

        # Mocking the fetch for now as WorkHistory.get_all_history() doesn't exist yet
        # But we can assume we can iterate over the DB or add a method
        # For this prototype, we will try to fetch from the DB connection directly if possible,
        # or fall back to simulation if DB is empty.

        episodes = []
        try:
            rows = self.history.get_focused_episodes(limit=100)
            for row in rows:
                # Format: "Operation: <op> Params: <params> Result: <res>"
                text = f"Operation: {row['operation']}\nParams: {row['params']}\nResult: {row['result_summary']}"
                episodes.append(text)
        except Exception as e:
            logger.warning(f"Failed to fetch history: {e}")

        if not episodes:
            logger.info("No focused memories found. Skipping replay.")
            return {"steps": 0, "avg_loss": 0.0}

        total_loss = 0.0
        steps = 0

        for epoch in range(epochs):
            for text in episodes:
                if not self.tokenizer:
                    break

                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    padding="max_length",
                )
                input_ids = inputs.input_ids.to(self.device)
                labels = input_ids.clone()

                # Train step
                loss = self.processor.train_step(input_ids, labels, self.optimizer)
                total_loss += loss
                steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0.0
        logger.info(f"Replay complete. Steps: {steps}, Avg Loss: {avg_loss:.4f}")

        return {"steps": steps, "avg_loss": avg_loss}

    def _crystallize_patterns(self) -> Dict[str, Any]:
        """
        Identify frequent patterns and convert to tools.
        """
        if not self.workspace:
            return {"status": "skipped", "reason": "No workspace connected"}

        logger.info("Crystallizing patterns (Tool Genesis)...")

        # Heuristic: Find synthesis nodes that haven't been compressed yet
        # This represents "consolidating insights"
        try:
            with self.workspace.neo4j_driver.session() as session:
                # Find a synthesis node not connected to a Tool
                result = session.run(
                    """
                    MATCH (n:ThoughtNode {cognitive_type: 'synthesis'})
                    OPTIONAL MATCH (t:Tool)-[:COMPRESSED_FROM]->(n)
                    WITH n, t
                    WHERE t IS NULL AND n.id <> 'thought_1763023485617159'
                    RETURN n.id as id, n.content as content
                    LIMIT 1
                    """
                ).single()

                if result:
                    node_id = result["id"]
                    content = result["content"]

                    # Generate a tool name (simple heuristic for now)
                    # Use LLM to generate a descriptive name
                    if self.workspace:
                        try:
                            prompt = (
                                f"Suggest a short, descriptive snake_case name (max 3 words) and a brief description "
                                f"for a cognitive tool derived from this thought: '{content}'. "
                                f"Format: Name: <name> | Description: <desc>"
                            )
                            llm_response = self.workspace._llm_generate(
                                system_prompt="You are a naming assistant for AI cognitive tools.",
                                user_prompt=prompt,
                            )
                            logger.info(f"Tool naming LLM response: {llm_response}")

                            if "|" in llm_response:
                                name_part, desc_part = llm_response.split("|", 1)
                                tool_name = name_part.replace("Name:", "").strip()
                                description = desc_part.replace("Description:", "").strip()
                            else:
                                logger.warning(
                                    "LLM response did not match format 'Name: ... | Description: ...'"
                                )
                                # Fallback: use numeric part of ID
                                suffix = node_id.split("_")[-1]
                                tool_name = f"tool_{suffix}"
                                description = f"Crystallized from synthesis: {content[:50]}..."
                        except Exception as e:
                            logger.warning(f"Failed to generate tool name: {e}")
                            suffix = node_id.split("_")[-1]
                            tool_name = f"tool_{suffix}"
                            description = f"Crystallized from synthesis: {content[:50]}..."
                    else:
                        suffix = node_id.split("_")[-1]
                        tool_name = f"tool_{suffix}"
                        description = f"Crystallized from synthesis: {content[:50]}..."

                    # Compress it
                    logger.info(f"Crystallizing node {node_id} into {tool_name}")
                    tool_result = self.workspace.compress_to_tool(
                        node_ids=[node_id], tool_name=tool_name, description=description
                    )

                    return {
                        "new_tools_created": 1,
                        "tool_id": tool_result.get("tool_id"),
                        "message": f"Crystallized tool {tool_name}",
                    }
                else:
                    return {"new_tools_created": 0, "message": "No new patterns to crystallize"}

        except Exception as e:
            logger.error(f"Crystallization failed: {e}")
            return {"error": str(e)}

    def diagrammatic_ruminator(self, focus_node_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Category-Theoretic Ruminator (Diagram Chasing).

        Treats the knowledge graph as a diagram category.
        Finds 'open' diagrams (e.g., non-commutative triangles) where:
        A -> B and A -> C exist, but the relationship between B and C is undefined.

        Acts as a Functor mapping this structural diagram to the LLM (Semantic Category)
        to deduce the missing morphism (relationship) that makes the diagram commute.
        """
        if not self.workspace:
            return {"status": "skipped", "reason": "No workspace"}

        logger.info(f"Ruminating on diagram topology (Focus: {focus_node_id})...")

        try:
            # 1. Select Focus Node (Object A)
            if not focus_node_id:
                with self.workspace.neo4j_driver.session() as session:
                    # Find potential 'span' roots: nodes with out-degree >= 2
                    # Relaxed to include ThoughtNodes
                    result = session.run(
                        """
                        MATCH (n)
                        WHERE (n:ConceptNode OR n:ThoughtNode)
                        WITH n, size((n)-->()) as out_degree
                        WHERE out_degree >= 2
                        RETURN n.id as id, n.name as name, n.content as content
                        ORDER BY rand()
                        LIMIT 1
                        """
                    ).single()

                    if not result:
                        return {
                            "status": "idle",
                            "message": "No suitable focus nodes found (graph too sparse).",
                        }

                    focus_node_id = result["id"]
                    focus_name = result["name"] or result["content"] or "Unnamed Node"
            else:
                with self.workspace.neo4j_driver.session() as session:
                    # Generic match for ID
                    result = session.run(
                        """
                        MATCH (n)
                        WHERE n.id = $id
                        RETURN n.name as name, n.content as content
                        """,
                        id=focus_node_id,
                    ).single()

                    if not result:
                        return {
                            "status": "error",
                            "message": f"Focus node {focus_node_id} not found.",
                        }
                    focus_name = result["name"] or result["content"] or f"Node {focus_node_id}"

            # 2. Find Open Triangle (The 'Span': B <- A -> C)
            # Find neighbors B and C that are NOT connected to each other
            span_query = """
                MATCH (a {id: $focus_id})-[r1]->(b)
                MATCH (a)-[r2]->(c)
                WHERE b.id < c.id  // Avoid duplicates
                AND NOT (b)--(c)   // The crucial 'Open' condition
                AND (b:ConceptNode OR b:ThoughtNode)
                AND (c:ConceptNode OR c:ThoughtNode)
                RETURN b.id as b_id, b.name as b_name, b.content as b_content,
                       c.id as c_id, c.name as c_name, c.content as c_content
                LIMIT 1
            """

            with self.workspace.neo4j_driver.session() as session:
                span_result = session.run(span_query, focus_id=focus_node_id).single()

            if not span_result:
                return {
                    "status": "idle",
                    "message": f"No open triangles found for {focus_name} (Diagram commutes).",
                }

            b_data = {
                "id": span_result["b_id"],
                "name": span_result["b_name"],
                "content": span_result["b_content"],
            }
            c_data = {
                "id": span_result["c_id"],
                "name": span_result["c_name"],
                "content": span_result["c_content"],
            }

            # 3. Functorial Mapping (LLM Query)
            prompt = (
                f"Perform a diagram chase on these concepts:\n"
                f"Object A (Source): {focus_name}\n"
                f"Object B: {b_data['name']} ({b_data['content']})\n"
                f"Object C: {c_data['name']} ({c_data['content']})\n\n"
                f"Topology: A -> B and A -> C.\n"
                f"Analysis: Does a direct relationship exist between B and C?\n"
                f"If yes, specify TYPE and DIRECTION.\n"
                f"Format: RELATION: <Yes/No> | TYPE: <type> | DIRECTION: <B->C or C->B> | REASON: <text>"
            )

            # Use dream/ruminator provider
            response = self.workspace.ruminator_provider.generate(
                system_prompt="You are a Category Theoretic Reasoner.", user_prompt=prompt
            )

            # 4. Parse Response (The Morphism)
            if "RELATION: Yes" in response:
                parts = response.split("|")
                rel_type = "RELATED_TO"
                direction = "B->C"
                reason = "Inferred via diagram chasing"

                for part in parts:
                    if "TYPE:" in part:
                        t = part.split(":")[1].strip().upper().replace(" ", "_")
                        if t.isalnum() or "_" in t:
                            rel_type = t
                    if "DIRECTION:" in part:
                        if "C->B" in part:
                            direction = "C->B"
                    if "REASON:" in part:
                        reason = part.split(":")[1].strip()

                # 5. Formal Verification (Categorical Engine)
                verification_result = {"result": "skipped"}
                try:
                    cat_engine = CategoryTheoryEngine()

                    # Assume simplistic relationship types for A->B and A->C for the prototype
                    # In production, these should be fetched from the 'span_query' result
                    # For now, we assume 'RELATED_TO' to test the engine flow

                    # Determine source/target for verification based on direction
                    v_source = b_data["id"] if direction == "B->C" else c_data["id"]
                    v_target = c_data["id"] if direction == "B->C" else b_data["id"]

                    verification_result = cat_engine.verify_triangle_commutativity(
                        a_id=focus_node_id,
                        b_id=v_source,
                        c_id=v_target,
                        path_ab_type="RELATED_TO",  # Placeholder: would need actual type from graph
                        path_ac_type="RELATED_TO",  # Placeholder
                        path_bc_type=rel_type,
                    )
                    logger.info(f"Formal Verification Result: {verification_result.get('result')}")
                except Exception as ve:
                    logger.warning(f"Verification failed: {ve}")
                    verification_result = {"result": "error", "reason": str(ve)}

                # 6. Apply the Morphism (Update Graph)
                source_id = b_data["id"] if direction == "B->C" else c_data["id"]
                target_id = c_data["id"] if direction == "B->C" else b_data["id"]

                with self.workspace.neo4j_driver.session() as session:
                    session.run(
                        f"""
                        MATCH (s:ConceptNode {{id: $sid}}), (t:ConceptNode {{id: $tid}})
                        MERGE (s)-[r:{rel_type}]->(t)
                        SET r.basis = 'diagram_chasing',
                            r.reason = $reason,
                            r.verified = $verified,
                            r.timestamp = timestamp()
                        """,
                        sid=source_id,
                        tid=target_id,
                        reason=reason,
                        verified=str(verification_result.get("result") == "proved"),
                    )

                # 7. Generate Categorical Report (Prototype)
                # To do this correctly, we'd gather all open triangles, but here we report on the one we processed
                triangle_info = {
                    "b_name": b_data["name"],
                    "c_name": c_data["name"],
                    "ab_type": "RELATED_TO",  # Placeholder
                    "ac_type": "RELATED_TO",  # Placeholder
                    "inferred": f"{direction} : {rel_type}",
                    "verification": verification_result,
                }

                cat_report = "Categorical Report Generated."
                try:
                    cat_report = cat_engine.generate_commutativity_report(
                        focus_node={"id": focus_node_id, "name": focus_name},
                        open_triangles=[triangle_info],
                    )
                except Exception as e:
                    logger.warning(f"Report generation failed: {e}")
                    pass

                return {
                    "status": "success",
                    "operation": "diagram_completion",
                    "triangle": f"{focus_name} -> ({b_data['name']}, {c_data['name']})",
                    "inferred_morphism": f"{direction} : {rel_type}",
                    "reason": reason,
                    "verification": verification_result.get("result"),
                    "report": cat_report,
                }

            return {
                "status": "success",
                "operation": "diagram_verified",
                "message": "No direct morphism found.",
            }

        except Exception as e:
            logger.error(f"Diagrammatic rumination failed: {e}")
            return {"status": "error", "reason": str(e)}

    def _ruminate_on_self_code(self) -> Dict[str, Any]:
        """
        Ruminator: Autonomously document the codebase during downtime.
        Uses a cheaper/free model (amazon/nova-2-lite-v1:free) with extra rate limiting.
        """
        if not self.workspace or not hasattr(self.workspace, "system_guide"):
            return {"status": "skipped", "reason": "No workspace or system_guide connected"}

        # Check if enabled
        if not getattr(self.workspace.config, "ruminator_enabled", False):
            return {"status": "skipped", "reason": "Ruminator disabled in config"}

        logger.info("Ruminating on codebase (Self-Documentation)...")

        # Extra rate limiting delay (seconds)
        ruminator_delay = getattr(self.workspace.config, "ruminator_delay", 2.0)

        try:
            # 1. Update AST bookmarks
            scan_result = self.workspace.system_guide.scan_codebase(".")
            logger.info(f"Rumination Scan: {scan_result}")

            # 2. Find undocumented bookmarks
            # We look for bookmarks where notes are "No docstring" or empty
            undocumented = []
            try:
                with self.workspace.neo4j_driver.session() as session:
                    result = session.run(
                        """
                        MATCH (b:CodeBookmark)
                        WHERE b.notes = 'No docstring' OR b.notes = ''
                        RETURN b.id as id, b.snippet as snippet, b.file as file, b.line as line
                        LIMIT 5
                        """
                    )
                    undocumented = [record for record in result]
            except Exception as e:  # Catch Neo4j errors
                logger.error(f"Ruminator DB Error: {e}")
                return {"status": "error", "reason": f"DB Error: {e}"}

            if not undocumented:
                return {"status": "idle", "message": "No undocumented code found."}

            processed_count = 0

            for record in undocumented:
                bid = record["id"]
                snippet = record["snippet"]
                file_path = record["file"]

                logger.info(f"Ruminating on {bid}...")

                # Generate documentation
                prompt = (
                    f"Analyze this code snippet from {file_path}:\n\n"
                    f"```python\n{snippet}\n```\n\n"
                    f"Write a concise but informative docstring explaining what this component does."
                )

                try:
                    # Use the dedicated Ruminator provider (supports backoff)
                    docstring = self.workspace.ruminator_provider.generate(
                        system_prompt="You are an expert software documentation assistant.",
                        user_prompt=prompt,
                    )

                    # Validate response
                    if docstring.strip().lower().startswith("error"):
                        logger.warning(f"Ruminator generated error message for {bid}: {docstring}")
                        continue

                    # Update the bookmark
                    with self.workspace.neo4j_driver.session() as session:
                        session.run(
                            """
                            MATCH (b:CodeBookmark {id: $id})
                            SET b.notes = $docstring, b.ruminated_at = timestamp()
                            """,
                            id=bid,
                            docstring=docstring,
                        )

                    processed_count += 1

                    # Extra rate limiting (on top of provider backoff)
                    import time  # Assuming time is imported at module level, but adding here for self-containment

                    time.sleep(ruminator_delay)

                except Exception as e:
                    logger.warning(f"Failed to ruminate on {bid}: {e}")

            return {
                "status": "active",
                "processed": processed_count,
                "model": self.workspace.ruminator_provider.model_name,
            }
        except Exception as e:
            logger.error(f"Rumination critical error: {e}")
            return {"status": "error", "reason": str(e)}

    def _explore_latent_space(self) -> Dict[str, Any]:
        """
        Curiosity-driven exploration (The Dreamer).
        Triggered when the system is bored or sees high utility gaps.
        """
        if not self.workspace.curiosity.should_explore():
            return {"status": "idle", "message": "Curiosity threshold not met."}

        logger.info("Curiosity triggered! Entering Dream State...")

        goal = self.workspace.curiosity.propose_goal()
        if not goal:
            return {"status": "idle", "message": "No interesting goals found."}

        logger.info(f"Dream Goal: {goal}")

        # Use the Dreamer (High-IQ) to explore this goal
        try:
            # 1. Ask the Dreamer what to do with this goal
            prompt = (
                f"You are in a deep sleep state (Dreaming). Your goal is: {goal}\n"
                f"You have access to 'hypothesize' and 'synthesize' tools.\n"
                f"Propose a single tool call to advance this goal."
            )

            # We use the Dreamer provider directly
            response = self.workspace.dreamer_provider.generate(
                system_prompt="You are an autonomous cognitive engine.", user_prompt=prompt
            )

            # 2. Execute the proposed action (Simplified for now)
            # In a full agent, we'd parse the tool call.
            # Here we'll just log the insight as a 'Dream' concept.

            with self.workspace.neo4j_driver.session() as session:
                session.run(
                    """
                    MERGE (c:ConceptNode {name: 'Dream Journal'})
                    MERGE (d:ConceptNode {name: $goal})
                    MERGE (c)-[:DREAMT_OF]->(d)
                    SET d.insight = $insight, d.timestamp = timestamp()
                    """,
                    goal=f"Dream: {goal[:50]}...",
                    insight=response,
                )

            self.workspace.curiosity.record_activity("dream", goal)

            return {
                "status": "active",
                "goal": goal,
                "insight": response[:100] + "...",
                "model": self.workspace.dreamer_provider.model_name,
            }

        except Exception as e:
            logger.error(f"Dream execution failed: {e}")
            return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    sleep = SleepCycle()
    print(sleep.dream())

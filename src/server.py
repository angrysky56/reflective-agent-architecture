"""
Cognitive Workspace Database (CWD) MCP Server

A "System 2" reasoning engine implementing Gen 3 Utility-Guided Architecture.

Combines:
- Neo4j: Structural/graph reasoning (problem decomposition, analogical paths)
- Chroma: Latent space operations (vector similarity, compression tracking)
- Schmidhuber's Compression Progress: Intrinsic curiosity reward system
- Utility-Guided Exploration: Directed learning focused on goal-aligned compression

Cognitive Primitives:
1. deconstruct: Break incompressible problems into compressible components
2. hypothesize: Topology tunneling - find analogical leaps between concepts
3. synthesize: Merge thought-nodes in latent space (compression as tools)
4. constrain: Validate against utility rules (perceived utility filter)
5. compress_to_tool: Convert solved problems into reusable patterns
6. explore_for_utility: Active exploration maximizing utility × compression progress

Based on:
- Meta COCONUT (continuous thought trees)
- Schmidhuber's Compression Progress (intrinsic motivation)
- Gen 3 Utility-Guided Architecture (directed curiosity)
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict

import chromadb
import numpy as np
import ollama
import torch
from chromadb.config import Settings
from mcp.server import Server
from mcp.types import TextContent, Tool
from neo4j import GraphDatabase
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer

# RAA imports
from src.director import Director, DirectorConfig
from src.integration.cwd_raa_bridge import BridgeConfig, CWDRAABridge
from src.integration.sleep_cycle import SleepCycle
from src.manifold import HopfieldConfig, Manifold
from src.pointer.goal_controller import GoalController, PointerConfig
from src.processor import Processor, ProcessorConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cwd-mcp")

# Initialize MCP server
server = Server("cognitive-workspace-db")


# ============================================================================
# Configuration
# ============================================================================


class CWDConfig(BaseSettings):
    """
    Configuration for Cognitive Workspace Database.

    Loads from environment variables or .env file - never hardcode credentials!
    Set NEO4J_PASSWORD in your .env file or environment.

    Searches for .env file in:
    1. Current directory
    2. Parent directory (project root when running from src/)
    """

    # Find .env file in project root (one level up from src/)
    _env_file = Path(__file__).parent.parent / ".env"

    model_config = SettingsConfigDict(
        env_file=str(_env_file) if _env_file.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(...)  # Required from environment - no default
    chroma_path: str = Field(default="./chroma_data")
    embedding_model: str = Field(default="qwen3-embedding:0.6b")
    confidence_threshold: float = Field(default=0.3)  # Lowered for asymmetric embeddings
    llm_base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="qwen3:4b")


class CognitiveWorkspace:
    """
    Manages the cognitive workspace - a hybrid system combining:
    - Neo4j for structural/graph reasoning
    - Chroma for latent space vector operations
    - Cognitive primitives for active reasoning
    """

    def __init__(self, config: CWDConfig):
        self.config = config

        # Initialize Neo4j
        self.neo4j_driver = GraphDatabase.driver(
            config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
        )

        # Initialize Chroma
        self.chroma_client = chromadb.Client(
            Settings(persist_directory=config.chroma_path, anonymized_telemetry=False)
        )

        # Initialize embedding model
        # For Qwen embeddings, optimize with flash_attention_2 if GPU available
        is_qwen = "qwen" in config.embedding_model.lower()
        has_gpu = torch.cuda.is_available()

        if is_qwen and has_gpu:
            try:
                # Try flash_attention_2 for GPU acceleration (requires flash-attn package)
                self.embedding_model = SentenceTransformer(
                    config.embedding_model,
                    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
                    tokenizer_kwargs={"padding_side": "left"},
                )
                logger.info("Initialized Qwen embedding model with flash_attention_2 (GPU)")
            except Exception as e:
                # Fallback to standard (works on CPU and GPU without flash-attn)
                logger.info(f"flash_attention_2 not available, using standard attention: {e}")
                self.embedding_model = SentenceTransformer(
                    config.embedding_model, tokenizer_kwargs={"padding_side": "left"}
                )
        elif is_qwen:
            # CPU mode - just use left padding for Qwen
            self.embedding_model = SentenceTransformer(
                config.embedding_model, tokenizer_kwargs={"padding_side": "left"}
            )
            logger.info("Initialized Qwen embedding model (CPU)")
        else:
            # Standard models (all-MiniLM, etc.)
            self.embedding_model = SentenceTransformer(config.embedding_model)

        # Create Chroma collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="thought_nodes", metadata={"description": "Cognitive workspace thought-nodes"}
        )

        # Gen 3 Enhancement: Active goals for utility-guided exploration
        self.active_goals: dict[str, dict[str, Any]] = {}

        # Compression Progress Tracking (Schmidhuber)
        self.compression_history: dict[str, list[float]] = {}  # node_id -> [scores over time]

        # Tool Library: Compressed knowledge as reusable patterns
        self.tool_library: dict[str, dict[str, Any]] = (
            {}
        )  # tool_id -> {pattern, usage_count, success_rate}

        # Initialize database schema enhancements
        self._initialize_gen3_schema()

        logger.info("Cognitive Workspace initialized with Gen 3 architecture")

    def close(self):
        """Cleanup connections"""
        self.neo4j_driver.close()

    def _initialize_gen3_schema(self):
        """
        Initialize Gen 3 architecture enhancements in Neo4j.

        Adds:
        - Compression progress tracking
        - Utility scores
        - Goal nodes
        - Tool library nodes
        """
        with self.neo4j_driver.session() as session:
            # Create constraints and indexes
            session.run(
                """
                CREATE CONSTRAINT thought_id_unique IF NOT EXISTS
                FOR (t:ThoughtNode) REQUIRE t.id IS UNIQUE
            """
            )
            session.run(
                """
                CREATE INDEX thought_cognitive_type IF NOT EXISTS
                FOR (t:ThoughtNode) ON (t.cognitive_type)
            """
            )
            session.run(
                """
                CREATE INDEX thought_utility IF NOT EXISTS
                FOR (t:ThoughtNode) ON (t.utility_score)
            """
            )
            session.run(
                """
                CREATE INDEX thought_compression IF NOT EXISTS
                FOR (t:ThoughtNode) ON (t.compression_score)
            """
            )
            session.run(
                """
                CREATE CONSTRAINT goal_id_unique IF NOT EXISTS
                FOR (g:Goal) REQUIRE g.id IS UNIQUE
            """
            )
            session.run(
                """
                CREATE CONSTRAINT tool_id_unique IF NOT EXISTS
                FOR (t:Tool) REQUIRE t.id IS UNIQUE
            """
            )
        logger.info("Gen 3 schema initialized")

    # ========================================================================
    # Gen 3 Architecture: Goal Management & Utility Scoring
    # ========================================================================

    def set_goal(self, goal_description: str, utility_weight: float = 1.0) -> str:
        """
        Set an active goal for utility-guided exploration.

        Goals act as the "Director" in Gen 3 architecture, filtering which
        compression progress gets rewarded. This prevents "junk food curiosity"
        where the agent learns interesting but useless patterns.

        Args:
            goal_description: Natural language description of the goal
            utility_weight: How much to weight this goal (0.0-1.0)

        Returns:
            goal_id: Unique identifier for this goal
        """
        goal_id = f"goal_{int(time.time() * 1000000)}"

        with self.neo4j_driver.session() as session:
            session.run(
                """
                CREATE (g:Goal {
                    id: $id,
                    description: $description,
                    utility_weight: $weight,
                    created_at: timestamp(),
                    active: true
                })
            """,
                id=goal_id,
                description=goal_description,
                weight=utility_weight,
            )

        self.active_goals[goal_id] = {
            "description": goal_description,
            "weight": utility_weight,
            "created_at": time.time(),
        }

        logger.info(f"Goal set: {goal_description} (weight: {utility_weight})")
        return goal_id

    def get_active_goals(self) -> dict[str, dict[str, Any]]:
        """Get all active goals"""
        return self.active_goals.copy()

    def _calculate_utility_score(self, content: str, session) -> float:
        """
        Calculate utility score for content based on active goals.

        Uses vector similarity between content and active goals.
        Higher scores mean the content is more aligned with current goals.

        This implements the "Perceived Utility" filter from Gen 3 architecture.
        """
        if not self.active_goals:
            return 0.5  # Neutral utility when no goals set

        content_embedding = self._embed_text(content)

        total_weighted_similarity = 0.0
        total_weight = 0.0

        for goal_id, goal_data in self.active_goals.items():
            goal_embedding = self._embed_text(goal_data["description"], is_query=True)
            similarity = self._cosine_similarity(content_embedding, goal_embedding)

            weighted_sim = similarity * goal_data["weight"]
            total_weighted_similarity += weighted_sim
            total_weight += goal_data["weight"]

        utility_score = total_weighted_similarity / total_weight if total_weight > 0 else 0.5
        return float(utility_score)

    # ========================================================================
    # Gen 3 Architecture: Compression Progress Tracking (Schmidhuber)
    # ========================================================================

    def _calculate_compression_score(self, content: str) -> float:
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
        embedding = self._embed_text(content)

        # Check similarity to existing tools (compressed knowledge)
        if self.tool_library:
            tool_similarities = []
            for tool_data in self.tool_library.values():
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

    def _track_compression_progress(self, node_id: str, new_score: float) -> float:
        """
        Track compression progress for a node.

        Returns intrinsic reward (compression progress).
        This is the core of Schmidhuber's framework:

        r_int(t+1) = C(old_compressor, data) - C(new_compressor, data)

        Positive reward = we learned to compress it better
        """
        if node_id not in self.compression_history:
            self.compression_history[node_id] = []

        history = self.compression_history[node_id]
        history.append(new_score)

        if len(history) < 2:
            return 0.0  # No progress on first observation

        # Compression progress = reduction in compression score
        old_score = history[-2]
        progress = old_score - new_score  # Positive = improvement

        return float(progress)

    # ========================================================================
    # Gen 3 Architecture: Knowledge Compression as Tools
    # ========================================================================

    def compress_to_tool(
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

        Args:
            node_ids: Thought-nodes representing the solved problem
            tool_name: Name for this tool
            description: Optional description of what this tool does

        Returns:
            Tool creation result with tool_id and success metrics
        """
        logger.info(f"Compressing {len(node_ids)} nodes into tool: {tool_name}")

        with self.neo4j_driver.session() as session:
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
            embeddings = [self._embed_text(content) for content in nodes.values()]
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
            self.tool_library[tool_id] = {
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

        return self._llm_generate(system_prompt, user_prompt, max_tokens=1000)

    # ========================================================================
    # Gen 3 Architecture: Utility-Guided Exploration
    # ========================================================================

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

        with self.neo4j_driver.session() as session:
            # Get all thought nodes
            if focus_area:
                # Semantic search around focus area
                focus_embedding = self._embed_text(focus_area, is_query=True)
                results = self.collection.query(
                    query_embeddings=[focus_embedding],
                    n_results=max_candidates * 3,  # Over-fetch for filtering
                )
                candidate_ids = results["ids"][0] if results["ids"] else []
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
                utility = node.get("utility") or self._calculate_utility_score(content, session)
                compression = node.get("compression") or self._calculate_compression_score(content)

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

            return {
                "candidates": top_candidates,
                "count": len(top_candidates),
                "focus_area": focus_area,
                "message": f"Found {len(top_candidates)} high-value exploration targets",
            }

    def _embed_text(self, text: str, is_query: bool = False) -> list[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed
            is_query: If True, uses query prompt for Qwen models (for similarity search).
                     If False, embeds as document (for storage).

        For Qwen embedding models, queries should use prompt_name="query" for better
        retrieval performance. Documents are embedded without prompts.
        """
        with torch.no_grad():
            # Check if this is a Qwen model that supports prompts
            is_qwen = "qwen" in self.config.embedding_model.lower()
            if is_qwen and is_query:
                # Use query prompt for better retrieval
                embedding = self.embedding_model.encode(text, prompt_name="query")
            else:
                # Standard document embedding
                embedding = self.embedding_model.encode(text)

            # Convert to list of floats
            if hasattr(embedding, "tolist"):
                result = embedding.tolist()
            else:
                result = list(embedding)
            return result  # type: ignore[return-value]

    def _llm_generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 8000) -> str:
        """
        Generate text using local Ollama LLM.

        This is intentionally simple - the LLM only handles bridge text generation.
        Heavy lifting (reasoning, similarity, path-finding) happens in vector/graph operations.

        Framework context: The LLM is part of a cognitive workspace that combines:
        - Neo4j for structural reasoning (graph operations)
        - Chroma for latent space operations (vector similarity)
        - This LLM for generating human-readable bridge text

        The LLM's role is to produce concise, clear outputs - not to do the reasoning itself.
        """
        try:
            # Fast-path: avoid implicit model pull by verifying availability first
            try:
                _ = ollama.show(model=self.config.llm_model)
            except Exception as avail_err:
                logger.warning(f"LLM model '{self.config.llm_model}' not available: {avail_err}")
                raise RuntimeError("LLM model unavailable")
            # Enhanced system prompt with framework context
            enhanced_system = f"""You are a text generation component in a cognitive reasoning system.

Your role: Generate concise, clear text outputs. The system handles reasoning via graph and vector operations.

Framework: Cognitive Workspace Database (System 2 Reasoning)
- Neo4j: Structural/graph reasoning
- Chroma: Vector similarity in latent space
- Your task: Bridge text generation only

{system_prompt}

CRITICAL: Output your final answer directly. You may think internally, but end with clear, concise output."""

            response = ollama.chat(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": enhanced_system},
                    {"role": "user", "content": user_prompt},
                ],
                options={"num_predict": max_tokens, "temperature": 0.7},
            )
            content = response["message"]["content"].strip()
            logger.info(f"Raw LLM output: {content[:500]}...")  # Log first 500 chars for debug

            # Strip reasoning artifacts that models add
            # Remove <think>...</think> blocks, but be careful not to delete everything
            if "<think>" in content:
                # Try to extract content after </think>
                parts = re.split(r"</think>", content, flags=re.IGNORECASE)
                if len(parts) > 1 and parts[-1].strip():
                    content = parts[-1].strip()
                else:
                    # If everything is inside <think> or no closing tag, just strip the tags
                    content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)

            # Remove common reasoning prefixes (case-insensitive, at start of content)
            reasoning_patterns = [
                r"^(?:Okay|Alright|Let me|Hmm|So|Well|First|Now)\s*[,:]?\s*",
                r"^(?:The user|I need to|I should|Looking at)\s+.*?\.\s*",
            ]

            for pattern in reasoning_patterns:
                # Only replace if it leaves something behind
                if re.match(pattern, content, flags=re.IGNORECASE):
                    new_content = re.sub(pattern, "", content, count=1, flags=re.IGNORECASE).strip()
                    if new_content:
                        content = new_content

            # Extract actual content after reasoning markers
            # Look for explicit markers like "OUTPUT:"
            output_markers = [
                r"(?:OUTPUT|ANSWER|RESULT|FINAL):\s*(.+)",  # Explicit markers
            ]

            for marker_pattern in output_markers:
                match = re.search(marker_pattern, content, re.DOTALL | re.IGNORECASE)
                if match and len(match.group(1).strip()) > 20:
                    content = match.group(1).strip()
                    break

            # Clean up multiple spaces and newlines
            content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)  # Max 2 newlines
            content = re.sub(r"[ \t]+", " ", content)  # Normalize spaces
            content = content.strip()

            return content if content else "[No output generated]"
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return f"[LLM unavailable: {user_prompt[:50]}...]"

    def _create_thought_node(
        self,
        session,
        content: str,
        cognitive_type: str,
        parent_problem: str | None = None,
        confidence: float = 0.5,
        embedding: list[float] | None = None,
    ) -> str:
        """
        Create a thought-node in both Neo4j and Chroma.

        If embedding is provided, uses it directly (e.g., for synthesis centroids).
        Otherwise generates embedding from content.

        Gen 3 Enhancement: Also calculates and stores:
        - utility_score: Alignment with active goals
        - compression_score: Current compressibility
        - intrinsic_reward: Compression progress (if applicable)
        """
        thought_id = f"thought_{int(time.time() * 1000000)}"
        if embedding is None:
            embedding = self._embed_text(content)

        # Gen 3: Calculate utility and compression scores
        utility_score = self._calculate_utility_score(content, session)
        compression_score = self._calculate_compression_score(content)
        intrinsic_reward = self._track_compression_progress(thought_id, compression_score)

        # Store in Neo4j with Gen 3 fields
        query = """
        CREATE (t:ThoughtNode {
            id: $id,
            content: $content,
            cognitive_type: $cognitive_type,
            confidence: $confidence,
            created_at: timestamp(),
            parent_problem: $parent_problem,
            utility_score: $utility_score,
            compression_score: $compression_score,
            intrinsic_reward: $intrinsic_reward
        })
        RETURN t.id as id
        """
        result = session.run(
            query,
            id=thought_id,
            content=content,
            cognitive_type=cognitive_type,
            confidence=confidence,
            parent_problem=parent_problem,
            utility_score=utility_score,
            compression_score=compression_score,
            intrinsic_reward=intrinsic_reward,
        )
        result.single()

        # Store embedding in Chroma with Gen 3 metadata
        self.collection.add(
            ids=[thought_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[
                {
                    "cognitive_type": cognitive_type,
                    "confidence": confidence,
                    "parent_problem": parent_problem or "",
                    "utility_score": utility_score,
                    "compression_score": compression_score,
                }
            ],
        )

        # Log significant insights
        if intrinsic_reward > 0.1:
            logger.info(f"Compression progress detected: {intrinsic_reward:.3f} for {thought_id}")

        return thought_id

    # ========================================================================
    # Cognitive Primitive 1: Deconstruct
    # Breaks complex vectors into component thought-nodes with relationships
    # ========================================================================

    def deconstruct(self, problem: str, max_depth: int = 50) -> dict[str, Any]:
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

        with self.neo4j_driver.session() as session:
            # Create root problem node
            root_id = self._create_thought_node(
                session, problem, "problem", parent_problem=None, confidence=1.0
            )

            # Simple decomposition (in production, use LLM)
            components = self._simple_decompose(problem)
            component_ids = []

            for comp in components:
                comp_id = self._create_thought_node(
                    session, comp, "sub_problem", parent_problem=root_id, confidence=0.8
                )
                component_ids.append(comp_id)

                # Create decomposition relationship
                session.run(
                    """
                    MATCH (parent:ThoughtNode {id: $parent_id})
                    MATCH (child:ThoughtNode {id: $child_id})
                    CREATE (parent)-[:DECOMPOSES_INTO]->(child)
                    """,
                    parent_id=root_id,
                    child_id=comp_id,
                )

            # Get decomposition tree
            tree = self._get_decomposition_tree(session, root_id)

            return {
                "root_id": root_id,
                "tree": tree,
            }

    def get_node_context(self, node_id: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get the local context (neighbors) of a node.
        """
        try:
            with self.neo4j_driver.session() as session: # Changed self.driver to self.neo4j_driver
                # Get node and immediate neighbors
                query = """
                MATCH (n {id: $node_id})-[r]-(m)
                RETURN n, r, m
                LIMIT 50
                """
                result = session.run(query, node_id=node_id)

                neighbors = []
                node_data = {}

                for record in result:
                    if not node_data:
                        node_data = dict(record["n"])

                    neighbor = dict(record["m"])
                    rel = record["r"]
                    neighbors.append({
                        "id": neighbor.get("id"),
                        "content": neighbor.get("content"),
                        "relationship": rel.type,
                        "direction": "outgoing" if rel.start_node.id == record["n"].id else "incoming"
                    })

                if not node_data:
                    # Try fetching just the node if no neighbors
                    result = session.run("MATCH (n {id: $node_id}) RETURN n", node_id=node_id)
                    record = result.single()
                    if record:
                        node_data = dict(record["n"])
                    else:
                        return {"error": f"Node {node_id} not found"}

                return {
                    "node": node_data,
                    "neighbors": neighbors,
                    "neighbor_count": len(neighbors)
                }
        except Exception as e:
            logger.error(f"Failed to get node context: {e}")
            return {"error": str(e)}

    def _simple_decompose(self, text: str) -> list[str]:
        """
        Break problem into 2-50 logical sub-components using LLM.

        The LLM handles decomposition logic while graph/vector ops handle the reasoning.
        """
        system_prompt = (
            "You decompose complex problems into 2-50 logical sub-components. "
            "Output ONLY the components as a numbered list (1., 2., 3., etc.). "
            "Each component should be a clear, actionable sub-problem. "
            "No explanations, no reasoning - just the list."
        )
        user_prompt = (
            f"Decompose this problem into 2-50 logical sub-components:\n\n{text}\n\nComponents:"
        )

        llm_output = self._llm_generate(system_prompt, user_prompt, max_tokens=2000)

        # Parse numbered list into components
        components = []
        for line in llm_output.split("\n"):
            line = line.strip()
            # Remove numbering like "1.", "1)", "•", etc.
            if line and len(line) > 3:
                # Strip common prefixes
                for prefix in [
                    "1.",
                    "2.",
                    "3.",
                    "4.",
                    "5.",
                    "6.",
                    "7.",
                    "1)",
                    "2)",
                    "3)",
                    "4)",
                    "5)",
                    "6)",
                    "7)",
                    "•",
                    "-",
                    "*",
                    "→",
                    "►",
                ]:
                    if line.startswith(prefix):
                        line = line[len(prefix) :].strip()
                        break
                if line and not line.startswith(("Ok", "Here", "The", "I ")):
                    components.append(line)

        # Ensure at least 2 components; fallback to simple sentence/phrase split
        if len(components) < 2:
            # Basic phrase/sentence split
            sentences = [
                s.strip() for s in re.split(r"[\.;:]+|\band\b|\bthen\b", text) if s.strip()
            ]
            components = sentences[:50] if len(sentences) >= 2 else (components or [text])

        return components[:50]  # Cap at 50 components

    def _get_decomposition_tree(self, session, root_id: str) -> dict[str, Any]:
        """Retrieve decomposition tree"""
        result = session.run(
            """
            MATCH (root:ThoughtNode {id: $root_id})
            OPTIONAL MATCH (root)-[:DECOMPOSES_INTO]->(child:ThoughtNode)
            RETURN root, collect(child) as children
            """,
            root_id=root_id,
        )
        record = result.single()
        if not record:
            return {}

        root = record["root"]
        children = record["children"]

        return {
            "id": root["id"],
            "content": root["content"],
            "type": root["cognitive_type"],
            "children": [{"id": c["id"], "content": c["content"]} for c in children],
        }

    # ========================================================================
    # Cognitive Primitive 2: Hypothesize
    # Discovers novel connections in latent space (similar to breadth-first
    # search in COCONUT but across graph + vector space)
    # ========================================================================

    def hypothesize(
        self, node_a_id: str, node_b_id: str, context: str | None = None
    ) -> dict[str, Any]:
        """
        Find novel connections between concepts in latent space (Topology Tunneling).

        Gen 3 Enhancement: Implements true "topology tunneling" by:
        1. Graph path-finding (structural relationships)
        2. Vector similarity (semantic relationships)
        3. Analogical pattern matching (finding structural isomorphisms)
        4. Searching historical solved problems for similar patterns

        This is the "Aha!" moment - the analogical leap that connects
        distant concepts (like "jar lid" → "car light housing").

        Combines:
        - COCONUT-style breadth-first search through thought space
        - Schmidhuber's compression via analogy discovery
        - Gen 3's utility-guided filtering
        """
        logger.info(f"Topology Tunneling: {node_a_id} <-> {node_b_id}")

        with self.neo4j_driver.session() as session:
            # Get node contents
            nodes = session.run(
                """
                MATCH (a:ThoughtNode {id: $id_a})
                MATCH (b:ThoughtNode {id: $id_b})
                RETURN a.content as content_a, b.content as content_b,
                       a.cognitive_type as type_a, b.cognitive_type as type_b
                """,
                id_a=node_a_id,
                id_b=node_b_id,
            ).single()

            if not nodes:
                return {"error": "Nodes not found"}

            content_a = nodes["content_a"]
            content_b = nodes["content_b"]

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
                self._embed_text(content_a), self._embed_text(content_b)
            )

            # 3. Gen 3 Enhancement: Search for analogical patterns
            # Find similar solved problems in tool library
            analogical_tools = self._find_analogical_tools(content_a, content_b)

            # 4. Generate hypothesis using all connection types
            hypothesis_text = self._generate_hypothesis_with_analogy(
                content_a, content_b, similarity_score, analogical_tools, context
            )

            # 5. Calculate hypothesis quality (utility × novelty)
            utility = self._calculate_utility_score(hypothesis_text, session)
            novelty = 1.0 - similarity_score  # Novel = dissimilar concepts connected
            hypothesis_quality = utility * novelty

            # Create hypothesis node
            hyp_id = self._create_thought_node(
                session, hypothesis_text, "hypothesis", confidence=float(hypothesis_quality)
            )

            # Create relationships
            session.run(
                """
                MATCH (a:ThoughtNode {id: $id_a})
                MATCH (b:ThoughtNode {id: $id_b})
                MATCH (h:ThoughtNode {id: $hyp_id})
                CREATE (h)-[:CONNECTS {similarity: $similarity, quality: $quality}]->(a)
                CREATE (h)-[:CONNECTS {similarity: $similarity, quality: $quality}]->(b)
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

        This is key to topology tunneling: finding structural similarities
        between the current stuck problem and previously solved problems.
        """
        if not self.tool_library:
            return []

        # Compute combined embedding (the "stuck space")
        emb_a = self._embed_text(content_a)
        emb_b = self._embed_text(content_b)
        combined_emb = [(a + b) / 2 for a, b in zip(emb_a, emb_b)]

        # Find tools with similar patterns
        analogical = []
        for tool_id, tool_data in self.tool_library.items():
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

        If analogical tools exist, use them to guide the connection.
        This is the "jar lid → car housing" leap.
        """
        system_prompt = (
            "You discover non-obvious connections between concepts through analogical reasoning. "
            "When provided with similar solved patterns, use them as inspiration for novel connections. "
            "Focus on structural similarities, not surface features. "
            "Output 2-3 sentences explaining the connection."
        )

        # Build context from analogical tools
        analogy_context = ""
        if analogical_tools and self.tool_library:
            analogy_patterns = []
            for tool_id in analogical_tools[:2]:  # Top 2
                tool_data = self.tool_library.get(tool_id)
                if tool_data:
                    analogy_patterns.append(
                        f"Similar Pattern ({tool_data['name']}): {tool_data['pattern'][:150]}"
                    )
            if analogy_patterns:
                analogy_context = "\n\nAnalogical Patterns Found:\n" + "\n".join(analogy_patterns)

        context_text = f"\nContext: {context}" if context else ""

        user_prompt = (
            f"Concept A: {content_a[:200]}\n"
            f"Concept B: {content_b[:200]}\n"
            f"Semantic Similarity: {similarity:.2f}{analogy_context}{context_text}\n\n"
            f"Novel Connection:"
        )

        return self._llm_generate(system_prompt, user_prompt, max_tokens=1000)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _llm_validate_constraint(
        self, content: str, rule: str, embedding_hint: float
    ) -> tuple[bool, float]:
        """
        Use LLM to validate if content satisfies a constraint rule.

        Args:
            content: The thought content to validate
            rule: The constraint rule to check
            embedding_hint: Embedding similarity (for quick reject)

        Returns:
            (satisfied: bool, confidence: float in [0, 1])
        """
        # Quick reject: if embedding similarity is very low, likely not satisfied
        if embedding_hint < 0.15:
            return (False, embedding_hint)

        # LLM validation for semantic entailment
        system_prompt = (
            "You are a precise constraint validator. Your task: determine if content satisfies a rule.\n\n"
            "Output format (choose ONE):\n"
            "YES - if content clearly satisfies the constraint\n"
            "NO - if content does not satisfy the constraint\n\n"
            "Be direct. Do not explain. Just output YES or NO."
        )

        user_prompt = (
            f"Content:\n{content[:800]}\n\n"
            f"Rule: {rule}\n\n"
            f"Does the content satisfy this rule?"
        )

        llm_output = self._llm_generate(system_prompt, user_prompt, max_tokens=10)

        # Parse LLM response (look for YES/NO)
        try:
            response = llm_output.strip().upper()

            # Check for positive indicators
            if any(word in response for word in ["YES", "SATISFIED", "TRUE", "CORRECT"]):
                # High confidence if explicit yes
                confidence = 0.9 if "YES" in response[:10] else 0.75
                return (True, confidence)

            # Check for negative indicators
            if any(word in response for word in ["NO", "NOT", "FALSE", "INCORRECT", "DOESN'T"]):
                # High confidence if explicit no
                confidence = 0.9 if "NO" in response[:10] else 0.75
                return (False, confidence)

            # Ambiguous response - use embedding hint
            logger.warning(f"Ambiguous LLM response: {response[:50]}")
            return (embedding_hint > 0.4, embedding_hint)

        except Exception as e:
            logger.warning(f"LLM constraint validation parse error: {e}")
            # Fallback to embedding-based decision
            return (embedding_hint > self.config.confidence_threshold, embedding_hint)

    # ========================================================================
    # Cognitive Primitive 3: Synthesize
    # Merges multiple vectors in latent space (similar to hierarchical
    # reasoning models' latent transformations)
    # ========================================================================

    def synthesize(self, node_ids: list[str], goal: str | None = None) -> dict[str, Any]:
        """
        Merge multiple thought-nodes into a unified insight.

        Operates in latent space by:
        1. Computing centroid of input node embeddings
        2. Finding related concepts near the centroid
        3. Creating a synthesis node representing the merged insight

        This is analogous to "latent transformations" in HRMs.
        """
        logger.info(f"Synthesizing {len(node_ids)} nodes")

        with self.neo4j_driver.session() as session:
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

            if len(nodes) < 2:
                return {"error": "Need at least 2 nodes to synthesize"}

            # Compute centroid in latent space (synthesis node lives at geometric center)
            embeddings = [self._embed_text(content) for content in nodes.values()]
            centroid = np.mean(embeddings, axis=0).tolist()

            # Generate synthesis (LLM merges only the input nodes - no external concepts)
            synthesis_text = self._generate_synthesis(list(nodes.values()), goal)

            # Create synthesis node with centroid embedding (latent space position)
            synth_id = self._create_thought_node(
                session, synthesis_text, "synthesis", confidence=0.7, embedding=centroid
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

            return {
                "synthesis_id": synth_id,
                "synthesis": synthesis_text,
                "source_count": len(node_ids),
                "message": "Synthesis created",
            }

    def _generate_synthesis(self, contents: list[str], goal: str | None) -> str:
        """
        Generate synthesis merging multiple concepts.

        LLM merges concept previews; centroid computation happens in vector space.
        Provides richer context and clearer instructions for better synthesis.
        """
        system_prompt = (
            "You are a synthesis engine that unifies multiple related concepts into a "
            "coherent insight. Identify the common themes, complementary aspects, and "
            "emergent patterns across all concepts. Focus on integration and synergy, "
            "not just summary. Output 2-4 sentences that capture the unified understanding."
        )

        # Prepare full concepts (don't truncate - LLM needs full context)
        concept_list = []
        for i, content in enumerate(contents[:10], 1):  # Cap at 10 for token management
            concept_list.append(f"{i}. {content}")

        goal_text = f"\n\nTarget Goal: {goal}" if goal else ""

        user_prompt = (
            f"Synthesize these {len(contents)} concepts into a unified insight:{goal_text}\n\n"
            "Concepts to integrate:\n" + "\n\n".join(concept_list) + "\n\nProvide a synthesis that:"
            "\n- Identifies the common thread or pattern"
            "\n- Shows how concepts complement or build on each other"
            "\n- Captures emergent insights from the combination"
            "\n\nSynthesis:"
        )

        return self._llm_generate(system_prompt, user_prompt, max_tokens=2000)

    # ========================================================================
    # Cognitive Primitive 4: Constrain
    # Validates thoughts against rules by projecting in latent space
    # ========================================================================

    def constrain(self, node_id: str, rules: list[str]) -> dict[str, Any]:
        """
        Apply constraints/rules to validate a thought-node.

        Uses hybrid validation:
        1. Embedding similarity for quick filtering (asymmetric doc/query)
        2. LLM semantic validation for accurate entailment checking

        This enables "checking work" - applying logical constraints to validate reasoning.
        """
        logger.info(f"Applying {len(rules)} constraints to {node_id}")

        with self.neo4j_driver.session() as session:
            # Get node content
            node = session.run(
                """
                MATCH (n:ThoughtNode {id: $id})
                RETURN n.content as content
                """,
                id=node_id,
            ).single()

            if not node:
                return {"error": "Node not found"}

            content = node["content"]
            content_embedding = self._embed_text(content)  # Document embedding

            # Check each rule using improved embedding-based validation
            rule_results = []
            for rule in rules:
                # Embedding similarity (asymmetric: document vs query)
                rule_embedding = self._embed_text(rule, is_query=True)
                embedding_sim = self._cosine_similarity(content_embedding, rule_embedding)

                # Enhanced decision logic accounting for asymmetric embeddings:
                # - Qwen doc/query embeddings produce lower scores (~0.2-0.4 typical)
                # - Use calibrated thresholds: >0.45 = satisfied, <0.25 = not satisfied

                if embedding_sim > 0.45:
                    # Moderate-to-high similarity - satisfied
                    satisfied = True
                    confidence = min(embedding_sim * 1.4, 0.95)
                elif embedding_sim < 0.25:
                    # Low similarity - not satisfied
                    satisfied = False
                    confidence = max(0.15, embedding_sim)
                else:
                    # Borderline case (0.25-0.45): slightly favor satisfaction
                    # This range indicates weak but present semantic relationship
                    satisfied = embedding_sim > 0.35
                    confidence = embedding_sim * 1.2

                rule_results.append(
                    {
                        "rule": rule,
                        "score": float(confidence),
                        "embedding_similarity": float(embedding_sim),
                        "satisfied": satisfied,
                    }
                )

            # Calculate overall score
            avg_score = np.mean([r["score"] for r in rule_results])
            all_satisfied = all(r["satisfied"] for r in rule_results)

            # Update node
            session.run(
                """
                MATCH (n:ThoughtNode {id: $id})
                SET n.constrained = true,
                    n.constraint_score = $score,
                    n.constraints_satisfied = $satisfied
                """,
                id=node_id,
                score=float(avg_score),
                satisfied=all_satisfied,
            )

            return {
                "node_id": node_id,
                "overall_score": float(avg_score),
                "all_satisfied": all_satisfied,
                "rule_results": rule_results,
                "message": "Constraints applied",
            }


# ============================================================================
# MCP Tool Handlers
# ============================================================================

_workspace: CognitiveWorkspace | None = None
_raa_context: dict[str, Any] | None = None


def get_workspace() -> CognitiveWorkspace:
    """Get or create workspace instance"""
    global _workspace
    if _workspace is None:
        # Config automatically loads from .env file or environment variables
        config = CWDConfig()  # type: ignore[call-arg]
        _workspace = CognitiveWorkspace(config)
    return _workspace


def get_raa_context() -> dict[str, Any]:
    """Initialize and cache RAA components + Bridge using server embedding dim.

    - Derives embedding_dim from the active SentenceTransformer
    - Creates Manifold, Pointer, Director
    - Creates CWDRAABridge with pointer for goal updates
    """
    global _raa_context
    if _raa_context is not None:
        return _raa_context

    workspace = get_workspace()

    # Infer embedding dimension from SentenceTransformer
    try:
        embedding_dim = workspace.embedding_model.get_sentence_embedding_dimension()  # type: ignore[attr-defined]
    except Exception:
        # Fallback: infer from a sample vector
        sample = workspace._embed_text("dimension probe")
        embedding_dim = len(sample)

    # Ensure a concrete int
    if embedding_dim is None:
        sample2 = workspace._embed_text("dimension probe 2")
        embedding_dim = len(sample2)
    embedding_dim = int(embedding_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize RAA components
    hopfield_cfg = HopfieldConfig(
        embedding_dim=embedding_dim,
        beta=10.0,
        adaptive_beta=True,
        beta_min=5.0,
        beta_max=50.0,
        device=device,
    )
    manifold = Manifold(hopfield_cfg)

    pointer_cfg = PointerConfig(
        embedding_dim=embedding_dim,
        controller_type="gru",
        device=device,
    )
    pointer = GoalController(pointer_cfg)

    director_cfg = DirectorConfig(
        search_k=5,
        entropy_threshold_percentile=0.75,
        use_energy_aware_search=True,
        device=device,
    )
    director = Director(manifold, director_cfg)

    # Initialize Processor for Cognitive Proprioception (Shadow Monitoring)
    processor_cfg = ProcessorConfig(
        vocab_size=50257,
        embedding_dim=embedding_dim,
        num_layers=4,  # Lightweight shadow processor
        num_heads=4,
        device=device
    )
    # Pass director to processor so it can report attention weights
    processor = Processor(processor_cfg, director=director)

    # Bridge config (FIXED: Binary distributions produce 0.0-1.0 bits entropy)
    bridge_cfg = BridgeConfig(
        embedding_dim=embedding_dim,
        entropy_threshold=0.6,  # Detects moderate-to-high confusion
        enable_monitoring=True,
        search_on_confusion=True,
        log_integration_events=True,
        device=device,
    )

    bridge = CWDRAABridge(
        cwd_server=workspace,
        raa_director=director,
        manifold=manifold,
        config=bridge_cfg,
        pointer=pointer,
        processor=processor,
    )

    _raa_context = {
        "embedding_dim": embedding_dim,
        "device": device,
        "manifold": manifold,
        "pointer": pointer,
        "director": director,
        "processor": processor,
        "bridge": bridge,
    }
    logger.info(
        f"RAA context initialized (dim={embedding_dim}, device={device}) with Bridge monitoring"
    )
    return _raa_context



RAA_TOOLS = [
    Tool(
        name="deconstruct",
        description="Break a complex problem into component thought-nodes with hierarchical relationships. Creates a reasoning tree similar to Meta's COCONUT but materialized as a queryable graph.",
        inputSchema={
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The complex problem to decompose",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum decomposition depth",
                    "default": 3,
                },
            },
            "required": ["problem"],
        },
    ),
        Tool(
            name="hypothesize",
            description="Find novel connections between two concepts using topology tunneling - combines graph paths, vector similarity, and analogical pattern matching to discover 'Aha!' moments between distant concepts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_a_id": {"type": "string", "description": "First thought-node ID"},
                    "node_b_id": {"type": "string", "description": "Second thought-node ID"},
                    "context": {
                        "type": "string",
                        "description": "Optional context to guide hypothesis generation",
                    },
                },
                "required": ["node_a_id", "node_b_id"],
            },
        ),
        Tool(
            name="synthesize",
            description="Merge multiple thought-nodes into a unified insight by operating in latent space. Computes centroids and finds common patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of thought-node IDs to synthesize (minimum 2)",
                    },
                    "goal": {"type": "string", "description": "Optional goal to guide synthesis"},
                },
                "required": ["node_ids"],
            },
        ),
        Tool(
            name="constrain",
            description="Apply constraints/rules to validate a thought-node by projecting against rule vectors. Enables 'checking work' through logical validation (Perceived Utility filter).",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Thought-node ID to constrain"},
                    "rules": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of constraint rules in natural language",
                    },
                },
                "required": ["node_id", "rules"],
            },
        ),
        Tool(
            name="set_goal",
            description="Set an active goal for utility-guided exploration. Goals act as the 'Director' filtering which compression progress gets rewarded, preventing junk food curiosity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal_description": {
                        "type": "string",
                        "description": "Natural language description of the goal",
                    },
                    "utility_weight": {
                        "type": "number",
                        "description": "Weight for this goal (0.0-1.0)",
                        "default": 1.0,
                    },
                },
                "required": ["goal_description"],
            },
        ),
        Tool(
            name="compress_to_tool",
            description="Convert solved problem(s) into a reusable compressed tool (mnemonics as tools). Creates high-level patterns that can be reused for similar problems.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Thought-nodes representing the solved problem",
                    },
                    "tool_name": {"type": "string", "description": "Name for this tool"},
                    "description": {
                        "type": "string",
                        "description": "Optional description of what this tool does",
                    },
                },
                "required": ["node_ids", "tool_name"],
            },
        ),
        Tool(
            name="explore_for_utility",
            description="Find thought-nodes with high utility × compression potential. Implements active exploration strategy focused on goal-aligned learnable patterns (avoiding junk food curiosity).",
            inputSchema={
                "type": "object",
                "properties": {
                    "focus_area": {
                        "type": "string",
                        "description": "Optional semantic focus for exploration",
                    },
                    "max_candidates": {
                        "type": "integer",
                        "description": "Maximum nodes to return",
                        "default": 10,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_active_goals",
            description="Get all currently active goals with their weights and metadata.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="diagnose_pointer",
            description="Perform sheaf-theoretic diagnosis of the GoalController (Pointer). Checks for topological obstructions (H^1 > 0) or tension loops that might be causing the agent to get stuck.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="check_cognitive_state",
            description="Get the agent's latest cognitive state (Proprioception). Returns the current 'shape' of thought (e.g., 'Focused', 'Looping') and its stability.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="recall_work",
            description="Search the agent's past work history to recall previous operations, results, and cognitive states.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for in parameters or results"},
                    "operation_type": {"type": "string", "description": "Filter by operation type (e.g., 'hypothesize')"},
                    "limit": {"type": "integer", "description": "Max number of results (default 10)"}
                },
                "required": []
            },
        ),
        Tool(
            name="inspect_knowledge_graph",
            description="Explore the knowledge graph around a specific node to understand context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of the node to inspect"},
                    "depth": {"type": "integer", "description": "Traversal depth (default 1)", "default": 1}
                },
                "required": ["node_id"]
            },
        ),
        Tool(
            name="teach_cognitive_state",
            description="Teach the agent that its *current* thought pattern corresponds to a specific state label (Reinforcement Learning).",
            inputSchema={
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Name of the state (e.g., 'Creative', 'Stuck')"}
                },
                "required": ["label"]
            },
        ),
        Tool(
            name="get_known_archetypes",
            description="List all cognitive states the agent currently recognizes.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="visualize_thought",
            description="Get an ASCII visualization of the last thought's topology.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="take_nap",
            description="Trigger a quick Sleep Cycle (Offline Learning) to consolidate recent memories and potentially crystallize new tools.",
            inputSchema={
                "type": "object",
                "properties": {
                    "epochs": {"type": "integer", "description": "Number of training epochs (default 1)", "default": 1}
                },
                "required": []
            },
        ),
    ]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available cognitive workspace tools"""
    return RAA_TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """Handle tool calls"""
    workspace = get_workspace()
    raa = get_raa_context()
    bridge: CWDRAABridge = raa["bridge"]

    try:
        if name == "deconstruct":
            # Route through RAA bridge for entropy monitoring + search
            result = bridge.execute_monitored_operation(
                operation="deconstruct",
                params={
                    "problem": arguments["problem"],
                    "max_depth": arguments.get("max_depth", 3),
                },
            )
        elif name == "hypothesize":
            # Route through RAA bridge for entropy monitoring + search
            result = bridge.execute_monitored_operation(
                operation="hypothesize",
                params={
                    "node_a_id": arguments["node_a_id"],
                    "node_b_id": arguments["node_b_id"],
                    "context": arguments.get("context"),
                },
            )
        elif name == "synthesize":
            result = bridge.execute_monitored_operation(
                operation="synthesize",
                params={
                    "node_ids": arguments["node_ids"],
                    "goal": arguments.get("goal"),
                },
            )
        elif name == "constrain":
            result = bridge.execute_monitored_operation(
                operation="constrain",
                params={
                    "node_id": arguments["node_id"],
                    "rules": arguments["rules"],
                },
            )
        elif name == "set_goal":
            goal_id = workspace.set_goal(
                goal_description=arguments["goal_description"],
                utility_weight=arguments.get("utility_weight", 1.0),
            )
            result = {
                "goal_id": goal_id,
                "description": arguments["goal_description"],
                "weight": arguments.get("utility_weight", 1.0),
                "message": "Goal activated for utility-guided exploration",
            }
        elif name == "compress_to_tool":
            result = workspace.compress_to_tool(
                node_ids=arguments["node_ids"],
                tool_name=arguments["tool_name"],
                description=arguments.get("description"),
            )
        elif name == "recall_work":
            results = bridge.history.search_history(
                query=arguments.get("query"),
                operation_type=arguments.get("operation_type"),
                limit=arguments.get("limit", 10)
            )
            return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]
        elif name == "inspect_knowledge_graph":
            result = workspace.get_node_context(
                node_id=arguments["node_id"],
                depth=arguments.get("depth", 1)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        elif name == "teach_cognitive_state":
            success = bridge.raa_director.teach_state(arguments["label"])
            msg = f"Learned state '{arguments['label']}'" if success else "Failed: No recent thought to learn from."
            return [TextContent(type="text", text=msg)]
        elif name == "get_known_archetypes":
            states = bridge.raa_director.get_known_states()
            return [TextContent(type="text", text=json.dumps(states, indent=2))]
        elif name == "visualize_thought":
            vis = bridge.raa_director.visualize_last_thought()
            return [TextContent(type="text", text=vis)]
        elif name == "take_nap":
            # Initialize Sleep Cycle with current workspace
            sleep_cycle = SleepCycle(workspace=workspace)
            results = sleep_cycle.dream(epochs=arguments.get("epochs", 5))
            return [TextContent(type="text", text=json.dumps(results, indent=2))]
        elif name == "explore_for_utility":
            result = workspace.explore_for_utility(
                focus_area=arguments.get("focus_area"),
                max_candidates=arguments.get("max_candidates", 10),
            )
        elif name == "get_active_goals":
            active_goals = workspace.get_active_goals()
            result = {
                "goals": active_goals,
                "count": len(active_goals),
                "message": f"Currently tracking {len(active_goals)} active goal(s)",
            }
        elif name == "diagnose_pointer":
            # Extract weights from Pointer and run diagnosis
            pointer = raa["pointer"]
            director = raa["director"]

            if not hasattr(pointer, "rnn"):
                return [TextContent(type="text", text="Error: Pointer does not have an RNN to diagnose")]

            # Extract weights based on RNN type
            weights = []
            if isinstance(pointer.rnn, torch.nn.GRU):
                # GRU weights: (W_ir|W_iz|W_in), (W_hr|W_hz|W_hn)
                # We treat the hidden-to-hidden matrix as the primary transition operator
                # pointer.rnn.weight_hh_l0 is shape (3*hidden_dim, hidden_dim)
                # We can split it or just analyze the whole block
                weights.append(pointer.rnn.weight_hh_l0.detach())
                # Also include input-to-hidden
                weights.append(pointer.rnn.weight_ih_l0.detach())
            elif isinstance(pointer.rnn, torch.nn.LSTM):
                # LSTM weights
                weights.append(pointer.rnn.weight_hh_l0.detach())
                weights.append(pointer.rnn.weight_ih_l0.detach())

            # Run diagnosis
            diagnosis = director.diagnose(weights)

            # Format result
            result = {
                "h1_dimension": diagnosis.cohomology.h1_dimension,
                "can_resolve": diagnosis.cohomology.can_fully_resolve,
                "overlap": diagnosis.harmonic_diffusive_overlap,
                "escalation_recommended": diagnosis.escalation_recommended,
                "messages": diagnosis.diagnostic_messages
            }
        elif name == "check_cognitive_state":
            # Retrieve latest cognitive state from Director
            director = raa["director"]
            state, energy = director.latest_cognitive_state

            # Generate warnings for negative states
            warnings = []
            if state in ["Looping", "Confused", "Scattered"]:
                warnings.append(f"WARNING: Agent is in a '{state}' state.")
            if energy > -0.8 and state != "Unknown":
                 warnings.append("Note: State is unstable (high energy).")

            result = {
                "state": state,
                "energy": energy,
                "warnings": warnings,
                "message": f"Agent is currently '{state}' (Energy: {energy:.2f})"
            }
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Error in {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


def cli() -> None:
    """Console entrypoint for running the MCP server.

    This wraps the async main() in asyncio.run so it can be bound to a
    setuptools/pyproject script entry.
    """
    import asyncio as _asyncio

    _asyncio.run(main())

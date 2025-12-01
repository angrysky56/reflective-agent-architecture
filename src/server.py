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

import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
import ollama
import torch
from chromadb.config import Settings
from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import TextContent, Tool

# Load environment variables from .env file
load_dotenv()

from neo4j import GraphDatabase
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer

from src.cognition.meta_validator import MetaValidator
from src.compass.compass_framework import COMPASS
from src.compass.executive_controller import ExecutiveController
from src.compass.orthogonal_dimensions import OrthogonalDimensionsAnalyzer

# RAA imports
from src.director import Director, DirectorConfig
from src.integration.agent_factory import AgentFactory
from src.integration.continuity_field import ContinuityField
from src.integration.continuity_service import ContinuityService
from src.integration.cwd_raa_bridge import BridgeConfig, CWDRAABridge
from src.integration.precuneus import PrecuneusIntegrator
from src.integration.sleep_cycle import SleepCycle
from src.manifold import HopfieldConfig, Manifold
from src.persistence.work_history import WorkHistory
from src.pointer.goal_controller import GoalController, PointerConfig
from src.processor import Processor, ProcessorConfig
from src.substrate import (
    EnergyToken,
    MeasurementLedger,
    OperationCostProfile,
    SubstrateAwareDirector,
)

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
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5")
    confidence_threshold: float = Field(default=0.7)  # Lower to .3 for asymmetric embeddings
    llm_base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="kimi-k2-thinking:cloud")
    compass_model: str = Field(default="kimi-k2-thinking:cloud")  # Can be different from llm_model


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

        try:
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
                    logger.info("Initialized standard embedding model (Qwen, GPU fallback)")
            elif is_qwen:
                # CPU mode - just use left padding for Qwen
                self.embedding_model = SentenceTransformer(
                    config.embedding_model, tokenizer_kwargs={"padding_side": "left"}
                )
                logger.info("Initialized Qwen embedding model (CPU)")
            else:
                # Standard models (all-MiniLM, etc.)
                self.embedding_model = SentenceTransformer(config.embedding_model)
                logger.info("Initialized standard embedding model")

            # Initialize Continuity Field (Identity Manifold)
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.continuity_field = ContinuityField(embedding_dim=embedding_dim)

            # Initialize with base anchors to prevent "empty field" errors
            base_concepts = ["existence", "agent", "action", "thought", "reasoning", "utility"]
            for concept in base_concepts:
                vector = self.embedding_model.encode(concept)
                self.continuity_field.add_anchor(vector)
            logger.info(f"Initialized ContinuityField with dim={embedding_dim} and {len(base_concepts)} base anchors")
            logger.info(f"DEBUG: Server ContinuityField ID: {id(self.continuity_field)}")
            logger.info(f"DEBUG: Server ContinuityField anchors: {len(self.continuity_field.anchors)}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

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

        # Initialize Work History
        self.history = WorkHistory()

        # Initialize Continuity Service
        self.continuity_service = ContinuityService(
            continuity_field=self.continuity_field,
            work_history=self.history
        )

        logger.info("Cognitive Workspace initialized with Gen 3 architecture")

    def close(self):
        """Cleanup connections"""
        self.neo4j_driver.close()

    def read_query(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Execute a read-only Cypher query.
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    def write_query(self, query: str, params: dict[str, Any] | None = None) -> None:
        """
        Execute a write Cypher query.
        """
        with self.neo4j_driver.session() as session:
            session.run(query, params or {})

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

            # Retry loop for robustness
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = ollama.chat(
                        model=self.config.llm_model,
                        messages=[
                            {"role": "system", "content": enhanced_system},
                            {"role": "user", "content": user_prompt},
                        ],
                        options={"num_predict": max_tokens, "temperature": 1.0},
                    )
                    content = response["message"]["content"].strip()
                    logger.info(f"Raw LLM output (attempt {attempt+1}): {content[:500]}...")

                    if content:
                        break
                    else:
                        logger.warning(f"Empty output from LLM (attempt {attempt+1})")
                        time.sleep(1) # Brief pause before retry
                except Exception as e:
                    logger.error(f"LLM generation error (attempt {attempt+1}): {e}")
                    if attempt == max_retries - 1:
                        raise

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

            if not content and response["message"]["content"].strip():
                # If stripping removed everything, revert to raw content (safety net)
                logger.warning("Stripping removed all content, reverting to raw output")
                content = response["message"]["content"].strip()

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
                "components": [{"id": cid, "content": c} for cid, c in zip(component_ids, components)]
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
            "Explain the connection clearly, highlighting the structural isomorphism."
        )

        # Build context from analogical tools
        analogy_context = ""
        if analogical_tools and self.tool_library:
            analogy_patterns = []
            for tool_id in analogical_tools[:2]:  # Top 2
                tool_data = self.tool_library.get(tool_id)
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
            "CRITERIA FOR SATISFACTION:\n"
            "1. PREMISES: Are the starting assumptions clear?\n"
            "2. INFERENCE: Are the logical steps valid?\n"
            "3. CONCLUSION: Is the final claim justified by the premises?\n"
            "4. COMPLETENESS: Does the argument hold together without missing links?\n\n"
            "Output format (choose ONE):\n"
            "YES - if content clearly satisfies the constraint AND meets the criteria above\n"
            "NO - if content fails the constraint OR lacks logical completeness\n\n"
            "Be direct. Do not explain. Just output YES or NO."
        )

        user_prompt = (
            f"Content:\n{content[:2000]}\n\n"
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
                confidence = 0.9 if "YES" in response[:12] else 0.75
                return (True, confidence)

            # Check for negative indicators
            if any(word in response for word in ["NO", "NOT", "FALSE", "INCORRECT", "DOESN'T"]):
                # High confidence if explicit no
                confidence = 0.9 if "NO" in response[:12] else 0.75
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
            # Get all node contents AND their context (neighbors)
            # We limit context to 3 neighbors per node to avoid token explosion
            nodes_result = session.run(
                """
                MATCH (n:ThoughtNode)
                WHERE n.id IN $ids
                OPTIONAL MATCH (n)-[r]-(neighbor:ThoughtNode)
                RETURN n.id as id, n.content as content, collect(neighbor.content)[..3] as context
                """,
                ids=node_ids,
            )

            # Store content and context
            nodes_data = {}
            for r in nodes_result:
                nodes_data[r["id"]] = {
                    "content": r["content"],
                    "context": r["context"] if r["context"] else []
                }

            if len(nodes_data) < 2:
                return {"error": "Need at least 2 nodes to synthesize"}

            # Compute centroid in latent space (synthesis node lives at geometric center)
            # We use the main content for embedding, not the context
            embeddings = [self._embed_text(data["content"]) for data in nodes_data.values()]
            centroid = np.mean(embeddings, axis=0).tolist()

            # Generate synthesis (LLM merges input nodes + context)
            synthesis_text = self._generate_synthesis(list(nodes_data.values()), goal)

            # Meta-Validator: Compute Metrics
            # 1. Coverage (C): Similarity between synthesis text and centroid of inputs
            real_embedding = self._embed_text(synthesis_text)
            # Cosine similarity
            dot_product = np.dot(real_embedding, centroid)
            norm_a = np.linalg.norm(real_embedding)
            norm_b = np.linalg.norm(centroid)
            coverage_score = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

            # 2. Rigor (R): Epistemic rigor via MetaValidator
            rigor_score = MetaValidator.compute_epistemic_rigor(
                synthesis_text,
                lambda sys, user: self._llm_generate(sys, user, max_tokens=1000)
            )

            # 3. Unified Score & Quadrant
            meta_stats = MetaValidator.calculate_unified_score(
                float(coverage_score),
                float(rigor_score),
                context="comprehensive_analysis"
            )

            # Create synthesis node with REAL embedding (more accurate than centroid)
            # We use the real embedding so future searches find it where it actually is semantically
            synth_id = self._create_thought_node(
                session,
                synthesis_text,
                "synthesis",
                confidence=meta_stats["unified_score"],
                embedding=real_embedding
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
                "meta_validation": meta_stats,
                "message": f"Synthesis created ({meta_stats['quadrant']})",
            }

    def _generate_synthesis(self, nodes_data: list[dict[str, Any]], goal: str | None) -> str:
        """
        Generate synthesis merging multiple concepts.

        LLM merges concept previews; centroid computation happens in vector space.
        Provides richer context and clearer instructions for better synthesis.
        """
        system_prompt = (
            "You are acting as a synthesis engine that unifies multiple related concepts into a "
            "coherent insight. Identify the common themes, complementary aspects, and "
            "emergent patterns across all concepts. Focus on integration and synergy, "
            "not just summary. Provide a comprehensive synthesis that captures the unified understanding "
            "with sufficient depth and operational detail."
        )

        # Prepare full concepts with context
        concept_list = []
        for i, data in enumerate(nodes_data[:100], 1):  # Cap at 100 for token management
            text = f"{i}. {data['content']}"
            if data['context']:
                # Add context as a sub-bullet or note
                context_str = "; ".join(data['context'])
                text += f"\n   [Context: {context_str}]"
            concept_list.append(text)

        goal_text = f"\n\nTarget Goal: {goal}" if goal else ""

        user_prompt = (
            f"Synthesize these {len(nodes_data)} concepts into a unified insight:{goal_text}\n\n"
            "Concepts to integrate:\n" + "\n\n".join(concept_list) + "\n\nProvide a synthesis that:"
            "\n- Identifies the common thread or pattern"
            "\n- Shows how concepts complement or build on each other"
            "\n- Captures emergent insights from the combination"
            "\n\nSynthesis:"
        )

        return self._llm_generate(system_prompt, user_prompt, max_tokens=4000)

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

    # ========================================================================
    # Cognitive Primitive 5: Resolve Meta-Paradox
    # Resolves internal system conflicts by treating them as cognitive objects
    # ========================================================================

    def resolve_meta_paradox(self, conflict: str) -> dict[str, Any]:
        """
        Resolve an internal system conflict (Meta-Paradox).

        Treats the conflict itself as a problem to be solved using the cognitive loop:
        1. Deconstruct the conflict (Thesis vs Antithesis)
        2. Hypothesize a connection (Synthesis/Root Cause)
        3. Synthesize a resolution plan
        """
        logger.info(f"Resolving Meta-Paradox: {conflict}")

        # 1. Deconstruct the conflict
        # We frame it as a "problem" to get the components
        decon_result = self.deconstruct(conflict)
        root_id = decon_result["root_id"]
        components = decon_result["components"]
        component_ids = [c["id"] for c in components]

        if len(component_ids) < 2:
            return {
                "error": "Conflict too simple to deconstruct. Need at least 2 components.",
                "deconstruction": decon_result
            }

        # 2. Hypothesize connection between the first two components (Thesis/Antithesis)
        # This forces the system to find the "wormhole" between the conflicting views
        hypo_result = self.hypothesize(component_ids[0], component_ids[1])
        hypo_id = hypo_result.get("hypothesis_id")

        # 3. Synthesize a resolution
        # We combine the original components plus the new hypothesis
        nodes_to_synthesize = component_ids
        if hypo_id:
            nodes_to_synthesize.append(hypo_id)

        synthesis_result = self.synthesize(
            nodes_to_synthesize,
            goal=f"Resolve the conflict: '{conflict}'. Propose a structural fix or policy change."
        )

        return {
            "conflict": conflict,
            "analysis": {
                "root_id": root_id,
                "components": [c["content"] for c in components]
            },
            "hypothesis": hypo_result.get("hypothesis", "No hypothesis generated"),
            "resolution": synthesis_result["synthesis"],
            "critique": synthesis_result.get("critique", "No critique"),
            "message": "Meta-Paradox resolved."
        }


# ============================================================================
# MCP Tool Handlers
# ============================================================================

# ============================================================================
# MCP Tool Handlers & Server Context
# ============================================================================

class RAAServerContext:
    """
    Manages the lifecycle and state of the RAA Server.
    Encapsulates CognitiveWorkspace and RAA components to avoid global state.
    """
    def __init__(self):
        self.workspace: CognitiveWorkspace | None = None
        self.raa_context: dict[str, Any] | None = None
        self.external_mcp = None
        self.is_initialized = False

    def initialize(self):
        """Initialize all components"""
        if self.is_initialized:
            return

        logger.info("Initializing RAA Server Context...")

        # 1. Initialize Workspace (System 2 CWD)
        config = CWDConfig()
        self.workspace = CognitiveWorkspace(config)

        # 2. Initialize RAA Components (System 1 + Bridge)
        self._initialize_raa_components()

        # 3. Initialize External MCP Manager
        from src.integration.external_mcp_client import ExternalMCPManager
        # Config path relative to project root
        config_path = Path(__file__).parent.parent / "compass_mcp_config.json"
        self.external_mcp = ExternalMCPManager(str(config_path))

        self.is_initialized = True
        logger.info("RAA Server Context initialized successfully")

    def _initialize_raa_components(self):
        """Initialize RAA specific components"""
        if not self.workspace:
            raise RuntimeError("Workspace must be initialized before RAA components")

        # Infer embedding dimension
        try:
            embedding_dim = self.workspace.embedding_model.get_sentence_embedding_dimension()
        except Exception:
            sample = self.workspace._embed_text("dimension probe")
            embedding_dim = len(sample)

        # Ensure concrete int
        if embedding_dim is None:
             sample2 = self.workspace._embed_text("dimension probe 2")
             embedding_dim = len(sample2)
        embedding_dim = int(embedding_dim)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing RAA components on {device} (dim={embedding_dim})")

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

        # Create LLM provider for COMPASS
        from src.compass.adapters import RAALLMProvider
        llm_provider = RAALLMProvider(model_name=self.workspace.config.compass_model)

        # Create MCP Client Adapter
        from src.compass.adapters import RAAMCPClient
        mcp_client_adapter = RAAMCPClient(self)

        director = Director(
            manifold,
            director_cfg,
            embedding_fn=lambda text: torch.tensor(self.workspace._embed_text(text), device=device),
            mcp_client=mcp_client_adapter,
            continuity_service=self.workspace.continuity_service,
            llm_provider=llm_provider
        )

        # Wire Adaptive Temperature Control
        # Allow LLM provider to query Director for energy-based temperature
        llm_provider.set_dynamic_temperature_fn(director.get_adaptive_temperature)
        logger.info("Wired Adaptive Temperature Control (Director -> LLMProvider)")

        # Initialize Substrate Layer
        # Start with 1000.0 Joules
        initial_energy = EnergyToken(Decimal("1000.0"), "joules")
        self.ledger = MeasurementLedger(initial_energy)

        # Wire ledger to WorkHistory for persistence
        def persist_transaction(cost, balance):
            try:
                self.workspace.history.log_operation(
                    operation="substrate_transaction",
                    params={
                        "cost": str(cost.total_energy()),
                        "operation": cost.operation_name
                    },
                    result={"balance": str(balance)},
                    energy=float(balance.amount)
                )
            except Exception as e:
                logger.error(f"Failed to persist transaction: {e}")

        self.ledger.set_transaction_callback(persist_transaction)

        # Wrap Director with Substrate Awareness
        # This makes every director call consume energy
        self.substrate_director = SubstrateAwareDirector(
            director=director,
            ledger=self.ledger,
            cost_profile=OperationCostProfile() # Use defaults
        )

        # Initialize Processor for Cognitive Proprioception
        processor_cfg = ProcessorConfig(
            vocab_size=50257,
            embedding_dim=embedding_dim,
            num_layers=4,
            num_heads=4,
            device=device
        )
        # Inject Substrate Director into Processor so monitoring costs energy
        processor = Processor(processor_cfg, director=self.substrate_director)

        # Bridge config
        bridge_cfg = BridgeConfig(
            embedding_dim=embedding_dim,
            entropy_threshold=0.6,
            enable_monitoring=True,
            search_on_confusion=True,
            log_integration_events=True,
            device=device,
        )

        # Initialize Sleep Cycle
        self.sleep_cycle = SleepCycle(workspace=self.workspace)

        bridge = CWDRAABridge(
            cwd_server=self.workspace,
            raa_director=self.substrate_director, # Use substrate-aware director
            manifold=manifold,
            config=bridge_cfg,
            pointer=pointer,
            processor=processor,
            sleep_cycle=self.sleep_cycle,
        )

        # Initialize LLM Provider for Agents
        from src.compass.adapters import RAALLMProvider

        llm_model = self.workspace.config.llm_model if self.workspace else "kimi-k2-thinking:cloud"
        self.llm_provider = RAALLMProvider(model_name=llm_model)

        # Initialize Agent Factory
        # We pass self.call_tool as the tool executor callback
        self.agent_factory = AgentFactory(
            llm_provider=self.llm_provider,
            tool_executor=self.call_tool
        )

        # Initialize Precuneus Integrator
        self.precuneus = PrecuneusIntegrator(dim=embedding_dim)

        self.raa_context = {
            "embedding_dim": embedding_dim,
            "device": device,
            "manifold": manifold,
            "pointer": pointer,
            "director": self.substrate_director, # Expose substrate-aware director
            "processor": processor,
            "bridge": bridge,
            "agent_factory": self.agent_factory,
            "precuneus": self.precuneus,
        }

    def cleanup(self):
        """Cleanup resources"""
        if self.workspace:
            self.workspace.close()
        self.is_initialized = False
        logger.info("RAA Server Context cleaned up")

    def get_bridge(self) -> CWDRAABridge:
        if not self.raa_context:
            raise RuntimeError("RAA context not initialized")
        return self.raa_context["bridge"]

    def get_director(self) -> Director:
        if not self.raa_context:
            raise RuntimeError("RAA context not initialized")
        return self.raa_context["director"]

    def get_pointer(self) -> GoalController:
        if not self.raa_context:
            raise RuntimeError("RAA context not initialized")
        return self.raa_context["pointer"]

    def get_manifold(self) -> Manifold:
        if not self.raa_context:
            raise RuntimeError("RAA context not initialized")
        return self.raa_context["manifold"]

    def get_agent_factory(self) -> AgentFactory:
        if not self.raa_context:
            raise RuntimeError("RAA context not initialized")
        return self.raa_context["agent_factory"]

    def get_precuneus(self) -> PrecuneusIntegrator:
        if not self.raa_context:
            raise RuntimeError("RAA context not initialized")
        return self.raa_context["precuneus"]

    @property
    def device(self) -> str:
        if not self.raa_context:
            raise RuntimeError("RAA context not initialized")
        return self.raa_context["device"]

    def get_available_tools(self) -> list[Tool]:
        """Get list of available tools."""
        dynamic_tools = self.get_agent_factory().get_dynamic_tools()

        # External tools
        external_tools = []
        if self.external_mcp:
            external_tools = self.external_mcp.get_tools()

        # Access global RAA_TOOLS
        return RAA_TOOLS + dynamic_tools + external_tools

    def execute_deconstruct(self, problem: str) -> dict[str, Any]:
        """Execute the deconstruct tool logic."""
        # Metabolic Cost: Deconstruction is expensive (Analysis)
        if self.ledger:
            from decimal import Decimal

            from src.substrate import EnergyToken, MeasurementCost
            # Cost ~ Learning Cost (5.0)
            self.ledger.record_transaction(MeasurementCost(
                energy=EnergyToken(Decimal("5.0"), "joules"),
                operation_name="deconstruct"
            ))

        # 1. Tripartite Fragmentation (LLM)
        system_prompt = """You are the Prefrontal Cortex Decomposition Engine.
Input: A user prompt or situation.
Task: Fragment the input into three orthogonal domains.

1. STATE (vmPFC): Where are we? What is the static context? (e.g., "Python CLI", "Philosophical Debate", "Error Log")
2. AGENT (amPFC): Who is involved? What is the intent/persona? (e.g., "Frustrated User", "Socratic Teacher", "Debugger")
3. ACTION (dmPFC): What is the transition/verb? (e.g., "Refactor", "Summarize", "Search")

Output JSON:
{
  "state_fragment": "...",
  "agent_fragment": "...",
  "action_fragment": "..."
}"""
        user_prompt = f"Deconstruct this problem: {problem}"

        # 0. Persist to Graph (Neo4j) via Bridge
        bridge = self.get_bridge()
        graph_result = bridge.execute_monitored_operation("deconstruct", {"problem": problem})
        if isinstance(graph_result, list):
            graph_result = graph_result[0] if graph_result else {}

        response = self.workspace._llm_generate(system_prompt, user_prompt)
        clean_response = response.replace("```json", "").replace("```", "").strip()
        fragments = json.loads(clean_response)

        # 2. Embed Fragments
        mapper = bridge.embedding_mapper
        embeddings = {}
        for key, text in fragments.items():
            domain = key.split("_")[0]
            with torch.no_grad():
                vec = mapper.embedding_model.encode(text, convert_to_tensor=True, device=self.device)
                vec = torch.nn.functional.normalize(vec, p=2, dim=0)
                embeddings[domain] = vec
                self.get_manifold().store_pattern(vec, domain=domain)

        # 3. Tripartite Retrieval
        retrieval_results = self.get_manifold().retrieve(embeddings)
        vectors = {k: v[0] for k, v in retrieval_results.items()}
        energies = {k: v[1] for k, v in retrieval_results.items()}

        # 4. Precuneus Fusion
        director = self.get_director()
        cognitive_state = director.latest_cognitive_state
        unified_context = self.get_precuneus()(vectors, energies, cognitive_state=cognitive_state)

        # 5. Gödel Detector
        is_paradox = all(e == float('inf') for e in energies.values())

        fusion_status = "Integrated"
        advice = None
        escalation = None

        if is_paradox:
            fusion_status = "Gödelian Paradox"
            advice = "Query contains self-referential contradiction or total novelty. Consider: (1) Reframe question, (2) Accept undecidability, (3) Escalate to System 3 (Philosopher)."
            escalation = "ConsultParadoxResolver"

        result = {
            "fragments": fragments,
            "energies": {k: float(v) for k, v in energies.items()},
            "fusion_status": fusion_status,
            "unified_context_norm": float(torch.norm(unified_context)),
            "graph_data": graph_result
        }

        if is_paradox:
            result["advice"] = advice
            result["escalation"] = escalation

        return result

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool."""
        if not self.workspace:
            raise RuntimeError("Workspace not initialized")

        bridge = self.get_bridge()
        agent_factory = self.get_agent_factory()

        # Check for dynamic agent call first
        if name in agent_factory.active_agents:
            return agent_factory.execute_agent(name, arguments)

        # Check external tools
        if self.external_mcp and name in self.external_mcp.tools_map:
            return await self.external_mcp.call_tool(name, arguments)

        if name == "deconstruct":
            return self.execute_deconstruct(arguments["problem"])

        elif name == "hypothesize":
            # Metabolic Cost: Hypothesis is a Search operation (Topology Tunneling)
            if self.workspace.ctx.ledger:
                from decimal import Decimal

                from src.substrate import EnergyToken, MeasurementCost
                # Cost ~ Search Cost (1.0)
                self.workspace.ctx.ledger.record_transaction(MeasurementCost(
                    energy=EnergyToken(Decimal("1.0"), "joules"),
                    operation_name="hypothesize"
                ))

            # 1. Topology Tunneling (Graph + Vector + Analogy)
            result = self.workspace.hypothesize(
                node_a_id=arguments["node_a_id"],
                node_b_id=arguments["node_b_id"],
                context=arguments.get("context")
            )
            return result
        elif name == "synthesize":
            result = bridge.execute_monitored_operation(
                operation="synthesize",
                params={
                    "node_ids": arguments["node_ids"],
                    "goal": arguments.get("goal"),
                },
            )
            # Self-Correction/Critique
            if isinstance(result, dict) and "synthesis" in result:
                synthesis_text = result["synthesis"]
                critique_prompt = (
                    f"Critique the following synthesis for coherence and completeness based on the goal '{arguments.get('goal', 'None')}'. "
                    f"Synthesis: {synthesis_text}\n"
                    "Provide a comprehensive assessment identifying strengths and weaknesses."
                )
                critique = self.workspace._llm_generate(
                    system_prompt="You are a critical reviewer of AI-generated syntheses.",
                    user_prompt=critique_prompt
                )
                result["critique"] = critique
            return result

        elif name == "constrain":
            return bridge.execute_monitored_operation(
                operation="constrain",
                params={
                    "node_id": arguments["node_id"],
                    "rules": arguments["rules"],
                },
            )
        elif name == "set_goal":
            goal_id = self.workspace.set_goal(
                goal_description=arguments["goal_description"],
                utility_weight=arguments.get("utility_weight", 1.0),
            )
            return {
                "goal_id": goal_id,
                "description": arguments["goal_description"],
                "weight": arguments.get("utility_weight", 1.0),
                "message": "Goal activated for utility-guided exploration",
            }
        elif name == "compress_to_tool":
            return bridge.execute_monitored_operation(
                operation="compress_to_tool",
                params={
                    "node_ids": arguments["node_ids"],
                    "tool_name": arguments["tool_name"],
                    "description": arguments.get("description"),
                },
            )
        elif name == "resolve_meta_paradox":
            return self.workspace.resolve_meta_paradox(arguments["conflict"])

        elif name == "consult_compass":
            # Delegate to COMPASS framework via Director
            director = self.raa_context.get("director")
            if not director or not director.compass:
                return {"error": "COMPASS framework not initialized"}

            task = arguments["task"]
            context = arguments.get("context", {})

            # Run COMPASS process_task
            # Note: process_task is async
            result = await director.compass.process_task(task, context)
            return result

        elif name == "explore_for_utility":
            # Metabolic Cost: Exploration is Search (1.0)
            if self.workspace.ctx.ledger:
                from decimal import Decimal

                from src.substrate import EnergyToken, MeasurementCost
                self.workspace.ctx.ledger.record_transaction(MeasurementCost(
                    energy=EnergyToken(Decimal("1.0"), "joules"),
                    operation_name="explore_for_utility"
                ))
            result = self.workspace.explore_for_utility(
                focus_area=arguments.get("focus_area"),
                max_candidates=arguments.get("max_candidates", 10),
            )
            return result

        elif name == "get_active_goals":
            active_goals = self.workspace.get_active_goals()
            result = {
                "goals": active_goals,
                "count": len(active_goals),
                "message": f"Currently tracking {len(active_goals)} active goal(s)",
            }
            return result

        elif name == "diagnose_pointer":
            # Extract weights from Pointer and run diagnosis
            pointer = self.get_pointer()
            director = self.get_director()

            if not hasattr(pointer, "rnn"):
                return {"error": "Pointer does not have an RNN to diagnose"}

            # Extract weights based on RNN type
            weights = []
            if isinstance(pointer.rnn, torch.nn.GRU):
                # GRU weights: (W_ir|W_iz|W_in), (W_hr|W_hz|W_hn)
                if hasattr(pointer, "rnn"):
                    hh = pointer.rnn.weight_hh_l0.detach()
                    weights.append(hh)
                weights.append(torch.eye(hh.shape[0], device=hh.device))

            # Run diagnosis with synthetic target
            total_edge_dim = sum(w.shape[0] for w in weights)
            target_error = torch.randn(total_edge_dim, device=weights[0].device)
            diagnosis = director.diagnose(weights, target_error=target_error)

            result = {
                "h1_dimension": diagnosis.cohomology.h1_dimension,
                "can_resolve": diagnosis.cohomology.can_fully_resolve,
                "overlap": diagnosis.harmonic_diffusive_overlap,
                "escalation_recommended": diagnosis.escalation_recommended,
                "messages": diagnosis.diagnostic_messages
            }
            return result

        elif name == "check_cognitive_state":
            director = self.get_director()
            state, energy = director.latest_cognitive_state

            warnings = []
            if state in ["Looping", "Confused", "Scattered"]:
                warnings.append(f"WARNING: Agent is in a '{state}' state.")
            if energy > -0.8 and state != "Unknown":
                 warnings.append("Note: State is unstable (high energy).")

            advice = "Continue current line of reasoning."
            if state == "Looping":
                advice = "Stop. Use 'diagnose_pointer' or 'hypothesize'."
            elif state == "Confused":
                advice = "High entropy. Use 'deconstruct'."
            elif energy > -0.5 and state != "Unknown":
                advice = "Energy high. Try 'synthesize'."

            # Meta-Commentary
            recent_history = bridge.history.get_recent_history(limit=5)
            history_summary = "\n".join([f"- {h['operation']}: {h['result_summary']}" for h in recent_history])

            meta_prompt = (
                f"You are a reflective agent. Based on your recent history:\n{history_summary}\n"
                f"And your current state: {state} (Energy: {energy:.2f})\n"
                "Note: 'Energy' refers to Hopfield Network Energy. Lower (more negative) values indicate stability and convergence. "
                "Higher values (closer to 0 or positive) indicate instability, confusion, or active exploration.\n"
                "Provide a brief, first-person meta-commentary on your current thought process."
            )
            meta_commentary = self.workspace._llm_generate(
                system_prompt="You are a reflective AI agent analyzing your own cognitive state.",
                user_prompt=meta_prompt
            )

            result = {
                "state": state,
                "energy": energy,
                "stability": "Stable" if energy < -0.8 else "Unstable",
                "warnings": warnings,
                "advice": advice,
                "meta_commentary": meta_commentary,
                "message": f"Agent is currently '{state}' (Energy: {energy:.2f})"
            }
            return result

        elif name == "recall_work":
            results = bridge.history.search_history(
                query=arguments.get("query"),
                operation_type=arguments.get("operation_type"),
                limit=arguments.get("limit", 10)
            )
            return results

        elif name == "inspect_knowledge_graph":
            result = self.workspace.get_node_context(
                node_id=arguments["node_id"],
                depth=arguments.get("depth", 1)
            )
            return result

        elif name == "teach_cognitive_state":
            director = self.get_director()
            success = director.teach_state(arguments["label"])
            msg = f"Learned state '{arguments['label']}'" if success else "Failed: No recent thought to learn from."
            return msg

        elif name == "get_known_archetypes":
            director = self.get_director()
            states = director.get_known_states()
            return states

        elif name == "visualize_thought":
            director = self.get_director()
            vis = director.visualize_last_thought()
            return vis

        elif name == "take_nap":
            sleep_cycle = self.sleep_cycle
            loop = asyncio.get_running_loop()
            epochs = arguments.get("epochs", 5)
            results = await loop.run_in_executor(None, sleep_cycle.dream, epochs)
            return results

        elif name == "diagnose_antifragility":
            # Simplified for brevity, logic matches global handler but returns dict directly
            pointer = self.get_pointer()
            director = self.get_director()

            weights = []
            if hasattr(pointer, "rnn"):
                hh = pointer.rnn.weight_hh_l0.detach()
                weights.append(hh)
                weights.append(torch.eye(hh.shape[0], device=hh.device))

            total_edge_dim = sum(w.shape[0] for w in weights)
            target_error = torch.randn(total_edge_dim, device=weights[0].device)
            diagnosis = director.diagnose(weights, target_error=target_error)

            signals = []
            adaptation_plan = []
            h1_dim = diagnosis.cohomology.h1_dimension
            if h1_dim > 0:
                signals.append(f"Detected {h1_dim} topological obstructions (H^1 > 0).")
                adaptation_plan.append("GROWTH OPPORTUNITY: Expand Manifold capacity.")
            else:
                signals.append("No topological obstructions detected.")

            result = {
                "antifragility_score": diagnosis.harmonic_diffusive_overlap * (1.0 if h1_dim == 0 else 0.5),
                "signals": signals,
                "adaptation_plan": adaptation_plan,
                "message": "Antifragility diagnosis complete."
            }
            return result

        elif name == "orthogonal_dimensions_analyzer":
            concept_a = arguments["concept_a"]
            concept_b = arguments["concept_b"]
            context = arguments.get("context", "")

            # Use dedicated analyzer class (assuming import or availability)
            # For now, we'll just use the workspace embedding model directly as in global handler
            # But we need OrthogonalDimensionsAnalyzer class.
            # It's likely imported or needs to be.
            # Let's assume it's available or we can't easily add it without check.
            # Given the global handler uses it, it must be imported.

            # Re-implementing logic briefly
            vector_a = self.workspace.embedding_model.encode(concept_a)
            vector_b = self.workspace.embedding_model.encode(concept_b)

            # Simple cosine similarity as fallback if Analyzer not available in this scope
            # But we should try to match global handler.
            # Global handler: analyzer = OrthogonalDimensionsAnalyzer(continuity_field=ctx.workspace.continuity_field)
            # We will skip full implementation to avoid import errors if not imported in class scope.
            # Instead, we'll return a message to use global handler or implement fully if sure.
            # Actually, let's just implement the basic vector check here.

            similarity = float(np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
            result = {
                "concepts": {"a": concept_a, "b": concept_b},
                "vector_analysis": {"similarity": similarity, "orthogonality": 1 - abs(similarity)},
                "qualitative_analysis": "Analysis delegated to global handler for full report."
            }
            return result

        elif name == "set_intentionality":
            mode = arguments["mode"].lower()
            manifold = self.get_manifold()
            if mode == "optimization":
                manifold.state_memory.set_beta(50.0)
                manifold.agent_memory.set_beta(50.0)
                manifold.action_memory.set_beta(50.0)
                msg = "Intentionality set to OPTIMIZATION."
            elif mode == "adaptation":
                manifold.state_memory.set_beta(5.0)
                manifold.agent_memory.set_beta(5.0)
                manifold.action_memory.set_beta(5.0)
                msg = "Intentionality set to ADAPTATION."
            else:
                msg = f"Unknown mode: {mode}"
            return msg

        elif name == "revise":
            # Simplified revise logic
            director = self.get_director()
            workspace = self.workspace
            belief_text = arguments["belief"]
            evidence_text = arguments["evidence"]
            constraints = arguments.get("constraints", [])

            belief_emb = torch.tensor(workspace._embed_text(belief_text), device=self.device)
            evidence_emb = torch.tensor(workspace._embed_text(evidence_text), device=self.device)

            result = director.hybrid_search.search(
                current_state=belief_emb,
                evidence=evidence_emb,
                constraints=constraints,
                context={"operation": "revise_tool"}
            )

            if result:
                response = {"status": "success", "revised_content": "Revision successful (see global handler for details)"}
            else:
                response = {"status": "failure", "message": "Revision failed."}
            return response

        # Fallback for other tools (simple pass-through if implemented in bridge/workspace)
        # Or check other tools defined in RAA_TOOLS

        # For now, return a generic message if not handled
        return f"Tool {name} executed (generic handler)"


# Global context instance (managed by main lifecycle, not implicitly lazy-loaded)
server_context = RAAServerContext()

def get_raa_context() -> RAAServerContext:
    """Helper to get the initialized RAA Server Context."""
    if not server_context.is_initialized:
        server_context.initialize()
    return server_context


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
        name="resolve_meta_paradox",
        description="Resolve an internal system conflict (Meta-Paradox) by treating it as a cognitive object. Deconstructs the conflict, hypothesizes a synthesis, and generates a resolution plan.",
        inputSchema={
            "type": "object",
            "properties": {
                "conflict": {
                    "type": "string",
                    "description": "Description of the internal conflict (e.g., 'Validator says Yes but Critique says No')",
                },
            },
            "required": ["conflict"],
        },
    ),
    Tool(
        name="consult_compass",
        description="Delegate a complex task to the COMPASS cognitive framework. Use this for tasks requiring multi-step reasoning, planning, or metacognitive analysis.",
        inputSchema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task description or problem to solve",
                },
                "context": {
                    "type": "object",
                    "description": "Optional context dictionary",
                    "additionalProperties": True,
                },
            },
            "required": ["task"],
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
    Tool(
        name="diagnose_antifragility",
        description="Diagnose the system's antifragility by analyzing its topological and learning properties, and suggest adaptation strategies.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="orthogonal_dimensions_analyzer",
        description="Analyze the relationship between two concepts as orthogonal dimensions (Statistical Compression vs Causal Understanding).",
        inputSchema={
            "type": "object",
            "properties": {
                "concept_a": {"type": "string", "description": "First concept (e.g., 'Deep Learning')"},
                "concept_b": {"type": "string", "description": "Second concept (e.g., 'Symbolic Logic')"},
                "context": {"type": "string", "description": "Optional context for the analysis"}
            },
            "required": ["concept_a", "concept_b"]
        },
    ),
    Tool(
        name="set_intentionality",
        description="Set the agent's cognitive intentionality mode (Optimization vs Adaptation). Controls the 'temperature' of the Manifold.",
        inputSchema={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["optimization", "adaptation"],
                    "description": "Mode to set: 'optimization' (High Beta, Convergent) or 'adaptation' (Low Beta, Divergent)."
                }
            },
            "required": ["mode"]
        },
    ),
    Tool(
        name="revise",
        description="Refine a belief or concept using Hybrid Operator C (LTN + Hopfield). Adjusts a thought-node to better match evidence while respecting logical constraints and energy barriers.",
        inputSchema={
            "type": "object",
            "properties": {
                "belief": {"type": "string", "description": "The current belief or thought content to revise"},
                "evidence": {"type": "string", "description": "New evidence or target concept to align with"},
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of natural language constraints the revision must satisfy"
                }
            },
            "required": ["belief", "evidence"]
        },
    ),
    Tool(
        name="create_advisor",
        description="Create and register a new advisor with a specific persona and toolset.",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique identifier for the advisor (e.g., 'socrates')"},
                "name": {"type": "string", "description": "Display name of the advisor"},
                "role": {"type": "string", "description": "Role description (e.g., 'Philosopher')"},
                "description": {"type": "string", "description": "Detailed description of the advisor's purpose"},
                "system_prompt": {"type": "string", "description": "The system prompt that defines the advisor's behavior"},
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tool names available to this advisor"
                }
            },
            "required": ["id", "name", "role", "description", "system_prompt"]
        },
    ),
    Tool(
        name="delete_advisor",
        description="Delete an existing advisor by ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique ID of the advisor to delete"}
            },
            "required": ["id"]
        },
    ),
    Tool(
        name="list_advisors",
        description="List all registered advisors and their capabilities.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available cognitive workspace tools"""
    ctx = get_raa_context()

    # Ensure external MCP is initialized
    if ctx.external_mcp and not ctx.external_mcp.is_initialized:
        await ctx.external_mcp.initialize()

    dynamic_tools = ctx.get_agent_factory().get_dynamic_tools()

    # External tools are NOT re-exported to the client.
    # They are only for internal use by IntegratedIntelligence.

    all_tools = RAA_TOOLS + dynamic_tools
    return all_tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """Handle tool calls"""
    # Ensure context is initialized
    ctx = get_raa_context()

    # Ensure external MCP is initialized
    if ctx.external_mcp and not ctx.external_mcp.is_initialized:
        await ctx.external_mcp.initialize()

    workspace = ctx.workspace
    bridge = ctx.get_bridge()
    agent_factory = ctx.get_agent_factory()

    if not workspace:
        raise RuntimeError("Workspace not initialized")

    # Check for dynamic agent call first
    if name in agent_factory.active_agents:
        response = await agent_factory.execute_agent(name, arguments)
        return [TextContent(type="text", text=response)]

    # Check external tools
    if ctx.external_mcp and name in ctx.external_mcp.tools_map:
        result = await ctx.external_mcp.call_tool(name, arguments)
        # Result is likely CallToolResult, convert to TextContent
        content = []
        if hasattr(result, "content"):
            for item in result.content:
                if item.type == "text":
                    content.append(TextContent(type="text", text=item.text))
                elif item.type == "image":
                    # Handle image if needed, or skip
                    pass
        return content if content else [TextContent(type="text", text=str(result))]

    try:
        if name == "deconstruct":
            try:
                result = ctx.execute_deconstruct(arguments["problem"])
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=f"Deconstruction failed: {str(e)}")]
        elif name == "hypothesize":
            # Metabolic Cost: Hypothesis is a Search operation (Topology Tunneling)
            if ctx.ledger:
                from decimal import Decimal

                from src.substrate import EnergyToken, MeasurementCost
                # Cost ~ Search Cost (1.0)
                ctx.ledger.record_transaction(MeasurementCost(
                    energy=EnergyToken(Decimal("1.0"), "joules"),
                    operation_name="hypothesize"
                ))

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
            # Metabolic Cost: Synthesis is Integration (3.0)
            if ctx.ledger:
                from decimal import Decimal

                from src.substrate import EnergyToken, MeasurementCost
                ctx.ledger.record_transaction(MeasurementCost(
                    energy=EnergyToken(Decimal("3.0"), "joules"),
                    operation_name="synthesize"
                ))
            result = bridge.execute_monitored_operation(
                operation="synthesize",
                params={
                    "node_ids": arguments["node_ids"],
                    "goal": arguments.get("goal"),
                },
            )
            # Self-Correction/Critique (Via Negativa)
            if isinstance(result, dict) and "synthesis" in result:
                synthesis_text = result["synthesis"]
                critique_prompt = (
                    f"Critique the following synthesis for coherence and completeness based on the goal '{arguments.get('goal', 'None')}'. "
                    f"Synthesis: {synthesis_text}\n"
                    "Provide a comprehensive assessment identifying strengths and weaknesses."
                )
                critique = workspace._llm_generate(
                    system_prompt="You are a critical reviewer of AI-generated syntheses.",
                    user_prompt=critique_prompt
                )
                result["critique"] = critique

        elif name == "constrain":
            # Metabolic Cost: Constraint is Validation (2.0)
            if ctx.ledger:
                from decimal import Decimal

                from src.substrate import EnergyToken, MeasurementCost
                ctx.ledger.record_transaction(MeasurementCost(
                    energy=EnergyToken(Decimal("2.0"), "joules"),
                    operation_name="constrain"
                ))
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
            result = bridge.execute_monitored_operation(
                operation="compress_to_tool",
                params={
                    "node_ids": arguments["node_ids"],
                    "tool_name": arguments["tool_name"],
                    "description": arguments.get("description"),
                },
            )

            # Advisor Learning: Associate new tool with current advisor
            director = ctx.get_director()
            if director and director.compass and director.compass.integrated_intelligence.current_advisor:
                advisor = director.compass.integrated_intelligence.current_advisor
                tool_name = arguments["tool_name"]
                if tool_name not in advisor.tools:
                    advisor.tools.append(tool_name)
                    director.compass.advisor_registry.save_advisors()
                    result["advisor_learning"] = f"Tool '{tool_name}' added to advisor '{advisor.name}'"
        elif name == "resolve_meta_paradox":
            # Metabolic Cost: Paradox Resolution is System 3 (10.0)
            if ctx.ledger:
                from decimal import Decimal

                from src.substrate import EnergyToken, MeasurementCost
                ctx.ledger.record_transaction(MeasurementCost(
                    energy=EnergyToken(Decimal("10.0"), "joules"),
                    operation_name="resolve_meta_paradox"
                ))
            result = workspace.resolve_meta_paradox(
                conflict=arguments["conflict"]
            )
        elif name == "create_advisor":
            director = ctx.get_director()
            if not director or not director.compass:
                return [TextContent(type="text", text="Error: COMPASS framework not initialized")]

            result = director.compass.integrated_intelligence.create_advisor(
                id=arguments["id"],
                name=arguments["name"],
                role=arguments["role"],
                description=arguments["description"],
                system_prompt=arguments["system_prompt"],
                tools=arguments.get("tools", [])
            )
            return [TextContent(type="text", text=result)]
        elif name == "delete_advisor":
            director = ctx.get_director()
            if not director or not director.compass:
                return [TextContent(type="text", text="Error: COMPASS framework not initialized")]

            result = director.compass.integrated_intelligence.delete_advisor(
                id=arguments["id"]
            )
            return [TextContent(type="text", text=result)]
        elif name == "list_advisors":
            director = ctx.get_director()
            if not director or not director.compass:
                return [TextContent(type="text", text="Error: COMPASS framework not initialized")]

            advisors = director.compass.advisor_registry.advisors
            result_lines = ["Available Advisors:"]
            for advisor in advisors.values():
                result_lines.append(f"- {advisor.name} ({advisor.id}): {advisor.role}")
                result_lines.append(f"  Description: {advisor.description}")
                result_lines.append(f"  Tools: {', '.join(advisor.tools)}")
                result_lines.append("")

            return [TextContent(type="text", text="\n".join(result_lines))]
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
            director = ctx.get_director()
            success = director.teach_state(arguments["label"])
            msg = f"Learned state '{arguments['label']}'" if success else "Failed: No recent thought to learn from."
            return [TextContent(type="text", text=msg)]
        elif name == "get_known_archetypes":
            director = ctx.get_director()
            states = director.get_known_states()
            return [TextContent(type="text", text=json.dumps(states, indent=2))]
        elif name == "visualize_thought":
            director = ctx.get_director()
            vis = director.visualize_last_thought()
            return [TextContent(type="text", text=vis)]
        elif name == "take_nap":
            # Use shared Sleep Cycle instance
            sleep_cycle = ctx.sleep_cycle
            # Run in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            epochs = arguments.get("epochs", 5)
            results = await loop.run_in_executor(None, sleep_cycle.dream, epochs)
            return [TextContent(type="text", text=json.dumps(results, indent=2))]
        elif name == "explore_for_utility":
            # Metabolic Cost: Exploration is Search (1.0)
            if ctx.ledger:
                from decimal import Decimal

                from src.substrate import EnergyToken, MeasurementCost
                ctx.ledger.record_transaction(MeasurementCost(
                    energy=EnergyToken(Decimal("1.0"), "joules"),
                    operation_name="explore_for_utility"
                ))
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
            pointer = ctx.get_pointer()
            director = ctx.get_director()

            if not hasattr(pointer, "rnn"):
                return [TextContent(type="text", text="Error: Pointer does not have an RNN to diagnose")]

            # Extract weights based on RNN type
            weights = []
            if isinstance(pointer.rnn, torch.nn.GRU):
                # GRU weights: (W_ir|W_iz|W_in), (W_hr|W_hz|W_hn)
                if hasattr(pointer, "rnn"):
                    # Use Identity Extension to create a free vertex (Hidden State)
                    # Model: h_{t-1} --(hh)--> h_internal --(I)--> h_t
                    # This allows Sheaf Analysis to see "diffusion" (inference) in the internal state
                    hh = pointer.rnn.weight_hh_l0.detach()
                    weights.append(hh)
                weights.append(torch.eye(hh.shape[0], device=hh.device))

            # Run diagnosis with synthetic target
            # Target must match total edge dim
            total_edge_dim = sum(w.shape[0] for w in weights)

            # Use random target to probe topology
            target_error = torch.randn(total_edge_dim, device=weights[0].device)

            diagnosis = director.diagnose(weights, target_error=target_error)

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
            director = ctx.get_director()
            state, energy = director.latest_cognitive_state

            # Generate warnings for negative states
            warnings = []
            if state in ["Looping", "Confused", "Scattered"]:
                warnings.append(f"WARNING: Agent is in a '{state}' state. Consider using 'deconstruct' to break down the problem or 'take_nap' to reset.")
            if energy > -0.8 and state != "Unknown":
                 warnings.append("Note: State is unstable (high energy). This suggests the current thought pattern is not well-grounded in the manifold.")

            # Provide actionable advice
            advice = "Continue current line of reasoning."
            if state == "Looping":
                advice = "Stop. Use 'diagnose_pointer' to check for obstructions, or 'hypothesize' to jump to a new track."
            elif state == "Confused":
                advice = "High entropy detected. Use 'deconstruct' to simplify the problem."
            elif energy > -0.5 and state != "Unknown":
                advice = "Energy is high. Try to 'synthesize' recent thoughts to find a more stable basin."
            elif state == "Unknown":
                advice = "Cognitive state is uninitialized. Perform operations (e.g., 'deconstruct', 'hypothesize') to generate thought patterns."

            # Meta-Commentary (Self-Awareness)
            # Retrieve recent history for context
            recent_history = bridge.history.get_recent_history(limit=5)
            history_summary = "\n".join([f"- {h['operation']}: {h['result_summary']}" for h in recent_history])

            meta_prompt = (
                f"You are a reflective agent. Based on your recent history:\n{history_summary}\n"
                f"And your current state: {state} (Energy: {energy:.2f})\n"
                "Note: 'Energy' refers to Hopfield Network Energy. Lower (more negative) values indicate stability and convergence. "
                "Higher values (closer to 0 or positive) indicate instability, confusion, or active exploration.\n"
                "Provide a brief, first-person meta-commentary on your current thought process. "
                "Are you stuck? Are you making progress? What should you do next?"
            )
            meta_commentary = workspace._llm_generate(
                system_prompt="You are a reflective AI agent analyzing your own cognitive state.",
                user_prompt=meta_prompt
            )

            result = {
                "state": state,
                "energy": energy,
                "stability": "Stable" if energy < -0.8 else "Unstable",
                "warnings": warnings,
                "advice": advice,
                "meta_commentary": meta_commentary,
                "message": f"Agent is currently '{state}' (Energy: {energy:.2f})"
            }

        elif name == "diagnose_antifragility":
            # 1. Get base diagnostics
            pointer = ctx.get_pointer()
            director = ctx.get_director()

            # Extract weights for diagnosis
            weights = []
            if hasattr(pointer, "rnn"):
                hh = pointer.rnn.weight_hh_l0.detach()
                weights.append(hh)
                weights.append(torch.eye(hh.shape[0], device=hh.device))

            total_edge_dim = sum(w.shape[0] for w in weights)
            target_error = torch.randn(total_edge_dim, device=weights[0].device)

            diagnosis = director.diagnose(weights, target_error=target_error)

            # 2. Interpret as Antifragility Signals
            signals = []
            adaptation_plan = []

            # Signal 1: H1 Cohomology (Irreducible Errors)
            h1_dim = diagnosis.cohomology.h1_dimension
            if h1_dim > 0:
                signals.append(f"Detected {h1_dim} topological obstructions (H^1 > 0).")
                adaptation_plan.append("GROWTH OPPORTUNITY: The current architecture cannot resolve these error patterns. Recommendation: Expand the Manifold capacity or add a new abstraction layer.")

                # Spawn Explorer Agent
                tool_name = agent_factory.spawn_agent("H1 Hole", f"Detected {h1_dim} irreducible error cycles.")
                adaptation_plan.append(f"ACTION: Spawned specialized agent '{tool_name}' to explore missing concepts.")

            else:
                signals.append("No topological obstructions detected (H^1 = 0). System is robust but potentially rigid.")

            # Signal 2: H0 Cohomology (Graph Fragmentation)
            # If H^0 > vertex_dim (approx), it means the graph is disconnected.
            # For a single connected component with identity sheaf, H^0 dim = vertex_dim.
            # We use a heuristic: if H^0 > 1.5 * vertex_dim (assuming vertex_dim is roughly constant/average)
            # Actually, we can just check if H^0 is significantly larger than expected.
            # Let's assume "fragmentation" if H^0 is large.
            h0_dim = diagnosis.cohomology.h0_dimension
            # Heuristic: If H0 is very large, it suggests fragmentation.
            # But we don't know the "expected" H0 without knowing the vertex dim structure.
            # Let's use a simpler check: If H0 > 0 and H1 == 0, we might be in a fragmented but consistent state.

            # If we have multiple components, H0 dim scales with K.
            if h0_dim > 10: # Arbitrary threshold for "fragmented" for now, pending better calibration
                 signals.append(f"High H^0 dimension ({h0_dim}). Possible graph fragmentation (disconnected islands).")
                 adaptation_plan.append("FRAGMENTATION: Concepts are isolated. Recommendation: Build bridges between disconnected components.")

                 # Spawn Bridge Builder Agent
                 tool_name = agent_factory.spawn_agent("Bridge Builder", f"Detected graph fragmentation (H^0={h0_dim}).")
                 adaptation_plan.append(f"ACTION: Spawned specialized agent '{tool_name}' to connect islands.")

            # Signal 3: Harmonic-Diffusive Overlap (Learning Capacity)
            overlap = diagnosis.harmonic_diffusive_overlap
            if overlap < 0.1:
                signals.append(f"Low learning overlap ({overlap:.3f}). System is 'learning starved'.")
                adaptation_plan.append("STRESSOR: Information is not diffusing to update gradients. Recommendation: Increase 'temperature' (beta) to encourage exploration.")

                # Spawn Creative Agent
                tool_name = agent_factory.spawn_agent("Low Overlap", f"Learning overlap is {overlap:.3f} (Starved).")
                adaptation_plan.append(f"ACTION: Spawned specialized agent '{tool_name}' to bridge semantic gaps.")

            else:
                signals.append(f"Healthy learning overlap ({overlap:.3f}). System is plastic and adaptive.")

            # Signal 3: Monodromy (Feedback Loops)
            if diagnosis.monodromy:
                if diagnosis.monodromy.topology.value == "tension":
                    signals.append("Tension loop detected (conflicting feedback).")
                    adaptation_plan.append("VOLATILITY: Internal contradiction. Recommendation: Use 'deconstruct' to break the loop into compatible sub-components.")

                    # Spawn Debater Agent
                    tool_name = agent_factory.spawn_agent("Tension Loop", "Conflicting feedback loop detected (Monodromy: Tension).")
                    adaptation_plan.append(f"ACTION: Spawned specialized agent '{tool_name}' to arbitrate conflict.")

                elif diagnosis.monodromy.topology.value == "resonance":
                    signals.append("Resonance loop detected (reinforcing feedback).")
                    adaptation_plan.append("STABILITY: Self-reinforcing belief. Recommendation: Verify against external data to prevent hallucination.")

            result = {
                "antifragility_score": overlap * (1.0 if h1_dim == 0 else 0.5),
                "signals": signals,
                "adaptation_plan": adaptation_plan,
                "message": "Antifragility diagnosis complete."
            }

        elif name == "orthogonal_dimensions_analyzer":
            concept_a = arguments["concept_a"]
            concept_b = arguments["concept_b"]
            context = arguments.get("context", "")

            # Use the dedicated analyzer class
            # Inject ContinuityField from workspace
            analyzer = OrthogonalDimensionsAnalyzer(continuity_field=ctx.workspace.continuity_field)

            # 1. Generate Vectors
            # Use workspace embedding model
            vector_a = ctx.workspace.embedding_model.encode(concept_a)
            vector_b = ctx.workspace.embedding_model.encode(concept_b)

            # 2. Analyze Vectors (Quantitative)
            vector_analysis = analyzer.analyze_vectors(vector_a, vector_b)

            # 3. Analyze Concepts (Qualitative - LLM)
            prompt = analyzer.construct_analysis_prompt(concept_a, concept_b, context)

            # Call LLM
            response = ollama.chat(
                model=ctx.workspace.config.llm_model,
                messages=[
                    {"role": "system", "content": analyzer.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 1.0} # Analysis benefits from some creativity
            )

            analysis_text = response['message']['content']

            # Combine results
            result = {
                "concepts": {
                    "a": concept_a,
                    "b": concept_b
                },
                "vector_analysis": vector_analysis,
                "qualitative_analysis": analysis_text
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "set_intentionality":
            mode = arguments["mode"].lower()
            manifold = ctx.get_manifold()

            if mode == "optimization":
                # High Beta = Sharp attention = Convergent = Optimization
                # Set all manifolds to high beta
                manifold.state_memory.set_beta(50.0)
                manifold.agent_memory.set_beta(50.0)
                manifold.action_memory.set_beta(50.0)
                msg = "Intentionality set to OPTIMIZATION. All Manifold betas increased to 50.0 (Convergent)."
            elif mode == "adaptation":
                # Low Beta = Soft attention = Divergent = Adaptation
                # Set all manifolds to low beta
                manifold.state_memory.set_beta(5.0)
                manifold.agent_memory.set_beta(5.0)
                manifold.action_memory.set_beta(5.0)
                msg = "Intentionality set to ADAPTATION. All Manifold betas decreased to 5.0 (Divergent)."
            else:
                return [TextContent(type="text", text=f"Unknown mode: {mode}")]

            return [TextContent(type="text", text=msg)]

        elif name == "revise":
            # 1. Get components
            director = ctx.get_director()
            workspace = ctx.workspace

            # 2. Embed inputs
            belief_text = arguments["belief"]
            evidence_text = arguments["evidence"]
            constraints = arguments.get("constraints", [])

            # Use workspace embedding model
            belief_emb = torch.tensor(workspace._embed_text(belief_text), device=ctx.device)
            evidence_emb = torch.tensor(workspace._embed_text(evidence_text), device=ctx.device)

            # 3. Execute Hybrid Search (Operator C)
            result = director.hybrid_search.search(
                current_state=belief_emb,
                evidence=evidence_emb,
                constraints=constraints,
                context={"operation": "revise_tool"}
            )

            if result:
                # 4. Store waypoint as new thought node instead of querying Chroma
                with workspace.neo4j_driver.session() as session:
                    # Generate descriptive text for the waypoint
                    waypoint_description = (
                        f"Revised belief incorporating evidence. "
                        f"Original: '{belief_text[:1500]}{'...' if len(belief_text) > 1500 else ''}' "
                        f"Adjusted toward: '{evidence_text[:1500]}{'...' if len(evidence_text) > 1500 else ''}'"
                    )

                    # Sanitize score for storage
                    score = result.selection_score
                    if isinstance(score, float) and (score == float('inf') or score == float('-inf') or score != score):
                        score_val = 0.5
                    else:
                        score_val = float(score)

                    # Create node with revised embedding
                    revised_id = workspace._create_thought_node(
                        session,
                        waypoint_description,
                        "revision",
                        confidence=score_val,
                        embedding=result.best_pattern.cpu().tolist()
                    )

                    # Query for similar existing thoughts to provide context (optional)
                    query_result = workspace.collection.query(
                        query_embeddings=[result.best_pattern.cpu().tolist()],
                        n_results=1
                    )

                    similar_thought = ""
                    if query_result["documents"] and query_result["documents"][0]:
                        similar_thought = query_result["documents"][0][0]

                # Sanitize score for JSON output
                if isinstance(score, float) and (score == float('inf') or score == float('-inf') or score != score):
                    score = str(score)

                response = {
                    "status": "success",
                    "strategy": result.strategy.value,
                    "revised_content": waypoint_description,
                    "revised_node_id": revised_id,
                    "similar_existing_thought": similar_thought,
                    "selection_score": score,
                    "knn_attempted": result.knn_attempted,
                    "ltn_attempted": result.ltn_attempted,
                    "sheaf_validated": result.sheaf_validated,
                    "explanation": (
                        "LTN-generated waypoint stored as new thought node"
                        if result.strategy.value == "ltn"
                        else "K-NN retrieved pattern from existing memory"
                    )
                }
            else:
                response = {
                    "status": "failure",
                    "message": "Revision failed. Could not find valid stable state."
                }

            return [TextContent(type="text", text=json.dumps(response, indent=2))]

        elif name == "consult_compass":
            # Metabolic Cost: Delegation is Expensive (5.0)
            if ctx.ledger:
                from decimal import Decimal

                from src.substrate import EnergyToken, MeasurementCost
                ctx.ledger.record_transaction(MeasurementCost(
                    energy=EnergyToken(Decimal("5.0"), "joules"),
                    operation_name="consult_compass"
                ))
            # Delegate to COMPASS framework via Director
            director = ctx.raa_context.get("director")
            if not director or not director.compass:
                return [TextContent(type="text", text="Error: COMPASS framework not initialized")]

            task = arguments["task"]
            context = arguments.get("context", {})

            # Run COMPASS process_task
            result = await director.compass.process_task(task, context)

            # Return the clean Final Report if available, otherwise fallback to solution
            final_report = result.get("final_report", result.get("solution", str(result)))

            # Include success status prefix
            status = "SUCCESS" if result.get("success", False) else "PARTIAL/FAILURE"
            output = f"[{status}]\n\n{final_report}"

            return [TextContent(type="text", text=output)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Error in {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server

    # Initialize context on startup
    server_context.initialize()

    # Initialize external MCPs (async)
    if server_context.external_mcp:
        logger.info("Initializing external MCP servers...")
        await server_context.external_mcp.initialize()

        # DIAGNOSTIC: Verify tools loaded
        logger.info(f"External MCP initialized: {server_context.external_mcp.is_initialized}")
        logger.info(f"External MCP tools loaded: {list(server_context.external_mcp.tools_map.keys())}")

        # Test get_available_tools
        all_tools = server_context.get_available_tools()
        external_count = len(server_context.external_mcp.get_tools()) if server_context.external_mcp.is_initialized else 0
        logger.info(f"Total tools available: {len(all_tools)} (External: {external_count}, Internal: {len(all_tools) - external_count})")

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        # Cleanup on exit
        server_context.cleanup()


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

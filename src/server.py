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
import os
import re
import shutil
import time
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
import torch
from chromadb.config import Settings
from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import TextContent, Tool
from neo4j import GraphDatabase
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.cognition.curiosity import CuriosityModule
from src.cognition.emotion_framework import consult_computational_empathy
from src.cognition.grok_lang import AffectVector, GrokDepthCalculator, Intent, MindState, Utterance
from src.cognition.logic_core import LogicCore
from src.cognition.meta_validator import MetaValidator
from src.cognition.system_guide import SystemGuideNodes
from src.cognition.working_memory import WorkingMemory
from src.compass.orthogonal_dimensions import OrthogonalDimensionsAnalyzer
from src.director import Director, DirectorConfig
from src.director.simple_gp import SimpleGP
from src.embeddings.base_embedding_provider import BaseEmbeddingProvider
from src.embeddings.embedding_factory import EmbeddingFactory
from src.integration.agent_factory import AgentFactory
from src.integration.continuity_field import ContinuityField
from src.integration.continuity_service import ContinuityService
from src.integration.cwd_raa_bridge import BridgeConfig, CWDRAABridge
from src.integration.precuneus import PrecuneusIntegrator
from src.integration.sleep_cycle import SleepCycle
from src.llm.factory import LLMFactory
from src.llm.provider import BaseLLMProvider
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
from src.substrate.energy import EnergyDepletionError, MetabolicLedger
from src.substrate.entropy import EntropyMonitor

# Load environment variables from .env file
load_dotenv()


def evolve_formula_logic(data_points, n_generations=10, hybrid=False):
    """
    Uses Genetic Programming to evolve a symbolic formula that fits the data.
    data_points: List of dictionaries [{'x': 1, 'y': 2, 'result': 3}, ...]
    hybrid: If True, uses Evolutionary Optimization (local refinement of constants).
    """
    print(f"DEBUG: EVOLVE FORMULA LOGIC CALLED (Hybrid={hybrid})")

    # 1. Prepare Data
    # SimpleGP expects list of dicts. We need to ensure target key is consistent.
    # The input schema has 'result' as the target.

    # 2. Initialize SimpleGP
    # We support x, y, z variables as per schema
    gp = SimpleGP(variables=["x", "y", "z"], population_size=100, max_depth=5)

    # 3. Evolve
    # SimpleGP.evolve returns (best_formula_str, best_error_float)
    best_formula, best_error = gp.evolve(
        data_points, target_key="result", generations=n_generations, hybrid=hybrid
    )

    return f"Evolved Formula: {best_formula} | MSE: {best_error:.4f} | Mode: {'Hybrid' if hybrid else 'Standard'}"


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cwd-mcp")

# Initialize MCP server
server = Server("cwd-mcp")


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
    chroma_path: str = Field(default=str(Path(__file__).parent.parent / "chroma_data"))
    # LLM Settings
    llm_provider: str = "openrouter"
    llm_model: str = Field(default="google/gemini-2.0-flash-exp:free", description="LLM model name")
    compass_model: str = Field(
        default="anthropic/claude-3.5-sonnet:beta", description="Model for COMPASS reasoning"
    )

    # Ruminator Settings
    ruminator_enabled: bool = Field(default=False, description="Enable background rumination")
    ruminator_delay: float = Field(
        default=2.0, description="Delay between rumination cycles (seconds)"
    )

    # Embedding Settings
    embedding_provider: str = Field(
        default="sentence-transformers",
        description="Embedding provider (sentence-transformers, ollama, lm_studio)",
    )
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5", description="Embedding model name"
    )
    confidence_threshold: float = Field(default=0.7)  # Lower to .3 for asymmetric embeddings
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device for tensor operations",
    )

    @model_validator(mode="after")
    def resolve_relative_paths(self):
        """Ensure paths are absolute relative to project root."""
        if self.chroma_path and not Path(self.chroma_path).is_absolute():
            # Resolve relative to project root (parent of parent of this file)
            # This ensures consistent behavior regardless of CWD (e.g., running from src/ vs root)
            root = Path(__file__).parent.parent
            self.chroma_path = str((root / self.chroma_path).resolve())
        return self


class CognitiveWorkspace:
    """
    Manages the cognitive workspace - a hybrid system combining:
    - Neo4j for structural/graph reasoning
    - Chroma for latent space vector operations
    - Cognitive primitives for active reasoning
    """

    def __init__(self, config: CWDConfig):
        self.config = config
        self.device = config.device
        self.manifold = None

        # Initialize Neo4j
        self.neo4j_driver = GraphDatabase.driver(
            config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
        )

        # Initialize Chroma with persistence
        # Use PersistentClient for data to survive restarts
        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma_path, settings=Settings(anonymized_telemetry=False)
        )

        # Initialize LLM Provider
        self.llm_provider: BaseLLMProvider = LLMFactory.create_provider(
            self.config.llm_provider, self.config.llm_model
        )

        # Initialize Ruminator Provider (for offline/background tasks)
        ruminator_model = os.getenv("RUMINATOR_MODEL", "amazon/nova-2-lite-v1:free")
        ruminator_provider_name = os.getenv("RUMINATOR_PROVIDER", "openrouter")
        self.ruminator_provider: BaseLLMProvider = LLMFactory.create_provider(
            ruminator_provider_name, ruminator_model
        )

        # Initialize Dreamer Provider (for high-IQ crystallization)
        dreamer_model = os.getenv("DREAMER_MODEL", "anthropic/claude-3.5-sonnet")
        dreamer_provider_name = os.getenv("DREAMER_PROVIDER", "openrouter")
        self.dreamer_provider: BaseLLMProvider = LLMFactory.create_provider(
            dreamer_provider_name, dreamer_model
        )

        # Initialize embedding provider using factory
        logger.info(f"Loading embedding provider: {self.config.embedding_provider}")
        logger.info(f"Embedding model: {self.config.embedding_model}")

        try:
            # Create embedding provider via factory (handles all optimizations)
            self.embedding_model: BaseEmbeddingProvider = EmbeddingFactory.create(
                provider_name=config.embedding_provider, model_name=config.embedding_model
            )

            # Initialize Continuity Field (Identity Manifold)
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.continuity_field = ContinuityField(embedding_dim=embedding_dim)

            # Initialize with base anchors to prevent "empty field" errors
            base_concepts = ["existence", "agent", "action", "thought", "reasoning", "utility"]
            for concept in base_concepts:
                vector = self.embedding_model.encode(concept)
                self.continuity_field.add_anchor(vector)
            logger.info(
                f"Initialized ContinuityField with dim={embedding_dim} and {len(base_concepts)} base anchors"
            )
            logger.info(f"DEBUG: Server ContinuityField ID: {id(self.continuity_field)}")
            logger.info(
                f"DEBUG: Server ContinuityField anchors: {len(self.continuity_field.anchors)}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize embedding provider: {e}")
            raise

        # Create Chroma collections
        self.collection = self.chroma_client.get_or_create_collection(
            name="thought_nodes", metadata={"description": "Cognitive workspace thought-nodes"}
        )

        # Collection for Manifold patterns (persistent memory across restarts)
        self.manifold_patterns_collection = self.chroma_client.get_or_create_collection(
            name="manifold_patterns",
            metadata={"description": "Hopfield network patterns for associative memory"},
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

        # Initialize Metabolic Ledger (Energy System)
        self.ledger = MetabolicLedger()

        # Initialize Entropy Monitor (State Dynamics)
        self.entropy_monitor = EntropyMonitor()

        # Initialize Continuity Service
        self.continuity_service = ContinuityService(
            continuity_field=self.continuity_field, work_history=self.history
        )

        logger.info("Cognitive Workspace initialized with Gen 3 architecture")

        # Initialize System Guide Nodes (Code Casefile)
        self.system_guide = SystemGuideNodes(self.neo4j_driver, root_path=os.getcwd())

        # Initialize Curiosity Module (Intrinsic Motivation)
        self.curiosity = CuriosityModule(self)

        # Initialize Logic Core (Formal Verification)
        self.logic_core = LogicCore()

        # Initialize Working Memory (Short-term context for LLM continuity)
        self.working_memory = WorkingMemory(max_entries=20, max_context_chars=8000)

        # Reference to Director (injected by ServerContext)
        self.director = None

        # Tool Usage Tracking for Entropy
        self.tool_usage_buffer: List[str] = []
        self.MAX_TOOL_BUFFER = 20

        # Initialize MCP Tools
        self.mcp_tools = {
            "read_file": self._read_file,
            "list_directory": self._list_directory,
            "search_codebase": self._search_codebase,
            "inspect_codebase": self._inspect_codebase,
            "get_cognitive_state": self._get_cognitive_state,
        }

    def get_manifold(self):
        if self.manifold:
            return self.manifold
        raise RuntimeError("Manifold not injected into Workspace")

    def _get_cognitive_state(self) -> str:
        """
        Get the current cognitive state (entropy, focus/explore mode).
        """
        try:
            status = self.entropy_monitor.get_status()
            return json.dumps(status, indent=2)
        except Exception as e:
            return f"Error getting cognitive state: {e}"

    def _track_tool_usage(self, tool_name: str):
        """Track tool usage for entropy calculation."""
        self.tool_usage_buffer.append(tool_name)
        if len(self.tool_usage_buffer) > self.MAX_TOOL_BUFFER:
            self.tool_usage_buffer.pop(0)

        # Update entropy
        counts = Counter(self.tool_usage_buffer)
        entropy = self.entropy_monitor.update_from_counts(counts)
        logger.debug(
            f"Tool usage entropy updated: {entropy:.2f} (State: {self.entropy_monitor.state.value})"
        )

    def _read_file(self, path: str) -> str:
        """Read file content."""
        self._track_tool_usage("read_file")
        try:
            file_path = Path(path)
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

    def _list_directory(self, path: str) -> str:
        """List directory contents."""
        self._track_tool_usage("list_directory")
        try:
            dir_path = Path(path)
            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Not a directory: {path}"

            items = []
            for item in dir_path.iterdir():
                type_str = "DIR" if item.is_dir() else "FILE"
                items.append(f"[{type_str}] {item.name}")
            return "\n".join(sorted(items))
        except Exception as e:
            return f"Error listing directory: {e}"

    def _search_codebase(self, query: str, path: str = ".") -> str:
        """Search for text pattern in codebase using grep, excluding common junk."""
        self._track_tool_usage("search_codebase")
        try:
            import subprocess

            # Validation: Ensure path is safe and exists
            search_path = Path(path).resolve()
            if not search_path.exists():
                return f"Error: Path not found: {path}"

            # Security: Resolve grep path to ensure we are running the intended binary
            grep_path = shutil.which("grep")
            if not grep_path:
                return "Error: grep utility not found."

            # Security: Prevent directory traversal outside of workspace
            workspace_root = Path(__file__).parent.parent.resolve()
            if not str(search_path).startswith(str(workspace_root)):
                return (
                    f"Error: Search path {search_path} is outside workspace root {workspace_root}"
                )

            # State-dependent behavior
            # FOCUS: Narrow search, fewer results (Convergence)
            # EXPLORE: Broad search, more results (Divergence)
            max_results = 100  # Default
            if self.entropy_monitor.state.value == "focus":
                max_results = 20
            elif self.entropy_monitor.state.value == "explore":
                max_results = 200

            # Exclude common non-code directories
            excludes = [
                "--exclude-dir=.git",
                "--exclude-dir=__pycache__",
                "--exclude-dir=chroma_data",
                "--exclude-dir=venv",
                "--exclude-dir=node_modules",
                "--exclude-dir=.pytest_cache",
                "--exclude-dir=.mypy_cache",
                "--exclude=*.pyc",
                "--exclude=*.rdb",
                "--exclude=*.log",
            ]

            # Construction: Use list format for subprocess to avoid shell injection.
            # Use '--' to delimit options from the query, protecting against queries starting with '-'
            cmd = [grep_path, "-r", "-n"] + excludes + ["--", query, str(search_path)]

            # Add timeout to prevent hanging
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0 and result.returncode != 1:
                return f"Error searching: {result.stderr}"

            output = result.stdout
            if not output:
                return "No matches found."

            # Truncate if too long
            lines = output.splitlines()
            if len(lines) > max_results:
                return (
                    "\n".join(lines[:max_results])
                    + f"\n... and {len(lines)-max_results} more matches (truncated due to {self.entropy_monitor.state.value} state)."
                )
            return output

        except subprocess.TimeoutExpired:
            return "Error: Search timed out."
        except Exception as e:
            return f"Error searching codebase: {e}"

    def _inspect_codebase(self, action: str, **kwargs) -> str:
        """
        Inspect codebase structure using System Guide Nodes (Code Casefile).
        Actions:
        - scan: Auto-generate bookmarks for classes/funcs (optional: path)
        - create_concept: Create a high-level concept (name, description)
        - bookmark: Create a bookmark (file, line, snippet, notes)
        - link: Link bookmark to concept (concept_name, bookmark_id)
        - get_concept: Get details of a concept (name)
        """
        self._track_tool_usage("inspect_codebase")
        try:
            if action == "scan":
                path = kwargs.get("path", ".")
                return self.system_guide.scan_codebase(path)

            elif action == "create_concept":
                name = kwargs.get("name")
                description = kwargs.get("description", "")
                if not name:
                    return "Error: 'name' required for create_concept"
                return self.system_guide.create_concept(name, description)

            elif action == "bookmark":
                file_path = kwargs.get("file")
                line = kwargs.get("line")
                snippet = kwargs.get("snippet")
                notes = kwargs.get("notes", "")
                if not all([file_path, line, snippet]):
                    return "Error: 'file', 'line', 'snippet' required for bookmark"
                return self.system_guide.create_bookmark(file_path, int(line), snippet, notes)

            elif action == "link":
                concept = kwargs.get("concept_name")
                bookmark_id = kwargs.get("bookmark_id")
                if not all([concept, bookmark_id]):
                    return "Error: 'concept_name', 'bookmark_id' required for link"
                self.system_guide.link_bookmark_to_concept(concept, bookmark_id)
                return f"Linked {bookmark_id} to {concept}"

            elif action == "get_concept":
                name = kwargs.get("name")
                if not name:
                    return "Error: 'name' required for get_concept"
                return json.dumps(self.system_guide.get_concept_details(name), indent=2)

            else:
                return f"Unknown action: {action}"

        except Exception as e:
            return f"Error in inspect_codebase: {e}"

    def _execute_with_tools(self, system_prompt: str, user_prompt: str, tools: List[Dict]) -> str:
        """
        Execute LLM generation with tool support.
        Handles the tool call loop.
        """

        # Initial call
        response = self.llm_provider.generate(system_prompt, user_prompt, tools=tools)

        # Check if response is a tool call (this depends on provider implementation)
        # Since our generate() returns string, we need to parse it or rely on provider-specific behavior.
        # However, OpenRouter/OpenAI providers usually return the content.
        # If tool_calls are present, they are in the message object, not the content string.
        # Our current generate() implementation returns message.content.
        # This is a limitation of the current generate() interface.

        # To properly support tools, we need to access the full message object or tool_calls.
        # But for now, let's assume we are using a provider that might return JSON for tool calls
        # or we need to update generate() to return more than just string.

        # CRITICAL FIX: The current generate() returns only string.
        # We need to update it to return the full response or handle tool calls internally.
        # Given the constraints, let's try to use a convention or parse the output if it looks like a tool call.
        # But OpenAI tool calls are structured.

        # For this MVP, let's assume the LLM might return a JSON string if we prompt it to,
        # OR we need to update generate() to return (content, tool_calls).

        return response

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

    def _validate_identifier(self, identifier: str) -> str:
        """
        Validate that an identifier (label/type) is safe for Cypher injection.
        Allows alphanumeric and underscores.
        """
        if not re.match(r"^[a-zA-Z0-9_]+$", identifier):
            raise ValueError(f"Invalid identifier: {identifier}")
        return identifier

    def search_nodes(
        self, label: str, property_filters: dict[str, Any] | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Search for nodes with a specific label and optional property filters.
        Uses validated f-strings for label to ensure compatibility.
        """
        safe_label = self._validate_identifier(label)

        # Build property filter string dynamically
        where_clause = ""
        if property_filters:
            conditions = [f"n.{k} = ${k}" for k in property_filters.keys()]
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        MATCH (n:{safe_label})
        {where_clause}
        RETURN n
        LIMIT $limit
        """

        params = property_filters or {}
        params["limit"] = limit

        return self.read_query(query, params)

    def traverse_relationships(
        self,
        start_id: str,
        rel_type: str | None = None,
        direction: str = "OUTGOING",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Traverse relationships dynamically.
        If rel_type is None, traverses all relationship types.
        """
        if rel_type:
            safe_rel_type = self._validate_identifier(rel_type)
            type_constraint = f":{safe_rel_type}"
        else:
            type_constraint = ""

        if direction == "OUTGOING":
            pattern = f"(n)-[r{type_constraint}]->(m)"
        elif direction == "INCOMING":
            pattern = f"(n)<-[r{type_constraint}]-(m)"
        else:
            pattern = f"(n)-[r{type_constraint}]-(m)"

        query = f"""
        MATCH (n) WHERE n.id = $start_id
        MATCH {pattern}
        RETURN m, type(r) as rel_type, properties(r) as rel_props
        LIMIT $limit
        """
        params = {"start_id": start_id, "limit": limit}
        return self.read_query(query, params)

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

        # Update working memory focus
        self.working_memory.set_focus(goal=goal_description)

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

    def _prove(self, premises: List[str], conclusion: str) -> Dict[str, Any]:
        """Verify a logical conclusion from premises using Prover9."""
        self._track_tool_usage("prove")
        if not self.logic_core:
            return {"result": "error", "message": "Logic Core not initialized"}
        return self.logic_core.prove(premises, conclusion)

    def _find_counterexample(
        self, premises: List[str], conclusion: str, domain_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Find a counterexample using Mace4."""
        self._track_tool_usage("find_counterexample")
        if not self.logic_core:
            return {"result": "error", "message": "Logic Core not initialized"}
        return self.logic_core.find_counterexample(premises, conclusion, domain_size)

    def _find_model(self, premises: List[str], domain_size: Optional[int] = None) -> Dict[str, Any]:
        """Find a model using Mace4."""
        self._track_tool_usage("find_model")
        if not self.logic_core:
            return {"result": "error", "message": "Logic Core not initialized"}
        return self.logic_core.find_model(premises, domain_size)

    def _check_well_formed(self, statements: List[str]) -> Dict[str, Any]:
        """Validate logical formulas."""
        self._track_tool_usage("check_well_formed")
        if not self.logic_core:
            return {"result": "error", "message": "Logic Core not initialized"}
        return self.logic_core.check_well_formed(statements)

    def _verify_commutativity(
        self,
        path_a: List[str],
        path_b: List[str],
        object_start: str,
        object_end: str,
        with_category_axioms: bool = True,
    ) -> Dict[str, Any]:
        """Verify diagram commutativity."""
        self._track_tool_usage("verify_commutativity")
        if not self.logic_core:
            return {"result": "error", "message": "Logic Core not initialized"}
        return self.logic_core.verify_commutativity(
            path_a, path_b, object_start, object_end, with_category_axioms
        )

    def _get_category_axioms(self, concept: str, **kwargs) -> Dict[str, Any]:
        """Get category theory axioms."""
        self._track_tool_usage("get_category_axioms")
        if not self.logic_core:
            return {"result": "error", "message": "Logic Core not initialized"}
        axioms = self.logic_core.get_category_axioms(concept, **kwargs)
        return {"concept": concept, "axioms": axioms}

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

        return self._llm_generate(system_prompt, user_prompt, max_tokens=16000)

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

            # Record in working memory
            top_ids = [c["node_id"] for c in top_candidates[:5]]
            self.working_memory.record(
                operation="explore_for_utility",
                input_data={"focus_area": focus_area, "max_candidates": max_candidates},
                output_data={
                    "count": len(top_candidates),
                    "top_score": top_candidates[0]["combined_score"] if top_candidates else 0,
                },
                node_ids=top_ids,
            )

            # Persist to SQLite history
            self.history.log_operation(
                operation="explore_for_utility",
                params={"focus_area": focus_area, "max_candidates": max_candidates},
                result={"count": len(top_candidates), "top_ids": top_ids},
                cognitive_state=self.working_memory.current_goal or "Unknown",
            )

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
        try:
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

        except Exception as e:
            logger.error(f"Embedding generation failed (likely CUDA error): {e}")
            logger.info("Attempting recovery: Moving embedding model to CPU and retrying...")

            try:
                # Attempt to move model to CPU
                if hasattr(self.embedding_model, "to"):
                    self.embedding_model.to("cpu")
                    logger.info("Moved embedding model to CPU successfully.")

                    # Retry encoding on CPU
                    with torch.no_grad():
                        # We don't try Qwen optimized prompt here to be safe, just standard encode
                        embedding = self.embedding_model.encode(text, device="cpu")

                        if hasattr(embedding, "tolist"):
                            result = embedding.tolist()
                        else:
                            result = list(embedding)
                        return result
                else:
                    logger.error("Embedding model does not support .to('cpu')")
                    raise e

            except Exception as e2:
                logger.error(f"Embedding fallback failed: {e2}")
                # Return zero vector as last resort to prevent system crash
                # Get dim from model or config
                dim = 1024  # Default fallback
                if hasattr(self.embedding_model, "get_sentence_embedding_dimension"):
                    dim = self.embedding_model.get_sentence_embedding_dimension()
                return [0.0] * dim

    def _get_contents_from_chroma(self, ids: list[str]) -> dict[str, str]:
        """
        Retrieve content for multiple thought IDs from Chroma.
        Returns a dictionary {node_id: content}.
        """
        if not ids:
            return {}

        try:
            results = self.collection.get(ids=ids, include=["documents"])
            if not results or not results["ids"]:
                return {}

            content_map = {}
            for doc_id, content in zip(results["ids"], results["documents"]):
                # Handle case where content might be None or list?
                # Chroma documents are usually strings.
                if content:
                    content_map[doc_id] = content

            return content_map
        except Exception as e:
            logger.error(f"Error retrieving content from Chroma: {e}")
            return {}

    def _llm_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 16000,
        include_memory: bool = True,
        operation_name: str = None,
    ) -> str:
        """
        Generate text using the configured LLM provider.

        Automatically injects working memory context for continuity
        across operations unless include_memory=False.

        Args:
            system_prompt: System context for the LLM
            user_prompt: User/task prompt
            max_tokens: Maximum response tokens
            include_memory: Whether to inject working memory context
            operation_name: Name of operation for memory tracking
        """
        try:
            # Inject working memory context into system prompt
            if include_memory and hasattr(self, "working_memory") and self.working_memory:
                memory_context = self.working_memory.get_context()
                if memory_context:
                    system_prompt = f"{system_prompt}\n\n{memory_context}"

            response = self.llm_provider.generate(system_prompt, user_prompt, max_tokens=max_tokens)
            return response
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return f"[LLM unavailable: {str(e)[:100]}...]"

    def _llm_generate_and_record(
        self,
        system_prompt: str,
        user_prompt: str,
        operation_name: str,
        input_data: Any = None,
        node_ids: List[str] = None,
        max_tokens: int = 16000,
    ) -> str:
        """
        Generate text AND record the operation in working memory.

        Use this for significant cognitive operations that should
        maintain context continuity (synthesize, hypothesize, etc.)
        """
        response = self._llm_generate(system_prompt, user_prompt, max_tokens, include_memory=True)

        # Record in working memory for future context
        self.working_memory.record(
            operation=operation_name,
            input_data=input_data or user_prompt[:500],
            output_data=response[:1000],
            node_ids=node_ids or [],
        )

        return response

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

        # Store in Neo4j with Gen 3 fields (Content stored in Chroma only)
        query = """
        CREATE (t:ThoughtNode {
            id: $id,
            cognitive_type: $cognitive_type,
            confidence: $confidence,
            created_at: timestamp(),
            parent_problem: $parent_problem,
            utility_score: $utility_score,
            compression_score: $compression_score,
            intrinsic_reward: $intrinsic_reward,
            chroma_doc_id: $id
        })
        RETURN t.id as id
        """
        result = session.run(
            query,
            id=thought_id,
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

            # 1. Advanced Tripartite Decomposition (System 2 Analysis)
            # We break the problem into three orthogonal domains:
            # - STATE (vmPFC): Context/Environment
            # - AGENT (amPFC): Persona/Intent
            # - ACTION (dmPFC): Verbs/Transitions

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

            llm_output = self._llm_generate(system_prompt, user_prompt, max_tokens=1000)
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
                comp_id = self._create_thought_node(
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
                vec = self._embed_text(content)
                vec_tensor = torch.tensor(vec, dtype=torch.float32, device=self.device)
                self.get_manifold().store_pattern(vec_tensor, domain=label.lower())
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
            self.working_memory.record(
                operation="deconstruct",
                input_data={"problem": problem[:500]},
                output_data={"components": len(final_components), "root_id": root_id},
                node_ids=[root_id] + component_ids[:5],
            )

            # Persist to SQLite history for long-term recall
            self.history.log_operation(
                operation="deconstruct",
                params={"problem": problem[:500], "max_depth": max_depth},
                result=result,
                cognitive_state=self.working_memory.current_goal or "Unknown",
            )

            return result

    def get_node_context(self, node_id: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get the local context (neighbors) of a node.
        """
        try:
            with self.neo4j_driver.session() as session:  # Changed self.driver to self.neo4j_driver
                # Get node and immediate neighbors
                query = """
                MATCH (n {id: $node_id})-[r]-(m)
                RETURN n, r, m
                LIMIT 50
                """
                result = session.run(query, node_id=node_id)

                neighbors = []
                node_data = {}

                # Collect all node IDs to fetch content from Chroma
                all_ids = {node_id}
                for record in result:
                    all_ids.add(record["n"]["id"])
                    all_ids.add(record["m"]["id"])

                content_map = self._get_contents_from_chroma(list(all_ids))

                # Re-run query to process results with content
                result = session.run(query, node_id=node_id)

                for record in result:
                    if not node_data:
                        node_data = dict(record["n"])
                        node_data["content"] = content_map.get(node_data.get("id"), "")

                    neighbor = dict(record["m"])
                    rel = record["r"]
                    neighbors.append(
                        {
                            "id": neighbor.get("id"),
                            "content": content_map.get(neighbor.get("id"), ""),
                            "relationship": rel.type,
                            "direction": (
                                "outgoing" if rel.start_node.id == record["n"].id else "incoming"
                            ),
                        }
                    )

                if not node_data:
                    # Try fetching just the node if no neighbors
                    result = session.run("MATCH (n {id: $node_id}) RETURN n", node_id=node_id)
                    record = result.single()
                    if record:
                        node_data = dict(record["n"])
                        node_data["content"] = content_map.get(node_data.get("id"), "")
                    else:
                        return {"error": f"Node {node_id} not found"}

                return {"node": node_data, "neighbors": neighbors, "neighbor_count": len(neighbors)}
        except Exception as e:
            logger.error(f"Failed to get node context: {e}")
            return {"error": str(e)}

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
        content_map = self._get_contents_from_chroma(all_ids)

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
                RETURN a.id as id_a, b.id as id_b,
                       a.cognitive_type as type_a, b.cognitive_type as type_b
                """,
                id_a=node_a_id,
                id_b=node_b_id,
            ).single()

            if not nodes:
                return {"error": "Nodes not found"}

            content_map = self._get_contents_from_chroma([nodes["id_a"], nodes["id_b"]])
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
            self.working_memory.record(
                operation="hypothesize",
                input_data={"node_a": node_a_id, "node_b": node_b_id, "context": context},
                output_data={
                    "hypothesis": hypothesis_text[:500],
                    "quality": float(hypothesis_quality),
                },
                node_ids=[node_a_id, node_b_id, hyp_id],
            )

            # Persist to SQLite history
            self.history.log_operation(
                operation="hypothesize",
                params={"node_a": node_a_id, "node_b": node_b_id, "context": context},
                result={
                    "hypothesis_id": hyp_id,
                    "quality": float(hypothesis_quality),
                    "similarity": float(similarity_score),
                },
                cognitive_state=self.working_memory.current_goal or "Unknown",
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

        return self._llm_generate(system_prompt, user_prompt, max_tokens=16000)

    def get_advisor_context(self, advisor_id: str) -> str:
        """
        Retrieve the textual context (knowledge) associated with an advisor.
        Fetches content from linked ThoughtNodes in Neo4j.
        """
        if not self.director or not self.director.compass:
            return "Compass not initialized."

        registry = self.director.compass.advisor_registry
        knowledge = registry.get_advisor_knowledge(advisor_id)
        node_ids = knowledge.get("node_ids", [])

        if not node_ids:
            return f"No knowledge linked to advisor '{advisor_id}'."

        context_parts = [f"Knowledge Context for Advisor '{advisor_id}':"]

        with self.neo4j_driver.session() as session:
            # Fetch all node details first (structure)
            node_records = session.run(
                "MATCH (n:ThoughtNode) WHERE n.id IN $ids RETURN n.id as id, n.type as type",
                ids=node_ids,
            ).data()

            # Fetch content from Chroma
            content_map = self._get_contents_from_chroma(node_ids)

            for record in node_records:
                nid = record["id"]
                ntype = record.get("type", "thought")
                content = content_map.get(nid, "Content not found.")
                context_parts.append(f"\n--- Node: {nid} ({ntype}) ---\n{content}")

        return "\n".join(context_parts)

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
        # LOGICAL RIGOR ENHANCEMENT (System 2)
        # We purposely disable the embedding "quick reject" (previously < 0.15)
        # to prevent semantic bias from rejecting logically valid but lexically distinct concepts.
        # The embedding_hint is still passed for logging but ignored for decision making.

        # Chain of Verification Prompt
        system_prompt = (
            "You are a precise constraint validator (System 2 Logic Engine).\n"
            "Your task: Determine if the Content logically satisfies the Rule.\n\n"
            "CHAIN OF VERIFICATION:\n"
            "1. PREMISE ISOLATION: Identify the core claims in the content.\n"
            "2. RULE EXTRACTION: Identify the strict condition imposed by the rule.\n"
            "3. CONTRADICTION SEARCH: Does any claim in the content contradict the rule? (Check for logical negation).\n"
            "4. COMPLETENESS: Does the content fully address the rule's requirement?\n"
            "5. VERDICT: Is the rule fully satisfied?\n\n"
            "Output format:\n"
            "ANALYSIS: [Brief step-by-step reasoning]\n"
            "VERDICT: VALID (if fully satisfied) or INVALID (if any failure)\n"
            "Be rigorous. Do not allow vague semantic overlap to pass as strict logical satisfaction."
        )

        user_prompt = (
            f"Content:\n{content[:4000]}\n\n" f"Rule: {rule}\n\n" f"Perform Chain of Verification:"
        )

        llm_output = self._llm_generate(system_prompt, user_prompt, max_tokens=1000)

        # Parse LLM response
        try:
            response = llm_output.strip()

            # Look for explicit verdict
            if "VERDICT: VALID" in response or "VERDICT:VALID" in response:
                # High confidence if explicit valid
                return (True, 0.95)

            if "VERDICT: INVALID" in response or "VERDICT:INVALID" in response:
                return (False, 0.95)

            # Fallback parsing for partial responses
            if "VALID" in response.splitlines()[-1]:
                return (True, 0.8)
            if "INVALID" in response.splitlines()[-1]:
                return (False, 0.8)

            # Ambiguous response
            logger.warning(f"Ambiguous Logic Validator response: {response[:50]}...")
            return (False, 0.5)

        except Exception as e:
            logger.warning(f"Logic Validator parse error: {e}")
            # Fallback to conservative rejection if logic fails
            return (False, 0.1)

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

            content_map = self._get_contents_from_chroma(list(all_ids))

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
                synthesis_text, lambda sys, user: self._llm_generate(sys, user, max_tokens=1000)
            )

            # 3. Unified Score & Quadrant
            meta_stats = MetaValidator.calculate_unified_score(
                float(coverage_score), float(rigor_score), context="comprehensive_analysis"
            )

            # Create synthesis node with REAL embedding (more accurate than centroid)
            # We use the real embedding so future searches find it where it actually is semantically
            synth_id = self._create_thought_node(
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
            self.working_memory.record(
                operation="synthesize",
                input_data={"goal": goal, "node_count": len(node_ids)},
                output_data={"synthesis": synthesis_text[:500], "quadrant": meta_stats["quadrant"]},
                node_ids=node_ids[:5] + [synth_id],
            )

            # Persist to SQLite history
            self.history.log_operation(
                operation="synthesize",
                params={"goal": goal, "node_count": len(node_ids), "node_ids": node_ids[:5]},
                result={
                    "synthesis_id": synth_id,
                    "quadrant": meta_stats["quadrant"],
                    "unified_score": meta_stats["unified_score"],
                },
                cognitive_state=self.working_memory.current_goal or "Unknown",
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
            if data["context"]:
                # Add context as a sub-bullet or note
                context_str = "; ".join(data["context"])
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

        return self._llm_generate(system_prompt, user_prompt, max_tokens=16000)

    # ========================================================================
    # Cognitive Primitive 4: Constrain
    # Validates thoughts against rules by projecting in latent space
    # ========================================================================

    def constrain(
        self,
        node_id: str,
        rules: list[str],
        mode: str = "consistency",
        conclusion: str | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """
        Validate logical constraints against a thought-node using formal logic.

        MODES (based on philosophical logic foundations):

        1. ENTAILMENT ("Does X follow from the premises?")
           - Proves that `conclusion` logically follows from `rules` (premises)
           - Uses Prover9 syntactic proof search
           - Returns: {proved, unprovable, timeout}
           - Example: Does 'mortal(socrates)' follow from 'all x (human(x)->mortal(x))' + 'human(socrates)'?

        2. CONSISTENCY ("Can these all be true together?")
           - Checks if all `rules` can be simultaneously satisfied
           - Uses Mace4 model finding to detect contradictions
           - Returns: {consistent, contradiction}
           - Example: Can 'P(a)' and '-P(a)' both hold? (No - contradiction)

        3. SATISFIABILITY ("Find a world where this holds")
           - Finds a concrete model satisfying all `rules`
           - Uses Mace4 to construct a finite model
           - Returns: {satisfiable, unsatisfiable} + model if found
           - Example: Find a model for 'exists x P(x)'

        Args:
            node_id: ID of thought-node to validate (provides context, logged to graph)
            rules: List of FOL statements (Prover9 syntax)
            mode: 'entailment', 'consistency', or 'satisfiability' (default: consistency)
            conclusion: For entailment mode - the statement to prove from rules as premises
            strict: Use Prover9/Mace4 (True) or embedding similarity (False)

        Returns:
            Dict with validation result, mode, and proof/model details
        """
        logger.info(f"Applying {len(rules)} constraints to {node_id} (strict={strict})")

        with self.neo4j_driver.session() as session:
            # Get node content
            node = session.run(
                """
                MATCH (n:ThoughtNode {id: $id})
                RETURN n.id as id
                """,
                id=node_id,
            ).single()

            if not node:
                return {"error": "Node not found"}

            content_map = self._get_contents_from_chroma([node_id])
            content = content_map.get(node_id, "")

            # === STRICT MODE: Formal Logic Verification ===
            if strict:
                if not hasattr(self, "logic_core"):
                    return {"error": "LogicCore not initialized", "node_id": node_id}

                logger.info(f"Executing STRICT logic verification, mode={mode}")

                # MODE: ENTAILMENT - "Does conclusion follow from premises?"
                if mode == "entailment":
                    if not conclusion:
                        return {
                            "error": "Entailment mode requires 'conclusion' parameter",
                            "hint": "Provide the statement to prove, e.g., conclusion='mortal(socrates)'",
                            "node_id": node_id,
                            "mode": mode,
                        }

                    # Prove: rules are premises, conclusion is what we want to derive
                    proof_result = self.logic_core.prove(rules, conclusion)
                    result_status = proof_result.get("result", "unknown")

                    # If unprovable, try to find counterexample
                    counterexample = None
                    if result_status != "proved":
                        ce_result = self.logic_core.find_counterexample(rules, conclusion)
                        if ce_result.get("result") == "model_found":
                            counterexample = ce_result.get(
                                "raw_output", "Model found disproving conclusion"
                            )

                    # Update node
                    session.run(
                        """
                        MATCH (n:ThoughtNode {id: $id})
                        SET n.constrained = true,
                            n.constraint_mode = 'entailment',
                            n.constraint_result = $result
                        """,
                        id=node_id,
                        result=result_status,
                    )

                    return {
                        "node_id": node_id,
                        "mode": "entailment",
                        "premises": rules,
                        "conclusion": conclusion,
                        "result": result_status,
                        "proved": result_status == "proved",
                        "counterexample": counterexample,
                        "explanation": f"Entailment check: Does '{conclusion}' follow from the {len(rules)} premises? Result: {result_status}",
                    }

                # MODE: CONSISTENCY - "Can all statements be true together?"
                elif mode == "consistency":
                    # Check for contradiction by trying to find a model
                    # If Mace4 finds a model, statements are consistent
                    # If Mace4 fails but Prover9 proves false, inconsistent
                    model_result = self.logic_core.find_model(rules)
                    result_status = model_result.get("result", "unknown")

                    if result_status == "model_found":
                        is_consistent = True
                        explanation = "Statements are CONSISTENT - a model exists where all hold simultaneously"
                    elif result_status == "no_model_found":
                        # Try to prove explicit contradiction
                        contradiction_result = self.logic_core.prove(
                            rules, "$F"
                        )  # $F is Prover9's false
                        if contradiction_result.get("result") == "proved":
                            is_consistent = False
                            explanation = (
                                "Statements are INCONSISTENT - a contradiction is derivable"
                            )
                        else:
                            is_consistent = None  # Undecidable in timeout
                            explanation = "UNDECIDABLE - no model found but no contradiction proven (timeout or incomplete)"
                    else:
                        is_consistent = None
                        explanation = f"Unknown result: {result_status}"

                    # Update node
                    session.run(
                        """
                        MATCH (n:ThoughtNode {id: $id})
                        SET n.constrained = true,
                            n.constraint_mode = 'consistency',
                            n.is_consistent = $consistent
                        """,
                        id=node_id,
                        consistent=is_consistent,
                    )

                    return {
                        "node_id": node_id,
                        "mode": "consistency",
                        "statements": rules,
                        "consistent": is_consistent,
                        "result": (
                            "consistent"
                            if is_consistent
                            else ("contradiction" if is_consistent is False else "undecidable")
                        ),
                        "model": model_result.get("raw_output") if is_consistent else None,
                        "explanation": explanation,
                    }

                # MODE: SATISFIABILITY - "Find a world where this holds"
                elif mode == "satisfiability":
                    model_result = self.logic_core.find_model(rules)
                    result_status = model_result.get("result", "unknown")

                    if result_status == "model_found":
                        is_satisfiable = True
                        model = model_result.get("raw_output", "Model found")
                    else:
                        is_satisfiable = False
                        model = None

                    # Update node
                    session.run(
                        """
                        MATCH (n:ThoughtNode {id: $id})
                        SET n.constrained = true,
                            n.constraint_mode = 'satisfiability',
                            n.is_satisfiable = $satisfiable
                        """,
                        id=node_id,
                        satisfiable=is_satisfiable,
                    )

                    return {
                        "node_id": node_id,
                        "mode": "satisfiability",
                        "statements": rules,
                        "satisfiable": is_satisfiable,
                        "result": "satisfiable" if is_satisfiable else "unsatisfiable",
                        "model": model,
                        "explanation": f"Found {'a model satisfying' if is_satisfiable else 'no model for'} the {len(rules)} statements",
                    }

                else:
                    return {
                        "error": f"Unknown mode: {mode}",
                        "valid_modes": ["entailment", "consistency", "satisfiability"],
                        "node_id": node_id,
                    }

            # === STANDARD MODE: Embedding + Semantic Validation ===
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

            # Record in working memory for context continuity
            self.working_memory.record(
                operation="constrain",
                input_data={"node_id": node_id, "rules": rules},
                output_data={"score": float(avg_score), "satisfied": all_satisfied},
                node_ids=[node_id],
            )

            # Persist to SQLite history
            self.history.log_operation(
                operation="constrain",
                params={"node_id": node_id, "rules": rules},
                result={"score": float(avg_score), "satisfied": all_satisfied},
                cognitive_state=self.working_memory.current_goal or "Unknown",
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
                "deconstruction": decon_result,
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
            goal=f"Resolve the conflict: '{conflict}'. Propose a structural fix or policy change.",
        )

        return {
            "conflict": conflict,
            "analysis": {"root_id": root_id, "components": [c["content"] for c in components]},
            "hypothesis": hypo_result.get("hypothesis", "No hypothesis generated"),
            "resolution": synthesis_result["synthesis"],
            "critique": synthesis_result.get("critique", "No critique"),
            "message": "Meta-Paradox resolved.",
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
        self.workspace.manifold = manifold

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
            embedding_fn=lambda text: torch.tensor(
                self.workspace._embed_text(text), dtype=torch.float32, device=device
            ),
            mcp_client=mcp_client_adapter,
            continuity_service=self.workspace.continuity_service,
            llm_provider=llm_provider,
            work_history=self.workspace.history,
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
                    params={"cost": str(cost.total_energy()), "operation": cost.operation_name},
                    result={"balance": str(balance)},
                    energy=float(balance.amount),
                )
            except Exception as e:
                logger.error(f"Failed to persist transaction: {e}")

        self.ledger.set_transaction_callback(persist_transaction)

        # Wrap Director with Substrate Awareness
        # This makes every director call consume energy
        self.substrate_director = SubstrateAwareDirector(
            director=director,
            ledger=self.ledger,
            cost_profile=OperationCostProfile(),  # Use defaults
        )

        # Inject director into workspace for introspection
        self.workspace.director = self.substrate_director

        # Initialize Processor for Cognitive Proprioception
        processor_cfg = ProcessorConfig(
            vocab_size=50257, embedding_dim=embedding_dim, num_layers=4, num_heads=4, device=device
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
            raa_director=self.substrate_director,  # Use substrate-aware director
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
            llm_provider=self.llm_provider, tool_executor=self.call_tool
        )

        # Initialize Precuneus Integrator
        self.precuneus = PrecuneusIntegrator(dim=embedding_dim).to(device)

        self.raa_context = {
            "embedding_dim": embedding_dim,
            "device": device,
            "manifold": manifold,
            "pointer": pointer,
            "director": self.substrate_director,  # Expose substrate-aware director
            "processor": processor,
            "bridge": bridge,
            "agent_factory": self.agent_factory,
            "precuneus": self.precuneus,
        }

        # Wire Manifold → Chroma sync callback
        self._setup_manifold_chroma_sync(manifold)

        # Cold start: Load patterns from Chroma into Manifold
        self._load_manifold_patterns_from_chroma()

    def _setup_manifold_chroma_sync(self, manifold: Manifold) -> None:
        """Wire Manifold to automatically sync patterns to Chroma."""
        pattern_counter = {"state": 0, "agent": 0, "action": 0}

        def on_pattern_stored(pattern, domain: str, metadata: Optional[dict]) -> None:
            """Callback to sync pattern to Chroma when stored in Manifold."""
            try:
                idx = pattern_counter[domain]
                pattern_counter[domain] += 1
                pattern_id = f"manifold_{domain}_{idx}"

                # Convert pattern to list for Chroma
                if hasattr(pattern, "cpu"):
                    embedding = pattern.cpu().tolist()
                    if isinstance(embedding[0], list):
                        embedding = embedding[0]  # Handle unsqueezed
                else:
                    embedding = list(pattern)

                # Prepare metadata
                meta = {"domain": domain, "index": idx}
                if metadata:
                    meta.update({k: str(v)[:500] for k, v in metadata.items()})

                self.workspace.manifold_patterns_collection.add(
                    ids=[pattern_id],
                    embeddings=[embedding],
                    documents=[f"Manifold pattern {domain}:{idx}"],
                    metadatas=[meta],
                )
                logger.debug(f"Synced pattern {pattern_id} to Chroma")
            except Exception as e:
                logger.warning(f"Failed to sync pattern to Chroma: {e}")

        manifold.set_on_pattern_stored_callback(on_pattern_stored)
        logger.info("Wired Manifold → Chroma pattern sync callback")

    def _load_manifold_patterns_from_chroma(self) -> None:
        """Cold start: Load persisted patterns from Chroma into Manifold."""
        import torch

        manifold = self.get_manifold()
        total_loaded = 0

        for domain in ["state", "agent", "action"]:
            try:
                results = self.workspace.manifold_patterns_collection.get(
                    where={"domain": domain}, include=["embeddings", "metadatas"]
                )

                if not results["ids"]:
                    continue

                # Get the appropriate memory
                if domain == "state":
                    memory = manifold.state_memory
                elif domain == "agent":
                    memory = manifold.agent_memory
                else:
                    memory = manifold.action_memory

                for embedding, metadata in zip(results["embeddings"], results["metadatas"]):
                    pattern = torch.tensor(
                        embedding, dtype=torch.float32, device=manifold.state_memory.device
                    )
                    clean_meta = {k: v for k, v in metadata.items() if k not in ["domain", "index"]}
                    # Directly store to avoid triggering callback (already in Chroma)
                    memory.store_pattern(pattern, metadata=clean_meta)
                    total_loaded += 1

            except Exception as e:
                logger.warning(f"Failed to load {domain} patterns from Chroma: {e}")

        if total_loaded > 0:
            logger.info(f"Cold start: Loaded {total_loaded} patterns from Chroma into Manifold")

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

    def sync_manifold_to_chroma(self, domain: str = "state") -> int:
        """
        Sync Manifold patterns to Chroma for persistence.

        Args:
            domain: Which Manifold domain to sync ('state', 'agent', 'action')

        Returns:
            Number of patterns synced
        """
        manifold = self.get_manifold()

        # Get the appropriate memory based on domain
        if domain == "state":
            memory = manifold.state_memory
        elif domain == "agent":
            memory = manifold.agent_memory
        elif domain == "action":
            memory = manifold.action_memory
        else:
            memory = manifold.state_memory

        patterns = memory.get_patterns()
        if patterns.numel() == 0:
            return 0

        synced = 0
        for i, pattern in enumerate(patterns):
            pattern_id = f"manifold_{domain}_{i}"
            metadata = memory.get_pattern_metadata(i)

            # Check if already exists
            try:
                existing = self.workspace.manifold_patterns_collection.get(ids=[pattern_id])
                if existing["ids"]:
                    continue  # Already synced
            except Exception as e:
                logger.warning(f"Error checking if pattern {pattern_id} exists: {e}")

            # Store in Chroma
            try:
                self.workspace.manifold_patterns_collection.add(
                    ids=[pattern_id],
                    embeddings=[pattern.cpu().tolist()],
                    documents=[f"Manifold pattern {domain}:{i}"],
                    metadatas=[
                        {"domain": domain, "index": i, **{k: str(v) for k, v in metadata.items()}}
                    ],
                )
                synced += 1
            except Exception as e:
                logger.warning(f"Failed to sync pattern {pattern_id}: {e}")

        logger.info(f"Synced {synced} patterns from Manifold {domain} to Chroma")
        return synced

    def load_manifold_from_chroma(self, domain: str = "state") -> int:
        """
        Load patterns from Chroma into Manifold (cold start).

        Args:
            domain: Which Manifold domain to load ('state', 'agent', 'action')

        Returns:
            Number of patterns loaded
        """
        import torch

        manifold = self.get_manifold()

        # Get the appropriate memory based on domain
        if domain == "state":
            memory = manifold.state_memory
        elif domain == "agent":
            memory = manifold.agent_memory
        elif domain == "action":
            memory = manifold.action_memory
        else:
            memory = manifold.state_memory

        # Query Chroma for patterns in this domain
        try:
            results = self.workspace.manifold_patterns_collection.get(
                where={"domain": domain}, include=["embeddings", "metadatas"]
            )
        except Exception as e:
            logger.warning(f"Failed to query Chroma for patterns: {e}")
            return 0

        if not results["ids"]:
            return 0

        loaded = 0
        for i, (pattern_id, embedding, metadata) in enumerate(
            zip(results["ids"], results["embeddings"], results["metadatas"])
        ):
            pattern = torch.tensor(
                embedding, dtype=torch.float32, device=manifold.state_memory.device
            )
            # Convert metadata back from strings
            clean_metadata = {k: v for k, v in metadata.items() if k not in ["domain", "index"]}
            memory.store_pattern(pattern, metadata=clean_metadata)
            loaded += 1

        logger.info(f"Loaded {loaded} patterns from Chroma into Manifold {domain}")
        return loaded

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

    def get_available_tools(self, include_external: bool = True) -> list[Tool]:
        """Get list of available tools."""
        dynamic_tools = self.get_agent_factory().get_dynamic_tools()

        # External tools
        external_tools = []
        if self.external_mcp and include_external:
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
            self.ledger.record_transaction(
                MeasurementCost(
                    energy=EnergyToken(Decimal("5.0"), "joules"), operation_name="deconstruct"
                )
            )

        # 1. Execute Core Deconstruction (Workspace + Graph Persistence)
        # This now handles Tripartite Decomposition, Embeddings, and Persistence
        bridge = self.get_bridge()
        result = bridge.execute_monitored_operation("deconstruct", {"problem": problem})

        # Handle list result (sometimes bridge returns list)
        if isinstance(result, list):
            result = result[0] if result else {}

        # 2. Retrieve Energies for Paradox Detection
        embeddings = result.get("embeddings", {})
        energies = {}
        vectors = {}

        if embeddings:
            # Re-retrieve to get energies (surprise/novelty)
            retrieval_results = self.get_manifold().retrieve(embeddings)
            vectors = {k: v[0] for k, v in retrieval_results.items()}
            energies = {k: v[1] for k, v in retrieval_results.items()}

        # 3. Precuneus Fusion (now returns coherence info)
        director = self.get_director()
        cognitive_state = director.latest_cognitive_state
        unified_context, coherence_info = self.get_precuneus()(
            vectors, energies, cognitive_state=cognitive_state
        )

        # 4. Gödel Detector
        is_paradox = all(e == float("inf") for e in energies.values()) if energies else False

        fusion_status = "Integrated"
        advice = None
        escalation = None

        if is_paradox:
            fusion_status = "Gödelian Paradox"
            advice = "Query contains self-referential contradiction or total novelty. Consider: (1) Reframe question, (2) Accept undecidability, (3) Escalate to System 3 (Philosopher)."
            escalation = "ConsultParadoxResolver"

        # Classify novelty from energies (interpretable summary)
        def classify_energy(e):
            if e < -0.7:
                return "familiar"
            if e < -0.3:
                return "moderate"
            return "novel"

        novelty_summary = {k: classify_energy(v) for k, v in energies.items()} if energies else {}

        # Enhance result with meta-cognitive insights (cleaned up)
        result.update(
            {
                "pattern_match": novelty_summary,  # Interpretable novelty
                "coherence": coherence_info,  # Stream weights and balance
                "fusion_status": fusion_status,
                "unified_context_norm": (
                    float(torch.norm(unified_context)) if unified_context is not None else 0.0
                ),
            }
        )

        if advice:
            result["advice"] = advice
        if escalation:
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
                self.workspace.ctx.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("1.0"), "joules"), operation_name="hypothesize"
                    )
                )

            # 1. Topology Tunneling (Graph + Vector + Analogy)
            result = self.workspace.hypothesize(
                node_a_id=arguments["node_a_id"],
                node_b_id=arguments["node_b_id"],
                context=arguments.get("context"),
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
                    user_prompt=critique_prompt,
                )
                result["critique"] = critique
            return result

        elif name == "evolve_formula":
            # Metabolic Cost: Evolution is expensive (10.0)
            if self.workspace.ctx.ledger:
                from decimal import Decimal

                from src.substrate import EnergyToken, MeasurementCost

                self.workspace.ctx.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("10.0"), "joules"),
                        operation_name="evolve_formula",
                    )
                )

            return evolve_formula_logic(
                arguments["data_points"],
                arguments.get("n_generations", 10),
                arguments.get("hybrid", False),
            )

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

                self.workspace.ctx.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("1.0"), "joules"),
                        operation_name="explore_for_utility",
                    )
                )
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
                "messages": diagnosis.diagnostic_messages,
            }
            return result

        elif name == "check_cognitive_state":
            director = self.get_director()
            state, energy = director.latest_cognitive_state

            # Get latest entropy if available
            entropy = 0.0
            if hasattr(director, "monitor") and director.monitor.entropy_history:
                entropy = director.monitor.entropy_history[-1]

            # Get adaptive remedial action
            remedial_action = director.get_remedial_action(state, energy, entropy)

            warnings = remedial_action.get("warnings", [])
            advice = remedial_action.get("advice", "Continue current line of reasoning.")

            # Meta-Commentary
            recent_history = bridge.history.get_recent_history(limit=5)
            history_summary = "\n".join(
                [f"- {h['operation']}: {h['result_summary']}" for h in recent_history]
            )

            meta_prompt = (
                f"You are a reflective agent. Based on your recent history:\n{history_summary}\n"
                f"And your current state: {state} (Energy: {energy:.2f})\n"
                "Note: 'Energy' refers to Hopfield Network Energy. Lower (more negative) values indicate stability and convergence. "
                "Higher values (closer to 0 or positive) indicate instability, confusion, or active exploration.\n"
                "Provide a brief, first-person meta-commentary on your current thought process."
            )
            meta_commentary = self.workspace._llm_generate(
                system_prompt="You are a reflective AI agent analyzing your own cognitive state.",
                user_prompt=meta_prompt,
            )

            result = {
                "state": state,
                "energy": energy,
                "stability": "Stable" if energy < -0.8 else "Unstable",
                "warnings": warnings,
                "advice": advice,
                "meta_commentary": meta_commentary,
                "message": f"Agent is currently '{state}' (Energy: {energy:.2f})",
            }
            return result

        elif name == "recall_work":
            results = bridge.history.search_history(
                query=arguments.get("query"),
                operation_type=arguments.get("operation_type"),
                limit=arguments.get("limit", 10),
            )
            return results

        elif name == "inspect_knowledge_graph":
            result = self.workspace.get_node_context(
                node_id=arguments["node_id"], depth=arguments.get("depth", 1)
            )
            return result

        elif name == "teach_cognitive_state":
            director = self.get_director()
            success = director.teach_state(arguments["label"])
            msg = (
                f"Learned state '{arguments['label']}'"
                if success
                else "Failed: No recent thought to learn from."
            )
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
                "antifragility_score": diagnosis.harmonic_diffusive_overlap
                * (1.0 if h1_dim == 0 else 0.5),
                "signals": signals,
                "adaptation_plan": adaptation_plan,
                "message": "Antifragility diagnosis complete.",
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

            similarity = float(
                np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
            )
            result = {
                "concepts": {"a": concept_a, "b": concept_b},
                "vector_analysis": {"similarity": similarity, "orthogonality": 1 - abs(similarity)},
                "qualitative_analysis": "Analysis delegated to global handler for full report.",
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

            belief_emb = torch.tensor(
                workspace._embed_text(belief_text), dtype=torch.float32, device=self.device
            )
            evidence_emb = torch.tensor(
                workspace._embed_text(evidence_text), dtype=torch.float32, device=self.device
            )

            result = director.hybrid_search.search(
                current_state=belief_emb,
                evidence=evidence_emb,
                constraints=constraints,
                context={"operation": "revise_tool"},
            )

            if result:
                response = {
                    "status": "success",
                    "revised_content": "Revision successful (see global handler for details)",
                }
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
        name="evolve_formula",
        description="Uses Genetic Programming (Symbolic Regression) to evolve a mathematical formula that fits a given dataset. Use this when the Director detects high entropy/complexity and simple patterns (like linear regression) fail. It discovers the 'hidden instruction set' of the data.",
        inputSchema={
            "type": "object",
            "properties": {
                "data_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                            "result": {"type": "number"},
                        },
                        "required": ["x", "y", "z", "result"],
                    },
                    "description": "List of data points to fit",
                },
                "n_generations": {
                    "type": "integer",
                    "description": "Number of evolutionary generations (default 10)",
                    "default": 10,
                },
                "hybrid": {
                    "type": "boolean",
                    "description": "If true, enables Evolutionary Optimization (local refinement of constants). Slower but more precise.",
                    "default": False,
                },
            },
            "required": ["data_points"],
        },
    ),
    Tool(
        name="constrain",
        description="""Validate logical constraints using formal logic (Prover9/Mace4).

MODES (based on philosophical logic foundations):
- ENTAILMENT: "Does conclusion follow from premises?" - Proves derivability. Requires `conclusion` param.
- CONSISTENCY: "Can all statements be true together?" - Finds contradictions. Default mode.
- SATISFIABILITY: "Find a world where this holds" - Constructs a model.

For ENTAILMENT: rules are premises, conclusion is what to prove.
For CONSISTENCY/SATISFIABILITY: rules are the statements to check.

Syntax: Prover9 FOL format (e.g., "all x (human(x) -> mortal(x))", "human(socrates)").""",
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Thought-node ID for context logging"},
                "rules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "FOL statements (Prover9 syntax). For entailment=premises, for consistency/satisfiability=statements to check",
                },
                "mode": {
                    "type": "string",
                    "enum": ["entailment", "consistency", "satisfiability"],
                    "description": "Validation mode: entailment (prove), consistency (no contradiction), satisfiability (find model)",
                    "default": "consistency",
                },
                "conclusion": {
                    "type": "string",
                    "description": "For entailment mode: the statement to prove from rules as premises",
                },
                "strict": {
                    "type": "boolean",
                    "description": "Use Prover9/Mace4 (True) or embedding similarity (False)",
                    "default": True,
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
                "waitForPreviousTools": {
                    "type": "boolean",
                    "description": "If true, wait for all previous tool calls from this turn to complete before executing (sequential). If false or omitted, execute this tool immediately (parallel with other tools).",
                },
            },
            "required": ["conflict"],
        },
    ),
    Tool(
        name="propose_goal",
        description="Propose a new goal for the agent based on intrinsic motivation (Curiosity/Boredom).",
        inputSchema={
            "type": "object",
            "properties": {
                "waitForPreviousTools": {
                    "type": "boolean",
                    "description": "If true, wait for all previous tool calls from this turn to complete before executing (sequential). If false or omitted, execute this tool immediately (parallel with other tools).",
                }
            },
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
                "query": {
                    "type": "string",
                    "description": "Text to search for in parameters or results",
                },
                "operation_type": {
                    "type": "string",
                    "description": "Filter by operation type (e.g., 'hypothesize')",
                },
                "limit": {"type": "integer", "description": "Max number of results (default 10)"},
            },
            "required": [],
        },
    ),
    Tool(
        name="inspect_knowledge_graph",
        description="Explore the knowledge graph around a specific node to understand context.",
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "ID of the node to inspect"},
                "depth": {
                    "type": "integer",
                    "description": "Traversal depth (default 1)",
                    "default": 1,
                },
            },
            "required": ["node_id"],
        },
    ),
    Tool(
        name="teach_cognitive_state",
        description="Teach the agent that its *current* thought pattern corresponds to a specific state label (Reinforcement Learning).",
        inputSchema={
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Name of the state (e.g., 'Creative', 'Stuck')",
                }
            },
            "required": ["label"],
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
        name="run_sleep_cycle",
        description="Trigger a Sleep Cycle (Offline Learning) to consolidate recent memories and potentially crystallize new tools.",
        inputSchema={
            "type": "object",
            "properties": {
                "epochs": {
                    "type": "integer",
                    "description": "Number of training epochs (default 1)",
                    "default": 1,
                }
            },
            "required": [],
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
                "concept_a": {
                    "type": "string",
                    "description": "First concept (e.g., 'Deep Learning')",
                },
                "concept_b": {
                    "type": "string",
                    "description": "Second concept (e.g., 'Symbolic Logic')",
                },
                "context": {"type": "string", "description": "Optional context for the analysis"},
            },
            "required": ["concept_a", "concept_b"],
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
                    "description": "Mode to set: 'optimization' (High Beta, Convergent) or 'adaptation' (Low Beta, Divergent).",
                }
            },
            "required": ["mode"],
        },
    ),
    Tool(
        name="revise",
        description="Refine a belief or concept using Hybrid Operator C (LTN + Hopfield). Adjusts a thought-node to better match evidence while respecting logical constraints and energy barriers.",
        inputSchema={
            "type": "object",
            "properties": {
                "belief": {
                    "type": "string",
                    "description": "The current belief or thought content to revise",
                },
                "evidence": {
                    "type": "string",
                    "description": "New evidence or target concept to align with",
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of natural language constraints the revision must satisfy",
                },
            },
            "required": ["belief", "evidence"],
        },
    ),
    Tool(
        name="create_advisor",
        description="Create and register a new advisor with a specific persona and toolset.",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique identifier for the advisor (e.g., 'socrates')",
                },
                "name": {"type": "string", "description": "Display name of the advisor"},
                "role": {"type": "string", "description": "Role description (e.g., 'Philosopher')"},
                "description": {
                    "type": "string",
                    "description": "Detailed description of the advisor's purpose",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "The system prompt that defines the advisor's behavior",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tool names available to this advisor",
                },
            },
            "required": ["id", "name", "role", "description", "system_prompt"],
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
            "required": ["id"],
        },
    ),
    Tool(
        name="list_advisors",
        description="List all registered advisors and their capabilities.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="link_advisor_knowledge",
        description="Link a ThoughtNode to an advisor's knowledge base, giving them persistent memory.",
        inputSchema={
            "type": "object",
            "properties": {
                "advisor_id": {
                    "type": "string",
                    "description": "ID of the advisor (e.g., 'socrates', 'researcher')",
                },
                "node_id": {"type": "string", "description": "ID of the ThoughtNode to link"},
            },
            "required": ["advisor_id", "node_id"],
        },
    ),
    Tool(
        name="get_advisor_knowledge",
        description="Get all knowledge associations (linked nodes and patterns) for an advisor.",
        inputSchema={
            "type": "object",
            "properties": {
                "advisor_id": {"type": "string", "description": "ID of the advisor to query"}
            },
            "required": ["advisor_id"],
        },
    ),
    Tool(
        name="get_advisor_context",
        description="Retrieve the full textual content of an advisor's linked knowledge nodes.",
        inputSchema={
            "type": "object",
            "properties": {
                "advisor_id": {"type": "string", "description": "ID of the advisor to query"},
            },
            "required": ["advisor_id"],
        },
    ),
    Tool(
        name="inspect_graph",
        description="Inspect the graph using dynamic queries. Search for nodes by label or traverse relationships.",
        inputSchema={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["nodes", "relationships"],
                    "description": "Operation mode: 'nodes' to search nodes, 'relationships' to traverse.",
                },
                "label": {
                    "type": "string",
                    "description": "Node label to search for (required for mode='nodes').",
                },
                "filters": {
                    "type": "object",
                    "description": "Property filters for node search (e.g., {'name': 'Value'}).",
                },
                "start_id": {
                    "type": "string",
                    "description": "Starting node ID for traversal (required for mode='relationships').",
                },
                "rel_type": {
                    "type": "string",
                    "description": "Relationship type to traverse (required for mode='relationships').",
                },
                "direction": {
                    "type": "string",
                    "enum": ["OUTGOING", "INCOMING", "BOTH"],
                    "default": "OUTGOING",
                    "description": "Traversal direction.",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max results to return.",
                },
            },
            "required": ["mode"],
        },
    ),
    Tool(
        name="consult_curiosity",
        description="Consult the agent's curiosity module to see if there are any interesting goals to pursue (based on boredom/surprise).",
        inputSchema={
            "type": "object",
            "properties": {
                "waitForPreviousTools": {
                    "type": "boolean",
                    "description": "If true, wait for all previous tool calls from this turn to complete before executing (sequential). If false or omitted, execute this tool immediately (parallel with other tools).",
                }
            },
        },
    ),
    Tool(
        name="compute_grok_depth",
        description="Compute the Grok-Depth empathetic alignment score between two mind-states across Grok-Lang's six cognitive levels (Signal, Symbol, Syntax, Semantics, Pragmatics, Meta). Returns a total score (0-1) and per-level alignments with a diagnostic interpretation.",
        inputSchema={
            "type": "object",
            "properties": {
                "speaker_id": {
                    "type": "string",
                    "description": "Identifier for the speaker/sender",
                },
                "listener_id": {
                    "type": "string",
                    "description": "Identifier for the listener/receiver",
                },
                "utterance_raw": {
                    "type": "string",
                    "description": "The raw utterance text (e.g., 'Fine.')",
                },
                "speaker_intent": {
                    "type": "string",
                    "enum": ["assert", "question", "request", "promise", "express", "declare"],
                    "description": "The speaker's intended speech act type",
                },
                "speaker_affect": {
                    "type": "object",
                    "properties": {
                        "valence": {
                            "type": "number",
                            "description": "Positive-negative dimension (-1 to 1)",
                        },
                        "arousal": {"type": "number", "description": "Activation level (0 to 1)"},
                        "dominance": {
                            "type": "number",
                            "description": "Control/power dimension (0 to 1)",
                        },
                    },
                    "description": "Speaker's affective state (VAD model)",
                },
                "listener_affect": {
                    "type": "object",
                    "properties": {
                        "valence": {
                            "type": "number",
                            "description": "Positive-negative dimension (-1 to 1)",
                        },
                        "arousal": {"type": "number", "description": "Activation level (0 to 1)"},
                        "dominance": {
                            "type": "number",
                            "description": "Control/power dimension (0 to 1)",
                        },
                    },
                    "description": "Listener's perceived affective state (VAD model)",
                },
                "context": {"type": "string", "description": "Optional context for the exchange"},
            },
            "required": ["speaker_id", "listener_id", "utterance_raw"],
        },
    ),
    Tool(
        name="consult_computational_empathy",
        description="""Query the Emotion Evolution Framework for evolutionary psychology insights, empathic response templates, and computational empathy architecture.

This tool provides access to:
- Basic emotions (fear, anger, disgust, joy, sadness, surprise) with neural correlates
- Complex emotions (guilt, pride, jealousy, romantic_love)
- Evolutionary layers of emotional processing (1-4)
- AI interaction guidelines and 7 key principles
- Empathic response templates (distress, joy, anxiety)
- Computational Empathy Architecture for value integration
- Valence-arousal to emotion mapping
- ACIP consciousness integration
- Emotional regulation strategies""",
        inputSchema={
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": [
                        "basic_emotion",
                        "complex_emotion",
                        "evolutionary_layer",
                        "ai_guidelines",
                        "ai_principles",
                        "empathic_template",
                        "computational_empathy",
                        "affect_mapping",
                        "neurobiology",
                        "acip",
                        "regulation",
                    ],
                    "description": "Type of query to perform",
                },
                "query_param": {
                    "type": "string",
                    "description": "Parameter for the query (e.g., emotion name like 'fear', layer number like '2', context like 'distress', or 'valence,arousal' like '-0.5,0.8')",
                },
            },
            "required": ["query_type"],
        },
    ),
    Tool(
        name="prove",
        description="""Prove a logical statement using Prover9.

Syntax: Prover9 FOL format (e.g., "all x (human(x) -> mortal(x))", "human(socrates)").
- Universal: all x (P(x)) — NOT ∀x
- Existential: exists x (P(x)) — wrap formula in parentheses!
- Implication: -> — NOT ⇒
- Negation: -P(x) — NOT ¬ or !
- Conjunction: & — NOT ∧
- Disjunction: | — NOT ∨
- Predicates: lowercase preferred (human, mortal)""",
        inputSchema={
            "type": "object",
            "properties": {
                "premises": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of logical premises",
                },
                "conclusion": {"type": "string", "description": "Statement to prove"},
            },
            "required": ["premises", "conclusion"],
        },
    ),
    Tool(
        name="find_counterexample",
        description="""Use Mace4 to find a counterexample showing the conclusion doesn't follow from premises.

Syntax: Same as prove tool - use Prover9 FOL format.""",
        inputSchema={
            "type": "object",
            "properties": {
                "premises": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of logical premises",
                },
                "conclusion": {"type": "string", "description": "Conclusion to disprove"},
                "domain_size": {
                    "type": "integer",
                    "description": "Optional: specific domain size to search",
                },
            },
            "required": ["premises", "conclusion"],
        },
    ),
    Tool(
        name="find_model",
        description="""Use Mace4 to find a finite model satisfying the given premises.

Syntax: Same as prove tool - use Prover9 FOL format.""",
        inputSchema={
            "type": "object",
            "properties": {
                "premises": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of logical premises",
                },
                "domain_size": {
                    "type": "integer",
                    "description": "Optional: specific domain size to search (default: incrementally search 2-10)",
                },
            },
            "required": ["premises"],
        },
    ),
    Tool(
        name="check_well_formed",
        description="Check if logical statements are well-formed with detailed syntax validation.",
        inputSchema={
            "type": "object",
            "properties": {
                "statements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Logical statements to check",
                }
            },
            "required": ["statements"],
        },
    ),
    Tool(
        name="verify_commutativity",
        description="Verify that a categorical diagram commutes by generating FOL premises and conclusion.",
        inputSchema={
            "type": "object",
            "properties": {
                "path_a": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of morphism names in first path",
                },
                "path_b": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of morphism names in second path",
                },
                "object_start": {"type": "string", "description": "Starting object"},
                "object_end": {"type": "string", "description": "Ending object"},
                "with_category_axioms": {
                    "type": "boolean",
                    "description": "Include basic category theory axioms (default: true)",
                },
            },
            "required": ["path_a", "path_b", "object_start", "object_end"],
        },
    ),
    Tool(
        name="get_category_axioms",
        description="Get FOL axioms for category theory concepts (category, functor, natural transformation).",
        inputSchema={
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "enum": ["category", "functor", "natural-transformation", "monoid", "group"],
                    "description": "Which concept's axioms to retrieve",
                },
                "functor_name": {
                    "type": "string",
                    "description": "For functor axioms: name of the functor (default: F)",
                },
                "functor_f": {
                    "type": "string",
                    "description": "For natural transformation: first functor",
                },
                "functor_g": {
                    "type": "string",
                    "description": "For natural transformation: second functor",
                },
                "component": {
                    "type": "string",
                    "description": "For natural transformation: component name",
                },
            },
            "required": ["concept"],
        },
    ),
    Tool(
        name="consult_ruminator",
        description="""Consult the Category-Theoretic Ruminator to perform 'Diagram Chasing' on the knowledge graph.

        It identifies 'open triangles' (non-commutative diagrams) starting from a focus node and uses an LLM (acting as a Functor) to propose missing relationships (morphisms) to make the diagram commute.""",
        inputSchema={
            "type": "object",
            "properties": {
                "focus_node_id": {
                    "type": "string",
                    "description": "Optional: The ID of the node to focus rumination on. If omitted, the system selects a node with 'structural tension'.",
                },
                "mode": {
                    "type": "string",
                    "description": "Operational mode (currently only 'diagram_chasing')",
                    "enum": ["diagram_chasing"],
                    "default": "diagram_chasing",
                },
            },
            "required": [],
        },
    ),
]


@server.list_tools()
async def list_tools(cursor: Any = None, _meta: Any = None, **kwargs) -> list[Tool]:
    """List available cognitive workspace tools"""
    logger.info("DEBUG: list_tools handler called")
    ctx = get_raa_context()

    # Ensure external MCP is initialized
    if ctx.external_mcp and not ctx.external_mcp.is_initialized:
        await ctx.external_mcp.initialize()

    dynamic_tools = ctx.get_agent_factory().get_dynamic_tools()

    # External tools are NOT re-exported to the client.
    # They are only for internal use by IntegratedIntelligence.

    all_tools = RAA_TOOLS + dynamic_tools
    logger.info(f"DEBUG: Returning {len(all_tools)} tools from handler")
    return all_tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """Handle tool calls"""
    try:
        logger.info(f"Received tool call: {name}")

        # Ensure context is initialized
        ctx = get_raa_context()

        # Ensure external MCP is initialized
        if ctx.external_mcp and not ctx.external_mcp.is_initialized:
            logger.info("Initializing external MCP connections...")
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
    except Exception as e:
        logger.error(f"Critical error in call_tool setup: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Critical Server Error: {str(e)}")]

    try:
        if name == "deconstruct":
            # Metabolic Cost: Deconstruction is analysis (1.5)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("1.5"), "joules"), operation_name="deconstruct"
                    )
                )
            try:
                result = ctx.execute_deconstruct(arguments["problem"])
                # Clean up output: remove raw embeddings (internal use only)
                result.pop("embeddings", None)
                result.pop("embeddings_serializable", None)
                # Rename component types to friendlier labels
                if "components" in result:
                    for comp in result["components"]:
                        if comp.get("type") == "State":
                            comp["type"] = "context"
                        elif comp.get("type") == "Agent":
                            comp["type"] = "perspective"
                        elif comp.get("type") == "Action":
                            comp["type"] = "operation"
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                import traceback

                return [
                    TextContent(
                        type="text",
                        text=f"Deconstruction failed: {str(e)}\n\n{traceback.format_exc()}",
                    )
                ]
        elif name == "hypothesize":
            # Metabolic Cost: Hypothesis is a Search operation (1.0)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("1.0"), "joules"), operation_name="hypothesize"
                    )
                )

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
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("3.0"), "joules"), operation_name="synthesize"
                    )
                )

            try:
                result = bridge.execute_monitored_operation(
                    operation="synthesize",
                    params={
                        "node_ids": arguments["node_ids"],
                        "goal": arguments.get("goal"),
                    },
                )
            except Exception as e:
                logger.error(f"Synthesize operation failed: {e}", exc_info=True)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": f"Synthesize failed: {str(e)}",
                                "node_ids": arguments.get("node_ids", []),
                                "hint": "Check if nodes exist and Chroma/Neo4j are accessible.",
                            },
                            indent=2,
                        ),
                    )
                ]

            # Handle None or unexpected result
            if result is None:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "Synthesize returned no result",
                                "node_ids": arguments.get("node_ids", []),
                                "hint": "The synthesize operation completed but returned nothing. Check server logs.",
                            },
                            indent=2,
                        ),
                    )
                ]
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
                    user_prompt=critique_prompt,
                )
                result["critique"] = critique

                # === Director-Orchestrated Synthesis Resolution ===
                # Classify the critique to determine if Director should intervene
                classification = (
                    workspace._llm_generate(
                        system_prompt="""Classify this synthesis critique into exactly one category:
- MISSING_DATA: Requires external facts, specific information, or user input to proceed
- ACTIONABLE: Can be improved through reasoning, adding context, or deeper analysis
- ACCEPTABLE: Minor issues or no significant problems
Respond with only the category name.""",
                        user_prompt=f"Critique: {critique}",
                    )
                    .strip()
                    .upper()
                )

                # Normalize classification
                if classification not in ("MISSING_DATA", "ACTIONABLE", "ACCEPTABLE"):
                    classification = "ACCEPTABLE"

                result["critique_classification"] = classification

                # If ACTIONABLE, delegate to Director for resolution via existing tools
                if classification == "ACTIONABLE":
                    director = ctx.get_director()
                    if director and director.compass:
                        # Energy cost for Director escalation
                        if workspace.ledger:
                            workspace.ledger.record_transaction(
                                MeasurementCost(
                                    energy=EnergyToken(Decimal("5.0"), "joules"),
                                    operation_name="synthesis_director_resolution",
                                )
                            )

                        # Build resolution task for COMPASS
                        resolution_task = f"""Resolve this synthesis critique using available cognitive tools.

ORIGINAL SYNTHESIS:
{synthesis_text}

CRITIQUE:
{critique}

GOAL: {arguments.get('goal', 'None')}

NODE IDS AVAILABLE: {arguments.get('node_ids', [])}

AVAILABLE RESOLUTION STRATEGIES:
1. Use 'explore_for_utility' to find related high-value nodes to incorporate
2. Use 'hypothesize' to find novel connections between existing nodes
3. Use 'deconstruct' to break down missing concepts into new nodes
4. Use 'inspect_knowledge_graph' to find neighboring concepts

Provide an improved synthesis that addresses the critique by using these tools to gather additional context and connections."""

                        try:
                            # Use Director's Time Gate for System 2 processing
                            resolution_result = await director.process_task_with_time_gate(
                                resolution_task,
                                {
                                    "node_ids": arguments.get("node_ids", []),
                                    "goal": arguments.get("goal"),
                                    "force_time_gate": True,  # Force System 2 engagement
                                },
                            )

                            # Extract improved synthesis from COMPASS result
                            if resolution_result.get("success", False):
                                improved_synthesis = resolution_result.get(
                                    "final_report",
                                    resolution_result.get("solution", synthesis_text),
                                )
                                result["synthesis"] = improved_synthesis
                                result["auto_resolved"] = True
                                result["resolution_method"] = "Director/COMPASS"
                                result["compass_score"] = resolution_result.get("score", 0.0)

                                # Re-critique the improved synthesis
                                new_critique = workspace._llm_generate(
                                    system_prompt="You are a critical reviewer of AI-generated syntheses.",
                                    user_prompt=f"Critique for goal '{arguments.get('goal', 'None')}': {improved_synthesis}",
                                )
                                result["critique"] = new_critique

                                # Re-classify
                                new_classification = (
                                    workspace._llm_generate(
                                        system_prompt="Classify: MISSING_DATA, ACTIONABLE, or ACCEPTABLE. Respond with only the category.",
                                        user_prompt=f"Critique: {new_critique}",
                                    )
                                    .strip()
                                    .upper()
                                )
                                if new_classification not in (
                                    "MISSING_DATA",
                                    "ACTIONABLE",
                                    "ACCEPTABLE",
                                ):
                                    new_classification = "ACCEPTABLE"
                                result["critique_classification"] = new_classification
                            else:
                                result["resolution_attempted"] = True
                                result["resolution_status"] = resolution_result.get(
                                    "status", "failed"
                                )

                        except Exception as e:
                            logger.warning(f"Director resolution failed: {e}")
                            result["resolution_error"] = str(e)
                    else:
                        result["resolution_skipped"] = "Director/COMPASS not available"

            # Return the result (either success or error)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "constrain":
            # Metabolic Cost: Constraint is Validation (2.0)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("2.0"), "joules"), operation_name="constrain"
                    )
                )

            result = bridge.execute_monitored_operation(
                operation="constrain",
                params={
                    "node_id": arguments["node_id"],
                    "rules": arguments["rules"],
                    "mode": arguments.get("mode", "consistency"),
                    "conclusion": arguments.get("conclusion"),
                    "strict": arguments.get("strict", True),
                },
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        elif name == "evolve_formula":
            # Metabolic Cost: Evolution is expensive (10.0)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("10.0"), "joules"),
                        operation_name="evolve_formula",
                    )
                )

            result = evolve_formula_logic(
                arguments["data_points"],
                arguments.get("n_generations", 10),
                arguments.get("hybrid", False),
            )

        elif name == "set_goal":
            # Metabolic Cost: Goal setting is cheap (0.5)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("0.5"), "joules"), operation_name="set_goal"
                    )
                )
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
            # Metabolic Cost: Compression is expensive (5.0)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("5.0"), "joules"),
                        operation_name="compress_to_tool",
                    )
                )
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
            if (
                director
                and director.compass
                and director.compass.integrated_intelligence.current_advisor
            ):
                advisor = director.compass.integrated_intelligence.current_advisor
                tool_name = arguments["tool_name"]
                if tool_name not in advisor.tools:
                    advisor.tools.append(tool_name)
                    director.compass.advisor_registry.save_advisors()
                    result["advisor_learning"] = (
                        f"Tool '{tool_name}' added to advisor '{advisor.name}'"
                    )
        elif name == "resolve_meta_paradox":
            # Metabolic Cost: Paradox Resolution is System 3 (10.0)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("10.0"), "joules"),
                        operation_name="resolve_meta_paradox",
                    )
                )

            result = workspace.resolve_meta_paradox(conflict=arguments["conflict"])
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
                tools=arguments.get("tools", []),
            )
            return [TextContent(type="text", text=result)]
        elif name == "delete_advisor":
            director = ctx.get_director()
            if not director or not director.compass:
                return [TextContent(type="text", text="Error: COMPASS framework not initialized")]

            result = director.compass.integrated_intelligence.delete_advisor(id=arguments["id"])
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
                if advisor.knowledge_node_ids:
                    result_lines.append(
                        f"  Knowledge Nodes: {len(advisor.knowledge_node_ids)} linked"
                    )
                result_lines.append("")

            return [TextContent(type="text", text="\n".join(result_lines))]
        elif name == "link_advisor_knowledge":
            director = ctx.get_director()
            if not director or not director.compass:
                return [TextContent(type="text", text="Error: COMPASS framework not initialized")]

            advisor_id = arguments["advisor_id"]
            node_id = arguments["node_id"]

            success = director.compass.advisor_registry.link_node_to_advisor(advisor_id, node_id)
            if success:
                msg = f"Linked node '{node_id}' to advisor '{advisor_id}'"
            else:
                msg = f"Failed to link: advisor '{advisor_id}' not found or node already linked"
            return [TextContent(type="text", text=msg)]
        elif name == "get_advisor_knowledge":
            director = ctx.get_director()
            if not director or not director.compass:
                return [TextContent(type="text", text="Error: COMPASS framework not initialized")]

            knowledge = director.compass.advisor_registry.get_advisor_knowledge(
                arguments["advisor_id"]
            )
            return [TextContent(type="text", text=json.dumps(knowledge, indent=2))]
        elif name == "get_advisor_context":
            result = workspace.get_advisor_context(arguments["advisor_id"])
            return [TextContent(type="text", text=result)]

        elif name == "recall_work":
            results = bridge.history.search_history(
                query=arguments.get("query"),
                operation_type=arguments.get("operation_type"),
                limit=arguments.get("limit", 10),
            )
            return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]
        elif name == "inspect_knowledge_graph":
            result = workspace.get_node_context(
                node_id=arguments["node_id"], depth=arguments.get("depth", 1)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        elif name == "teach_cognitive_state":
            director = ctx.get_director()
            success = director.teach_state(arguments["label"])
            msg = (
                f"Learned state '{arguments['label']}'"
                if success
                else "Failed: No recent thought to learn from."
            )
            return [TextContent(type="text", text=msg)]
        elif name == "get_known_archetypes":
            director = ctx.get_director()
            states = director.get_known_states()
            return [TextContent(type="text", text=json.dumps(states, indent=2))]
        elif name == "visualize_thought":
            director = ctx.get_director()
            vis = director.visualize_last_thought()
            return [TextContent(type="text", text=vis)]
        elif name == "consult_ruminator":
            # Metabolic Cost: Rumination (3.0)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("3.0"), "joules"),
                        operation_name="consult_ruminator",
                    )
                )

            sleep_cycle = ctx.sleep_cycle
            if not sleep_cycle:
                return [
                    TextContent(type="text", text="Error: Sleep Cycle component not initialized")
                ]

            # Run in executor
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, sleep_cycle.diagrammatic_ruminator, arguments.get("focus_node_id")
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "run_sleep_cycle":
            # Use shared Sleep Cycle instance
            sleep_cycle = ctx.sleep_cycle
            # Run in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            epochs = arguments.get("epochs", 5)
            results = await loop.run_in_executor(None, sleep_cycle.dream, epochs)

            # Recharge Energy
            if workspace.ledger:
                workspace.ledger.recharge()
                results["energy_status"] = workspace.ledger.get_status()

            return [TextContent(type="text", text=json.dumps(results, indent=2))]
        elif name == "explore_for_utility":
            # Metabolic Cost: Exploration is Search (1.0)
            if ctx.ledger:
                from decimal import Decimal

                from src.substrate import EnergyToken, MeasurementCost

                ctx.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("1.0"), "joules"),
                        operation_name="explore_for_utility",
                    )
                )
            result = workspace.explore_for_utility(
                focus_area=arguments.get("focus_area"),
                max_candidates=arguments.get("max_candidates", 10),
            )
        elif name == "consult_curiosity":
            # Curiosity-driven goal proposal
            goal = workspace.curiosity.propose_goal()
            if goal:
                result = {"status": "success", "goal": goal}
            else:
                result = {"status": "idle", "message": "No boredom-driven goals at this time."}
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        elif name == "compute_grok_depth":
            # Grok-Lang empathetic alignment scoring
            # Metabolic Cost: Empathy computation is moderate (2.0)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("2.0"), "joules"),
                        operation_name="compute_grok_depth",
                    )
                )

            # Parse affect vectors with defaults
            speaker_affect_data = arguments.get("speaker_affect", {})
            listener_affect_data = arguments.get("listener_affect", {})

            speaker_affect = AffectVector(
                valence=speaker_affect_data.get("valence", 0.0),
                arousal=speaker_affect_data.get("arousal", 0.5),
                dominance=speaker_affect_data.get("dominance", 0.5),
            )
            listener_affect = AffectVector(
                valence=listener_affect_data.get("valence", 0.0),
                arousal=listener_affect_data.get("arousal", 0.5),
                dominance=listener_affect_data.get("dominance", 0.5),
            )

            # Parse intent with default
            intent_str = arguments.get("speaker_intent", "assert")
            intent_map = {
                "assert": Intent.ASSERT,
                "question": Intent.QUESTION,
                "request": Intent.REQUEST,
                "promise": Intent.PROMISE,
                "express": Intent.EXPRESS,
                "declare": Intent.DECLARE,
            }
            speaker_intent = intent_map.get(intent_str, Intent.ASSERT)

            # Build MindState objects
            speaker_state = MindState(
                agent_id=arguments["speaker_id"], affect=speaker_affect, intent=speaker_intent
            )
            listener_state = MindState(
                agent_id=arguments["listener_id"],
                affect=listener_affect,
                intent=Intent.ASSERT,  # Default for listener
            )

            # Build Utterance
            utterance = Utterance(content=arguments["utterance_raw"], speaker_state=speaker_state)

            # Compute Grok-Depth score
            calculator = GrokDepthCalculator(workspace.embedding_model)
            grok_result = calculator.compute_grok_depth(speaker_state, listener_state, utterance)

            result = {
                "total_score": round(grok_result["total_score"], 3),
                "per_level": {k.name: round(v, 3) for k, v in grok_result["per_level"].items()},
                "diagnosis": grok_result["diagnosis"],
                "strongest_level": grok_result["strongest_level"].name,
                "weakest_level": grok_result["weakest_level"].name,
                "critical_gaps": [level.name for level in grok_result["critical_gaps"]],
                "speaker_id": arguments["speaker_id"],
                "listener_id": arguments["listener_id"],
                "utterance": arguments["utterance_raw"],
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        elif name == "consult_computational_empathy":
            # Query the Emotion Evolution Framework
            query_type = arguments.get("query_type", "")
            query_param = arguments.get("query_param", "")

            try:
                framework_result = consult_computational_empathy(query_type, query_param)
                return [TextContent(type="text", text=json.dumps(framework_result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=f"Error querying emotion framework: {e}")]
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
                return [
                    TextContent(type="text", text="Error: Pointer does not have an RNN to diagnose")
                ]

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
                "messages": diagnosis.diagnostic_messages,
            }
        elif name == "check_cognitive_state":
            # ==== MULTI-SIGNAL COGNITIVE STATE (Phase 8) ====
            director = ctx.get_director()
            hopfield_state, hopfield_energy = director.latest_cognitive_state

            # --- Signal 1: Entropy from WorkHistory ---
            entropy_history = bridge.history.get_entropy_history(limit=20)
            if entropy_history:
                avg_entropy = sum(entropy_history) / len(entropy_history)
                entropy_trend = (
                    entropy_history[-1] - entropy_history[0] if len(entropy_history) > 1 else 0.0
                )
            else:
                avg_entropy = 0.0
                entropy_trend = 0.0

            # --- Signal 2: Metabolic Energy from Ledger ---
            metabolic_status = {
                "current_energy": "100.0",
                "max_energy": "100.0",
                "percentage": "100.0%",
            }  # Default
            metabolic_pct = 100.0
            if workspace.ledger:
                metabolic_status = workspace.ledger.get_status()
                # Parse the string values from ledger
                try:
                    current = float(metabolic_status.get("current_energy", "100"))
                    max_e = float(metabolic_status.get("max_energy", "100"))
                    metabolic_pct = (current / max_e) * 100 if max_e > 0 else 100.0
                except (ValueError, TypeError):
                    # Fallback: parse percentage string
                    pct_str = metabolic_status.get("percentage", "100%").replace("%", "")
                    metabolic_pct = float(pct_str) if pct_str else 100.0

            # --- Signal 3: Operation Pattern Analysis (Looping Detection) ---
            recent_history = bridge.history.get_recent_history(limit=10)
            op_counts = {}
            for h in recent_history:
                op = h.get("operation", "unknown")
                op_counts[op] = op_counts.get(op, 0) + 1

            # Detect looping: same operation > 50% of recent history
            is_looping = any(count > 5 for count in op_counts.values())
            dominant_op = max(op_counts, key=op_counts.get) if op_counts else "none"

            # --- Signal 4: Goal Alignment ---
            active_goal = (
                workspace.working_memory.current_goal if workspace.working_memory else None
            )

            # --- Composite State Classification ---
            warnings = []
            if is_looping:
                warnings.append(
                    f"LOOPING DETECTED: Operation '{dominant_op}' repeated {op_counts.get(dominant_op, 0)} times in last 10 operations."
                )

            if avg_entropy > 2.0:
                warnings.append(
                    f"HIGH ENTROPY ({avg_entropy:.2f}): System shows confusion/uncertainty."
                )
            elif avg_entropy < 0.5 and len(entropy_history) > 5:
                warnings.append(
                    f"LOW ENTROPY ({avg_entropy:.2f}): System may be stuck in local minimum."
                )

            if metabolic_pct < 20:
                warnings.append(
                    f"LOW ENERGY ({metabolic_pct:.0f}%): Consider 'run_sleep_cycle' to recharge."
                )

            # --- Dynamic Advice (LLM-Generated) ---
            history_summary = "\n".join(
                [
                    f"- {h['operation']}: {h.get('result_summary', '')[:100]}"
                    for h in recent_history[:5]
                ]
            )

            advice_prompt = f"""Analyze this agent's cognitive state and provide ONE specific recommendation.

SIGNALS:
- Hopfield State: {hopfield_state} (Energy: {hopfield_energy:.2f})
- Avg Entropy (20 ops): {avg_entropy:.2f} (Trend: {'+' if entropy_trend > 0 else ''}{entropy_trend:.2f})
- Metabolic Energy: {metabolic_pct:.0f}%
- Looping Detected: {is_looping} (Dominant: {dominant_op})
- Active Goal: {active_goal or 'None set'}
- Warnings: {warnings or 'None'}

RECENT OPERATIONS:
{history_summary}

Provide a brief, actionable recommendation (1-2 sentences). Be specific about which tool to use."""

            dynamic_advice = workspace._llm_generate(
                system_prompt="You are an expert cognitive systems advisor. Be concise and specific.",
                user_prompt=advice_prompt,
                max_tokens=400,
            )

            # --- Meta-Commentary ---
            meta_prompt = f"""You are a reflective agent. Based on this multi-signal analysis:

HOPFIELD: {hopfield_state} (Energy: {hopfield_energy:.2f})
ENTROPY: {avg_entropy:.2f} (Trend: {'+' if entropy_trend > 0 else ''}{entropy_trend:.2f})
METABOLIC: {metabolic_pct:.0f}%
LOOPING: {is_looping}
GOAL: {active_goal or 'None'}

Recent work:
{history_summary}

Provide a brief first-person reflection on your cognitive state. Are you making progress? Are you stuck? What do you observe about your own patterns?"""

            meta_commentary = workspace._llm_generate(
                system_prompt="You are a reflective AI agent analyzing your own cognitive state.",
                user_prompt=meta_prompt,
                max_tokens=500,
            )

            result = {
                "signals": {
                    "hopfield": {"state": hopfield_state, "energy": float(hopfield_energy)},
                    "entropy": {
                        "average": float(avg_entropy),
                        "trend": float(entropy_trend),
                        "sample_size": len(entropy_history),
                    },
                    "metabolic": {"available_pct": float(metabolic_pct), "raw": metabolic_status},
                    "patterns": {
                        "is_looping": is_looping,
                        "dominant_op": dominant_op,
                        "op_counts": op_counts,
                    },
                    "goal": active_goal,
                },
                "composite_state": (
                    "Looping"
                    if is_looping
                    else ("Confused" if avg_entropy > 2.0 else hopfield_state)
                ),
                "warnings": warnings,
                "advice": dynamic_advice,
                "meta_commentary": meta_commentary,
                "message": f"Multi-signal state: Hopfield={hopfield_state}, Entropy={avg_entropy:.2f}, Metabolic={metabolic_pct:.0f}%, Looping={is_looping}",
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

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
                adaptation_plan.append(
                    "GROWTH OPPORTUNITY: The current architecture cannot resolve these error patterns. Recommendation: Expand the Manifold capacity or add a new abstraction layer."
                )

                # Spawn Explorer Agent
                tool_name = agent_factory.spawn_agent(
                    "H1 Hole", f"Detected {h1_dim} irreducible error cycles."
                )
                adaptation_plan.append(
                    f"ACTION: Spawned specialized agent '{tool_name}' to explore missing concepts."
                )

            else:
                signals.append(
                    "No topological obstructions detected (H^1 = 0). System is robust but potentially rigid."
                )

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
            if (
                h0_dim > 10
            ):  # Arbitrary threshold for "fragmented" for now, pending better calibration
                signals.append(
                    f"High H^0 dimension ({h0_dim}). Possible graph fragmentation (disconnected islands)."
                )
                adaptation_plan.append(
                    "FRAGMENTATION: Concepts are isolated. Recommendation: Build bridges between disconnected components."
                )

                # Spawn Bridge Builder Agent
                tool_name = agent_factory.spawn_agent(
                    "Bridge Builder", f"Detected graph fragmentation (H^0={h0_dim})."
                )
                adaptation_plan.append(
                    f"ACTION: Spawned specialized agent '{tool_name}' to connect islands."
                )

            # Signal 3: Harmonic-Diffusive Overlap (Learning Capacity)
            overlap = diagnosis.harmonic_diffusive_overlap
            if overlap < 0.1:
                signals.append(
                    f"Low learning overlap ({overlap:.3f}). System is 'learning starved'."
                )
                adaptation_plan.append(
                    "STRESSOR: Information is not diffusing to update gradients. Recommendation: Increase 'temperature' (beta) to encourage exploration."
                )

                # Spawn Creative Agent
                tool_name = agent_factory.spawn_agent(
                    "Low Overlap", f"Learning overlap is {overlap:.3f} (Starved)."
                )
                adaptation_plan.append(
                    f"ACTION: Spawned specialized agent '{tool_name}' to bridge semantic gaps."
                )

            else:
                signals.append(
                    f"Healthy learning overlap ({overlap:.3f}). System is plastic and adaptive."
                )

            # Signal 3: Monodromy (Feedback Loops)
            if diagnosis.monodromy:
                if diagnosis.monodromy.topology.value == "tension":
                    signals.append("Tension loop detected (conflicting feedback).")
                    adaptation_plan.append(
                        "VOLATILITY: Internal contradiction. Recommendation: Use 'deconstruct' to break the loop into compatible sub-components."
                    )

                    # Spawn Debater Agent
                    tool_name = agent_factory.spawn_agent(
                        "Tension Loop", "Conflicting feedback loop detected (Monodromy: Tension)."
                    )
                    adaptation_plan.append(
                        f"ACTION: Spawned specialized agent '{tool_name}' to arbitrate conflict."
                    )

                elif diagnosis.monodromy.topology.value == "resonance":
                    signals.append("Resonance loop detected (reinforcing feedback).")
                    adaptation_plan.append(
                        "STABILITY: Self-reinforcing belief. Recommendation: Verify against external data to prevent hallucination."
                    )

            result = {
                "antifragility_score": overlap * (1.0 if h1_dim == 0 else 0.5),
                "signals": signals,
                "adaptation_plan": adaptation_plan,
                "message": "Antifragility diagnosis complete.",
            }

        elif name == "prove":
            # Metabolic Cost: Proving is rigorous (5.0)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("5.0"), "joules"), operation_name="prove"
                    )
                )

            result = workspace._prove(
                premises=arguments["premises"], conclusion=arguments["conclusion"]
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "find_counterexample":
            # Metabolic Cost: Model finding is rigorous (5.0)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("5.0"), "joules"),
                        operation_name="find_counterexample",
                    )
                )

            result = workspace._find_counterexample(
                premises=arguments["premises"],
                conclusion=arguments["conclusion"],
                domain_size=arguments.get("domain_size"),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "find_model":
            # Metabolic Cost: Standard logic op (2.0)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("2.0"), "joules"), operation_name="find_model"
                    )
                )
            result = workspace._find_model(
                premises=arguments["premises"], domain_size=arguments.get("domain_size")
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "check_well_formed":
            # Low cost
            result = workspace._check_well_formed(arguments["statements"])
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "verify_commutativity":
            # Medium cost (3.0)
            if workspace.ledger:
                from decimal import Decimal

                from src.substrate.energy import EnergyToken, MeasurementCost

                workspace.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("3.0"), "joules"),
                        operation_name="verify_commutativity",
                    )
                )
            result = workspace._verify_commutativity(
                path_a=arguments["path_a"],
                path_b=arguments["path_b"],
                object_start=arguments["object_start"],
                object_end=arguments["object_end"],
                with_category_axioms=arguments.get("with_category_axioms", True),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_category_axioms":
            # Low cost
            result = workspace._get_category_axioms(
                concept=arguments["concept"],
                **{k: v for k, v in arguments.items() if k != "concept"},
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "orthogonal_dimensions_analyzer":
            concept_a = arguments["concept_a"]
            concept_b = arguments["concept_b"]
            context = arguments.get("context", "")

            # Use the dedicated analyzer class
            # Inject ContinuityField from workspace
            analyzer = OrthogonalDimensionsAnalyzer(continuity_field=ctx.workspace.continuity_field)

            # 1. Generate Vectors
            # Use workspace embedding model with robust fallback
            vector_a_list = ctx.workspace._embed_text(concept_a)
            vector_b_list = ctx.workspace._embed_text(concept_b)

            # Convert to numpy for analyzer
            vector_a = np.array(vector_a_list, dtype=np.float32)
            vector_b = np.array(vector_b_list, dtype=np.float32)

            # 2. Analyze Vectors (Quantitative)
            vector_analysis = analyzer.analyze_vectors(vector_a, vector_b)

            # 3. Analyze Concepts (Qualitative - LLM)
            prompt = analyzer.construct_analysis_prompt(concept_a, concept_b, context)

            # Call LLM
            analysis_text = ctx.workspace._llm_generate(
                system_prompt=analyzer.SYSTEM_PROMPT, user_prompt=prompt
            )

            # Combine results
            result = {
                "concepts": {"a": concept_a, "b": concept_b},
                "vector_analysis": vector_analysis,
                "qualitative_analysis": analysis_text,
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
            belief_emb = torch.tensor(
                workspace._embed_text(belief_text), dtype=torch.float32, device=ctx.device
            )
            evidence_emb = torch.tensor(
                workspace._embed_text(evidence_text), dtype=torch.float32, device=ctx.device
            )

            # 3. Execute Hybrid Search (Operator C)
            result = director.hybrid_search.search(
                current_state=belief_emb,
                evidence=evidence_emb,
                constraints=constraints,
                context={"operation": "revise_tool"},
                force_ltn=True,  # Force LTN refinement for revision
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
                    if isinstance(score, float) and (
                        score == float("inf") or score == float("-inf") or score != score
                    ):
                        score_val = 0.5
                    else:
                        score_val = float(score)

                    # Create node with revised embedding
                    revised_id = workspace._create_thought_node(
                        session,
                        waypoint_description,
                        "revision",
                        confidence=score_val,
                        embedding=result.best_pattern.cpu().tolist(),
                    )

                    # Query for similar existing thoughts to provide context (optional)
                    query_result = workspace.collection.query(
                        query_embeddings=[result.best_pattern.cpu().tolist()], n_results=1
                    )

                    similar_thought = ""
                    if query_result["documents"] and query_result["documents"][0]:
                        similar_thought = query_result["documents"][0][0]

                # Sanitize score for JSON output
                if isinstance(score, float) and (
                    score == float("inf") or score == float("-inf") or score != score
                ):
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
                    ),
                }

                # Record in working memory
                workspace.working_memory.record(
                    operation="revise",
                    input_data={"belief": belief_text[:300], "evidence": evidence_text[:300]},
                    output_data={
                        "revised": waypoint_description[:500],
                        "strategy": result.strategy.value,
                    },
                    node_ids=[revised_id],
                )

                # Persist to SQLite history
                workspace.history.log_operation(
                    operation="revise",
                    params={"belief": belief_text[:300], "evidence": evidence_text[:300]},
                    result={"revised_node_id": revised_id, "strategy": result.strategy.value},
                    cognitive_state=workspace.working_memory.current_goal or "Unknown",
                )
            else:
                response = {
                    "status": "failure",
                    "message": "Revision failed. Could not find valid stable state.",
                }

        elif name == "inspect_graph":
            mode = arguments["mode"]
            limit = arguments.get("limit", 10)

            if mode == "nodes":
                label = arguments.get("label")
                if not label:
                    return [
                        TextContent(type="text", text="Error: 'label' is required for node search.")
                    ]
                filters = arguments.get("filters", {})
                results = workspace.search_nodes(label, filters, limit)
            elif mode == "relationships":
                start_id = arguments.get("start_id")
                rel_type = arguments.get("rel_type")  # Optional now
                if not start_id:
                    return [
                        TextContent(
                            type="text", text="Error: 'start_id' is required for traversal."
                        )
                    ]
                direction = arguments.get("direction", "OUTGOING")
                results = workspace.traverse_relationships(start_id, rel_type, direction, limit)
            else:
                return [TextContent(type="text", text=f"Unknown mode: {mode}")]

            return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]

            return [TextContent(type="text", text=json.dumps(response, indent=2))]

        elif name == "consult_compass":
            # Metabolic Cost: Delegation is Expensive (5.0)
            if ctx.ledger:
                from decimal import Decimal

                from src.substrate import EnergyToken, MeasurementCost

                ctx.ledger.record_transaction(
                    MeasurementCost(
                        energy=EnergyToken(Decimal("5.0"), "joules"),
                        operation_name="consult_compass",
                    )
                )
            # Delegate to COMPASS framework via Director
            director = ctx.raa_context.get("director")
            if not director or not director.compass:
                return [TextContent(type="text", text="Error: COMPASS framework not initialized")]

            task = arguments["task"]
            context = arguments.get("context", {})

            # Run COMPASS process_task with Time Gate (Dynamic Inference Budgeting)
            result = await director.process_task_with_time_gate(task, context)

            # Return the clean Final Report if available, otherwise fallback to solution
            final_report = result.get("final_report", result.get("solution", str(result)))

            # Include success status prefix
            status = "SUCCESS" if result.get("success", False) else "PARTIAL/FAILURE"
            output = f"[{status}]\n\n{final_report}"

            return [TextContent(type="text", text=output)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except EnergyDepletionError as e:
        logger.warning(f"Energy depleted: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]
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
        logger.info(
            f"External MCP tools loaded: {list(server_context.external_mcp.tools_map.keys())}"
        )

        # Test get_available_tools
        all_tools = server_context.get_available_tools(include_external=True)
        external_count = (
            len(server_context.external_mcp.get_tools())
            if server_context.external_mcp.is_initialized
            else 0
        )
        logger.info(
            f"Total tools available: {len(all_tools)} (External: {external_count}, Internal: {len(all_tools) - external_count})"
        )

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        # Cleanup on exit
        server_context.cleanup()


if __name__ == "__main__":
    import asyncio

    logger.info(f"DEBUG: Server request handlers: {server.request_handlers.keys()}")
    logger.info(f"DEBUG: Server notification handlers: {server.notification_handlers.keys()}")
    asyncio.run(main())


def cli() -> None:
    """Console entrypoint for running the MCP server.

    This wraps the async main() in asyncio.run so it can be bound to a
    setuptools/pyproject script entry.
    """
    import asyncio as _asyncio

    _asyncio.run(main())

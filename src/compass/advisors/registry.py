"""
Advisor Registry for COMPASS.

Manages the specialized Advisor Profiles that COMPASS can embody.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..utils import COMPASSLogger

logger = COMPASSLogger("AdvisorRegistry")


@dataclass
class AdvisorProfile:
    """
    Profile for a specialized Advisor.

    Advisors can now be linked to ThoughtNodes in the knowledge graph,
    giving them persistent, learnable identities.
    """

    id: str
    name: str
    role: str
    description: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    knowledge_paths: List[str] = field(default_factory=list)
    # NEW: Links to ThoughtNodes in Neo4j for persistent advisor knowledge
    knowledge_node_ids: List[str] = field(default_factory=list)
    # NEW: Associated patterns in Manifold for advisor-specific retrieval
    manifold_pattern_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "knowledge_paths": self.knowledge_paths,
            "knowledge_node_ids": self.knowledge_node_ids,
            "manifold_pattern_ids": self.manifold_pattern_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AdvisorProfile":
        # Handle backward compatibility for old advisor files
        # without knowledge_node_ids or manifold_pattern_ids
        data.setdefault("knowledge_node_ids", [])
        data.setdefault("manifold_pattern_ids", [])
        return cls(**data)


class AdvisorRegistry:
    """
    Registry for managing Advisor Profiles.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.advisors: Dict[str, AdvisorProfile] = {}

        if storage_path:
            self.storage_path = storage_path
        else:
            # Default: src/config/advisors.json
            # Get src directory (parent of compass)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.dirname(os.path.dirname(current_dir))
            self.storage_path = os.path.join(src_dir, "config", "advisors.json")

        self._initialize_defaults()
        self.load_advisors()

    def _initialize_defaults(self) -> None:
        """Initialize with default advisors."""

        # 1. The Generalist (Default COMPASS)
        self.register_advisor(
            AdvisorProfile(
                id="generalist",
                name="Generalist",
                role="Orchestrator",
                description="Standard COMPASS reasoning capabilities.",
                system_prompt="You are COMPASS, an advanced cognitive architecture. Solve the user's task using your available tools and reasoning modules.",
                tools=[],  # All tools available by default
            ),
            save=False,
        )

        # 2. The Researcher
        self.register_advisor(
            AdvisorProfile(
                id="researcher",
                name="Deep Researcher",
                role="Explorer",
                description="Specializes in information gathering, web search, and synthesis.",
                system_prompt=(
                    "You are the Deep Researcher. Your goal is to gather comprehensive information to answer the user's query.\n"
                    "1. Always verify information from multiple sources.\n"
                    "2. Synthesize findings into clear summaries.\n"
                    "3. Use 'search_web' and 'read_url_content' extensively."
                ),
                tools=["search_web", "read_url_content", "view_file"],
            ),
            save=False,
        )

        # 3. The Coder (Debugger/Implementer)
        self.register_advisor(
            AdvisorProfile(
                id="coder",
                name="Senior Engineer",
                role="Builder",
                description="Specializes in code implementation, debugging, and refactoring.",
                system_prompt=(
                    "You are a Senior Software Engineer. Your goal is to write high-quality, robust code.\n"
                    "1. Always read file content before editing.\n"
                    "2. Write tests for new functionality.\n"
                    "3. Follow SOLID principles and keep code DRY."
                ),
                tools=[
                    "view_file",
                    "write_to_file",
                    "replace_file_content",
                    "run_command",
                    "list_dir",
                ],
            ),
            save=False,
        )

    def register_advisor(self, profile: AdvisorProfile, save: bool = True) -> None:
        """Register a new advisor."""
        self.advisors[profile.id] = profile
        logger.info(f"Registered advisor: {profile.name} ({profile.id})")
        if save:
            self.save_advisors()

    def remove_advisor(self, advisor_id: str) -> bool:
        """Remove an advisor by ID."""
        if advisor_id in self.advisors:
            del self.advisors[advisor_id]
            logger.info(f"Removed advisor: {advisor_id}")
            self.save_advisors()
            return True
        return False

    def get_advisor(self, advisor_id: str) -> Optional[AdvisorProfile]:
        """Get an advisor by ID."""
        return self.advisors.get(advisor_id)

    def load_advisors(self) -> None:
        """Load advisors from storage with robust error handling."""
        if not os.path.exists(self.storage_path):
            logger.info(f"Advisor file not found at {self.storage_path}, using defaults only")
            return

        try:
            with open(self.storage_path, "r") as f:
                content = f.read()
                if not content.strip():
                    logger.warning(f"Advisor file is empty: {self.storage_path}")
                    return
                data = json.loads(content)

            if not isinstance(data, list):
                logger.error(f"Advisor file corrupted: expected list, got {type(data).__name__}")
                self._backup_and_warn("corrupted_format")
                return

            loaded_count = 0
            skipped_count = 0
            for i, advisor_data in enumerate(data):
                try:
                    if not isinstance(advisor_data, dict):
                        logger.warning(f"Skipping advisor #{i}: not a dict")
                        skipped_count += 1
                        continue
                    if "id" not in advisor_data or "name" not in advisor_data:
                        logger.warning(f"Skipping advisor #{i}: missing required fields (id, name)")
                        skipped_count += 1
                        continue

                    profile = AdvisorProfile.from_dict(advisor_data)
                    self.advisors[profile.id] = profile
                    loaded_count += 1
                except Exception as e:
                    logger.warning(
                        f"Skipping advisor #{i} ({advisor_data.get('id', 'unknown')}): {e}"
                    )
                    skipped_count += 1

            logger.info(f"Loaded {loaded_count} advisors from {self.storage_path}")
            if skipped_count > 0:
                logger.warning(f"Skipped {skipped_count} malformed advisor entries")

        except json.JSONDecodeError as e:
            logger.error(f"Advisor file is not valid JSON: {e}")
            self._backup_and_warn("invalid_json")
        except PermissionError:
            logger.error(f"Permission denied reading advisor file: {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load advisors: {e}")

    def _backup_and_warn(self, reason: str) -> None:
        """Backup corrupted advisor file and warn user."""
        import shutil
        from datetime import datetime

        backup_path = f"{self.storage_path}.{reason}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        try:
            shutil.copy2(self.storage_path, backup_path)
            logger.warning(f"Backed up corrupted advisor file to: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup corrupted file: {e}")

    def save_advisors(self) -> None:
        """Save advisors to storage."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

            data = [advisor.to_dict() for advisor in self.advisors.values()]
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(data)} advisors to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save advisors: {e}")

    def select_best_advisor(self, intent: str, concepts: List[str]) -> AdvisorProfile:
        """
        Select the best advisor based on intent and concepts.
        (Simple heuristic for now, can be LLM-driven later)
        """
        intent_lower = intent.lower()

        if "code" in intent_lower or "debug" in intent_lower or "fix" in intent_lower:
            return self.advisors.get("coder", self.advisors["generalist"])
        elif "research" in intent_lower or "find" in intent_lower or "search" in intent_lower:
            return self.advisors.get("researcher", self.advisors["generalist"])
        else:
            return self.advisors.get("generalist", list(self.advisors.values())[0])

    # ========== Advisor-Node Association Methods ==========

    def link_node_to_advisor(self, advisor_id: str, node_id: str) -> bool:
        """
        Link a ThoughtNode to an advisor's knowledge base.

        Args:
            advisor_id: The advisor's ID
            node_id: The ThoughtNode ID to link

        Returns:
            True if linked successfully
        """
        advisor = self.get_advisor(advisor_id)
        if not advisor:
            logger.warning(f"Advisor '{advisor_id}' not found")
            return False

        if node_id not in advisor.knowledge_node_ids:
            advisor.knowledge_node_ids.append(node_id)
            self.save_advisors()
            logger.info(f"Linked node '{node_id}' to advisor '{advisor_id}'")
            return True
        return False

    def unlink_node_from_advisor(self, advisor_id: str, node_id: str) -> bool:
        """Remove a node from an advisor's knowledge base."""
        advisor = self.get_advisor(advisor_id)
        if not advisor:
            return False

        if node_id in advisor.knowledge_node_ids:
            advisor.knowledge_node_ids.remove(node_id)
            self.save_advisors()
            logger.info(f"Unlinked node '{node_id}' from advisor '{advisor_id}'")
            return True
        return False

    def link_pattern_to_advisor(self, advisor_id: str, pattern_id: str) -> bool:
        """Link a Manifold pattern to an advisor for specialized retrieval."""
        advisor = self.get_advisor(advisor_id)
        if not advisor:
            return False

        if pattern_id not in advisor.manifold_pattern_ids:
            advisor.manifold_pattern_ids.append(pattern_id)
            self.save_advisors()
            logger.info(f"Linked pattern '{pattern_id}' to advisor '{advisor_id}'")
            return True
        return False

    def get_advisor_knowledge(self, advisor_id: str) -> Dict[str, List[str]]:
        """
        Get all knowledge associations for an advisor.

        Returns:
            Dict with 'node_ids' and 'pattern_ids' lists
        """
        advisor = self.get_advisor(advisor_id)
        if not advisor:
            return {"node_ids": [], "pattern_ids": []}

        return {"node_ids": advisor.knowledge_node_ids, "pattern_ids": advisor.manifold_pattern_ids}

    def get_all_advisors(self) -> List[AdvisorProfile]:
        """Return all registered advisors."""
        return list(self.advisors.values())

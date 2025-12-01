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
    """
    id: str
    name: str
    role: str
    description: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    knowledge_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "knowledge_paths": self.knowledge_paths
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AdvisorProfile":
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

    def _initialize_defaults(self):
        """Initialize with default advisors."""

        # 1. The Generalist (Default COMPASS)
        self.register_advisor(AdvisorProfile(
            id="generalist",
            name="Generalist",
            role="Orchestrator",
            description="Standard COMPASS reasoning capabilities.",
            system_prompt="You are COMPASS, an advanced cognitive architecture. Solve the user's task using your available tools and reasoning modules.",
            tools=[] # All tools available by default
        ), save=False)

        # 2. The Researcher
        self.register_advisor(AdvisorProfile(
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
            tools=["search_web", "read_url_content", "view_file"]
        ), save=False)

        # 3. The Coder (Debugger/Implementer)
        self.register_advisor(AdvisorProfile(
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
            tools=["view_file", "write_to_file", "replace_file_content", "run_command", "list_dir"]
        ), save=False)

    def register_advisor(self, profile: AdvisorProfile, save: bool = True):
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

    def load_advisors(self):
        """Load advisors from storage."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for advisor_data in data:
                    profile = AdvisorProfile.from_dict(advisor_data)
                    self.advisors[profile.id] = profile
            logger.info(f"Loaded {len(data)} advisors from {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load advisors: {e}")

    def save_advisors(self):
        """Save advisors to storage."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

            data = [advisor.to_dict() for advisor in self.advisors.values()]
            with open(self.storage_path, 'w') as f:
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

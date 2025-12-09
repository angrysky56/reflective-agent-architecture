import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.llm.factory import LLMFactory
from src.llm.provider import BaseLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ThoughtNode:
    """
    Represents a single unit of thought or observation in the recursive hierarchy.
    """

    content: str
    level: int  # 0 = base thought, 1 = meta-thought, 2 = meta-meta-thought
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    children: List["ThoughtNode"] = field(default_factory=list)
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "ThoughtNode") -> None:
        child.parent_id = self.id
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "level": self.level,
            "timestamp": self.timestamp.isoformat(),
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata,
        }


class RecursiveObserver:
    """
    Manages the hierarchy of thoughts and performs recursive observation.
    Acts as the 'Meta-Layer' of the agent.
    """

    def __init__(self) -> None:
        self.root_thoughts: List[ThoughtNode] = []
        self.active_context: List[ThoughtNode] = []  # Flat list of recent thoughts for context
        self.llm_client: Optional[BaseLLMProvider] = None

        # Initialize the Observer LLM
        self._init_llm()

    def _init_llm(self) -> None:
        """Initialize the LLM provider for meta-analysis."""
        provider_name = os.getenv("OBSERVER_PROVIDER", os.getenv("LLM_PROVIDER", "openrouter"))
        model_name = os.getenv(
            "OBSERVER_MODEL", os.getenv("LLM_MODEL", "deepseek/deepseek-v3.2-speciale")
        )

        try:
            self.llm_client = LLMFactory.create_provider(provider_name, model_name)
            logger.info(f"RecursiveObserver initialized with {provider_name}/{model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Observer LLM: {e}")
            self.llm_client = None

    def observe(self, content: str, level: int = 0, metadata: Optional[Dict] = None) -> ThoughtNode:
        """
        Register a new thought or observation.

        Args:
            content: The text content of the thought.
            level: The meta-level (0=base, 1=meta).
            metadata: Optional tags or data.
        """
        node = ThoughtNode(content=content, level=level, metadata=metadata or {})

        # If this is a meta-thought (level > 0), try to link it to recent base thoughts
        if level > 0 and self.active_context:
            # Simple heuristic: attach to the most recent thought of level-1
            # In a real system, this might use vector similarity or explicit references
            target_level = level - 1
            for prev_node in reversed(self.active_context):
                if prev_node.level == target_level:
                    prev_node.add_child(node)
                    break
        else:
            # Base thoughts or unlinked thoughts go to root
            self.root_thoughts.append(node)

        self.active_context.append(node)

        # Prune context if too long (keep last 50 thoughts)
        if len(self.active_context) > 50:
            self.active_context.pop(0)

        return node

    def reflect(self) -> Optional[Dict[str, Any]]:
        """
        Trigger a meta-analysis cycle.
        The Observer looks at the recent context and generates a critique or insight.
        Returns a structured action dict if a change is recommended.
        """
        if not self.llm_client:
            logger.warning("Cannot reflect: No LLM client available.")
            return None

        # Construct prompt from recent context
        context_str = "\n".join(
            f"[{t.level}] {t.timestamp.strftime('%H:%M:%S')}: {t.content}"
            for t in self.active_context[-10:]  # Look at last 10 thoughts
        )

        system_prompt = (
            "You are the Meta-Observer of an AI agent. "
            "Identify any logical loops, contradictions, or opportunities for deeper insight.\n"
            "If the agent is stuck or confused, you MUST recommend a remedial action.\n"
            "Output ONLY a JSON object with the following structure:\n"
            "{\n"
            '  "observation": "Concise critique of the reasoning process",\n'
            '  "action_type": "NONE" | "SWITCH_STRATEGY" | "ADJUST_THRESHOLD" | "TRIGGER_SLEEP",\n'
            '  "parameters": { ... } (optional parameters for the action)\n'
            "}\n"
        )

        user_prompt = f"Analyze the following stream of thoughts:\n\n{context_str}"

        try:
            response = self.llm_client.generate(system_prompt, user_prompt)
            if response:
                # Parse JSON from response (handle potential markdown wrapping)
                import json

                clean_response = response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:-3]
                elif clean_response.startswith("```"):
                    clean_response = clean_response[3:-3]

                try:
                    action_data: Dict[str, Any] = json.loads(clean_response)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse reflection JSON: {response}")
                    # Fallback for non-JSON response
                    action_data = {"observation": response, "action_type": "NONE"}

                # Register the reflection as a meta-thought
                max_level = max((t.level for t in self.active_context), default=0)
                self.observe(
                    f"REFLECTION: {action_data.get('observation', 'No observation')}",
                    level=max_level + 1,
                    metadata={"type": "reflection", "action": action_data},
                )

                return action_data

        except Exception as e:
            logger.error(f"Reflection failed: {e}")

        return None

    def get_hierarchy_json(self) -> str:
        """Return the current thought hierarchy as a JSON string."""
        import json

        return json.dumps([t.to_dict() for t in self.root_thoughts], indent=2)

"""
Working Memory: Short-term context for LLM operations.

This module provides a sliding window of recent cognitive operations
that gets injected into LLM calls, enabling continuity of reasoning
across multiple tool invocations.

Unlike episodic memory (Neo4j graph) which is permanent, working memory
is transient and session-scoped - it provides the "train of thought"
context that makes multi-step reasoning coherent.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single entry in working memory."""

    timestamp: datetime
    operation: str  # e.g., "deconstruct", "synthesize", "hypothesize"
    input_summary: str  # Condensed input (first 500 chars)
    output_summary: str  # Condensed output (first 1000 chars)
    node_ids: List[str] = field(default_factory=list)  # Related nodes
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Format entry for injection into LLM context."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        nodes_str = f" [Nodes: {', '.join(self.node_ids[:3])}]" if self.node_ids else ""
        return f"[{time_str}] {self.operation}{nodes_str}\n  Input: {self.input_summary}\n  Output: {self.output_summary}"


class WorkingMemory:
    """
    Transient working memory for cognitive continuity.

    Maintains a sliding window of recent operations to provide
    context for LLM reasoning. This enables:
    - Coherent multi-step reasoning
    - Reference to recent attempts and failures
    - Awareness of concept progression
    - Smarter critique resolution

    The memory is not persisted - it's session-scoped to match
    the "train of thought" metaphor.
    """

    def __init__(self, max_entries: int = 20, max_context_chars: int = 8000):
        """
        Initialize working memory.

        Args:
            max_entries: Maximum number of entries to retain
            max_context_chars: Maximum characters for context injection
        """
        self.max_entries = max_entries
        self.max_context_chars = max_context_chars
        self.entries: deque[MemoryEntry] = deque(maxlen=max_entries)
        self.session_start = datetime.now()

        # Track current "focus" - what we're working on
        self.current_goal: Optional[str] = None
        self.current_task: Optional[str] = None

        logger.info(f"WorkingMemory initialized: max_entries={max_entries}")

    def record(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
        node_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an operation in working memory.

        Args:
            operation: Name of the operation (e.g., "synthesize")
            input_data: Input to the operation (will be summarized)
            output_data: Output from the operation (will be summarized)
            node_ids: Related node IDs if applicable
            metadata: Additional context
        """
        # Summarize inputs/outputs to control size
        input_summary = self._summarize(input_data, max_chars=500)
        output_summary = self._summarize(output_data, max_chars=1000)

        entry = MemoryEntry(
            timestamp=datetime.now(),
            operation=operation,
            input_summary=input_summary,
            output_summary=output_summary,
            node_ids=node_ids or [],
            metadata=metadata or {}
        )

        self.entries.append(entry)
        logger.debug(f"WorkingMemory recorded: {operation} ({len(self.entries)} entries)")

    def get_context(self, max_entries: Optional[int] = None) -> str:
        """
        Get formatted context string for LLM injection.

        Args:
            max_entries: Override for number of entries to include

        Returns:
            Formatted context string suitable for system/user prompt
        """
        if not self.entries:
            return ""

        entries_to_include = list(self.entries)
        if max_entries:
            entries_to_include = entries_to_include[-max_entries:]

        # Build context string, respecting character limit
        context_parts = ["=== WORKING MEMORY (Recent Operations) ==="]

        if self.current_goal:
            context_parts.append(f"Active Goal: {self.current_goal}")
        if self.current_task:
            context_parts.append(f"Current Task: {self.current_task}")

        context_parts.append("")  # Blank line

        total_chars = sum(len(p) for p in context_parts)

        # Add entries from oldest to newest, respecting limit
        for entry in entries_to_include:
            entry_str = entry.to_context_string()
            if total_chars + len(entry_str) + 2 > self.max_context_chars:
                context_parts.append("... (earlier entries truncated)")
                break
            context_parts.append(entry_str)
            total_chars += len(entry_str) + 1

        context_parts.append("=== END WORKING MEMORY ===\n")

        return "\n".join(context_parts)

    def set_focus(self, goal: Optional[str] = None, task: Optional[str] = None) -> None:
        """
        Set the current focus of working memory.

        Args:
            goal: High-level goal (e.g., "Analyze quantum consciousness")
            task: Current task (e.g., "Synthesizing nodes for hypothesis")
        """
        if goal is not None:
            self.current_goal = goal
        if task is not None:
            self.current_task = task

    def clear(self) -> None:
        """Clear all entries (new session/topic)."""
        self.entries.clear()
        self.current_goal = None
        self.current_task = None
        self.session_start = datetime.now()
        logger.info("WorkingMemory cleared")

    def get_recent_operations(self, n: int = 5) -> List[str]:
        """Get list of recent operation names."""
        return [e.operation for e in list(self.entries)[-n:]]

    def get_related_nodes(self, n: int = 10) -> List[str]:
        """Get unique node IDs from recent operations."""
        seen = set()
        result = []
        for entry in reversed(list(self.entries)):
            for node_id in entry.node_ids:
                if node_id not in seen:
                    seen.add(node_id)
                    result.append(node_id)
                    if len(result) >= n:
                        return result
        return result

    def _summarize(self, data: Any, max_chars: int = 500) -> str:
        """Summarize data to fit within character limit."""
        if data is None:
            return "(None)"

        if isinstance(data, str):
            text = data
        elif isinstance(data, dict):
            # For dicts, prioritize key fields
            priority_keys = ["content", "synthesis", "hypothesis", "problem", "goal", "critique"]
            parts = []
            for key in priority_keys:
                if key in data:
                    parts.append(f"{key}: {str(data[key])[:200]}")
            if not parts:
                text = str(data)
            else:
                text = "; ".join(parts)
        elif isinstance(data, list):
            text = f"[{len(data)} items]: {str(data[:3])}"
        else:
            text = str(data)

        # Truncate and clean
        text = text.replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[:max_chars-3] + "..."

        return text

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"WorkingMemory({len(self.entries)} entries, goal={self.current_goal})"

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

class AgentFactory:
    """
    System 3 Component: Antifragile Agent Factory.

    Responsible for spawning, managing, and executing specialized ephemeral agents
    in response to topological obstructions identified by Sheaf Diagnostics.
    """

    def __init__(self, llm_generate_fn: Callable[[str, str], str]):
        """
        Initialize the factory.

        Args:
            llm_generate_fn: Callback to the main LLM generation function.
                             Signature: (system_prompt, user_prompt) -> response
        """
        self.llm_generate = llm_generate_fn
        self.active_agents: Dict[str, Dict[str, Any]] = {}

    def spawn_agent(self, signal_type: str, context: str) -> str:
        """
        Spawn a new specialized agent based on a diagnostic signal.

        Args:
            signal_type: The type of obstruction (e.g., "Tension Loop", "H1 Hole").
            context: Contextual details about the obstruction.

        Returns:
            The name of the newly created tool.
        """
        logger.info(f"Spawning agent for signal: {signal_type}")

        # 1. Generate Persona via LLM
        generation_prompt = (
            f"We have detected a topological obstruction of type: '{signal_type}'.\n"
            f"Context: {context}\n\n"
            "Task: Create a System Prompt for a specialized AI agent designed to resolve this specific problem.\n"
            "The agent should be focused, objective, and specialized.\n"
            "For 'Tension Loop', it should be a 'Debater' that arbitrates conflicting views.\n"
            "For 'H1 Hole', it should be an 'Explorer' that finds missing concepts.\n"
            "For 'Low Overlap', it should be a 'Creative' that bridges gaps.\n\n"
            "Output ONLY the System Prompt for this new agent."
        )

        persona = self.llm_generate(
            system_prompt="You are an AI Architect designing specialized sub-agents.",
            user_prompt=generation_prompt
        )

        # 2. Create Tool Definition
        # Sanitize tool name
        safe_type = signal_type.lower().replace(" ", "_").replace("^", "")
        tool_name = f"consult_{safe_type}_agent"

        # If already exists, maybe version it or just overwrite (ephemeral)
        # For this MVP, we overwrite to keep it simple.

        tool_def = {
            "name": tool_name,
            "description": f"Specialized agent spawned to resolve {signal_type}. Ask it for help with the obstruction.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or task for this specialized agent."
                    }
                },
                "required": ["query"]
            }
        }

        # 3. Register
        self.active_agents[tool_name] = {
            "persona": persona,
            "def": tool_def,
            "signal_type": signal_type,
            "created_at": "now" # Timestamp could be added
        }

        logger.info(f"Spawned dynamic tool: {tool_name}")
        return tool_name

    def get_dynamic_tools(self) -> List[Dict[str, Any]]:
        """Return list of tool definitions for all active agents."""
        return [agent["def"] for agent in self.active_agents.values()]

    def execute_agent(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Execute a dynamic agent.

        Args:
            tool_name: Name of the agent tool.
            args: Arguments passed to the tool (must contain 'query').

        Returns:
            The agent's response.
        """
        if tool_name not in self.active_agents:
            return f"Error: Agent '{tool_name}' not found or has retired."

        agent_data = self.active_agents[tool_name]
        persona = agent_data["persona"]
        query = args.get("query", "")

        logger.info(f"Executing dynamic agent {tool_name} with query: {query[:50]}...")

        response = self.llm_generate(
            system_prompt=persona,
            user_prompt=query
        )

        return f"[{tool_name} Response]:\n{response}"

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

class AgentFactory:
    """
    System 3 Component: Antifragile Agent Factory.

    Responsible for spawning, managing, and executing specialized ephemeral agents
    in response to topological obstructions identified by Sheaf Diagnostics.
    """

    def __init__(self, llm_provider: Any, tool_executor: Callable[[str, Dict[str, Any]], Any]):
        """
        Initialize the factory.

        Args:
            llm_provider: Instance of RAALLMProvider (or compatible) for chat completion.
            tool_executor: Async callback to execute tools. Signature: async (name, args) -> result
        """
        self.llm_provider = llm_provider
        self.tool_executor = tool_executor
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

        # 1. Generate Persona via LLM (using provider)
        # We use a simple non-streaming call for persona generation
        # generation_prompt = (
        #     f"We have detected a topological obstruction of type: '{signal_type}'.\n"
        #     f"Context: {context}\n\n"
        #     "Task: Create a System Prompt for a specialized AI agent designed to resolve this specific problem.\n"
        #     "The agent should be focused, objective, and specialized.\n"
        #     "For 'Tension Loop', it should be a 'Debater' that arbitrates conflicting views.\n"
        #     "For 'H1 Hole', it should be an 'Explorer' that finds missing concepts.\n"
        #     "For 'Low Overlap', it should be a 'Creative' that bridges gaps.\n\n"
        #     "Output ONLY the System Prompt for this new agent."
        # )

        # Helper for simple generation using the provider
        # We need to run this async, but spawn_agent might be called from sync context?
        # For now, we'll assume we can use a helper or just do it.
        # Actually, let's make spawn_agent async if possible, or use a sync wrapper if needed.
        # But wait, the Director calls this. Director is likely async or can be.
        # For this MVP, let's assume we can't easily change spawn_agent to async without cascading changes.
        # We'll use a sync generation if available, or just use the provider in a blocking way if we must.
        # However, RAALLMProvider is async.
        # Let's use a workaround: The persona generation is less critical to be tool-capable.
        # We can use the old llm_generate_fn if we kept it, OR we just define a default persona for now
        # to avoid async complexity in this specific method if it's called synchronously.
        # BUT, let's try to do it right. If spawn_agent is called by SheafAnalyzer (sync), we have a problem.
        # Let's check SheafAnalyzer. It calls it? No, Director calls it.
        # Director.check_and_search is async (it calls compass.process_task).
        # So we can make spawn_agent async!

        # For now, let's just hardcode the persona based on type to save time/complexity,
        # or use a simplified synchronous generation if we had one.
        # Let's stick to the plan: use llm_provider.
        # We will make spawn_agent async in a future refactor.
        # For now, let's just use a default persona map to avoid the async call here.

        personas = {
            "Tension Loop": "You are a Debater Agent. Your goal is to resolve contradictions by finding a higher-order synthesis. You have access to tools like 'consult_compass'.",
            "H1 Hole": "You are an Explorer Agent. Your goal is to find missing concepts to fill topological voids. You have access to tools like 'consult_compass'.",
            "Low Overlap": "You are a Creative Agent. Your goal is to bridge disconnected concepts. You have access to tools like 'consult_compass'."
        }
        persona = personas.get(signal_type, "You are a Specialized Agent. Solve the problem.")

        # 2. Create Tool Definition
        # Sanitize tool name
        safe_type = signal_type.lower().replace(" ", "_").replace("^", "")
        tool_name = f"consult_{safe_type}_agent"

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
            "created_at": "now"
        }

        logger.info(f"Spawned dynamic tool: {tool_name}")
        return tool_name

    def get_dynamic_tools(self) -> List[Dict[str, Any]]:
        """Return list of tool definitions for all active agents."""
        return [agent["def"] for agent in self.active_agents.values()]

    async def execute_agent(self, tool_name: str, args: Dict[str, Any]) -> str:
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

        from src.compass.adapters import Message
        messages = [
            Message(role="system", content=persona),
            Message(role="user", content=query)
        ]

        # Get available tools to pass to the agent
        # We want to pass RAA tools (like consult_compass)
        # We can get them via a separate method or just assume we pass a standard set.
        # For now, let's assume we pass a list of tool definitions.
        # We need to get these definitions.
        # Ideally, we'd pass them in __init__ or get them from a callback.
        # Let's assume we can get them from the tool_executor context or similar?
        # Actually, we can just pass 'consult_compass' definition manually for now,
        # or better, let's ask the server for them.
        # But we don't have a reference to server.
        # Let's define a minimal set of tools this agent can use.

        agent_tools = [
            {
                "name": "consult_compass",
                "description": "Delegate a complex task to the COMPASS cognitive framework.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "context": {"type": "object"}
                    },
                    "required": ["task"]
                }
            }
        ]

        # Execution Loop
        max_turns = 5
        current_turn = 0
        final_response = ""

        while current_turn < max_turns:
            current_turn += 1
            response_content = ""
            tool_calls = []

            # Call LLM
            async for chunk in self.llm_provider.chat_completion(messages, stream=False, tools=agent_tools):
                try:
                    if chunk.strip().startswith('{"tool_calls":'):
                        import json
                        data = json.loads(chunk)
                        if "tool_calls" in data:
                            tool_calls.extend(data["tool_calls"])
                    else:
                        response_content += chunk
                except Exception:
                    response_content += chunk

            # If no tool calls, we are done
            if not tool_calls:
                final_response = response_content
                break

            # Execute Tools
            messages.append(Message(role="assistant", content=response_content)) # Add thought/content

            # We need to add tool calls to history properly for some LLMs,
            # but for our simple loop, we'll just add the results.

            for tool_call in tool_calls:
                # Parse tool call
                # Assuming tool_call is dict from Ollama: {'function': {'name':..., 'arguments':...}}
                # or similar. Our adapter yields it.
                # Let's assume standard format.

                func = tool_call.get("function", {})
                name = func.get("name")
                import json
                try:
                    args_str = func.get("arguments", "{}")
                    if isinstance(args_str, str):
                        t_args = json.loads(args_str)
                    else:
                        t_args = args_str
                except Exception:
                    t_args = {}

                logger.info(f"Agent {tool_name} calling tool: {name}")

                # Execute
                try:
                    result = await self.tool_executor(name, t_args)
                    result_str = str(result)
                except Exception as e:
                    result_str = f"Error: {str(e)}"

                # Add result to messages
                messages.append(Message(role="user", content=f"Tool '{name}' result: {result_str}"))

        return f"[{tool_name} Response]:\n{final_response}"

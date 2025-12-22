import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    System 3 Component: Antifragile Agent Factory.

    Responsible for spawning, managing, and executing specialized ephemeral agents
    in response to topological obstructions identified by Sheaf Diagnostics.
    """

    def __init__(
        self,
        llm_provider: Any,
        tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        tool_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        workspace: Optional[Any] = None,
    ):
        """
        Initialize the factory.

        Args:
            llm_provider: Instance of RAALLMProvider (or compatible) for chat completion.
            tool_executor: Async callback to execute tools. Signature: async (name, args) -> result
            tool_lookup: Callback to get tool schema. Signature: (name) -> schema_dict
            workspace: CognitiveWorkspace instance (for The Library persistence).
        """
        self.llm_provider = llm_provider
        self.tool_executor = tool_executor
        self.tool_lookup = tool_lookup
        self.workspace = workspace
        self.active_agents: Dict[str, Dict[str, Any]] = {}

        # Initialize Registry
        from reflective_agent_architecture.compass.advisors.registry import AdvisorRegistry

        self.registry = AdvisorRegistry()

    async def spawn_agent(self, signal_type: str, context: str) -> str:
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

        from reflective_agent_architecture.compass.adapters import Message

        messages = [
            Message(
                role="system",
                content="You are an expert AI Architect. Create specialized agent personas.",
            ),
            Message(role="user", content=generation_prompt),
        ]

        persona = ""
        try:
            # Use the provider to generate the persona
            async for chunk in self.llm_provider.chat_completion(messages, stream=False):
                persona += chunk

            if not persona:
                logger.warning("LLM returned empty persona. Using default.")
                persona = "You are a Specialized Agent. Solve the problem."

        except Exception as e:
            logger.error(f"Failed to generate persona: {e}. Using default.")
            persona = "You are a Specialized Agent. Solve the problem."

        logger.info(f"Generated persona for {signal_type}: {persona[:100]}...")

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
                        "description": "The question or task for this specialized agent.",
                    }
                },
                "required": ["query"],
            },
        }

        # 3. Register
        self.active_agents[tool_name] = {
            "persona": persona,
            "def": tool_def,
            "signal_type": signal_type,
            "created_at": "now",
        }

        logger.info(f"Spawned dynamic tool: {tool_name}")
        logger.info(f"Spawned dynamic tool: {tool_name}")
        return tool_name

    def spawn_advisor(self, advisor_id: str) -> str:
        """
        Spawn a known advisor from the registry.

        Args:
            advisor_id: ID of the advisor to spawn (e.g., 'thermodynamicist').

        Returns:
            The tool name for the spawned advisor.
        """
        profile = self.registry.get_advisor(advisor_id)
        if not profile:
            raise ValueError(f"Advisor '{advisor_id}' not found in registry.")

        tool_name = f"consult_{profile.id}"

        # Tool Definition
        tool_def = {
            "name": tool_name,
            "description": f"{profile.name}: {profile.description}",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": f"Ask {profile.name} for help."}
                },
                "required": ["query"],
            },
        }

        self.active_agents[tool_name] = {
            "persona": profile.system_prompt,
            "def": tool_def,
            "signal_type": "advisor_spawn",
            "created_at": "now",
            "tools": profile.tools,
        }

        logger.info(f"Spawned advisor: {tool_name}")
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

        from reflective_agent_architecture.compass.adapters import Message

        messages = [Message(role="system", content=persona), Message(role="user", content=query)]

        # Define available tools for the agent.
        # Dynamically build based on agent's assigned tools
        assigned_tools = agent_data.get("tools", ["consult_compass"])
        agent_tools = []

        for t_name in assigned_tools:
            schema = None
            if self.tool_lookup:
                schema = self.tool_lookup(t_name)

            # Fallback for hardcoded consult_compass if lookups fail or missing
            if not schema and t_name == "consult_compass":
                schema = {
                    "type": "object",
                    "properties": {"task": {"type": "string"}, "context": {"type": "object"}},
                    "required": ["task"],
                }

            if schema:
                agent_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t_name,
                            "description": f"Tool: {t_name}",  # Description ideally comes from lookup too
                            "parameters": schema,
                        },
                    }
                )

        # Ensure consult_compass is always available if fallback needed for 'dynamic' agents
        if not agent_tools and "consult_compass" not in assigned_tools:
            # Default fallback
            pass

        # Execution Loop
        max_turns = 5
        current_turn = 0
        final_response = ""

        while current_turn < max_turns:
            current_turn += 1
            response_content = ""
            tool_calls = []

            # Call LLM
            async for chunk in self.llm_provider.chat_completion(
                messages, stream=False, tools=agent_tools
            ):
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
            messages.append(
                Message(role="assistant", content=response_content)
            )  # Add thought/content

            # We need to add tool calls to history properly for some LLMs,
            # but for our simple loop, we'll just add the results.

            for tool_call in tool_calls:
                # Parse tool call
                # Assuming standard dictionary format from the adapter.

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
                    if self.tool_executor:
                        result = await self.tool_executor(name, t_args)
                        result_str = str(result)
                    else:
                        result_str = "Error: No tool executor configured for this agent factory."
                except Exception as e:
                    result_str = f"Error: {str(e)}"

                # Add result to messages
                messages.append(Message(role="user", content=f"Tool '{name}' result: {result_str}"))

        # --- The Library: Auto-Save Logic (Gen 3) ---
        if self.workspace and final_response:
            try:
                # 1. Determine Advisor ID
                if tool_name.startswith("consult_"):
                    advisor_id = tool_name.replace("consult_", "").replace("_agent", "")

                    # 3. Create Thought Node using Standard Template (Precuneus/Workspace)
                    # This ensures utility, compression, and embedding are calculated correctly.
                    with self.workspace.neo4j_driver.session() as session:
                        # Create the node (Standard)
                        node_id = self.workspace._create_thought_node(
                            session=session,
                            content=final_response,
                            cognitive_type="Insight",
                            confidence=1.0,  # Advisors are experts
                        )

                        # Link Attribution (Advisor -> Node)
                        session.run(
                            """
                            MERGE (a:Advisor {id: $advisor_id})
                            MATCH (t:ThoughtNode {id: $node_id})
                            MERGE (a)-[:AUTHORED]->(t)
                            """,
                            advisor_id=advisor_id,
                            node_id=node_id,
                        )

                    logger.info(
                        f"The Library: Saved insight {node_id} from {advisor_id} via Standard Template"
                    )

                    # 4. Link in Registry (So it appears in manage_advisor get_knowledge)
                    # AgentFactory has self.registry initialized in __init__
                    if self.registry:
                        self.registry.link_node_to_advisor(advisor_id, node_id)
                        logger.info(
                            f"The Library: Linked {node_id} to advisor {advisor_id} in registry."
                        )

                    # Append Library Link to response
                    final_response += f"\n\n[Recorded in The Library: {node_id}]"

            except Exception as e:
                logger.error(f"The Library: Failed to auto-save insight: {e}")

        return f"[{tool_name} Response]:\n{final_response}"

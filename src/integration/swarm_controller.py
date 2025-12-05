import asyncio
import logging
from typing import Any, Dict, List

from .agent_factory import AgentFactory

logger = logging.getLogger(__name__)

class SwarmController:
    """
    Orchestrates "Hive Mind" dynamics by managing multiple specialized advisors
    in parallel and synthesizing their outputs.
    """

    def __init__(self, agent_factory: AgentFactory):
        self.factory = agent_factory

    async def run_swarm(self, task: str, advisor_ids: List[str], context: str = "") -> str:
        """
        Run a swarm of advisors on a specific task.

        Args:
            task: The problem statement or query.
            advisor_ids: List of advisor IDs to recruit (e.g., ["linearist", "thermodynamicist"]).
            context: Optional context string.

        Returns:
            A synthesized consensus string.
        """
        logger.info(f"Initiating Swarm for task: {task[:50]}... with advisors: {advisor_ids}")

        # 1. Recruit (Spawn) Advisors
        active_tools = []
        for aid in advisor_ids:
            try:
                tool_name = self.factory.spawn_advisor(aid)
                active_tools.append(tool_name)
            except Exception as e:
                logger.error(f"Failed to spawn advisor {aid}: {e}")

        if not active_tools:
            return "Swarm Failure: No advisors could be recruited."

        # 2. Broadcast Task (Parallel Execution)
        # We use the factory's execute_agent method
        async def query_agent(tool_name: str) -> Dict[str, str]:
            try:
                response = await self.factory.execute_agent(tool_name, {"query": f"{context}\n\nTask: {task}"})
                return {"agent": tool_name, "response": response}
            except Exception as e:
                return {"agent": tool_name, "error": str(e)}

        tasks = [query_agent(tool) for tool in active_tools]
        results = await asyncio.gather(*tasks)

        # 3. Synthesize Results
        synthesis = await self._synthesize_consensus(task, results)

        return synthesis

    async def _synthesize_consensus(self, task: str, results: List[Dict[str, str]]) -> str:
        """
        Synthesize divergent advisor outputs into a coherent consensus.
        """
        prompt = f"Task: {task}\n\n"
        prompt += "Advisor Responses:\n"

        for res in results:
            agent = res.get("agent", "Unknown")
            content = res.get("response", "")
            error = res.get("error")

            if error:
                prompt += f"--- {agent} (FAILED) ---\n{error}\n\n"
            else:
                prompt += f"--- {agent} ---\n{content}\n\n"

        prompt += (
            "You are the Swarm Synthesizer. Analyze the above responses.\n"
            "1. Identify points of consensus.\n"
            "2. Highlight unique insights from specific perspectives.\n"
            "3. Resolve contradictions using Hegelian Dialectics (Thesis-Antithesis-Synthesis).\n"
            "4. Provide a final unified answer."
        )

        from src.compass.adapters import Message
        messages = [
            Message(role="system", content="You are an expert dialetic synthesizer."),
            Message(role="user", content=prompt)
        ]

        # Use the factory's provider for synthesis
        final_output = ""
        try:
            async for chunk in self.factory.llm_provider.chat_completion(messages, stream=False):
                final_output += chunk
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            final_output = "Error generating synthesis."

        return final_output

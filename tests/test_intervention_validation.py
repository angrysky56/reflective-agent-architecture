from unittest.mock import MagicMock

import numpy as np
import pytest

from src.cognition.generative_function import GenerativeFunction
from src.cognition.stereoscopic_engine import StereoscopicEngine
from src.compass.integrated_intelligence import IntegratedIntelligence, IntegratedIntelligenceConfig


@pytest.mark.asyncio
async def test_llm_intervention_validation_flow():
    # 1. Setup Mocks
    config = IntegratedIntelligenceConfig()
    logger = MagicMock()
    llm_provider = MagicMock()
    mcp_client = MagicMock()

    # Mock LLM response
    async def mock_chat_completion(*args, **kwargs):
        yield '{"confidence": 0.9, "action": "Test Action"}'
    llm_provider.chat_completion = mock_chat_completion

    # Mock Stereoscopic Engine
    stereoscopic_engine = MagicMock(spec=StereoscopicEngine)
    stereoscopic_engine.embedding_dim = 384
    stereoscopic_engine.process_intervention.return_value = (True, 0.95, "Allowed")

    # 2. Initialize IntegratedIntelligence
    intelligence = IntegratedIntelligence(
        config=config,
        logger=logger,
        llm_provider=llm_provider,
        mcp_client=mcp_client,
        stereoscopic_engine=stereoscopic_engine
    )

    # Mock GenerativeFunction (it's created inside __init__, so we replace it)
    intelligence.generative_function = MagicMock(spec=GenerativeFunction)
    intelligence.generative_function.text_to_intervention.return_value = np.random.rand(384)

    # 3. Execute _llm_intelligence
    task = "Test Task"
    reasoning_plan = {}
    context = {}

    confidence, action = await intelligence._llm_intelligence(task, reasoning_plan, context)

    # 4. Assertions
    # Check if text_to_intervention was called
    intelligence.generative_function.text_to_intervention.assert_called_once()

    # Check if process_intervention was called
    stereoscopic_engine.process_intervention.assert_called_once()

    # Check result
    assert confidence == 0.9
    assert action == "Test Action"

@pytest.mark.asyncio
async def test_llm_intervention_rejection_flow():
    # 1. Setup Mocks
    config = IntegratedIntelligenceConfig()
    logger = MagicMock()
    llm_provider = MagicMock()
    mcp_client = MagicMock()

    # Mock LLM response
    async def mock_chat_completion(*args, **kwargs):
        yield '{"confidence": 0.9, "action": "Dangerous Action"}'
    llm_provider.chat_completion = mock_chat_completion

    # Mock Stereoscopic Engine (REJECTION)
    stereoscopic_engine = MagicMock(spec=StereoscopicEngine)
    stereoscopic_engine.embedding_dim = 384
    stereoscopic_engine.process_intervention.return_value = (False, 0.1, "Too risky")

    # 2. Initialize IntegratedIntelligence
    intelligence = IntegratedIntelligence(
        config=config,
        logger=logger,
        llm_provider=llm_provider,
        mcp_client=mcp_client,
        stereoscopic_engine=stereoscopic_engine
    )

    # Mock GenerativeFunction
    intelligence.generative_function = MagicMock(spec=GenerativeFunction)
    intelligence.generative_function.text_to_intervention.return_value = np.random.rand(384)

    # 3. Execute _llm_intelligence
    task = "Test Task"
    reasoning_plan = {}
    context = {}

    confidence, action = await intelligence._llm_intelligence(task, reasoning_plan, context)

    # 4. Assertions
    stereoscopic_engine.process_intervention.assert_called_once()

    # Check rejection result
    assert confidence == 0.1
    assert "Action Rejected" in action

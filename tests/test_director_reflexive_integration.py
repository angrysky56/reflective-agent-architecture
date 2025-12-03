from unittest.mock import MagicMock, patch

import pytest
import torch

from src.director.director_core import DirectorConfig, DirectorMVP
from src.director.reflexive_closure_engine import ReflexiveClosureEngine


class MockManifold:
    def __init__(self):
        self.beta = 1.0

    def compute_adaptive_beta(self, entropy, max_entropy=None):
        return 0.5

    def set_beta(self, beta):
        self.beta = beta

@pytest.fixture
def mock_director():
    config = DirectorConfig(
        enable_reflexive_closure=True,
        reflexive_analysis_interval=10
    )
    manifold = MockManifold()

    # Mock dependencies
    with patch('src.director.director_core.EntropyMonitor') as MockMonitor, \
         patch('src.director.director_core.SheafAnalyzer') as MockSheaf, \
         patch('src.director.director_core.MatrixMonitor') as MockMatrix, \
         patch('src.director.director_core.LTNRefiner') as MockLTN, \
         patch('src.director.director_core.HybridSearchStrategy') as MockHybrid, \
         patch('src.director.director_core.COMPASS') as MockCOMPASS, \
         patch('src.integration.agent_factory.AgentFactory') as MockAgentFactory:

        director = DirectorMVP(manifold=manifold, config=config)

        # Configure COMPASS mock
        mock_compass_instance = MockCOMPASS.return_value
        mock_compass_instance.omcd_controller.determine_resource_allocation.return_value = {
            "amount": 50.0,
            "confidence": 0.8
        }

        # Mock the reflexive engine to track calls
        director.reflexive_engine = MagicMock(spec=ReflexiveClosureEngine)
        director.reflexive_engine.get_threshold.return_value = 1.5
        director.reflexive_engine.record_intervention_start.return_value = "test_episode_id"

        # Mock monitor to return high entropy (clash)
        director.monitor.check_logits.return_value = (True, 2.5)
        director.monitor.get_threshold.return_value = 2.0

        # Mock search result
        mock_result = MagicMock()
        mock_result.best_pattern = torch.tensor([1.0, 0.0])
        mock_result.selection_score = 0.9
        director.search = MagicMock(return_value=mock_result)

        return director

def test_reflexive_initialization(mock_director):
    """Test that ReflexiveClosureEngine is initialized."""
    assert mock_director.reflexive_engine is not None

def test_check_entropy_uses_dynamic_threshold(mock_director):
    """Test that _check_entropy uses the engine's threshold."""
    # Setup
    mock_director.reflexive_engine.get_threshold.return_value = 3.0

    # Test with entropy 2.5 (should be False if threshold is 3.0)
    is_clash = mock_director._check_entropy(entropy=2.5, energy=0.0)

    # Verify
    mock_director.reflexive_engine.get_threshold.assert_called()
    assert not is_clash

def test_intervention_recording_flow(mock_director):
    """Test that interventions are recorded during check_and_search."""
    # Setup context
    current_state = torch.tensor([0.1, 0.1])
    logits = torch.tensor([0.1, 0.9])

    # Execute
    mock_director.check_and_search(current_state, logits)

    # Verify start recorded
    mock_director.reflexive_engine.record_intervention_start.assert_called_once()
    call_kwargs = mock_director.reflexive_engine.record_intervention_start.call_args[1]
    assert call_kwargs['entropy'] == 2.5
    assert call_kwargs['intervention_type'] == "search"

    # Verify end recorded
    mock_director.reflexive_engine.record_intervention_end.assert_called_once()
    end_kwargs = mock_director.reflexive_engine.record_intervention_end.call_args[1]
    assert end_kwargs['episode_id'] == "test_episode_id"
    assert end_kwargs['task_success'] is True
    assert end_kwargs['outcome_quality'] == 0.9

def test_intervention_failure_recording(mock_director):
    """Test that failed searches are recorded as failures."""
    # Setup search failure
    mock_director.search.return_value = None

    # Execute
    mock_director.check_and_search(torch.tensor([0.1]), torch.tensor([0.1]))

    # Verify end recorded with failure
    mock_director.reflexive_engine.record_intervention_end.assert_called_once()
    end_kwargs = mock_director.reflexive_engine.record_intervention_end.call_args[1]
    assert end_kwargs['task_success'] is False
    assert end_kwargs['outcome_quality'] == 0.0

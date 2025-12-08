from unittest.mock import MagicMock, patch

import pytest

from src.integration.sleep_cycle import SleepCycle


@pytest.fixture
def mock_workspace():
    ws = MagicMock()
    ws.neo4j_driver = MagicMock()
    ws.ruminator_provider = MagicMock()
    return ws

def test_diagrammatic_ruminator_success(mock_workspace):
    """Verify Diagrammatic Ruminator successfully identifies and completes an open triangle."""
    sc = SleepCycle(workspace=mock_workspace)

    # Mock Neo4j Session
    mock_session = MagicMock()
    mock_workspace.neo4j_driver.session.return_value.__enter__.return_value = mock_session

    # 1. Mock Focus Node Query (Step 1)
    # Returns a node with out_degree >= 2
    mock_focus_result = MagicMock()
    mock_focus_result.single.return_value = {"id": "node_A", "name": "ConceptA"}

    # 2. Mock Open Triangle Query (Step 2)
    # Returns B and C which are connected to A but not to each other
    mock_span_result = MagicMock()
    mock_span_result.single.return_value = {
        "b_id": "node_B", "b_name": "ConceptB", "b_content": "ContentB",
        "c_id": "node_C", "c_name": "ConceptC", "c_content": "ContentC"
    }

    # 3. Setup side_effect for session.run to return different results for different queries
    def side_effect(*args, **kwargs):
        query = args[0].strip()
        if "MATCH (n:ConceptNode)" in query and "out_degree >= 2" in query:
            return mock_focus_result
        if "MATCH (a:ConceptNode {id: $focus_id})" in query and "NOT (b)--(c)" in query:
            return mock_span_result
        return MagicMock()

    mock_session.run.side_effect = side_effect

    # 4. Mock LLM Response (The Functor)
    # Proposes a specific morphism B->C
    mock_workspace.ruminator_provider.generate.return_value = (
        "RELATION: Yes | TYPE: IMPLIES | DIRECTION: B->C | REASON: Transitive property via A"
    )

    # Execute
    result = sc.diagrammatic_ruminator(focus_node_id=None)

    # Verify
    assert result["status"] == "success"
    assert result["operation"] == "diagram_completion"
    assert result["inferred_morphism"] == "B->C : IMPLIES"

    # Verify Graph Update
    # Check if the last call was the MERGE operation
    calls = mock_session.run.call_args_list
    merge_called = False
    for call in calls:
        query = call[0][0]
        params = call[1]
        if "MERGE (s)-[r:IMPLIES]->(t)" in query:
            assert params["sid"] == "node_B"
            assert params["tid"] == "node_C"
            assert params["reason"] == "Transitive property via A"
            merge_called = True
            break

    assert merge_called, "Neo4j MERGE query was not called with correct parameters"

def test_diagrammatic_ruminator_commutes(mock_workspace):
    """Verify Ruminator handles fully commutative diagrams (no open triangles)."""
    sc = SleepCycle(workspace=mock_workspace)

    mock_session = MagicMock()
    mock_workspace.neo4j_driver.session.return_value.__enter__.return_value = mock_session

    # 1. Mock Focus Node Found
    mock_focus_result = MagicMock()
    mock_focus_result.single.return_value = {"id": "node_A", "name": "ConceptA"}

    # 2. Mock Open Triangle Query returns None (Diagram already commutes)
    mock_span_result = MagicMock()
    mock_span_result.single.return_value = None

    def side_effect(*args, **kwargs):
        query = args[0].strip()
        if "out_degree >= 2" in query:
            return mock_focus_result
        if "NOT (b)--(c)" in query:
            return mock_span_result
        return MagicMock()

    mock_session.run.side_effect = side_effect

    # Execute
    result = sc.diagrammatic_ruminator(focus_node_id="node_A")

    # Verify
    assert result["status"] == "idle"
    assert "Diagram commutes" in result["message"]
    # LLM should not be called
    mock_workspace.ruminator_provider.generate.assert_not_called()

def test_diagrammatic_ruminator_explicit_focus(mock_workspace):
    """Verify Ruminator works with an explicit focus_node_id."""
    sc = SleepCycle(workspace=mock_workspace)
    mock_session = MagicMock()
    mock_workspace.neo4j_driver.session.return_value.__enter__.return_value = mock_session

    # Mock finding the explicit node
    mock_node_result = MagicMock()
    mock_node_result.single.return_value = {"name": "ExplicitConcept"}

    # Mock finding span (success case)
    mock_span_result = MagicMock()
    mock_span_result.single.return_value = {
        "b_id": "B", "b_name": "B", "b_content": "C_B",
        "c_id": "C", "c_name": "C", "c_content": "C_C"
    }

    def side_effect(*args, **kwargs):
        query = args[0].strip()
        if "MATCH (n:ConceptNode {id: $id})" in query:
            return mock_node_result
        return mock_span_result

    mock_session.run.side_effect = side_effect

    # Mock LLM
    mock_workspace.ruminator_provider.generate.return_value = "RELATION: No" # No relationship found

    # Execute
    result = sc.diagrammatic_ruminator(focus_node_id="explicit_id_123")

    # Verify
    assert result["status"] == "success"
    assert result["operation"] == "diagram_verified"
    assert "No direct morphism" in result["message"]

def test_diagrammatic_ruminator_with_verification(mock_workspace):
    """Verify that CategoryTheoryEngine is invoked for formal verification."""
    sc = SleepCycle(workspace=mock_workspace)

    mock_session = MagicMock()
    mock_workspace.neo4j_driver.session.return_value.__enter__.return_value = mock_session

    # 1. Mock Focus and Span
    mock_focus = MagicMock()
    mock_focus.single.return_value = {"id": "A", "name": "ConceptA"}

    mock_span = MagicMock()
    mock_span.single.return_value = {
        "b_id": "B", "b_name": "ConceptB", "b_content": "ContentB",
        "c_id": "C", "c_name": "ConceptC", "c_content": "ContentC"
    }

    def side_effect(*args, **kwargs):
        query = args[0].strip()
        if "out_degree >= 2" in query: return mock_focus
        if "NOT (b)--(c)" in query: return mock_span
        return MagicMock()
    mock_session.run.side_effect = side_effect

    # 2. Mock LLM Response
    mock_workspace.ruminator_provider.generate.return_value = (
        "RELATION: Yes | TYPE: IMPLIES | DIRECTION: B->C | REASON: Logic"
    )

    # 3. Patch Category Theory Engine
    with patch('src.integration.sleep_cycle.CategoryTheoryEngine') as MockEngine:
        mock_engine_instance = MockEngine.return_value
        mock_engine_instance.verify_triangle_commutativity.return_value = {"result": "proved"}
        mock_engine_instance.generate_commutativity_report.return_value = "Formal Report"

        result = sc.diagrammatic_ruminator(focus_node_id=None)

        # Verify Engine Call
        mock_engine_instance.verify_triangle_commutativity.assert_called_once()
        _, kwargs = mock_engine_instance.verify_triangle_commutativity.call_args
        # Check inferred type passed to verification
        assert kwargs['path_bc_type'] == "IMPLIES"

        # Verify Report in Result
        assert result["verification"] == "proved"
        assert result["report"] == "Formal Report"

        # Verify Graph Update includes verified=True
        calls = mock_session.run.call_args_list
        merge_call = [c for c in calls if "MERGE" in c[0][0]][0]
        assert merge_call[1]["verified"] == "True"

"""
Integration Test: COMPASS × RAA Phase 1
Tests oMCD resource allocation with RAA cognitive states.
"""

import sys
sys.path.insert(0, '/home/ty/Repositories/ai_workspace/reflective-agent-architecture/src')

from compass.compass_integration import COMPASSOrchestrator, CognitiveState, ResourceAllocation
from compass.config import COMPASSConfig


def test_resource_allocation_low_complexity():
    """Test allocation for simple tasks."""
    print("\n=== Test 1: Low Complexity Task ===")
    
    orchestrator = COMPASSOrchestrator()
    
    state = CognitiveState(
        energy=-0.3,
        entropy=0.2,
        confidence=0.8,
        stability="Stable",
        state_type="Focused"
    )
    
    allocation = orchestrator.allocate_resources(
        cognitive_state=state,
        task_complexity=0.2,
        importance=10.0
    )
    
    print(f"Optimal Resources: {allocation.optimal_resources:.2f}")
    print(f"Expected Benefit: {allocation.expected_benefit:.2f}")
    print(f"Expected Cost: {allocation.expected_cost:.2f}")
    print(f"Recommendation: {allocation.recommendation}")
    
    assert allocation.optimal_resources < 50, "Low complexity should require few resources"
    assert allocation.recommendation == "LOW_EFFORT: Simple heuristic sufficient"
    print("✅ Test passed!")


def test_resource_allocation_high_complexity():
    """Test allocation for complex tasks."""
    print("\n=== Test 2: High Complexity Task ===")
    
    orchestrator = COMPASSOrchestrator()
    
    state = CognitiveState(
        energy=-0.6,
        entropy=0.8,
        confidence=0.3,
        stability="Unstable",
        state_type="Looping"
    )
    
    allocation = orchestrator.allocate_resources(
        cognitive_state=state,
        task_complexity=0.9,
        importance=15.0
    )
    
    print(f"Optimal Resources: {allocation.optimal_resources:.2f}")
    print(f"Expected Benefit: {allocation.expected_benefit:.2f}")
    print(f"Expected Cost: {allocation.expected_cost:.2f}")
    print(f"Recommendation: {allocation.recommendation}")
    
    # Check System 3 escalation
    should_escalate = orchestrator.should_escalate_to_system3(allocation)
    print(f"System 3 Escalation: {should_escalate}")
    
    assert allocation.optimal_resources > 50, "High complexity should require more resources"
    print("✅ Test passed!")


def test_custom_configuration():
    """Test with custom COMPASS configuration."""
    print("\n=== Test 3: Custom Configuration ===")
    
    config = COMPASSConfig()
    config.omcd.alpha = 0.15  # Lower cost per unit effort
    config.omcd.R = 20.0      # Higher importance weight
    
    orchestrator = COMPASSOrchestrator(config)
    
    state = CognitiveState(
        energy=-0.5,
        entropy=0.5,
        confidence=0.5,
        stability="Stable",
        state_type="Broad"
    )
    
    allocation = orchestrator.allocate_resources(
        cognitive_state=state,
        task_complexity=0.5,
        importance=20.0
    )
    
    print(f"Optimal Resources: {allocation.optimal_resources:.2f}")
    print(f"Confidence at Optimal: {allocation.confidence_at_optimal:.3f}")
    print("✅ Test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("COMPASS × RAA Integration Tests (Phase 1)")
    print("=" * 60)
    
    try:
        test_resource_allocation_low_complexity()
        test_resource_allocation_high_complexity()
        test_custom_configuration()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

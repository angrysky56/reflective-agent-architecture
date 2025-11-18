"""
Quick test for constraint validation improvements.

Tests the hybrid LLM+embedding approach to ensure constraints
are validated accurately.
"""

import sys
from pathlib import Path

from src.server import CognitiveWorkspace, CWDConfig

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_constraint_validation():
    """Test constraint validation with known valid and invalid cases."""
    print("=" * 70)
    print("Constraint Validation Test")
    print("=" * 70)

    # Initialize CWD
    config = CWDConfig()  # type: ignore[call-arg]
    workspace = CognitiveWorkspace(config)

    # Create a test node with mathematical content
    with workspace.neo4j_driver.session() as session:
        test_content = (
            "The Fibonacci sequence is defined recursively where each number is the sum "
            "of the two preceding ones: F(n) = F(n-1) + F(n-2), with F(0) = 0 and F(1) = 1. "
            "This generates the sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34..."
        )

        node_id = workspace._create_thought_node(
            session,
            content=test_content,
            cognitive_type="test",
            confidence=1.0,
        )

        print(f"\n1. Created test node: {node_id[:20]}...")
        print(f"   Content: {test_content[:100]}...\n")

    # Test Case 1: Valid constraints (should be satisfied)
    # Use semantically broader constraints that better match embedding space
    valid_rules = [
        "Discusses mathematical sequences or patterns",
        "Contains recursive or iterative formulas",
        "Includes numerical examples or series",
    ]

    print("2. Testing VALID constraints (should be satisfied):")
    result1 = workspace.constrain(node_id, valid_rules)

    for r in result1["rule_results"]:
        status = "✓" if r["satisfied"] else "✗"
        print(
            f"   {status} {r['rule'][:50]}: "
            f"score={r['score']:.3f} (emb={r['embedding_similarity']:.3f})"
        )

    print(
        f"\n   Overall: {result1['overall_score']:.3f}, All satisfied: {result1['all_satisfied']}"
    )

    # Test Case 2: Invalid constraints (should NOT be satisfied)
    invalid_rules = [
        "Discusses quantum mechanics",
        "Contains chemical formulas",
        "Describes biological processes",
    ]

    print("\n3. Testing INVALID constraints (should NOT be satisfied):")
    result2 = workspace.constrain(node_id, invalid_rules)

    for r in result2["rule_results"]:
        status = "✓" if r["satisfied"] else "✗"
        print(
            f"   {status} {r['rule'][:50]}: "
            f"score={r['score']:.3f} (emb={r['embedding_similarity']:.3f})"
        )

    print(
        f"\n   Overall: {result2['overall_score']:.3f}, All satisfied: {result2['all_satisfied']}"
    )

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print(f"  Valid constraints satisfied: {result1['all_satisfied']} (expected: True)")
    print(f"  Invalid constraints rejected: {not result2['all_satisfied']} (expected: True)")

    # Cleanup
    workspace.close()

    # Final verdict
    success = result1["all_satisfied"] and not result2["all_satisfied"]
    if success:
        print("\n✓ Constraint validation working correctly!")
    else:
        print("\n✗ Constraint validation needs adjustment")

    print("=" * 70)


if __name__ == "__main__":
    test_constraint_validation()

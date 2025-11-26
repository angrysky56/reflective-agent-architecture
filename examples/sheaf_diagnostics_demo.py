"""
Example: Sheaf Diagnostics Integration with RAA

Demonstrates how to use sheaf-theoretic analysis for:
1. Detecting topological obstructions to learning
2. Diagnosing stuck states beyond entropy monitoring
3. Principled escalation decisions

Based on "Sheaf Cohomology of Linear Predictive Coding Networks" (Seely, 2025)
"""

import torch
import torch.nn as nn

from src.director import (
    CognitiveTopology,
    Director,
    DirectorConfig,
    SheafAnalyzer,
    SheafConfig,
    SheafDiagnostics,
    create_supervision_target,
)
from src.manifold import HopfieldConfig, ModernHopfieldNetwork


def example_basic_diagnosis():
    """
    Basic example: Analyze a simple feedforward network.
    """
    print("=" * 60)
    print("Example 1: Basic Feedforward Network Diagnosis")
    print("=" * 60)
    
    analyzer = SheafAnalyzer(SheafConfig(device="cpu"))
    
    # Create a simple 3-layer network
    # Input (4) -> Hidden (8) -> Hidden (8) -> Output (4)
    weights = [
        torch.randn(8, 4),
        torch.randn(8, 8),
        torch.randn(4, 8)
    ]
    
    # Create supervision target
    x = torch.randn(4)
    y = torch.randn(4)
    b = create_supervision_target(x, y, weights)
    
    # Run full diagnosis
    diagnosis = analyzer.full_diagnosis(weights, target_error=b)
    
    print(f"\nCohomology Analysis:")
    print(f"  H^1 dimension: {diagnosis.cohomology.h1_dimension}")
    print(f"  Can fully resolve: {diagnosis.cohomology.can_fully_resolve}")
    
    print(f"\nLearning Metrics:")
    print(f"  Harmonic-Diffusive Overlap: {diagnosis.harmonic_diffusive_overlap:.4f}")
    print(f"  Learning can proceed: {diagnosis.learning_can_proceed}")
    print(f"  Escalation recommended: {diagnosis.escalation_recommended}")
    
    if diagnosis.diagnostic_messages:
        print(f"\nDiagnostic Messages:")
        for msg in diagnosis.diagnostic_messages:
            print(f"  - {msg}")
    
    return diagnosis


def example_feedback_analysis():
    """
    Example: Analyze network with feedback connections.
    
    This demonstrates monodromy analysis for detecting resonance vs tension.
    """
    print("\n" + "=" * 60)
    print("Example 2: Feedback Loop Analysis (Monodromy)")
    print("=" * 60)
    
    analyzer = SheafAnalyzer(SheafConfig(device="cpu"))
    
    # Network with square first layer (for monodromy)
    W1 = torch.eye(8) * 0.5  # Forward connection
    W2 = torch.randn(8, 8)
    W3 = torch.randn(4, 8)
    weights = [W1, W2, W3]
    
    print("\n--- Case A: Resonant Feedback (Φ ≈ I) ---")
    # Feedback that creates resonance
    W_fb_resonance = torch.eye(8) * 2.0  # Φ = W_fb @ W1 = 2*0.5*I = I
    
    mono_resonance = analyzer.analyze_monodromy(W1, W_fb_resonance)
    print(f"  Topology: {mono_resonance.topology.value}")
    print(f"  Spectral radius: {mono_resonance.spectral_radius:.4f}")
    print(f"  Interpretation: Slow inference, but learnable")
    
    print("\n--- Case B: Tension Feedback (Φ ≈ -I) ---")
    # Feedback that creates tension
    W_fb_tension = -torch.eye(8) * 2.0  # Φ = -I
    
    mono_tension = analyzer.analyze_monodromy(W1, W_fb_tension)
    print(f"  Topology: {mono_tension.topology.value}")
    print(f"  Spectral radius: {mono_tension.spectral_radius:.4f}")
    print(f"  Interpretation: Internal contradictions may stall learning")
    
    # Full diagnosis with tension feedback
    diagnosis = analyzer.full_diagnosis(
        weights, 
        feedback_weights=[W_fb_tension]
    )
    
    print(f"\nFull Diagnosis with Tension Feedback:")
    print(f"  Escalation recommended: {diagnosis.escalation_recommended}")
    
    return diagnosis


def example_integrated_director():
    """
    Example: Integrate sheaf diagnostics with the RAA Director.
    
    Shows how to combine entropy monitoring with topological analysis.
    """
    print("\n" + "=" * 60)
    print("Example 3: Integrated Director with Sheaf Diagnostics")
    print("=" * 60)
    
    # Create Manifold
    manifold = ModernHopfieldNetwork(HopfieldConfig(
        embedding_dim=64,
        beta=10.0,
        adaptive_beta=True,
        device="cpu"
    ))
    
    # Store some patterns
    for _ in range(10):
        manifold.store_pattern(torch.randn(64))
    
    # Create Director
    director = Director(manifold, DirectorConfig(
        entropy_threshold_percentile=0.75,
        use_energy_aware_search=True
    ))
    
    # Create Sheaf Analyzer
    sheaf_analyzer = SheafAnalyzer(SheafConfig(device="cpu"))
    
    # Simulate a network configuration
    weights = [
        torch.randn(64, 32),
        torch.randn(64, 64),
        torch.randn(32, 64)
    ]
    
    # Create test inputs
    current_state = torch.randn(1, 64)
    
    # Simulate high-entropy processor output (confusion)
    vocab_size = 1000
    confused_logits = torch.randn(1, 10, vocab_size)  # Uniform-ish
    
    print("\n--- Combined Entropy + Sheaf Analysis ---")
    
    # Step 1: Check entropy (standard Director)
    is_clash, entropy = director.check_entropy(confused_logits)
    print(f"  Entropy check: clash={is_clash}, entropy={entropy:.4f}")
    
    # Step 2: Sheaf diagnosis
    diagnosis = sheaf_analyzer.full_diagnosis(weights)
    print(f"  Sheaf diagnosis: H^1 dim={diagnosis.cohomology.h1_dimension}")
    print(f"  Learning can proceed: {diagnosis.learning_can_proceed}")
    
    # Step 3: Combined decision
    should_escalate = is_clash and diagnosis.escalation_recommended
    print(f"\n  COMBINED DECISION: Escalate = {should_escalate}")
    
    if should_escalate:
        print("  Reason: High entropy AND topological obstruction")
    elif is_clash:
        # Standard Director search
        new_goal = director.check_and_search(current_state, confused_logits)
        if new_goal is not None:
            print(f"  Reason: High entropy but topology OK, found alternative goal")
    else:
        print("  Reason: Neither entropy nor topology indicate stuck state")
    
    return director, sheaf_analyzer


def example_attention_analysis():
    """
    Example: Analyze attention patterns through sheaf lens.
    
    Useful for diagnosing transformer-based processors.
    """
    print("\n" + "=" * 60)
    print("Example 4: Attention Pattern Sheaf Analysis")
    print("=" * 60)
    
    from src.director import AttentionSheafAnalyzer
    from src.processor import GoalBiasedAttention
    
    # Create attention module
    attention = GoalBiasedAttention(
        embedding_dim=64,
        num_heads=4,
        dropout=0.0
    )
    
    # Generate test inputs
    batch_size, seq_len = 2, 16
    query = torch.randn(batch_size, seq_len, 64)
    key = torch.randn(batch_size, seq_len, 64)
    value = torch.randn(batch_size, seq_len, 64)
    goal = torch.randn(64)
    
    # Forward pass to get attention weights
    with torch.no_grad():
        _, attn_weights = attention(query, key, value, goal_state=goal)
    
    print(f"\nAttention weights shape: {attn_weights.shape}")
    print(f"  (batch={batch_size}, heads=4, seq={seq_len}, seq={seq_len})")
    
    # Analyze through sheaf lens
    attn_analyzer = AttentionSheafAnalyzer(SheafConfig(device="cpu"))
    results = attn_analyzer.diagnose_attention(attn_weights)
    
    print(f"\nPer-Head Analysis:")
    for head_result in results["per_head"]:
        print(f"  Head {head_result['head']}: "
              f"H^1={head_result['h1_dim']}, "
              f"overlap={head_result['overlap']:.4f}, "
              f"can_learn={head_result['can_learn']}")
    
    print(f"\nAggregate Metrics:")
    print(f"  Max H^1 dimension: {results['aggregate']['max_h1_dim']}")
    print(f"  Mean overlap: {results['aggregate']['mean_overlap']:.4f}")
    print(f"  Problematic heads: {results['aggregate']['num_problematic_heads']}")
    
    return results


def example_escalation_decision():
    """
    Example: Use sheaf diagnostics for escalation decisions.
    
    Demonstrates the new topological criterion for System 3 escalation.
    """
    print("\n" + "=" * 60)
    print("Example 5: Topological Escalation Criteria")
    print("=" * 60)
    
    analyzer = SheafAnalyzer(SheafConfig(
        h1_escalation_threshold=0,  # Any non-trivial H^1 triggers escalation
        overlap_warning_threshold=0.1,
        device="cpu"
    ))
    
    scenarios = [
        ("Well-conditioned feedforward", 
         [torch.randn(4, 4), torch.randn(4, 4)], 
         None),
        ("Feedforward with tension feedback", 
         [torch.eye(4), torch.eye(4)], 
         [-torch.eye(4)]),
        ("Feedforward with resonance feedback", 
         [torch.eye(4), torch.eye(4)], 
         [torch.eye(4)]),
    ]
    
    for name, weights, feedback in scenarios:
        print(f"\n--- {name} ---")
        diagnosis = analyzer.full_diagnosis(weights, feedback_weights=feedback)
        
        print(f"  H^1 dimension: {diagnosis.cohomology.h1_dimension}")
        if diagnosis.monodromy:
            print(f"  Monodromy topology: {diagnosis.monodromy.topology.value}")
        print(f"  Escalation recommended: {diagnosis.escalation_recommended}")
        
        if diagnosis.diagnostic_messages:
            for msg in diagnosis.diagnostic_messages[:2]:  # First 2 messages
                print(f"  Message: {msg[:60]}...")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SHEAF DIAGNOSTICS FOR RAA - EXAMPLES")
    print("Based on Seely (2025): Sheaf Cohomology of Linear PC Networks")
    print("=" * 60)
    
    example_basic_diagnosis()
    example_feedback_analysis()
    example_integrated_director()
    example_attention_analysis()
    example_escalation_decision()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)
    
    print("""
Key Takeaways:
1. H^1 ≠ 0 indicates irreducible errors that inference cannot eliminate
2. Low harmonic-diffusive overlap means learning is starved
3. Tension monodromy (Φ ≈ -I) can cause learning to stall
4. Combine entropy monitoring with sheaf diagnostics for robust stuck detection
5. Use topological criteria for principled escalation to System 3
    """)


if __name__ == "__main__":
    main()

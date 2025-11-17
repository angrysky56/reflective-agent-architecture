#!/usr/bin/env python3
"""Diagnostic script to identify RAA issues"""
import torch
import torch.nn.functional as f

print("=" * 70)
print("RAA Diagnostic Report")
print("=" * 70)

# Test 1: Convergence tolerance
print("\n1. CONVERGENCE TOLERANCE TEST")
print("-" * 70)
current = torch.randn(1, 128)
current = f.normalize(current, p=2, dim=-1)

# After one Hopfield step, normalized vectors can be very close
for noise_level in [0.0001, 0.001, 0.01, 0.05]:
    retrieved = current + torch.randn(1, 128) * noise_level
    retrieved = f.normalize(retrieved, p=2, dim=-1)

    state_change = torch.norm(current - retrieved).item()
    print(
        f"Noise {noise_level:.4f}: change={state_change:.6f}, "
        f"converges(1e-4)={state_change < 1e-4}, "
        f"converges(1e-2)={state_change < 1e-2}"
    )

# Test 2: Energy threshold
print("\n2. ENERGY THRESHOLD TEST")
print("-" * 70)
print("Typical Hopfield energy range: [-2.0, 0.5]")
print("Energy threshold -1e6: UNREACHABLE (disables early exit)")
print("Energy threshold 0.1: Reasonable for good solutions")
print("Energy threshold 1.0: Very lenient")

# Test 3: Shape handling
print("\n3. SHAPE HANDLING TEST")
print("-" * 70)
batched = torch.randn(1, 128)
print(f"Batched tensor shape: {batched.shape}")
print(f"shape[0] = {batched.shape[0]} (batch size)")
print(f"shape[1] = {batched.shape[1]} (embedding dim)")
print("Test expects shape[0] == embedding_dim: WRONG!")
print("Should check shape[-1] or shape[1] for embedding dim")

# Test 4: Pattern diversity
print("\n4. PATTERN DIVERSITY TEST")
print("-" * 70)
# Simulate clustered patterns
patterns = []
for cluster_id in range(5):
    center = torch.randn(128)
    center = f.normalize(center, p=2, dim=0)
    for _ in range(4):
        noise = torch.randn(128) * 0.1
        pattern = center + noise
        pattern = f.normalize(pattern, p=2, dim=0)
        patterns.append(pattern)

patterns_tensor = torch.stack(patterns)
print(f"Created {len(patterns)} patterns in 5 clusters")

# Check pattern similarity within and across clusters
similarities = torch.matmul(patterns_tensor, patterns_tensor.T)
within_cluster_sim = []
across_cluster_sim = []

for i in range(5):
    start = i * 4
    end = start + 4
    # Within cluster
    cluster_sims = similarities[start:end, start:end]
    within_cluster_sim.extend(cluster_sims[cluster_sims < 0.99].flatten().tolist())
    # Across clusters
    for j in range(5):
        if i != j:
            other_start = j * 4
            other_end = other_start + 4
            cross_sims = similarities[start:end, other_start:other_end]
            across_cluster_sim.extend(cross_sims.flatten().tolist())

print(f"Within-cluster similarity: {sum(within_cluster_sim)/len(within_cluster_sim):.3f}")
print(f"Across-cluster similarity: {sum(across_cluster_sim)/len(across_cluster_sim):.3f}")
print(
    f"Diversity score (lower is more diverse): {sum(similarities.flatten())/(len(patterns)**2):.3f}"
)

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print("1. Increase convergence_tolerance to 1e-2 or 5e-3")
print("2. Set energy_threshold to reasonable value like 0.5")
print("3. Fix test to check shape[-1] instead of shape[0]")
print("4. Increase pattern noise (0.3-0.5) for more diversity")
print("=" * 70)

#!/usr/bin/env python3
"""Compare ORT intermediate tensors between seq=1 and seq=2 to find first divergence.

Instead of comparing hologram vs ORT (which requires profile feature),
compare ORT seq=1 vs seq=2 at position 0 — in a causal model, position 0's
representation should be identical regardless of sequence length.

This finds the first ONNX node where position 0 outputs differ between seq=1 and seq=2,
which is the same node where hologram diverges.

Usage: python3 scripts/compare_intermediates.py
"""
import numpy as np

seq1 = np.load("/tmp/ort_intermediates_seq1.npz")
seq2 = np.load("/tmp/ort_intermediates_seq2.npz")

print(f"seq1: {len(seq1)} tensors, seq2: {len(seq2)} tensors")

# Find common tensors.
common = sorted(set(seq1.keys()) & set(seq2.keys()))
print(f"Common: {len(common)} tensors\n")

# For each tensor, check if the values at position 0 match.
# Position 0 in seq=1 has shape [..., 1, ...] and seq=2 has [..., 2, ...].
# We compare the first slice (position 0).
divergent = []
for name in common:
    a = seq1[name].flatten()
    b = seq2[name]

    # For seq=2, take only the first half (position 0).
    # This works for shapes like [1, 2, D] where D is fixed.
    if b.size == 2 * a.size and a.size > 0:
        b_pos0 = b.flatten()[:a.size]
        max_diff = np.max(np.abs(a - b_pos0))
        if max_diff > 1e-4:
            divergent.append((name, a.shape, b.shape, max_diff))
    elif b.size == a.size:
        # Same size — should be identical (e.g., weight-like constants).
        max_diff = np.max(np.abs(a.flatten() - b.flatten()))
        if max_diff > 1e-6:
            divergent.append((name, a.shape, b.shape, max_diff))

print(f"Divergent at position 0: {len(divergent)} tensors")
for name, s1, s2, diff in divergent[:30]:
    print(f"  {name}: shape1={s1} shape2={s2} max_diff={diff:.6f}")
if len(divergent) > 30:
    print(f"  ... ({len(divergent) - 30} more)")

# Also check: does ORT itself produce identical position-0 values?
# In a correct causal model, position 0 should be identical at seq=1 and seq=2.
print(f"\n=== ORT causal consistency check ===")
print("(Position 0 should be identical between seq=1 and seq=2)")
ort_divergent = 0
for name in common:
    a = seq1[name]
    b = seq2[name]
    if b.size == 2 * a.size and a.size > 0:
        b_pos0 = b.flatten()[:a.size]
        max_diff = np.max(np.abs(a.flatten() - b_pos0))
        if max_diff > 1e-3:
            ort_divergent += 1
            if ort_divergent <= 5:
                print(f"  ORT DIVERGENT: {name} max_diff={max_diff:.6f}")
print(f"Total ORT-internal divergences: {ort_divergent}")

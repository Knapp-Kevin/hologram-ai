#!/usr/bin/env python3
"""Find first divergent node between hologram and ORT intermediates.

Reads ORT npz and hologram's captured node buffers (dumped by the Rust test),
matches by ONNX tensor name, and finds the first node where values diverge.

This script reads the hologram intermediates from a generated dump file.
First run the Rust test to generate the dump, then run this script.

Usage:
  1. Run: ORT_STRATEGY=system cargo test -p hologram-ai-conformance \
       --features "conformance,profile" --test exec_conformance \
       -- --ignored tinyllama_node_divergence --nocapture 2> /tmp/hologram_nodes.txt
  2. Run: python3 scripts/find_divergent_node.py
"""
import numpy as np
import re
import sys

# Load ORT intermediates.
ort = np.load("/tmp/ort_intermediates_seq2.npz")
print(f"ORT: {len(ort)} tensors")

# Parse hologram node dump from test output.
# Format: "  node N (onnx_name): shape=[...] elems=M range=[min, max]"
# We need to match ONNX names between hologram and ORT.

# For now, just compare the FINAL logits and work backwards.
# ORT logits:
ort_logits = ort.get("logits")
if ort_logits is not None:
    last_pos = ort_logits[0, -1, :]
    ort_top5 = np.argsort(last_pos)[-5:][::-1]
    print(f"ORT top-5: {ort_top5.tolist()}")
    print(f"ORT logit values: {[f'{last_pos[i]:.4f}' for i in ort_top5]}")
    print(f"ORT logit range: [{last_pos.min():.4f}, {last_pos.max():.4f}]")

# Find tensors that look like they're from attention layers.
# Look for intermediate hidden states after each layer.
print(f"\n=== ORT tensors with shape (1, 2, 2048) — hidden states ===")
for name in sorted(ort.keys()):
    arr = ort[name]
    if arr.shape == (1, 2, 2048):
        rng = f"[{arr.min():.4f}, {arr.max():.4f}]"
        print(f"  {name}: range={rng}")

# Find the final hidden state before lm_head.
print(f"\n=== ORT tensor names containing 'norm' or 'lm_head' ===")
for name in sorted(ort.keys()):
    if 'norm' in name.lower() or 'lm_head' in name.lower() or 'logit' in name.lower():
        arr = ort[name]
        print(f"  {name}: shape={arr.shape} range=[{arr.min():.4f}, {arr.max():.4f}]")

# The key comparison: find the embedding output and first RmsNorm output.
print(f"\n=== Key intermediate comparison (ORT) ===")
for name in ['val_59', 'val_60', 'add', 'add_1', 'add_2', 'input_layernorm', 'logits']:
    for key in sorted(ort.keys()):
        if key == name or key.startswith(name + '_'):
            arr = ort[key]
            if arr.size < 100000:
                print(f"  {key}: shape={arr.shape} range=[{arr.min():.4f}, {arr.max():.4f}]")
            break

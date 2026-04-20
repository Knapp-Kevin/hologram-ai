#!/usr/bin/env python3
"""Find the first divergent node between ORT and hologram.

Requires:
  1. ORT intermediates: run scripts/compare_ort.py --dump-all first
  2. Hologram dump: HOLOGRAM_DUMP_DIR=/tmp/qwen2-dump hologram-ai run ...

Usage:
    python3 scripts/find_divergence.py models/Qwen2-0.5B/model.onnx /tmp/qwen2-dump

Strategy: Since hologram's compiled node indices don't directly correspond to
ONNX node names, we compare specific checkpoints:
  - Node 1 (Gather): embedding lookup
  - Node 3 (RmsNorm): post-embedding norm
  - Final node (Gemm): logits
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np


def load_hologram_manifest(dump_dir: str) -> list[dict]:
    """Load hologram dump manifest."""
    manifest = []
    with open(f"{dump_dir}/manifest.csv") as f:
        for row in csv.reader(f):
            manifest.append({
                "instr_idx": int(row[0]),
                "output_idx": int(row[1]),
                "size_bytes": int(row[2]),
                "kernel": row[3],
            })
    return manifest


def load_hologram_output(dump_dir: str, instr_idx: int, output_idx: int) -> np.ndarray:
    """Load a hologram instruction output as f32 array."""
    path = f"{dump_dir}/node_{instr_idx:05d}_{output_idx}.bin"
    raw = np.fromfile(path, dtype=np.uint8)
    # Try f32 first
    if len(raw) % 4 == 0:
        return np.frombuffer(raw, dtype=np.float32)
    elif len(raw) % 8 == 0:
        return np.frombuffer(raw, dtype=np.int64)
    return raw


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/find_divergence.py <model.onnx> <hologram_dump_dir>")
        sys.exit(1)

    model_path = sys.argv[1]
    dump_dir = sys.argv[2]
    model_dir = str(Path(model_path).parent)

    # Load ORT intermediates
    ort_dir = f"{model_dir}/_ort_intermediates"
    if not Path(ort_dir).exists():
        print(f"ORT intermediates not found at {ort_dir}")
        print(f"Run: python3 scripts/compare_ort.py {model_path} --dump-all --prompt 'The'")
        sys.exit(1)

    with open(f"{ort_dir}/manifest.json") as f:
        ort_manifest = json.load(f)

    # Load hologram manifest
    holo_manifest = load_hologram_manifest(dump_dir)
    print(f"Hologram: {len(holo_manifest)} instructions")
    print(f"ORT: {len(ort_manifest)} tensors")
    print()

    # Key comparison points based on the Qwen2 graph structure:
    # - Hologram node 1 (Gather, output_idx=300, 3584 bytes) = embedding [1,1,896]
    # - ORT "embedding" = embedding [1,1,896]
    comparisons = [
        ("Embedding", 1, "embedding"),
        ("Post-embed RmsNorm", 3, "mul_97"),  # post-embed normalization
    ]

    # Find embedding by kernel type
    for entry in holo_manifest:
        if entry["kernel"] == "Gather" and entry["size_bytes"] == 3584:
            comparisons[0] = ("Embedding", entry["instr_idx"], "embedding")
            break

    for name, holo_idx, ort_name in comparisons:
        print(f"=== {name} ===")

        # Load hologram
        holo_entry = holo_manifest[holo_idx]
        holo_data = load_hologram_output(dump_dir, holo_entry["instr_idx"], holo_entry["output_idx"])

        # Load ORT
        if ort_name in ort_manifest:
            ort_file = ort_manifest[ort_name]["file"]
            ort_data = np.load(f"{ort_dir}/{ort_file}").flatten()
        else:
            print(f"  ORT tensor '{ort_name}' not found")
            continue

        # Truncate to matching length
        min_len = min(len(holo_data), len(ort_data))
        if min_len == 0:
            print(f"  Empty data (hologram: {len(holo_data)}, ORT: {len(ort_data)})")
            continue

        h = holo_data[:min_len].astype(np.float32)
        o = ort_data[:min_len].astype(np.float32)

        # Compare
        abs_diff = np.abs(h - o)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()

        print(f"  Hologram: shape={holo_data.shape}, first 8: {holo_data[:8]}")
        print(f"  ORT:      shape={ort_data.shape}, first 8: {ort_data[:8]}")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        if max_diff < 1e-4:
            print(f"  MATCH (within tolerance)")
        elif max_diff < 1e-2:
            print(f"  CLOSE (minor divergence)")
        else:
            print(f"  DIVERGED! Max diff = {max_diff:.4f}")
            # Find first divergent element
            first_div = np.argmax(abs_diff > 1e-2)
            print(f"  First divergence at index {first_div}: hologram={h[first_div]:.6f}, ORT={o[first_div]:.6f}")

        print()

    # Also compare all instructions by iterating through hologram nodes
    # and comparing against ORT where possible
    print("=== Full scan (first 50 instructions) ===")
    print()

    for entry in holo_manifest[:50]:
        idx = entry["instr_idx"]
        holo_data = load_hologram_output(dump_dir, entry["instr_idx"], entry["output_idx"])
        kernel = entry["kernel"]
        size = entry["size_bytes"]

        if len(holo_data) == 0:
            continue

        if holo_data.dtype == np.float32:
            finite = holo_data[np.isfinite(holo_data)]
            nan_count = np.isnan(holo_data).sum()
            inf_count = np.isinf(holo_data).sum()

            if nan_count or inf_count:
                print(f"  [{idx:4d}] {kernel:30s} out={entry['output_idx']:4d} size={size:8d}  NaN={nan_count} Inf={inf_count} ⚠️")
            elif len(finite) > 0 and (finite.max() > 1e6 or finite.min() < -1e6):
                print(f"  [{idx:4d}] {kernel:30s} out={entry['output_idx']:4d} size={size:8d}  range=[{finite.min():.2e}, {finite.max():.2e}] ⚠️ EXTREME")
            else:
                stats = f"range=[{finite.min():.4f}, {finite.max():.4f}]" if len(finite) > 0 else "empty"
                print(f"  [{idx:4d}] {kernel:30s} out={entry['output_idx']:4d} size={size:8d}  {stats}")


if __name__ == "__main__":
    main()

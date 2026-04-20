#!/usr/bin/env python3
"""Node-by-node comparison: ORT intermediates dump.

Dumps all intermediate node outputs from an ONNX model via ORT,
saving them as .npy files for comparison with hologram.

Usage:
    python3 scripts/compare_ort.py models/Qwen2-0.5B/model.onnx
    python3 scripts/compare_ort.py models/Qwen2-0.5B/model.onnx --prompt "The capital of France is"

Requires: onnx, onnxruntime, numpy, transformers (for tokenizer)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto


def run_ort_with_all_intermediates(
    model_path: str, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Run ORT and capture all intermediate node outputs.

    Uses ORT's ability to request specific output names without modifying
    the model graph.
    """
    import onnxruntime as ort

    # ORT can produce any named tensor as output via run() output_names param.
    # But we need to register them first by modifying the graph.
    model = onnx.load(model_path)

    # Collect all node output names and their producing op
    node_outputs = []
    for node in model.graph.node:
        for out in node.output:
            if out:
                node_outputs.append((out, node.op_type, node.name))

    # Infer shapes/types so we can add intermediates with correct types.
    try:
        model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception:
        pass  # shape inference may fail on some ops, but we try

    # Build a map from tensor name to inferred type
    type_map = {}
    for vi in model.graph.value_info:
        type_map[vi.name] = vi.type
    for o in model.graph.output:
        type_map[o.name] = o.type

    existing_outputs = {o.name for o in model.graph.output}
    for name, op_type, _ in node_outputs:
        if name not in existing_outputs:
            if name in type_map:
                # Use the inferred type info
                out = model.graph.output.add()
                out.name = name
                out.type.CopyFrom(type_map[name])
            # Skip tensors with no type info — they'd cause ORT errors

    # Save modified model next to original (for external data resolution)
    model_dir = os.path.dirname(os.path.abspath(model_path))
    tmp_path = os.path.join(model_dir, "_compare_tmp.onnx")
    try:
        onnx.save(model, tmp_path)

        # Disable graph optimization to preserve node names
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        sess = ort.InferenceSession(tmp_path, opts)

        # Run and get all outputs
        output_names = [o.name for o in sess.get_outputs()]
        all_outputs = sess.run(output_names, inputs)

        result = {}
        for name, arr in zip(output_names, all_outputs):
            result[name] = arr
        return result, node_outputs
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def find_key_probes(node_outputs: list[tuple[str, str, str]]) -> list[tuple[str, str, str, str]]:
    """Identify key probe points for a transformer model."""
    probes = []
    seen_ops = set()

    for name, op_type, node_name in node_outputs:
        tag = None

        # Embedding
        if node_name == "node_embedding":
            tag = "EMBEDDING"
        # First RMSNorm chain (post-embedding)
        elif "pow_1" == node_name and op_type == "Pow":
            tag = "FIRST_POW"
        elif "mul_97" in node_name:
            tag = "POST_EMBED_NORM_MUL"
        # First layer attention
        elif node_name == "node_MatMul_248":
            tag = "L0_QK_SCORES"
        elif node_name == "node_Add_249":
            tag = "L0_QK_PLUS_MASK"
        elif node_name == "node_Softmax_250":
            tag = "L0_ATTN_WEIGHTS"
        elif node_name == "node_scaled_dot_product_attention":
            tag = "L0_ATTN_OUTPUT"
        # First residual
        elif node_name == "node_add_335":
            tag = "L0_RESIDUAL"
        # First FFN
        elif node_name == "node_mul_2200":
            tag = "L0_FFN_GATE"
        # Q/K/V projections for layer 0
        elif node_name == "node_linear":
            tag = "L0_Q_PROJ"
        elif node_name == "node_linear_1":
            tag = "L0_K_PROJ"
        elif node_name == "node_linear_2":
            tag = "L0_V_PROJ"
        # Final output
        elif name == "logits":
            tag = "LOGITS"

        if tag:
            probes.append((name, op_type, node_name, tag))

    return probes


def main():
    parser = argparse.ArgumentParser(description="Dump ORT intermediates for comparison")
    parser.add_argument("model", help="Path to ONNX model")
    parser.add_argument("--prompt", default="The", help="Text prompt")
    parser.add_argument("--dump-all", action="store_true", help="Dump all intermediates to .npy files")
    args = parser.parse_args()

    # Tokenize
    model_dir = os.path.dirname(args.model)
    if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        token_ids = tokenizer.encode(args.prompt)
    else:
        token_ids = [785]

    print(f"Prompt: {args.prompt!r}")
    print(f"Token IDs: {token_ids}")
    print(f"Seq len: {len(token_ids)}")
    print()

    # Build inputs
    input_ids = np.array([token_ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    print("Running ORT with all intermediate outputs...")
    ort_results, node_outputs = run_ort_with_all_intermediates(args.model, inputs)
    print(f"Captured {len(ort_results)} intermediate tensors")
    print()

    # Find and display key probes
    probes = find_key_probes(node_outputs)
    if not probes:
        # Fallback: sample every 100th node
        probes = [(n, op, nn, f"NODE_{i}") for i, (n, op, nn) in enumerate(node_outputs) if i % 100 == 0]

    print("=== Key Intermediate Values ===")
    print()
    for name, op_type, node_name, tag in probes:
        if name not in ort_results:
            print(f"  [{tag}] {node_name} ({op_type}): NOT CAPTURED")
            continue
        arr = ort_results[name]
        print(f"  [{tag}] {node_name} ({op_type}) -> {name}")
        print(f"    shape={arr.shape}, dtype={arr.dtype}")
        if arr.dtype in (np.float32, np.float64, np.float16):
            finite = arr[np.isfinite(arr)]
            if len(finite) > 0:
                print(f"    range=[{finite.min():.6f}, {finite.max():.6f}]")
                print(f"    mean={finite.mean():.6f}, std={finite.std():.6f}")
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            if nan_count or inf_count:
                print(f"    nan={nan_count}, inf={inf_count}")
            flat = arr.flatten()
            print(f"    first 8: {flat[:8]}")
        elif arr.dtype in (np.int64, np.int32):
            print(f"    values: {arr.flatten()[:10]}")
        elif arr.dtype == np.bool_:
            print(f"    true%: {arr.sum() / arr.size * 100:.1f}%")
        print()

    # Dump all intermediates if requested
    if args.dump_all:
        dump_dir = Path(model_dir) / "_ort_intermediates"
        dump_dir.mkdir(exist_ok=True)

        manifest = {}
        for name, op_type, node_name in node_outputs:
            if name in ort_results:
                arr = ort_results[name]
                safe_name = name.replace("/", "_").replace(":", "_")
                npy_path = dump_dir / f"{safe_name}.npy"
                np.save(npy_path, arr)
                manifest[name] = {
                    "file": str(npy_path.name),
                    "op_type": op_type,
                    "node_name": node_name,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }

        with open(dump_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Saved {len(manifest)} intermediates to {dump_dir}/")

    # Print summary for comparison with hologram
    print("=== Summary for hologram comparison ===")
    print()
    if "embedding" in ort_results:
        emb = ort_results["embedding"]
        print(f"Embedding output: shape={emb.shape}")
        print(f"  first 8: {emb.flatten()[:8]}")
        print(f"  norm: {np.linalg.norm(emb.flatten()):.6f}")
    if "logits" in ort_results:
        logits = ort_results["logits"]
        last = logits[0, -1, :]
        top5 = np.argsort(last)[-5:][::-1]
        print(f"Logits at last position top-5: {top5}")
        print(f"  scores: {last[top5]}")


if __name__ == "__main__":
    main()

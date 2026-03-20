#!/usr/bin/env python3
"""Capture ALL intermediate tensors from TinyLlama ONNX model via ORT.

Modifies the ONNX graph to expose every node's output as a graph output,
runs inference, and saves all tensors to a numpy archive for comparison.

Usage: python3 scripts/ort_intermediates.py [seq_len]
"""
import sys
import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("pip install onnx onnxruntime", file=sys.stderr)
    sys.exit(1)

MODEL = "models/TinyLlama-1.1B-Chat-v1.0/model_causal.onnx"
SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 2

print(f"Loading {MODEL}...")
model = onnx.load(MODEL)
graph = model.graph

# Collect all intermediate tensor names.
all_names = set()
for node in graph.node:
    for out in node.output:
        if out:
            all_names.add(out)

# Build a map from tensor name → dtype from node outputs and value_info.
dtype_map = {}
for vi in graph.value_info:
    if vi.type.tensor_type.elem_type:
        dtype_map[vi.name] = vi.type.tensor_type.elem_type

# Add missing outputs to graph (use known dtype or UNDEFINED to let ORT infer).
existing_outputs = {o.name for o in graph.output}
for name in sorted(all_names):
    if name not in existing_outputs:
        dt = dtype_map.get(name, onnx.TensorProto.UNDEFINED)
        graph.output.append(onnx.helper.make_empty_tensor_value_info(name))

print(f"Graph has {len(graph.node)} nodes, {len(graph.output)} outputs")

# Save modified model.
modified_path = "/tmp/tinyllama_all_outputs.onnx"
onnx.save(model, modified_path, save_as_external_data=True, all_tensors_to_one_file=True, location="tinyllama_all_outputs.data")
print(f"Saved modified model to {modified_path}")

# Run inference.
print(f"Running ORT at seq={SEQ}...")
sess = ort.InferenceSession(modified_path)
input_ids = np.array([[1] + list(range(2, SEQ + 1))], dtype=np.int64)
attention_mask = np.ones([1, SEQ], dtype=np.int64)

outputs = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
output_names = [o.name for o in sess.get_outputs()]

# Save to npz.
out_path = f"/tmp/ort_intermediates_seq{SEQ}.npz"
data = {}
for name, arr in zip(output_names, outputs):
    if isinstance(arr, np.ndarray) and arr.dtype in (np.float32, np.float64):
        data[name] = arr.astype(np.float32)

np.savez(out_path, **data)
print(f"Saved {len(data)} f32 tensors to {out_path}")

# Print first 20 tensors as summary.
for i, (name, arr) in enumerate(sorted(data.items())[:20]):
    print(f"  {name}: shape={arr.shape} range=[{arr.min():.4f}, {arr.max():.4f}]")
if len(data) > 20:
    print(f"  ... ({len(data) - 20} more)")

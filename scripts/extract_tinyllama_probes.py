#!/usr/bin/env python3
"""Extract TinyLlama sub-models for layer-by-layer conformance testing.

Creates ONNX sub-models that terminate at specific layer boundaries,
enabling binary-search of compute divergence between hologram and ORT.

Usage:
    python3 scripts/extract_tinyllama_probes.py

Requires:
    pip install onnx
    models/TinyLlama-1.1B-Chat-v1.0/model_causal.onnx (from hologram-ai download)

Output:
    crates/hologram-ai-conformance/fixtures/tinyllama_embed.onnx
    crates/hologram-ai-conformance/fixtures/tinyllama_norm0.onnx
    crates/hologram-ai-conformance/fixtures/tinyllama_layer0.onnx
"""
import sys
from pathlib import Path

try:
    from onnx.utils import extract_model
except ImportError:
    print("pip install onnx", file=sys.stderr)
    sys.exit(1)

MODEL = "models/TinyLlama-1.1B-Chat-v1.0/model_causal.onnx"
FIXTURES = Path("crates/hologram-ai-conformance/fixtures")

if not Path(MODEL).exists():
    print(f"Model not found: {MODEL}", file=sys.stderr)
    print("Download with: hologram-ai download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --format onnx", file=sys.stderr)
    sys.exit(1)

FIXTURES.mkdir(parents=True, exist_ok=True)

probes = [
    # (output_name, inputs, output_file, description)
    ("embedding", ["input_ids"], "tinyllama_embed.onnx",
     "Embedding lookup only (Gather on vocab weight)"),
    ("mul_99", ["input_ids"], "tinyllama_norm0.onnx",
     "Embedding + first RmsNorm (input_layernorm)"),
    ("add_320", ["input_ids", "attention_mask"], "tinyllama_layer0.onnx",
     "Full layer 0 (embedding + attention + FFN + residual)"),
]

for output_name, inputs, filename, desc in probes:
    out_path = FIXTURES / filename
    print(f"Extracting {filename}: {desc}")
    try:
        extract_model(MODEL, str(out_path), inputs, [output_name])
        size_mb = out_path.stat().st_size / 1e6
        print(f"  -> {out_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)

print("\nDone. Run conformance tests with:")
print("  ORT_STRATEGY=system cargo test -p hologram-ai-conformance \\")
print("    --features conformance -- tinyllama --nocapture --ignored")

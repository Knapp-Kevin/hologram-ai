# Plan 014: ONNX TinyLlama Coherent Generation

## Context

ONNX TinyLlama compiled and ran without errors but was not detected as LLM
(missing metadata), so it compiled as a single graph without KV cache or pipeline.

## What was done

### Step 1: `infer_llm_metadata_from_graph()` — compiler.rs

After AttentionFusion runs, scans for `GroupedQueryAttention` nodes and infers:
- `arch` = "llama"
- `n_layers` = count of GQA nodes
- `n_kv_heads`, `head_dim` from GQA params
- `n_embd` = num_heads × head_dim
- `context_length` = 2048 (default)

Only sets metadata not already present (no-op for GGUF which sets these at import).

### Results

Using `model_causal.onnx` (265MB, f32 weights, native logits):
- **Detected as LLM**: `model: TextLlm arch=llama`
- **Pipeline compilation**: KV cache enabled, prefill + decode sub-archives
- **Prefill**: 213ms for 40 tokens (vs 4.5s GGUF Q4_0)
- **Generation**: 29.5 tok/s (vs 1.1 tok/s GGUF Q4_0)
- **Prefill logits correct**: Top-1 = newline (id=13, val=12.24)
- **Decode produces English**: `so`, `and`, `I` etc. — real words, not foreign garbage

### Remaining issue

Decode tokens are English but not fully coherent — same KV cache decode
consistency issue seen in GGUF. The prefill computation is correct.

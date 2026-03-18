# Plan 012: Attention Fusion + KV Cache + LUT-GEMM

## Context

Generation degenerates after ~7 tokens despite provably correct computation
(causal cos_sim=1.0). Without KV cache, every step recomputes the full sequence.
ollama (llama.cpp) produces perfect output with the same models using KV cache.

**Three changes, in priority order:**
1. **Attention Fusion** (ONNX) — fuse decomposed Q@K^T→Softmax→V into
   `AiOp::GroupedQueryAttention` so ONNX and GGUF share one attention path
2. **KV Cache** — prefill/decode split; decode processes 1 token with cached K/V
3. **LUT-GEMM** — compile-time Q4_0→k-means re-quantization for GGUF weights

**ONNX first**, then GGUF benefits from the same infrastructure.

---

## Phase 1: ONNX Attention Fusion Pass

### Goal
Detect the ONNX attention pattern and fuse into `AiOp::GroupedQueryAttention`.
After this, ONNX and GGUF graphs both use the fused attention op, enabling a
single KV cache implementation.

### Pattern to detect
TinyLlama ONNX attention per layer (after Reshape+Transpose):
```
Q[1,heads,seq,dim] @ K^T[1,heads,dim,seq] → scores[1,heads,seq,seq]
  → Mul(scale) → Add(mask) → Softmax → @ V[1,heads,seq,dim]
  → Transpose → Reshape
```

### Implementation

**P1.1. Add `AttentionFusion` optimization pass** (hologram-ai-common/src/opt/)
- New file: `attention_fusion.rs`
- Pattern match: find `MatMul → Mul(scalar) → Softmax → MatMul` chains
- Extract: num_heads, num_kv_heads, head_dim, scale, causal (from mask)
- Replace with: `AiOp::GroupedQueryAttention { ... }`
- Fold the surrounding Reshape+Transpose into the fused op's input/output

**P1.2. Register in OptPipeline** (hologram-ai-common/src/opt/mod.rs)
- Add `AttentionFusion` to `OptPipeline::mvp()` after existing passes

**P1.3. Verification**
- Compile TinyLlama ONNX, verify fused `Attention` ops in graph (fewer nodes)
- Run conformance: output matches ORT at all positions

---

## Phase 2: KV Cache

### Phase 2a: hologram base crate changes

**B1. Add `FloatOp::KvWrite` / `FloatOp::KvRead`** (hologram-core/op/float_op.rs)
- `KvWrite { layer: u32, n_kv_heads: u32, head_dim: u32 }` — stores K,V at pos
- `KvRead { layer: u32, n_kv_heads: u32, head_dim: u32 }` — reads cached K,V

**B2. Add `KvCacheState`** (hologram-exec, new file)
- Per-layer K/V buffers: `Vec<f32>` sized `[max_seq × n_kv_heads × head_dim]`
- `write_pos: usize` — advances by seq_len per call
- `write(layer, k_data, v_data)` — append to cache
- `read(layer) → (&[f32], &[f32])` — return K,V up to write_pos
- `reset()` — clear for new sequence

**B3. Add dispatch** (hologram-exec/kv/store.rs)
- KvWrite: copy K/V into KvCacheState, return K/V unchanged (pass-through for prefill)
- KvRead: return full cached K/V concatenated with current K/V

**B4. Add `execute_with_kv_state()`** (hologram-exec/eval/executor.rs)
- Like `execute_with_shape_hints` but also takes `&mut KvCacheState`
- Thread KvCacheState through dispatch loop

### Phase 2b: hologram-ai changes

**B5. Fix GGUF metadata** (hologram-ai-gguf/src/arch/llama.rs)
- Add `n_kv_heads`, `head_dim` to `AiGraph::metadata`
- Makes `compute_kv_layout()` → non-zero → `is_llm = true`

**B6. Inject KvSlot ops in AiGraph** (hologram-ai-gguf/src/arch/llama.rs)
- Before `GroupedQueryAttention`: insert `KvSlotWrite { layer }` on K and V
- During Decode lowering: also insert `KvSlotRead { layer }` before attention
- Prefill: KvWrite stores K/V, attention sees full sequence
- Decode: KvWrite stores new K/V, KvRead retrieves full cache → attention

**B7. ONNX: same injection after fusion** (hologram-ai-common/src/opt/ or lower/)
- After AttentionFusion, the ONNX graph has `GroupedQueryAttention` ops
- Same KvSlot injection logic applies

**B8. Phase-aware lowering** (hologram-ai-common/src/lower/builder.rs)
- Use `kv_layout` parameter (currently `_kv_layout`)
- Prefill phase: KvSlotWrite → FloatOp::KvWrite (store and pass through)
- Decode phase: KvSlotWrite → FloatOp::KvWrite + KvRead → concatenated K/V

**B9. Generation loop** (hologram-ai/src/commands/run_cmd.rs)
- Load pipeline archive (prefill + decode)
- Step 0: run prefill model → fills KvCacheState + gets logits
- Steps 1+: run decode model with 1 token → updates KvCacheState + gets logits

**B10. HoloRunner pipeline support** (hologram-ai/src/compiler.rs)
- Detect pipeline archives
- Hold KvCacheState across execute() calls
- Route to prefill (first call) vs decode (subsequent calls)

---

## Phase 3: LUT-GEMM for GGUF Q4_0

**L1. `q4_0_to_lut4()` converter** (hologram-ai-quant or hologram-ai-common)
- Dequantize Q4_0 → f32 at compile time
- `quantize_4bit(f32, rows, cols)` → `QuantizedWeights4`
- rkyv serialize → ConstantData::Bytes

**L2. Lowering strategy update** (hologram-ai-common/src/lower/strategy.rs)
- Q4_0 Gemm → `GraphOp::MatMulLut4(cid)` instead of `FloatOp::Gemm { quant_b: 1 }`
- Handle trans_b: transpose f32 weights before k-means

**L3. Builder integration** (hologram-ai-common/src/lower/builder.rs)
- Read weight bytes from AiParam::Mmap, convert, store as constant

---

## Execution Order

1. **Phase 1** (ONNX attention fusion) — self-contained in hologram-ai
2. **Phase 2a** (KV cache base) — hologram base crate
3. **Phase 2b** (KV cache integration) — hologram-ai
4. **Phase 3** (LUT-GEMM) — hologram-ai only, independent of KV cache

---

## Verification

```bash
# Phase 1
cargo test -p hologram-ai-common -- attention_fusion --nocapture
cargo test -p hologram-ai --features e2e -- tinyllama_onnx --nocapture

# Phase 2
cargo test -p hologram-exec  # KvCacheState + dispatch
cargo test -p hologram-ai --features e2e -- tinyllama --nocapture

# Phase 3
cargo test -p hologram-ai --features e2e -- tinyllama_gguf --nocapture

# Full
cargo test && cargo clippy -- -D warnings
```

## Critical Files

| Phase | File | Change |
|-------|------|--------|
| 1 | `hologram-ai-common/src/opt/attention_fusion.rs` (new) | SDPA pattern → GQA fusion |
| 1 | `hologram-ai-common/src/opt/mod.rs` | Register fusion pass |
| 2a | `hologram/hologram-core/src/op/float_op.rs` | FloatOp::KvWrite/KvRead |
| 2a | `hologram/hologram-exec/src/kv_cache.rs` (new) | KvCacheState |
| 2a | `hologram/hologram-exec/src/kv/store.rs` | KV dispatch |
| 2a | `hologram/hologram-exec/src/eval/executor.rs` | execute_with_kv_state |
| 2b | `hologram-ai-gguf/src/arch/llama.rs` | Metadata + KvSlot ops |
| 2b | `hologram-ai-common/src/lower/builder.rs` | Phase-aware KV lowering |
| 2b | `hologram-ai/src/commands/run_cmd.rs` | Pipeline decode loop |
| 2b | `hologram-ai/src/compiler.rs` | HoloRunner pipeline support |
| 3 | `hologram-ai-common/src/lower/strategy.rs` | Q4_0 → MatMulLut4 |

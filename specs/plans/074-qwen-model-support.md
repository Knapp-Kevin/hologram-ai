# Plan 074: Architectural Patterns from Qwen — Compiler Improvements + Cross-Family Validation

## Context

Studying Qwen models (Qwen2, Qwen2.5, Qwen3) reveals architectural patterns that
represent broader trends in modern LLMs. Rather than just "supporting Qwen," this plan
adopts six patterns that make hologram-ai a better compiler for **all** model families.
Qwen2-0.5B serves as the validation model to prove these patterns work end-to-end.

**Design principle:** hologram-ai is a compiler only (ADR-0016). These patterns are all
compile-time or runtime optimizations — no model weights change, no training required.

---

## Pattern 1: Fused RoPE + Context Scaling

**What Qwen teaches:** Every modern LLM uses RoPE. Qwen extends context from 8K → 128K+
**without retraining** using NTK-aware interpolation and YaRN — purely by modifying RoPE
frequency computation at inference time. This is a perfect fit for hologram-ai's
compiler-only design: context scaling becomes a compile-time flag.

**What hologram-ai gains:** Any RoPE-based model (LLaMA, Mistral, Gemma, Qwen, Phi)
compiled with `--context-scale yarn --factor 4.0` gets extended context for free.

### Current State
- `AiOp::GroupedQueryAttention` has `rope: bool`, `rope_base: f32` — hardcoded to
  `false`/`0.0` in `AttentionFusion` (line 178)
- Lowering correctly passes these to `FloatOp::Attention`
- `FloatOp::Attention` has the fields but `tape_builder.rs` destructures with `..` to
  skip them (line 731)
- `AttentionParams` doesn't include rope fields
- Standalone `dispatch_rope()` kernel exists (lines 643-679 of attention.rs)

### Changes

**Phase 1: Basic RoPE kernel (hologram base)**

1. **`crates/hologram-exec/src/float_dispatch/attention.rs`**
   - Add `rope: bool`, `rope_base: f32` to `AttentionParams`
   - In `dispatch_attention()`, if `rope == true`: apply rotary embedding to Q and K
     slices before computing `Q @ K^T`
   - Extract cos/sin computation from existing `dispatch_rope()` into shared helper

2. **`crates/hologram-exec/src/tape_builder.rs`** (~line 731)
   - Stop ignoring rope fields in `FloatOp::Attention` match arm
   - Pass `rope`, `rope_base` through to `TapeKernel::InlineAttention`

3. **`crates/hologram-exec/src/tape.rs`**
   - Update `TapeKernel::InlineAttention` to include `rope: bool`, `rope_base: u32`
   - Wire through to `AttentionParams` at dispatch time

**Phase 2: RoPE scaling variants (hologram base + hologram-ai)**

4. **Add `RopeScaling` enum** to `FloatOp::Attention` and `AiOp::GroupedQueryAttention`:
   ```rust
   enum RopeScaling {
       None,
       Linear { factor: f32 },
       Ntk { factor: f32 },
       Yarn { factor: f32, original_max_position: u32 },
   }
   ```

5. **Implement scaling in `dispatch_attention()`**:
   - `Linear`: scale frequencies by `1/factor`
   - `NTK`: adjust base dynamically: `base * factor^(dim/(dim-2))`
   - `YaRN`: NTK + attention scaling + linear ramp between low/high frequencies

6. **CLI flags** (`--context-scale`, `--context-factor`):
   - Extract scaling config from HuggingFace `config.json` `rope_scaling` field
   - Allow override at compile time

**Phase 3: RoPE detection in ONNX (hologram-ai)**

7. **`crates/hologram-ai-common/src/opt/attention_fusion.rs`**
   - Detect RoPE pattern: Sin/Cos + multiply + interleave before Q/K matmuls
   - Set `rope: true`, extract `rope_base` from constants, absorb RoPE nodes

8. **Re-enable `PreAttentionFusion` pass** in pipeline.rs if pattern detection
   is better done as a separate pass

### Verification
- Conformance: `dispatch_attention(rope=true)` vs `dispatch_rope() + dispatch_attention(rope=false)` — match within atol=1e-5
- TinyLlama e2e unchanged (LLaMA bakes RoPE into graph, so `rope` stays false)
- NTK: verify cos/sin frequencies differ at positions > original_max_position

---

## Pattern 2: LogN Attention Scaling

**What Qwen teaches:** Qwen uses `log(n)/log(n_train)` scaling on attention logits to
prevent attention entropy degradation at long sequences. Complementary to RoPE scaling.

**What hologram-ai gains:** Better long-context quality for any model, with a single
multiply per attention head. Trivial to implement, high value.

### Changes

1. **`FloatOp::Attention`** — add `logn_scaling: bool`, `training_context_len: u32`
2. **`dispatch_attention()`** — when enabled, multiply attention scores by
   `log(current_pos + 1) / log(training_context_len)`
3. **CLI flag** — `--logn-attention` (auto-detect from `config.json` when available)

### Verification
- Unit test: verify scaling factor changes with position
- E2E: compile model with `--logn-attention`, verify output doesn't degrade at
  long sequences compared to unscaled

---

## Pattern 3: QK-Norm (Pre-Attention Normalization)

**What Qwen teaches:** Some Qwen variants (and Gemma2) apply RMSNorm to Q and K
before attention. This stabilizes attention distribution, especially important for
models with large head dimensions or at high quantization levels.

**What hologram-ai gains:** Support for QK-Norm models + better numerical stability
in quantized attention.

### Current State
- `qk_norm: bool` field exists on `AiOp::GroupedQueryAttention` and `FloatOp::Attention`
- Hardcoded to `false`, kernel ignores it

### Changes

1. **`dispatch_attention()`** — if `qk_norm == true`: apply RMSNorm to Q and K head
   slices after projection, before Q@K^T
2. **`AttentionParams`** — add `qk_norm: bool`
3. **`TapeKernel::InlineAttention`** — add `qk_norm: bool`
4. **`tape_builder.rs`** — stop skipping `qk_norm` in `..` destructure
5. **`attention_fusion.rs`** — detect RMSNorm nodes between projection and attention

### Verification
- Conformance: `dispatch_attention(qk_norm=true)` vs manual RMSNorm on Q/K +
  `dispatch_attention(qk_norm=false)` — match within atol=1e-5

---

## Pattern 4: KV Cache Quantization as First-Class Feature

**What Qwen teaches:** Qwen3.5 treats KV cache compression as seriously as weight
quantization — block-128 INT8 with near-lossless quality. KV cache memory often
dominates at long context (e.g., 128K tokens × 32 layers × 128 dim × 2 KV × f32 =
4 GB just for cache).

**What hologram-ai gains:** hologram base **already has** Q8/Q4 KV cache with WHT
rotation, boundary layers, and NEON SIMD. CLI flags exist (`--kv-cache`, etc.). The
gap is making this as easy as `--quantize q4_0` and baking preferences into archives.

### Current State (already done in hologram base)
- `KvBits::Q8` and `KvBits::Q4` fully implemented in `kv_cache.rs` (1,982 lines)
- Per-channel affine quantization with scale/zero-point
- Walsh-Hadamard Transform rotation for V (improves Q4 quality)
- Boundary layer protection (first/last N layers stay F32)
- NEON SIMD on aarch64
- CLI flags wired in Plan 040 Tier 1

### Changes

1. **`crates/hologram-ai/src/compiler.rs`**
   - Add `kv_cache_config: Option<KvCacheConfig>` to `ModelCompiler`
   - Serialize into archive metadata so runtime reads it automatically

2. **Archive metadata** — store `KvCacheConfig` in `ModelMetaSection` or new section

### Verification
- Compile TinyLlama with `--kv-quant q8`, run decode, verify quality
- Memory benchmark: Q8 should use ~4x less KV cache than F32

---

## Pattern 5: SwiGLU Clamping for Numerical Stability

**What Qwen teaches:** Qwen clamps SiLU values to prevent overflow in mixed-precision
compute. Without clamping, `silu(x) = x * sigmoid(x)` can produce large values when
`x > 10`, which overflow in f16/Q8 intermediate precision.

**What hologram-ai gains:** Better numerical stability in quantized models, especially
at Q4. Prevents rare but catastrophic overflow in FFN layers.

### Changes

1. **`dispatch_swiglu()`** in hologram base — add optional clamping:
   `silu(clamp(x, -10, 10)) * up` (or make threshold configurable)
2. **AiOp/FloatOp** — add `clamp_gate: Option<f32>` field to `FusedSwiGLU`
3. **Detect from model config** — Qwen's config specifies activation clamping

### Verification
- Unit test: verify clamped vs unclamped output for values near boundaries
- E2E: run Q4 model with and without clamping, verify no NaN/Inf in FFN layers

---

## Pattern 6: Per-Layer Quantization Sensitivity

**What Qwen teaches:** Qwen's architecture is designed for clean quantization. Different
layers have different sensitivity — attention projections and output projections are more
sensitive than FFN layers. Qwen's KV cache boundary layers (protect first/last N layers
at F32) is this principle applied to caching.

**What hologram-ai gains:** Smarter weight quantization. Instead of uniform Q4 across
all layers, allow `--quantize q4_0 --protect-layers first:2,last:1` to keep critical
layers at higher precision. This pattern already exists for KV cache (`boundary_layers`)
but not for weights.

### Changes

1. **`QuantStrategy` enum** — extend with per-layer overrides:
   ```rust
   enum QuantStrategy {
       None,
       Q4_0,
       Q8_0,
       Mixed { default: QuantLevel, overrides: Vec<LayerOverride> },
   }
   ```

2. **`try_convert_f32_to_lut4`** in `builder.rs` — check layer index against overrides,
   skip quantization for protected layers

3. **CLI** — `--protect-layers first:2,last:1` or `--quant-sensitivity auto`

### Verification
- Compile TinyLlama Q4 with first/last layers protected at f32
- Compare perplexity/output quality vs uniform Q4

---

## Validation: Qwen2-0.5B Cross-Family Test

All six patterns above benefit any model family. Qwen2-0.5B serves as the validation
target because it exercises all of them (RoPE, GQA, SwiGLU, post-embedding RMSNorm)
and is the first non-LLaMA LLM in the test suite.

### V.1: Download + Compile + Decode

1. Download: `hologram-ai download Qwen/Qwen2-0.5B --format onnx -o models/Qwen2-0.5B`
2. New test file: `crates/hologram-ai/tests/qwen2_e2e.rs` (`#[cfg(feature = "e2e")]`)
   - `qwen2_onnx_compiles()` — verify compilation, check node count
   - `qwen2_onnx_decode()` — single-token generation, output shape, no NaN
   - `qwen2_variable_seq_len()` — test seq=1, 7, 128
   - Follow pattern from `tinyllama_e2e.rs`

### V.2: Tokenizer — Byte-Level BPE

Qwen uses BBPE with ~151K vocab. Verify:
- `tokenizer.json` loads without error
- Encode → decode = identity for ASCII, CJK, emoji, mixed scripts
- Special tokens: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`

### V.3: Position IDs

Qwen ONNX export likely emits `position_ids` input. Commit `aa65654` added support.
- Verify `position_ids` in graph inputs
- Verify decode step increments correctly from prefill length

### V.4: Architecture Detection

Currently hardcodes `"llama"` for any model with GQA.
- Detect `"qwen2"` from `config.json` companion file or ONNX metadata
- Map in `infer_llm_metadata_from_graph()` in `compiler.rs`
- Fall back to `"llama"` if unknown

### V.5: Post-Embedding RMSNorm

Qwen applies RMSNorm after embedding (LLaMA doesn't). Verify correctness unfused.
- SPRINT.md already flags `Embed + Norm fusion` as Wave 3 future work
- This validation confirms the unfused path works correctly

### Verification
- `cargo test -p hologram-ai --features e2e -- qwen2 --nocapture`
- Output logits shape: `[1, vocab_size]` where vocab_size ≈ 151,936
- Generated tokens not garbage (temperature=0)
- Tokenizer round-trips multilingual text
- Arch metadata reads `"qwen2"` not `"llama"`

---

## Execution Order

1. **Validation (V.1-V.5)** — download Qwen2-0.5B, attempt compilation. This
   reveals which patterns are actually blocking vs already-work-by-accident.
2. **Pattern 1 (RoPE)** — highest impact, unblocks the most models
3. **Pattern 3 (QK-Norm)** — same code paths as Pattern 1, bundle together
4. **Pattern 4 (KV cache exposure)** — smallest change, highest immediate value
5. **Pattern 2 (LogN)** — trivial to add once Pattern 1 is in
6. **Pattern 5 (SwiGLU clamping)** — surgical change, do when Q4 stability
   issues surface
7. **Pattern 6 (per-layer quant)** — design work, do when quality benchmarks
   justify it

## Key Files

| File | Pattern | Purpose |
|------|---------|---------|
| `hologram/crates/hologram-exec/src/float_dispatch/attention.rs` | 1,2,3 | Fused RoPE, LogN scaling, QK-Norm in attention kernel |
| `hologram/crates/hologram-exec/src/tape_builder.rs` | 1,2,3 | Wire new fields through TapeKernel |
| `hologram/crates/hologram-exec/src/tape.rs` | 1,2,3 | Update InlineAttention variant |
| `hologram/crates/hologram-core/src/op/float_op.rs` | 1,2,3 | RopeScaling enum, LogN fields |
| `hologram-ai/crates/hologram-ai-common/src/opt/attention_fusion.rs` | 1,3 | Detect RoPE + QK-Norm in ONNX |
| `hologram-ai/crates/hologram-ai-common/src/ir/op.rs` | 1,2,3 | Add rope_scaling, logn to GQA |
| `hologram-ai/crates/hologram-ai/src/compiler.rs` | 4,6,V | KvCacheConfig, arch detection, per-layer quant |
| `hologram/crates/hologram-exec/src/float_dispatch/elementwise.rs` | 5 | SwiGLU clamping |
| `hologram/crates/hologram-exec/src/kv_cache.rs` | 4 | Already done — reference |
| `hologram-ai/crates/hologram-ai/tests/qwen2_e2e.rs` | V | Qwen2 validation test |
| `hologram-ai/crates/hologram-ai/tests/tinyllama_e2e.rs` | V | Pattern to follow |
| `hologram-ai/crates/hologram-ai-tokenizer/` | V | BBPE verification |
| `hologram-ai/crates/hologram-ai/src/download/convert.rs` | V | Already supports qwen2 |

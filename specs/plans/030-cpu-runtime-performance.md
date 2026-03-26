# Plan 030 — CPU Runtime Performance: Wire LUT-GEMM + Reduce Overhead

## Context

The `hologram-ai run --prompt ...` operation takes ~2s end-to-end. The user wants sub-second, CPU-only.

**Critical finding**: hologram base has a complete LUT-GEMM execution path (`GraphOp::MatMulLut4/8` → `TapeKernel::MatMulLut4/8` → Psumbook → lut_gemm kernel) that reduces MatMul to cache-line-resident table lookups. But **hologram-ai never emits these opcodes**. 100% of MatMuls currently go through `FloatOp::MatMul` / `FloatOp::Gemm` — the slow f32 path. This is Plan 012 Phase 3, designed but never completed.

The Psumbook Q4 accumulator is 64 bytes (one cache line, fits in L1). Instead of streaming 4.1 GB of f32 weights through the memory hierarchy, LUT-GEMM uses 4-bit centroid indices + a 16-entry table. The working set is tiny. This bypasses the DDR bandwidth ceiling entirely.

**Two parallel workstreams**:
1. **Wire LUT-GEMM** (the transformative change — both repos)
2. **ExecutionSession** (eliminate per-step overhead — both repos)

---

## Phase 0: Instrumentation (Effort: S)

### 0.1 Wall-clock tracing spans in generation loop

Add `tracing::info_span!` around each phase in `run_cmd.rs`:
- Archive load (`HoloRunner::from_path`)
- Prefill (first `execute_with_kv`)
- Each decode step
- Print TTFT + tok/s summary at end

**File**: `crates/hologram-ai/src/commands/run_cmd.rs`

### 0.2 Per-phase timing in tape execution (hologram base)

Add `tracing::debug_span!` in `execute_tape` / `execute_tape_with_kv` around:
- `node_dtypes_map()` / `node_shapes_map()` construction
- `seed_arena` + `prewarm_arena`
- `tape.execute()` (actual compute)

**File**: hologram base `crates/hologram-exec/src/mmap/mod.rs`

---

## Phase 1: Wire LUT-GEMM Path (Effort: M, Impact: Transformative)

This is the single biggest lever. Connects hologram-ai's compiler to hologram base's existing LUT-GEMM infrastructure.

### 1.1 Compile-time weight quantization pass (hologram-ai)

New optimization pass that runs after the standard opt pipeline, before lowering:

1. Walk the graph; find MatMul/Gemm nodes whose weight input is an `AiParam`
2. For each weight tensor:
   - Read f32 data (from Inline or Mmap)
   - Run k-means clustering to produce Q centroids (Q=16 for Q4, Q=256 for Q8)
   - Encode weights as centroid indices (4-bit packed for Q4, 8-bit for Q8)
   - Store the quantized weight + centroid table as a new `AiParam::Inline`
3. Replace the `AiOp::MatMul` node with `AiOp::QuantizedMatMul { scheme: Q4_0 }`
4. Mark the weight `TensorInfo` with the quantization descriptor

**Existing infrastructure to reuse**:
- `hologram-ai-quant/src/q4_0.rs` — Q4_0 quantizer (has `quantize_q4_0`)
- `hologram-ai-quant/src/q8_0.rs` — Q8_0 quantizer
- hologram base `crates/hologram-exec/src/lut_gemm/quantize.rs` — k-means clustering (`quantize_weights_q4`, `quantize_weights_q8`), centroid table generation

**CLI integration**: Add `--quantize q4_0` / `--quantize q8_0` flag to `hologram-ai compile`.

**Files**:
- New: `crates/hologram-ai-common/src/opt/weight_quantize.rs`
- Modify: `crates/hologram-ai/src/cli.rs` (new `--quantize` flag)
- Modify: `crates/hologram-ai/src/compiler.rs` (insert pass)
- Reuse: `crates/hologram-ai-quant/src/q4_0.rs`, `q8_0.rs`

### 1.2 Lowering: emit GraphOp::MatMulLut4/8 (hologram-ai)

Currently `dispatch.rs:151-166` marks `QuantizedMatMul` as `Unsupported`. Change this:

1. In `dispatch.rs`: Route `AiOp::QuantizedMatMul { Q4_0 }` → `DispatchTarget::GraphOp(GraphOp::MatMulLut4(constant_id))` (same for Q8)
2. In `builder.rs`: When emitting a `MatMulLut4` node, register the quantized weight blob as a `ConstantData` in the graph and pass its `ConstantId` to the `GraphOp`
3. The weight blob format must match what hologram base's `lut_gemm_4bit()` expects: packed 4-bit indices + centroid table (see `psumbook.rs` and `matmul.rs` in hologram-exec)

**Files**:
- Modify: `crates/hologram-ai-common/src/lower/dispatch.rs` (route QuantizedMatMul)
- Modify: `crates/hologram-ai-common/src/lower/builder.rs` (emit MatMulLut4/8 with ConstantId)
- Modify: `crates/hologram-ai-common/src/lower/strategy.rs` (handle quantized weight shape)

### 1.3 GGUF quantized weights → LUT-GEMM (hologram-ai)

GGUF models already carry Q4_0/Q8_0 weights with `QuantDescriptor`. Currently these get lowered to `FloatOp::Gemm { quant_b: 1 }` which hologram base handles via the float path (with dequant overhead).

Instead: convert GGUF Q4_0 block format → hologram LUT-GEMM format (centroid indices + centroid table) during lowering, and emit `GraphOp::MatMulLut4`.

**Files**:
- Modify: `crates/hologram-ai-common/src/lower/strategy.rs` (detect quant_b > 0, emit MatMulLut)
- Reuse: hologram base `lut_gemm/quantize.rs` for format conversion

### 1.4 Verify end-to-end (both repos)

- Compile TinyLlama ONNX with `--quantize q4_0`
- Compile TinyLlama GGUF (already Q4_0)
- Run both with `--prompt "Tell me a joke"`
- Verify coherent output + measure tok/s
- Expected: MatMul goes through `TapeKernel::MatMulLut4` → Psumbook → lut_gemm_4bit kernel

---

## Phase 2: ExecutionSession — Eliminate Per-Step Overhead (Effort: M, Impact: ~100-200ms)

### 2.1 Persistent ExecutionSession (hologram base)

Every `execute_tape_with_kv` call currently rebuilds:
- `node_dtypes_map` HashMap (~500 insertions)
- `node_shapes_map` HashMap (~500 insertions)
- `BufferArena` + seeds all constants
- `prewarm_arena` (iterates all instructions)
- `TapeContext` with `mem::replace` KV swap dance

For 21 steps, that's 21x redundant work.

**Fix**: New `ExecutionSession` struct:

```rust
pub struct ExecutionSession<'a> {
    tape: &'a EnumTape,
    plan: &'a LoadedPlan,
    arena: BufferArena<'a>,
    dtype_map: HashMap<NodeId, FloatDType>,
    shape_map: HashMap<NodeId, Vec<usize>>,
    weight_cache: WeightCache,
    kv_state: Option<KvCacheState>,
}

impl ExecutionSession {
    fn new(tape, plan) -> Self;
    fn prefill(&mut self, inputs) -> GraphOutputs;
    fn decode_step(&mut self, inputs) -> GraphOutputs;
}
```

Arena keeps constants pinned; clears only intermediates between steps. Maps built once. KV state owned (no swap).

**Files**:
- hologram base: new `crates/hologram-exec/src/session.rs`
- hologram base: `crates/hologram-exec/src/mmap/mod.rs` (public `create_session` API)

### 2.2 Wire ExecutionSession from HoloRunner (hologram-ai)

Replace per-call `execute_tape_with_kv` in the generation loop with session-based API:
- `HoloRunner::create_session()` → `ExecutionSession`
- Generation loop calls `session.prefill()` then `session.decode_step()` in a loop
- Session owns KvCacheState — no separate init/pass/swap

**Files**:
- `crates/hologram-ai/src/compiler.rs` (HoloRunner gains `create_session` method)
- `crates/hologram-ai/src/commands/run_cmd.rs` (generation loop uses session)

### 2.3 Dual tape: prefill + decode

Build two tapes from the same `LoadedPlan`:
- Prefill tape: variable seq via resolve_size
- Decode tape: all shapes pre-resolved for seq=1, batch=1

The decode tape eliminates all `resolve_size()` overhead in the hot path. Memory cost is negligible (tapes are kernel pointers + metadata, not weights).

**Files**:
- hologram base: `build_tape_for_seq` API variant
- hologram-ai: `compiler.rs` HoloRunner holds `decode_tape: Option<EnumTape>`

---

## Phase 3: Additional CPU Optimizations (Effort: M)

### 3.1 QK-Norm + RoPE + KV-Store fusion

Last unimplemented fusion from Plan 019. Fuses 5-7 ops per attention layer into 1 dispatch. AiOp fields (`qk_norm`, `rope`) already exist as placeholders.

**Blocked on**: hologram base wiring the flags in `dispatch_attention()`.

**Files**:
- hologram-ai: `crates/hologram-ai-common/src/opt/` — re-add PreAttentionFusion pass
- hologram base: wire `qk_norm`/`rope` flags in attention kernel

### 3.2 Remaining clone elimination

SPRINT.md lists remaining `.clone()` calls. Profile with Phase 0 instrumentation first.

---

## Expected Outcome

| Scenario | Current | After Phase 1 | After Phase 1+2 |
|----------|---------|---------------|-----------------|
| f32 ONNX, 20 tok | ~2.0s | ~2.0s (no change, still f32) | ~1.7-1.8s |
| Q4_0 (via --quantize), 20 tok | N/A | **~0.3-0.5s** | **~0.2-0.4s** |
| Q4_0 GGUF, 20 tok | ~2.0s (using float path!) | **~0.3-0.5s** | **~0.2-0.4s** |
| TTFT (Q4 prefill) | ~170ms | ~30-60ms | ~25-50ms |

**The path to sub-second**: Wire LUT-GEMM (Phase 1) + ExecutionSession (Phase 2).

---

## Verification

1. **Phase 0**: `RUST_LOG=hologram_ai=info hologram-ai run ...` shows per-phase timing
2. **Phase 1**:
   - `hologram-ai compile --quantize q4_0 -m tinyllama.onnx -o tinyllama-q4.holo`
   - `hologram-ai run tinyllama-q4.holo --prompt "Tell me a joke"` → coherent text
   - Tracing shows `MatMulLut4` dispatches (not `InlineMatMul`)
   - Tok/s > 40 (vs 13.6 for f32)
3. **Phase 2**: Decode step criterion benchmark shows overhead reduced; generation timing improves
4. **All phases**: `cargo test`, `cargo clippy -- -D warnings` pass; no `.unwrap()`, no `println!`

## Critical Files

| File | Repo | Changes |
|------|------|---------|
| `crates/hologram-ai-common/src/opt/weight_quantize.rs` | hologram-ai | **New**: compile-time weight quantization pass |
| `crates/hologram-ai-common/src/lower/dispatch.rs` | hologram-ai | Route QuantizedMatMul → GraphOp::MatMulLut4/8 |
| `crates/hologram-ai-common/src/lower/builder.rs` | hologram-ai | Emit MatMulLut4/8 with ConstantId |
| `crates/hologram-ai-common/src/lower/strategy.rs` | hologram-ai | Handle quantized weight lowering |
| `crates/hologram-ai/src/cli.rs` | hologram-ai | `--quantize` flag |
| `crates/hologram-ai/src/compiler.rs` | hologram-ai | Quantize pass insertion, HoloRunner session |
| `crates/hologram-ai/src/commands/run_cmd.rs` | hologram-ai | Tracing spans, session-based generation loop |
| `crates/hologram-ai-quant/src/q4_0.rs` | hologram-ai | Reuse: existing Q4_0 quantizer |
| hologram `crates/hologram-exec/src/session.rs` | hologram base | **New**: ExecutionSession |
| hologram `crates/hologram-exec/src/mmap/mod.rs` | hologram base | Session API, tracing spans |
| hologram `crates/hologram-exec/src/lut_gemm/quantize.rs` | hologram base | Reuse: k-means + centroid table |
| hologram `crates/hologram-exec/src/lut_gemm/matmul.rs` | hologram base | Reuse: lut_gemm_4bit/8bit kernels |
| hologram `crates/hologram-exec/src/tape.rs` | hologram base | Already handles MatMulLut4/8 dispatch |
| hologram `crates/hologram-graph/src/graph/mod.rs` | hologram base | Already defines GraphOp::MatMulLut4/8 |

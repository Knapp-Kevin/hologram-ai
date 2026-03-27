# Plan 034 — Runtime Acceleration Roadmap

## Context

TinyLlama generates "The capital of France is Paris." at 20.5 tok/s
(compiled seq=2048, variable-length prefill, decode prewarm skip).
The f32 path is memory-bandwidth-bound at ~15 tok/s theoretical max.
The 20.5 tok/s includes the prewarm skip optimization.

Three acceleration paths are ready to wire but disconnected:
1. LUT-GEMM quantization (Psumbook, 64-byte L1-resident accumulators)
2. Epilogue fusion (MatMul+Activation, bias fusion — hologram base Sprints 23-24)
3. ShapeContextGraph for true shape-polymorphic execution

## Current State

| Component | Status | Location |
|-----------|--------|----------|
| `--quantize q4_0` CLI flag | Done | cli.rs |
| `try_convert_f32_to_lut4` | Done | builder.rs |
| GGUF Q4_0 → `MatMulLut4` interception | Done | builder.rs:340 |
| `TapeKernel::MatMulLut4` dispatch | Done | hologram base tape.rs |
| Psumbook Q4/Q8 kernels | Done | hologram base lut_gemm/ |
| `MatMulLut4Activation` tape kernel | Done | hologram base tape.rs |
| `FusedMatMulBiasActivation` tape kernel | Done | hologram base tape.rs |
| `InlineMatMulActivation` tape kernel | Done | hologram base tape.rs |
| ShapeContextGraph walker | Done | shape_spec_bridge.rs:835 |
| ShapeProjection trait (100+ ops) | Done | shape_spec_bridge.rs:42 |
| Pre→post fusion node ID mapping | Not done | — |
| Decode prewarm skip | Done | hologram base mmap/mod.rs |
| Variable-length at full context | Works | verified at seq=2048 |
| Variable-length at intermediate seq | Broken | Reshape meta ambiguity |

---

## Sprint A: LUT-GEMM End-to-End (highest impact)

### Goal
GGUF Q4_0 TinyLlama generates coherent text via the Psumbook LUT-GEMM path.

### Why
f32 decode at 20.5 tok/s is near the DDR bandwidth ceiling. Q4 weights
are 8× smaller — the Psumbook accumulator (64 bytes) fits in a single
cache line. This changes decode from bandwidth-bound to compute-bound.

### Tasks

1. **Verify GGUF Q4_0 compilation fires `MatMulLut4`**
   - Compile `model_causal.onnx` equivalent GGUF at full context
   - Check tracing output for "LUT-GEMM: converted Q4_0 Gemm → MatMulLut4"
   - File: GGUF compiler path in builder.rs

2. **Verify GGUF end-to-end generation**
   - `hologram-ai run <gguf.holo> --prompt "..." --max-tokens 20`
   - Expect coherent English output
   - Measure tok/s (target: >40 tok/s)

3. **Verify `--quantize q4_0` on ONNX causal model**
   - `hologram-ai compile --quantize q4_0 -m model_causal.onnx`
   - Run and verify output quality
   - Compare tok/s vs f32 baseline

### Files
- `crates/hologram-ai-common/src/lower/builder.rs` (LUT-GEMM interception)
- `crates/hologram-ai-quant/src/q4_0.rs` (quantizer)
- hologram base `crates/hologram-exec/src/lut_gemm/` (kernels)

---

## Sprint B: Epilogue Fusion Wiring

### Goal
Wire hologram-ai's fused AiOp variants to hologram base's fused TapeKernel
variants, eliminating intermediate buffers.

### Why
hologram-ai has `AiOp::MatMulRelu/Gelu/Silu` and `AiOp::FusedLayerNormResidual`.
hologram base has `InlineMatMulActivation`, `MatMulLut4Activation`,
`InlineMatMulBiasActivation`, `InlineRmsNormActivation`. But hologram-ai's
lowering maps everything as plain `MatMul` — the fused kernels are never used.

### Tasks

1. **Lower `MatMulRelu/Gelu/Silu` → `FusedMatMulActivation` graph ops**
   - In `strategy.rs`: when lowering fused MatMul variants, emit
     `GraphOp::FusedMatMulActivation { m, k, n, activation }`
   - Tape builder already resolves these to `InlineMatMulActivation`

2. **Lower quantized fused variants → `FusedMatMulLut4Activation`**
   - In `builder.rs`: when `--quantize q4_0` is used and the MatMul has
     a fused activation, emit `MatMulLut4Activation`

3. **Verify fusion fires in compiled graph**
   - Count fused ops in TinyLlama compilation output
   - Benchmark decode tok/s before/after

### Files
- `crates/hologram-ai-common/src/lower/strategy.rs`
- `crates/hologram-ai-common/src/lower/builder.rs`
- hologram base `crates/hologram-exec/src/tape_builder.rs`

---

## Sprint C: ShapeContextGraph Post-Fusion (Plan 033 completion)

### Goal
Compute ShapeContextGraph from the POST-fusion graph so node IDs match
the runtime tape. This enables true shape-polymorphic execution at ANY
compiled seq_len, eliminating the Reshape meta ambiguity.

### Why
The ShapeContextGraph walker produces correct shapes (validated: 1034 nodes
resolved). But it uses pre-fusion node IDs. The fusion pass adds/removes
nodes, invalidating the mapping. Option B from Plan 033: maintain a pre→post
node ID mapping.

### Design

The fusion pass (`hologram::compile()`) already returns `CompilationOutput`.
Add a `node_id_mapping: HashMap<NodeId, NodeId>` field that records which
pre-fusion nodes map to which post-fusion nodes:
- Surviving nodes: identity mapping (same ID)
- Removed nodes (absorbed by fusion): map to the fused node's ID
- New nodes (fused ops): no pre-image needed

The mapping flows through `CompilationOutput` → hologram-ai `compile()` →
used to remap `ShapeContextGraph` entries before embedding in the archive.

### Tasks

1. **Add `node_id_mapping` to `CompilationOutput`** (hologram base)
   - Fusion pass records mappings in `FusionStats` or a new struct
   - `compile()` returns the mapping

2. **Remap ShapeContextGraph after compilation** (hologram-ai)
   - After `hologram::compile()`, apply mapping to all `node_id` fields
     in `ShapeContextGraph.seeds` and `ShapeContextGraph.projections`
   - Remove dangling entries (nodes that were deleted by fusion)

3. **Re-enable shape overrides in `HoloRunner`** (hologram-ai)
   - `resolve_shapes()` returns the walker's shape_map (currently disabled)
   - `build_tape_from_plan_with_shapes()` uses adjusted kernel params

4. **Conformance gate**
   - `onnx_kv_decode_variable_length` test passes at seq=32 actual=7

### Files
- hologram base `crates/hologram-compiler/src/compiler/mod.rs` (mapping)
- hologram base `crates/hologram-graph/src/fusion/` (record mappings)
- `crates/hologram-ai/src/compiler.rs` (remap + re-enable)
- `crates/hologram-ai/tests/mini_fixture.rs` (conformance)

---

## Execution Order

| Sprint | Effort | Impact | Depends On |
|--------|--------|--------|------------|
| **A: LUT-GEMM** | S (verify + fix) | Transformative (4-5× tok/s) | — |
| **B: Epilogue Fusion** | M (lowering changes) | Medium (buffer elimination) | — |
| **C: ShapeContextGraph** | L (cross-repo) | Architectural (any-seq) | — |

A and B are independent and can run in parallel.
C is architectural and can start after A/B are validated.

---

## Verification

- Sprint A: `hologram-ai run <gguf.holo> --prompt "Tell me a joke"` → coherent text at >40 tok/s
- Sprint B: TinyLlama compilation shows fused op counts; benchmark shows improvement
- Sprint C: `cargo test -p hologram-ai --features e2e -- onnx_kv_decode_variable` passes
- All: `cargo test`, `cargo clippy -- -D warnings` clean

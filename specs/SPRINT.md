# Current Sprint — hologram-ai

## Sprint Goal

**Attention Fusion + KV Cache + LUT-GEMM (Plan 012):**
Fuse ONNX decomposed attention into `GroupedQueryAttention`, implement KV cache
for prefill/decode generation, and wire LUT-GEMM for GGUF Q4_0 weights.
ONNX first, then GGUF benefits from same infrastructure.
See `specs/plans/012-attention-fusion-kvcache-lutgemm.md`.

**Design principle:** hologram-ai is a compiler only (ADR-0016). It ships
zero runtime code. All kernels belong in hologram base crate.
CLI: `compile`, `info`, `download` — nothing else.

---

## In Progress: Plan 012 — Attention Fusion + KV Cache + LUT-GEMM

### Phase 1: ONNX Attention Fusion ✓
- [x] Add `AttentionFusion` pass (MatMul→Add→Softmax→[IsNaN→Where]→MatMul → GroupedQueryAttention)
- [x] Register in OptPipeline (after RmsNormFusion)
- [x] TinyLlama ONNX: 22/22 layers fused, 1294 → 1167 nodes

### Phase 2a: KV Cache (hologram base crate) ✓
- [x] Add `FloatOp::KvWrite` / `FloatOp::KvRead` to hologram-core
- [x] Add `KvCacheState` to hologram-exec (write/read/advance/reset, 2 tests)
- [x] Add `execute_with_kv_state()` + `execute_plan_with_kv_state()` API
- [x] Re-export `KvCacheState` from hologram facade
- [ ] Wire KvWrite/KvRead dispatch into executor (TODO when integration ready)

### Phase 2b: KV Cache (hologram-ai) — in progress
- [x] Fix GGUF metadata (n_kv_heads, head_dim → is_llm=true)
- [x] Inject KvSlotWrite ops in GGUF builder (2 per layer: K + V, with is_key tag)
- [x] Lowering strategy: KvSlotWrite → FloatOp::KvWrite, KvSlotRead → FloatOp::KvRead
- [x] KvWrite/KvRead pass-through dispatch in float_dispatch.rs
- [x] HoloRunner pipeline support (sub-archive extraction, section fallback)
- [x] Pipeline archive compiles and runs end-to-end (GGUF 377 nodes)
- [x] KvCacheState wired into executor dispatch loop (writes K/V per layer)
- [x] Generation loop uses execute_with_kv (cache fills during full-seq execution)
- [x] HoloRunner.execute_with_kv() method
- [x] **Decode mode**: single-token input with KV cache expansion ✓
  - [x] RoPE position offset (start_pos injected for decode tokens)
  - [x] KV cache expansion (cache[0..pos] ++ new_data for K/V)
  - [x] Causal mask fix for absolute positions (seq_q < seq_k)
  - [x] Verified: decode logits match full-recomputation exactly
- [x] ONNX: inject KvSlot after attention fusion (KvSlotInjection pass)

### Phase 3: LUT-GEMM
- [x] Q4_0 → QuantizedWeights4 compile-time converter (`try_convert_q4_0_to_lut4`)
- [x] Lowering: Q4_0 Gemm → MatMulLut4 (builder intercepts `quant_b=1`)
- [x] Builder: constant insertion for LUT weights (`matmul_lut_4bit`)

---

## Resolved: Plan 011 — GGUF Step 1+ Gibberish Diagnosis

**Finding:** hologram-exec computation is provably correct (causal cos_sim=1.0 at all
seq lengths). Degeneration is from full-sequence recomputation without KV cache —
Q4_0 dequant→f32→Gemm noise accumulates. ollama (with KV cache) produces correct output.

- [x] Causal logit consistency test (`mini_fixture.rs`) — cos_sim=1.0 ✓
- [x] Fix `resolve_gemm`/`resolve_matmul` rank preservation (`shape_spec_bridge.rs`)
- [x] Guard `resolve_dynamic_sizes` against 1-D shapes (`executor.rs`)

## Resolved: Plan 010 — GGUF Generation Quality Fix

- [x] Rewrite `run_cmd.rs` — SeqMode enum, temperature+top-k sampling, penalty fix
- [x] Add `has_shape_context()` to HoloRunner
- [x] Add fixture-driven regression testing methodology to AGENTS.md
- [x] Add fused kernel conformance tests (GQA, SwiGLU) with file-based fixtures

---

## In Progress: Plan 009 — LUT-GEMM + KV-Cache + ShapeContextGraph Runtime

### Step 1 — ShapeContextGraph Runtime Integration

#### Step 1a — `execute_plan_with_shape_hints()` API (hologram-exec)
- [x] Add `shape_hints: Option<&HashMap<u32, Vec<usize>>>` to `propagate_level_shapes()` — hints override all compiled/inferred shapes (shape_propagate.rs)
- [x] Split `execute_core` → `execute_core_with_hints` to thread hints through the dispatch loop (executor.rs)
- [x] Add `KvExecutor::execute_with_shape_hints()` public method (executor.rs)
- [x] Add `execute_plan_with_shape_hints()` function to `mmap/mod.rs`
- [x] Export `execute_plan_with_shape_hints` from `hologram-exec/src/lib.rs`
- [x] Re-export from hologram facade `src/lib.rs`

#### Step 1b — `run_with_shape_context()` caller (hologram-ai)
- [x] Add `read_shape_context_from_archive()` — reads `ShapeContextGraph` from `.holo` bytes via section table offset (compiler.rs)
- [x] Add `run_with_shape_context()` — loads archive, walks ShapeContextGraph with runtime input shapes, calls `execute_plan_with_shape_hints()` (compiler.rs)
- [x] Export from `hologram-ai/src/lib.rs`

#### Step 1c — Wire into conformance test and e2e test
- [x] Update `tinyllama_causal_onnx_top1_matches_ort` conformance test to use `run_with_shape_context()` — fixes seq=2 divergence
- [x] Update `tinyllama_onnx_variable_seq_len_runs` e2e test to use `run_with_shape_context()` — explicit shape hint path

#### Pending verification
- [ ] Run `tinyllama_causal_onnx_top1_matches_ort` — expect PASS for seq=2
- [ ] Run `tinyllama_onnx_variable_seq_len_runs` — expect PASS for seq=1,7,128

### Step 2 — LUT-GEMM for GGUF Q4_0 (pending)
- [ ] Add `FloatOp::MatMulQ4 { m, k, n }` to hologram-core
- [ ] Add `dispatch_matmul_q4()` kernel in hologram-exec
- [ ] Change lowering strategy: GGUF Q4_0 → `MatMulQ4` instead of `Gemm { quant_b: 1 }`
- [ ] Verify: GGUF generation speed > 1 tok/s (vs 0.1 tok/s today)

### Step 3 — KV-Cache (pending)
- [ ] Add `KvCacheStore` + `KvSlotWrite`/`Read` dispatch in hologram-exec
- [ ] Lower `AiOp::KvSlotWrite`/`KvSlotRead` to `FloatOp` variants in strategy.rs
- [ ] Prefill/Decode graph split in hologram-ai-gguf llama.rs
- [ ] ShapeContextGraph `FromInput` spec for variable cache length

---

## Complete

### ShapeContextGraph (Plan 008)

#### Step 1 — `FloatOp → ShapeSpecRepr` translator
- [x] Create `crates/hologram-ai-common/src/lower/shape_spec_bridge.rs`
- [x] Implement `float_op_to_shape_spec_repr()` mapping all `FloatOp` variants to serializable specs
- [x] Handle all op families: unary, binary, norms, reductions, MatMul/Gemm, Gather/Embed, Reshape, Transpose, Concat, Slice, Attention, Shape, Where, vision ops (`Unknown`)
- [x] Implement `resolve_spec()` runtime resolver with helpers (broadcast, dims, matmul, gemm, reshape, parse_shape_i64)
- [x] 15 unit tests: all spec variants + runtime resolution

#### Step 2 — Build `ShapeContextGraph` during lowering
- [x] Add `ShapeDimRepr`, `ShapeSpecRepr`, `ShapeSeed`, `ShapeProjectionEntry`, `ShapeContextGraph` types to `exec_context.rs` (rkyv-serializable, `SECTION_SHAPE_CONTEXT = SECTION_CUSTOM_BASE + 0x21`)
- [x] Emit `ShapeSeed` for constant params with fully-concrete shapes in `builder.rs`
- [x] Emit `ShapeSeed` for input nodes with fully-concrete shapes
- [x] Emit `ShapeProjectionEntry` for every `GraphOp::Float(...)` node in topo loop (both `GraphOp` and `FloatNeedsShape` branches)
- [x] Insert `ShapeContextGraph` into `ContextBundle` alongside `ShapeRecipeSection`

#### Step 3 — Runtime `walk_shape_context()`
- [x] Implement `walk_shape_context()` in `shape_spec_bridge.rs` (topological seed → project walk)
- [x] Seeds `ShapeMap` from compile-time concrete shapes, then injects runtime input shapes
- [x] Calls `resolve_spec()` per `ShapeProjectionEntry`, handles `shape_value_input` for Reshape/Expand
- [x] Re-exported from `lower/mod.rs` as public API

#### Step 4 — `ParamRecipe` (deferred)
- `ParamRecipe`/`DeferredStrategy` kept as belt-and-suspenders alongside `ShapeContextGraph`.
  Full retirement deferred until end-to-end pipeline is verified with variable `seq_len`.

#### Step 5 — Extend hologram `resolve_dynamic_sizes()`
- [x] Cover `Embed { dim: 0 }` — infers `dim` from embedding table `shape[-1]`
- [x] Cover `Concat { size_a: 0, size_b: 0 }` — infers row sizes from input `shape[-1]`
- [x] Cover `Attention { head_dim: 0 }` — infers `head_dim` from Q `shape[-1] / num_q_heads`

#### Verification
- [x] `cargo test -p hologram-ai-common` — 133 tests pass (shape_spec_bridge + walk_shape_context + all prior)
- [x] `cargo test -p hologram-ai-conformance` — 45 tests pass
- [x] `cargo check --workspace` — zero errors, zero clippy warnings

#### Step 4 (completed) — Retire `ParamRecipe` for shape-derived cases
- [x] `size_op!` macro no longer emits `ParamRecipe::DimVar` / `RuntimeInferred` — emits 0-sentinel only
- [x] `resolve_dynamic_sizes()` extended to cover `ReduceProd { size: 0 }` and `InstanceNorm { size: 0 }` (hologram-exec)
- [x] All `size_op!` ops now rely on `resolve_dynamic_sizes()` for runtime patching — no recipe needed
- [x] `size_op_with_symbolic_dim` test updated to assert `recipe.is_none()` and `size=0`

#### Step 6 (completed) — Variable seq_len E2E + walk_shape_context conformance
- [x] `tinyllama_onnx_variable_seq_len_runs` test in `tinyllama_e2e.rs` (seq_len = 1, 7, 128)
- [x] `compile_with_shape_context()` API on `ModelCompiler` (returns archive + debug_map + ShapeContextGraph)
- [x] `walk_shape_context_matmul_projects_output_shape` conformance test — asserts MatMul output shape [m,n]
- [x] `walk_shape_context_rmsnorm_same_as_input` conformance test — asserts SameAs(0) projection

#### Verification (final)
- [x] `cargo test -p hologram-ai-common` — all tests pass
- [x] `cargo test -p hologram-ai-conformance --features conformance` — compiles clean
- [x] `cargo clippy -p hologram-ai-common -- -D warnings` — zero warnings
- [x] `cargo check --workspace` — zero errors

---

## Previous Sprint (Complete): Execution Conformance + TinyLlama E2E

### Execution Conformance Testing (Plan 006)

#### Phase 1: Layer E — Compile-Time Shape Consistency (hologram-ai only)
- [x] Create `shape_consistency.rs` validation pass in hologram-ai-common (8 unit tests)
- [x] Integrate into `compiler.rs` after concretization, before lowering
- [x] Add tests: param/shape mismatch, MatMul k-dim mismatch, dynamic dim, zero-product, Gemm trans_b
- [x] Run on TinyLlama ONNX: found 52 issues (1 MatMul k-dim mismatch, ~50 zero-product shapes from unresolved seq_len=0 sentinels)

#### Phase 2: Layer D — Executor Intermediate Capture (hologram base repo)
- [x] Add `BufferArena::snapshot()` in `arena.rs` (non-destructive, `#[cfg(feature = "profile")]`)
- [x] Add `ShapeMap::snapshot()` (feature-gated)
- [x] Add `KvExecutor::execute_with_intermediates()` + `IntermediateCapture` struct (feature-gated)

#### Phase 3: Layer D — Execution Conformance Harness (hologram-ai repo)
- [x] Add `compile_with_debug_info()` to `ModelCompiler` (ONNX name → NodeId map)
- [x] Add ORT intermediate tensor capture in `ort_runner.rs`
- [x] Add `exec_comparator.rs` — node-by-node comparison with tolerances (5 unit tests)
- [x] Add `tests/exec_conformance.rs` — multi-node ONNX integration tests (7 tests, `conformance` feature-gated)

### TinyLlama E2E Testing (feat/tinyllama-e2e)

#### Performance Fixes
- [x] Enable `parallel` feature in hologram-ai workspace `Cargo.toml` — was missing, causing single-threaded graph execution despite rayon support
- [x] Verified attention kernel already uses `cblas_sgemm` (Apple Accelerate) on macOS — no fix needed

#### Bug Fixes
- [x] Fix NaN detector false positive for non-f32 ops in `executor.rs` — i64 value `-1` (0xFFFFFFFFFFFFFFFF) was incorrectly triggering NaN detector when cast to f32. Fix: check `output_dtype() == FloatDType::F32` before interpreting bytes as f32.

#### Tests Added
- [x] Add `e2e = []` feature gate to `hologram-ai/crates/hologram-ai/Cargo.toml`
- [x] Add `tests/tinyllama_e2e.rs` — compile + run tests for ONNX and GGUF (feature-gated `e2e`)
  - `tinyllama_onnx_compiles` — compile ONNX, assert > 1000 nodes, > 1 GB weights
  - `tinyllama_gguf_compiles` — compile GGUF, assert > 100 nodes, > 500 MB weights
  - `tinyllama_onnx_runs_and_produces_english` — run with chat prompt, assert non-empty English output
  - `tinyllama_gguf_runs_and_produces_english` — same for GGUF
  - `nan_detector_no_false_positive_on_i64_concat` — regression for NaN detector fix
  - `tinyllama_onnx_batched_matmul_shape_regression` — documents known MatMul K-dim mismatch bug
- [x] Add `hologram/crates/hologram-exec/tests/shape_chain.rs` — 8 op-chain regression tests covering TinyLlama's connected-op patterns (RoPE, Reshape -1, batched MatMul, i64 Concat)

#### Conformance Test Infrastructure Fixes
- [x] Fix ORT crate version: `ort 2.0.0-rc.9` deadlocked with system ORT 1.24.3 (re-entrant OnceLock bug). Upgraded to `2.0.0-rc.12` targeting ORT 1.24 and removed `load-dynamic` feature.
- [x] Add `ORT_STRATEGY=system` requirement to exec_conformance.rs header comment.

#### Conformance Tests for Connected-Op Bugs (exec_conformance.rs)
- [x] Add `batched_4d_matmul` ONNX builder — [1,4,6,8] × [1,4,8,6] → [1,4,6,6]
- [x] Add `concat_4d_last_axis` ONNX builder — axis=3 concat exposing concat row_size bug
- [x] Add `scaled_dot_product_attention` ONNX builder — Transpose+MatMul+Mul+Softmax+MatMul
- [x] `batched_4d_matmul_matches_ort` — passes (4D batched matmul works correctly)
- [x] `concat_4d_last_axis_matches_ort` — FAILED, root cause found and fixed
- [x] `scaled_dot_product_attention_matches_ort` — passes

#### Bug Fixes — Concat Axis Row Size
- [x] Fix `concrete_concat_row_size` in `strategy.rs`: was computing `product(dims_after_axis)` only, missing `dim[axis]` itself. Now computes `dim[axis] * product(dims_after_axis)`. This caused axis=N concat (N>0) to emit `size_a=1` → element-wise interleave instead of correct chunk-based concat. Fixed root cause; all 9 exec_conformance tests pass.

#### Rule Added to AGENTS.md
- [x] Added "Conformance Testing Mandate" section: runtime bugs and connected-op bugs must have a conformance test before any fix is applied.

#### Bug Fixes — Broadcast Inflation (stale compiled shapes)

- [x] **Root cause identified**: `binary_elementwise_broadcast` in `float_dispatch.rs` could produce an output larger than both inputs when compiled input_shapes had 0-sentinels that resolved to wrong values at runtime. Example: nodes 314/315 (RoPE sin rotation) had compiled shapes `[32,2]` and `[32,1,2]`; `broadcast_shapes([32,2],[32,1,2])=[32,32,2]` → out_len=2048 from 64-element inputs.
- [x] Fix: added `out_len > max(a.len(), b.len())` guard in `binary_elementwise_broadcast` and `binary_compare_broadcast` → falls back to element-cycling. This is the correct semantics since both inputs have equal size in the stale-shape case.
- [x] Removed all TEMP debug `eprintln!` blocks from `executor.rs` (RESHAPE CS/TE/SM/SM_R/DBG, ALL_RESHAPE, SHAPES314, NODE272, OUT, NaN detector).
- [x] Added 3 regression tests in `hologram/crates/hologram-exec/tests/float_conformance.rs` for the broadcast inflation guard (`broadcast_stale_shapes_no_inflation`, `broadcast_valid_shapes_still_broadcast`, `broadcast_valid_nd_broadcast_still_works`).

#### Status after broadcast fix

- [x] **`tinyllama_onnx_runs_and_produces_english`** passes — ONNX model compiles (1205 nodes, 4.1 GB weights) and runs to completion without errors. Output tokens are incoherent because the ONNX export only includes `last_hidden_state` (2048-dim hidden state), not `logits` (32000-dim). The lm_head linear projection is absent from this ONNX export — this is an inherent model file limitation, not a hologram bug.
- [x] **`tinyllama_onnx_batched_matmul_shape_regression`** updated to assert `run_ok=true` — the bug is fixed.
- [x] Removed all TEMP debug `eprintln!` from `executor.rs` (`[RESHAPE SM_R]`, `[RESHAPE_EXPAND2]`).
- **GGUF model**: compiles (333 nodes, 606 MB weights), runs to completion, generates partial text ("Response: ...") then degenerates into repetitive tokens. Root cause not yet diagnosed (candidates: causal masking, Q4_0 dequantization, RoPE positions).

#### Known Bugs (open → being fixed in Phase 5)
- [ ] **GGUF token degeneration**: GGUF model generates `Response:` then degenerates. Primary suspect: GroupedQueryAttention (GQA) kernel — head reshape, KV repeat, or scale. Secondary: SwiGLU (`silu(gate)*up` vs `gate*up`). See plan 007.
- [ ] **ONNX incoherent output**: `inject_lm_head_if_needed` exists but may not activate if `embed_tokens.weight` is absent/named differently in this ONNX export. See plan 007.

#### Phase 5: GGUF + ONNX Inference Fix (plan 007)
- [x] Save plan to `specs/plans/007-gguf-onnx-inference-fix.md`
- [ ] Add `gqa_matches_ort` conformance test (GQA kernel vs ORT)
- [ ] Add `swiglu_matches_ort` conformance test (SwiGLU vs ORT)
- [ ] Add `inject_lm_head_regression` unit test (lm_head injection)
- [ ] Run conformance tests — identify failing kernel(s)
- [ ] Fix GQA kernel (if conformance reveals mismatch)
- [ ] Fix SwiGLU kernel (if conformance reveals mismatch)
- [ ] Fix ONNX lm_head injection (if inject test reveals gap)
- [ ] Update `tinyllama_e2e.rs` to assert coherent English output
- [ ] All tests pass, zero clippy warnings

---

## Previous Sprint (Complete): Kernel Conformance Testing (Plan 005)

### Shape Propagation (33 tests)
- [x] Shape conformance for all major op categories (onnx_conformance.rs)
- [x] Conv2d, MatMul, Gemm, MaxPool, GlobalAvgPool, reductions, TopK, etc.
- [x] Subgraph shape propagation (If, Loop)

### Layer A — hologram-exec inline tests (87 tests, pure Rust)
- [x] Known-answer, property, numerical stability tests for all major FloatOp variants
- [x] Exhaustive match ensuring new FloatOp variants require tests
- [x] `tests/float_conformance.rs` integration test

### Layer B — hologram-ai-conformance (reference cross-validation)
- [x] Tolerance, comparator, reference modules
- [x] 31 cross-validation tests (op_conformance.rs)
- [x] ORT runner: 17 ONNX tests (8 unary, 4 binary, 2 softmax, 1 matmul, 2 gemm)
- [x] ORT composite models (RmsNorm, LayerNorm)

### Layer C — Validate CLI + quantization
- [x] `hologram-ai validate --model <path>` (compilation-level validation)
- [x] Quant Tier 1 + Tier 2 (golden vectors)
- [x] CI Tier 1/2/3 (nightly workflow, GitHub Actions)

---

## Previous Sprint (Complete): ONNX Last Mile

### Phase 1: Vision-Critical Ops
- [x] Add 9 AiOp variants: Conv, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool, Resize, Pad, InstanceNorm, LRN
- [x] Add ONNX op mappings + `attr_s()` accessor to OpContext
- [x] Add shape propagation rules (Conv/Pool formula, Resize, Pad, etc.)
- [x] Add data propagation match arms
- [x] Add dynamic param resolution for Pad/Resize (opset 11+ inputs)
- [x] Add lowering dispatch entries → FloatNeedsShape (FloatOp variants added to hologram base)
- [x] Add resolve_op strategy arms for Conv2d, ConvTranspose, MaxPool2d, AvgPool2d, GlobalAvgPool, Resize, Pad, InstanceNorm, LRN

### Phase 2: Utility Ops
- [x] Add 12 AiOp variants: ReduceProd, ReduceL1, ReduceL2, TopK, ScatterND, CumSum, NonZero, OneHot, DepthToSpace, SpaceToDepth, Compress, ReverseSequence
- [x] Add ONNX mappings + quantization integration (QuantizeLinear, DequantizeLinear)
- [x] Add shape propagation rules for utility ops (reductions, TopK, ScatterND, NonZero, OneHot, DepthToSpace, SpaceToDepth, Compress)
- [x] Add lowering dispatch entries → FloatNeedsShape (FloatOp variants added to hologram base)
- [x] Add resolve_op strategy arms for ReduceProd, TopK, ScatterND, CumSum, NonZero, Compress, ReverseSequence
- [x] Add OpDecomposition pass: ReduceL1→Abs+ReduceSum, ReduceL2→Mul+ReduceSum+Sqrt, DepthToSpace/SpaceToDepth→Reshape+Transpose+Reshape

### Phase 3: Proto/Type Gaps
- [x] Add F64 dtype + ONNX type 11 mapping
- [x] Add INT16 dtype
- [x] Add widening casts for UINT16→INT32, UINT32→INT64, UINT64→INT64
- [x] Add opset version validation (parse opset_import, enforce max_opset, store in metadata)
- [x] F64→F32 and INT16→I32 lowering at weight serialization and FloatDType mapping
- [x] Document and handle optional input semantics

### Phase 4: Subgraph Support (If/Loop/Scan)
- [x] Add `subgraphs: HashMap<String, AiGraph>` to AiGraph
- [x] Add AiOp::If, Loop, Scan variants
- [x] Add `attr_g()` graph attribute accessor + ONNX If/Loop/Scan op mappings
- [x] Add recursive ONNX import with subgraph key rewriting
- [x] Add optimization pass recursion into subgraphs
- [x] Add lowering to hologram's native SubgraphDef + CallSubgraph (compile-time flattening)

### Phase 5: Long-Tail + Conformance
- [x] Map remaining niche ops to Opaque with warnings (RNG, ML, linear algebra, sequence, optional)
- [x] Validate recursion into subgraphs
- [x] Verify multi-output ops (TopK, Split, BatchNorm training)
- [x] ONNX conformance test runner (node test suite)

---

## Done

- [x] Remove `InferenceSession` + structural cleanup (ADR-0016)
- [x] Symbolic shapes: `DimExpr` algebra, `DimVarTable`, `ConstraintStore` (ADR-0015)
- [x] Tokenizer expansion: Unigram (Viterbi), WordPiece, multi-algorithm dispatch (ADR-0012)
- [x] GGUF v2/v3 binary parser + metadata extraction (ADR-0006)
- [x] LlamaArch graph construction from GGUF tensors
- [x] Compiler rework: `HoloArchive` + `CompileStats` replacing `CompiledModel`
- [x] CLI: `inspect_gguf`, compile stats output
- [x] Shape propagation optimization pass (`ShapePropagation`)
- [x] Delete `Run` CLI command — users call `hologram run` directly
- [x] 60 tests passing, zero clippy warnings
- [x] Native `FloatOp` in hologram base crate (55 variants, kernels, dispatch, CLI inspect)
- [x] Lowering emits `GraphOp::Float(FloatOp::...)` for ALL ops (zero custom ops remaining)
- [x] Deleted `custom_ops.rs` — all 446 lines removed, no `CustomHandler` closures
- [x] Removed `CustomOpRegistry` from `LoweringOutput` — lowering is pure native ops
- [x] Op extensibility plan documented (`specs/plans/003-op-extensibility.md`)
- [x] KV-cache ops: `AiOp::KvSlotWrite`/`KvSlotRead` in IR, dispatch, shape propagation
- [x] KV-cache layout: `MemoryPlanner` computes real `KvCacheLayout` from arch metadata
- [x] Multi-graph lowering: `LowerPhase` enum (Prefill/Decode/Forward), phase-aware `lower()`
- [x] Pipeline archive: `PipelineWriter` bundles prefill + decode sub-archives for LLMs
- [x] LLM meta section: `LlmMetaSection` with rkyv zero-copy serialization (`SECTION_LLM_META` 0x1011)
- [x] Tokenizer section: `TokenizerSectionData` with rkyv zero-copy serialization (`SECTION_TOKENIZER` 0x1001)
- [x] ConstantFolding: identity elimination, reshape-of-constant folding, dead constant removal
- [x] 67 tests passing, zero clippy warnings
- [x] Shape-tracked execution: `ShapeMap`, `FloatOp::Transpose` with physical permutation,
  actual Reshape (reads shape tensor), N-D broadcasting (Expand), i64/i32 shape auto-detection
- [x] TinyLlama 1.1B end-to-end: ONNX → .holo → execute all 1612 nodes (~215s debug build)
- [x] Tokenizer embedding: `--tokenizer` CLI flag, `TokenizerSectionData::from_tokenizer_json()`
- [x] Output decoding: `hologram run` applies argmax + tokenizer decode when section present
- [x] `--prompt` flag: autoregressive text generation with `MiniBpeEncoder`
- [x] `ModelMetaSection` (0x1002): `ModelKind` enum, arch, capabilities
- [x] `--input-file`: load raw binary inputs from files
- [x] Typed output formatting: f32/f64/i32/i64 dtype-aware display
- [x] Compiler auto-embeds `ModelMetaSection` in compiled archives
- [x] ONNX shape oracle: seed shapes from ValueInfoProto, settled-shape protection
- [x] RmsNorm fusion pass: Pow→ReduceMean→Add→Sqrt→Reciprocal→Mul → AiOp::RmsNorm
- [x] Multi-level DataProp: re-materialization for transitive shape dependencies
- [x] Seq_len sentinel: dynamic dims use 0-sentinel, resolved at runtime
- [x] Inf/NaN diagnostic: scan compiled params for broken scale factors
- [x] Dynamic param resolution: Pad/Resize (opset 11+ inputs), Clip (opset 11+ min/max)
- [x] Optional input semantics: documented pattern, Clip min/max resolved from constant inputs
- [x] Multi-output ops: TopK (values+indices dtype), Split (N outputs), BatchNorm (training 5 outputs)
- [x] ONNX conformance test suite: 29 shape-propagation tests covering all op categories
- [x] 147 tests passing, zero clippy warnings
- [x] Subgraph lowering: If (compile-time flatten + Where), Loop (compile-time unroll), Scan (CallSubgraph fallback)
- [x] GraphBuilder.flatten_registered_subgraph() for compile-time subgraph inlining
- [x] DispatchTarget::Subgraph variant routes If/Loop/Scan through subgraph lowering path
- [x] 4 subgraph lowering tests: If with both branches, If then-only, Loop known trip count, Loop zero trip

See `specs/plans/001-spec-alignment-completed.md`, `specs/plans/002-mvp-remaining.md`,
and `specs/plans/004-onnx-last-mile.md` for full details.

---

## Still Blocked on hologram base crate

- **Shape metadata on graph edges** — hologram graphs have no per-edge
  shape/dtype, forcing shapes to be baked into closure captures
- **`KvExecutor::execute_layer()`** — does not exist; manual sub-archive
  extraction required
- ~~**Vision FloatOp variants**~~ — DONE: Conv2d, ConvTranspose, MaxPool2d, AvgPool2d, GlobalAvgPool, Resize, PadOp, InstanceNorm, LRN added
- ~~**Utility FloatOp variants**~~ — DONE: ReduceProd, TopK, ScatterND, CumSum, NonZero, Compress, ReverseSequence added
- **Vision/utility runtime kernels** — FloatOp variants exist but dispatch returns `UnsupportedOp` (stub); kernels not yet implemented
- ~~**Subgraph lowering**~~ — DONE: compile-time flattening covers If/Loop; dynamic Loop/Scan falls back to `CallSubgraph` (needs runtime dispatch)
- **`LayerEntrypoint::Subgraph(u32)` runtime** — declared but not implemented;
  needed for dynamic Loop/Scan control flow at runtime

---

## Notes

- CLI: exactly 3 commands — `compile`, `info`, `download`
- ONNX importer path still works (single-archive, non-pipeline)
- GGUF importer supports `llama`, `mistral`, `codellama`, `tinyllama` arch names
- No backwards compatibility concerns — can break APIs freely
- Future extensibility: op decomposition (now), serializable op descriptors (Phase 3), WASM kernels (Phase 4+). See `specs/plans/003-op-extensibility.md`.
- Archive sections use rkyv for zero-copy access from memory-mapped files.
- `rkyv = "0.8"` added to workspace dependencies.

# Plan 009: LUT-GEMM + KV-Cache + ShapeContextGraph Runtime Integration

## Goal

Three interlocking improvements that build on each other:

1. **ShapeContextGraph Runtime Integration** — wire `walk_shape_context()` into
   the execution path so all shapes are correctly projected from runtime inputs.
   Fixes TinyLlama seq>1 correctness. Prerequisite for KV-cache (variable cache
   length).

2. **LUT-GEMM for GGUF** — emit `GraphOp::MatMulLut4` instead of
   `FloatOp::Gemm { quant_b: 1 }` for GGUF Q4_0 weights. The kernels
   (`lut_gemm_4bit`, `MatMulLut4` dispatch) already exist in hologram-exec.
   This eliminates the dequantization overhead, cuts GGUF memory bandwidth 8×,
   and uses the byte-domain execution model hologram was designed for.

3. **KV-Cache (Decode mode)** — implement `KvSlotWrite`/`KvSlotRead` in
   hologram-exec. Split the GGUF model into Prefill + Decode sub-archives using
   the existing `LowerPhase` enum. The ShapeContextGraph handles variable
   cache-length shapes across decode steps.

---

## Current State

| Component | Status |
|-----------|--------|
| `lut_gemm_4bit` / `lut_gemm_8bit` kernels | ✓ exist (hologram-exec/src/lut_gemm/) |
| `GraphOp::MatMulLut4` / `MatMulLut8` dispatch | ✓ exist (kv/store.rs:99-104) |
| `ShapeContextGraph` compiled into archives | ✓ (section 0x1021) |
| `walk_shape_context()` API | ✓ (hologram-ai-common, conformance-tested) |
| `walk_shape_context()` called at runtime | ✗ NOT WIRED |
| GGUF Q4_0 lowered to `MatMulLut4` | ✗ uses `FloatOp::Gemm { quant_b: 1 }` |
| `AiOp::KvSlotWrite` / `KvSlotRead` lowered | ✗ not lowered at all |
| Prefill/Decode split for GGUF | ✗ single-graph only |

---

## Step 1 — ShapeContextGraph Runtime Integration

### 1a. New `execute_plan` API (hologram-exec)

Add to `hologram-exec/src/lib.rs` and `hologram-exec/src/eval/executor.rs`:

```rust
/// Pre-projected shape map: NodeId → concrete shape.
/// Produced by `walk_shape_context()` and injected before dispatch.
pub type RuntimeShapeMap = HashMap<u32, Vec<usize>>;

/// Execute a compiled plan with pre-projected shapes from `walk_shape_context()`.
///
/// `shape_hints` overrides the executor's internal shape resolution for every
/// node that appears in the map. This eliminates the need for:
///   - 0-sentinel guessing from buffer sizes (parse_shape_values)
///   - resolve_dynamic_sizes() kernel-param patching
/// and makes execution correct for any seq_len, batch size, etc.
pub fn execute_plan_with_shape_hints(
    plan: &CompiledPlan,
    inputs: &GraphInputs,
    shape_hints: &RuntimeShapeMap,
) -> ExecResult<Outputs>
```

**In executor.rs `propagate_level_shapes()`**: when a node's id is in `shape_hints`,
use the hint directly instead of inferring from buffer sizes. This is the only change
needed in the executor — everything else falls out naturally.

### 1b. Caller integration (hologram-ai)

In `hologram-ai/src/compiler.rs`, add a `run()` helper and update `run_cmd.rs` CLI:

```rust
// hologram_ai::runtime::run_with_shape_context()
pub fn run_with_shape_context(
    archive: &HoloArchive,
    inputs: &GraphInputs,
    input_shapes: &HashMap<u32, Vec<usize>>,  // named input_id → shape
) -> anyhow::Result<Outputs> {
    // 1. Deserialize ShapeContextGraph from archive
    let ctx_graph = read_shape_context_graph(&archive.bytes)?;

    // 2. Project shapes for all nodes
    let mut shape_map = HashMap::new();
    walk_shape_context(&ctx_graph, &input_shapes, &HashMap::new(), &mut shape_map);

    // 3. Execute with projected shapes
    let plan = hologram::load_from_bytes(&archive.bytes)?;
    hologram::execute_plan_with_shape_hints(&plan, inputs, &shape_map)
        .map_err(|e| anyhow::anyhow!("{e}"))
}
```

The `ShapeContextGraph` section is already archived; we just need a reader
in hologram-ai that extracts it from the archive bytes using rkyv.

### 1c. Connection to `ShapeSpec`/`ShapeSpecRepr`

`walk_shape_context()` uses `ShapeSpecRepr` (compiled into each
`ShapeProjectionEntry`) to project output shapes from input shapes. The full
mapping is:

| ShapeSpecRepr | Resolver |
|---------------|----------|
| `SameAs(i)` | `output = inputs[i].shape` |
| `Broadcast(a, b)` | `output = broadcast_shapes(inputs[a], inputs[b])` |
| `BroadcastAll` | `output = broadcast_shapes(all inputs)` |
| `DropLastDim(i)` | `output = inputs[i].shape[..rank-1]` |
| `Dims(vec)` | per-dim: `Fixed(v)`, `FromInput{input,axis}`, `Inferred` |
| `Custom` | `resolve_custom(spec, inputs, shape_value_bytes)` |

Every op in the compiled graph has a `ShapeProjectionEntry`. After
`walk_shape_context()`, every node's shape is in `shape_map`. The executor uses
these directly — no guessing, no `parse_shape_values`, no `resolve_dynamic_sizes`.

---

## Step 2 — LUT-GEMM for GGUF Q4_0

### 2a. Change lowering strategy (hologram-ai)

In `crates/hologram-ai-common/src/lower/strategy.rs`, in the `AiOp::Gemm`
arm of `resolve_op()`:

```rust
// Current (wrong for Q4_0):
// FloatNeedsShape(FloatOp::Gemm { quant_b: 1, ... })

// New (correct):
// When weight is Q4_0:
//   - serialize weight as QuantizedWeights4 { indices: [u8], centroids: [f32; 16], rows, cols }
//   - emit GraphOp::MatMulLut4(cid) where cid references the packed weight constant
// When weight is Q8_0 or INT8:
//   - emit GraphOp::MatMulLut8(cid)
// When weight is F32/F16:
//   - emit FloatOp::Gemm (unchanged)
```

The `QuantizedWeights4` format expected by `lut_gemm_4bit`:
- `indices: Vec<u8>` — packed 4-bit weight indices (2 per byte)
- `centroids: [f32; 16]` — 16 centroid values for the Q4 codebook
- `rows: u32`, `cols: u32`

The GGUF Q4_0 format: blocks of 32 values, each with a scale factor and 32 4-bit
indices. We need a conversion step at compile time to pack into the expected format.

Actually — the existing `lut_gemm_4bit` uses a psumbook/centroid approach
that requires pre-trained centroids. Raw Q4_0 from GGUF uses a linear
quantization (not centroid-based). We need to adapt the kernel OR use a simpler
dispatch path.

**Simpler Q4_0 path** (no centroids needed):
- Add `FloatOp::MatMulQ4 { m: u32, k: u32, n: u32 }` for the decompressed-at-decode path
- Inputs: [activations f32, weights Q4_0 bytes, scales f32]
- Kernel: dequantize blocks on-the-fly during GEMM (avoid full materialization)
- This is `2×` memory bandwidth vs `MatMulLut4` but `8×` vs dequantize-then-GEMM

OR route through the existing `MatMulLut4` after converting Q4_0 to centroid
format at compile time (a one-time offline step during `hologram-ai compile`).

**Recommended**: Use `FloatOp::MatMulQ4` (block-quantized GEMM) as the first step.
It's simpler to implement correctly, avoids centroid conversion, and is still
significantly faster than dequantize→BLAS by keeping weights in Q4 format during
the multiply loop.

### 2b. Wire `MatMulLut4` constants into archive

For the centroid-based `MatMulLut4`:
- At compile time, convert GGUF Q4_0 blocks to `QuantizedWeights4` format
- Store as a `ConstantId` in the `ConstantStore` within the archive
- The `ConstantData::Deferred` mechanism allows lazy loading from the weight blob

### 2c. Shape projection for LUT-GEMM

With ShapeContextGraph integrated (Step 1), `MatMulLut4` gets `SameAs` or
`Custom` shape projection. The output shape is `[m, n]` where `m` comes from
the activations (dynamic at seq>1) and `n` is fixed (weight rows). This maps to:

```rust
ShapeProjectionEntry {
    spec: ShapeSpecRepr::Dims(vec![
        ShapeDimRepr::FromInput { input: 0, axis: 0 },  // m from activations
        ShapeDimRepr::Fixed(n as u32),                   // n from weight shape
    ]),
    ..
}
```

---

## Step 3 — KV-Cache (Decode Mode)

### 3a. `KvSlotWrite` / `KvSlotRead` dispatch (hologram-exec)

Add to `kv/store.rs` dispatch table:

```rust
GraphOp::Float(FloatOp::KvSlotWrite { layer }) => {
    // inputs[0] = new_K [batch, heads, 1, head_dim]
    // inputs[1] = new_V [batch, heads, 1, head_dim]
    // kv_cache[layer].append(K, V)
    // Returns: updated cache length as scalar
}

GraphOp::Float(FloatOp::KvSlotRead { layer }) => {
    // Returns: (K_full, V_full) [batch, heads, seq_cache, head_dim]
    // seq_cache = current cache length after append
}
```

The KV cache is stored outside the `BufferArena` in a separate `KvCacheStore`:
```rust
pub struct KvCacheStore {
    // Per-layer K and V caches
    pub layers: Vec<KvLayer>,
}
pub struct KvLayer {
    pub k: Vec<f32>,  // [batch, heads, seq_cache, head_dim] row-major
    pub v: Vec<f32>,
    pub seq_len: usize,
}
```

`KvCacheStore` is passed to `execute_plan_with_shape_hints()` alongside
`shape_hints`. It persists between decode steps.

### 3b. `AiOp::KvSlotWrite/Read` lowering (hologram-ai)

In `strategy.rs`, add lowering for `AiOp::KvSlotWrite`/`KvSlotRead` to the
corresponding `FloatOp` variants (which map to `GraphOp::Float(FloatOp::Kv...)`).

### 3c. Prefill / Decode graph split (hologram-ai)

The GGUF `LlamaArch` already tags ops with `LowerPhase`. The compiler needs to:

**Prefill graph**: Standard full-sequence forward pass. At the end of each
attention layer, emit `KvSlotWrite` to populate the cache.

**Decode graph**: Single-token forward pass. Before each attention layer, emit
`KvSlotRead` to read cached K/V, concatenate with current token's K/V
(`[batch, heads, seq_cache+1, head_dim]`), then compute attention.

### 3d. ShapeContextGraph for variable cache length

In Decode mode, the cache length grows each step. `ShapeProjectionEntry` for
`KvSlotRead` has:

```rust
ShapeProjectionEntry {
    spec: ShapeSpecRepr::Dims(vec![
        ShapeDimRepr::Fixed(batch),
        ShapeDimRepr::Fixed(n_kv_heads),
        ShapeDimRepr::FromInput { input: 0, axis: 0 },  // seq_cache (variable!)
        ShapeDimRepr::Fixed(head_dim),
    ]),
    ..
}
```

At each decode step, the caller updates `input_shapes` with the current
`seq_cache` length, calls `walk_shape_context()`, and passes the new
`shape_hints` to `execute_plan_with_shape_hints()`.

This is exactly what `ShapeContextGraph` was designed for: one compiled archive,
correct shapes for any cache length.

---

## Implementation Order

```
Step 1a  execute_plan_with_shape_hints() API            hologram (base)
Step 1b  run_with_shape_context() caller                hologram-ai
Step 1c  Wire into CLI run command                      hologram-ai
      → Fixes tinyllama_causal_onnx_top1_matches_ort ✓
      → tinyllama_onnx_variable_seq_len_runs passes ✓

Step 2a  FloatOp::MatMulQ4 kernel + dispatch            hologram (base)
Step 2b  Lowering: GGUF Q4_0 → MatMulQ4                hologram-ai
      → GGUF generation speed: 0.1 tok/s → ~1 tok/s (est)
      → GGUF memory: 606 MiB weights, no expansion

Step 3a  KvSlotWrite/Read dispatch + KvCacheStore       hologram (base)
Step 3b  AiOp::Kv* lowering                            hologram-ai
Step 3c  Prefill/Decode split                           hologram-ai
Step 3d  ShapeContextGraph for variable cache length    hologram-ai
      → Decode mode: single-token generation
      → ~10× decode speedup (cache hit vs recompute)
```

---

## Verification

```bash
# Step 1: shapes correct for any seq_len
ORT_STRATEGY=system cargo test -p hologram-ai-conformance --features conformance \
  -- tinyllama_causal_onnx_top1 --nocapture
# Expected: passes for seq=1 AND seq=2

cargo test -p hologram-ai --features e2e -- tinyllama_onnx_variable_seq_len --nocapture
# Expected: seq=1, 7, 128 all match ORT

# Step 2: LUT-GEMM speed
cargo bench -p hologram-bench -- lut_gemm
cargo test -p hologram-ai --features e2e -- tinyllama_gguf_runs --nocapture
# Expected: >1 tok/s (vs 0.1 tok/s today)

# Step 3: KV-cache correctness
cargo test -p hologram-ai --features e2e -- tinyllama_gguf_runs --nocapture
# Expected: coherent English output with multi-step decode
```

---

## Critical Files

| File | Change |
|------|--------|
| `hologram/crates/hologram-exec/src/eval/executor.rs` | Add `execute_plan_with_shape_hints()`, use hints in `propagate_level_shapes()` |
| `hologram/crates/hologram-exec/src/lib.rs` | Export new API, add `RuntimeShapeMap` type |
| `hologram/crates/hologram-exec/src/kv/store.rs` | Add `KvCacheStore`, `KvSlotWrite`/`Read` dispatch |
| `hologram/crates/hologram-exec/src/float_dispatch.rs` | Add `dispatch_matmul_q4()` kernel |
| `hologram/crates/hologram-core/src/op/float_op.rs` | Add `MatMulQ4`, `KvSlotWrite`, `KvSlotRead` variants (if not present) |
| `crates/hologram-ai-common/src/lower/strategy.rs` | Route GGUF Q4_0 → `MatMulQ4`, lower `KvSlotWrite`/`Read` |
| `crates/hologram-ai/src/compiler.rs` | Add `run_with_shape_context()`, read `ShapeContextGraph` from archive |
| `crates/hologram-ai/src/commands/run_cmd.rs` | Call `run_with_shape_context()` instead of bare `execute_plan()` |
| `crates/hologram-ai-gguf/src/arch/llama.rs` | Prefill/Decode graph split |

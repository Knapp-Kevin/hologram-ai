# Plan 008 — ShapeContextGraph: Compile-Time Shape Projection

## Status: Planned

## Problem

hologram-ai currently handles symbolic shapes (e.g., `seq_len`, `batch`) using two
overlapping mechanisms:

1. **`DeferredStrategy`** — lowers `AiOp → FloatOp` with 0-sentinels for unknown dims,
   and records `ParamRecipe` entries in a `ShapeRecipeSection` archive section.
2. **hologram's `resolve_dynamic_sizes()`** — rewrites `FloatOp` params before dispatch
   when it detects 0-sentinels (partially covers the same ground).

The result is scattered, brute-force shape resolution: the runtime guesses shapes from
buffer sizes, corrects stale sentinels, and applies recipes individually. There is no
single structure that says "this op's output shape comes from *these* inputs in *this*
structural way."

**Goal**: Replace `ParamRecipe`-based shape patching with a `ShapeContextGraph` — a
compile-time map that records, for each operation, how its output shape structurally
derives from its inputs. This is encoded in hologram's `ShapeSpec`/`ShapeDim` language
and stored in the `.holo` archive. At runtime, a single topological walk projects input
shapes forward through the entire graph before any dispatch happens.

**Key property**: The same compiled archive executes correctly for any input shape
(any `seq_len`, `batch`, etc.) — full shape-polymorphic compilation.

---

## Background: hologram's Existing Shape Machinery

hologram's executor already has:
- `ShapeSpec` / `ShapeDim` — declarative per-op output shape specification
- `resolve_float_shape(op, ctx)` — unified resolver, single entry point
- `ShapeMap` — `HashMap<NodeId, Vec<usize>>`, populated at runtime
- `shape_propagate.rs` — pre-dispatch level-by-level propagation
- `correct_stale_shape()` — corrects stale 0-sentinels using buffer elem counts
- `resolve_dynamic_sizes()` — rewrites `FloatOp` params before dispatch for 0-sentinel fields

What's missing: a compile-time serialized structure that tells hologram's runtime *how*
shapes flow through the graph, expressed in the `ShapeSpec`/`ShapeDim` vocabulary.

---

## Formal Mapping

### `DimExpr → ShapeDim`

| hologram-ai `DimExpr` | hologram `ShapeDim` | Meaning |
|-----------------------|---------------------|---------|
| `Concrete(v)` | `Fixed(v)` | Baked-in constant (e.g., `hidden=2048`) |
| `Var(id)` | `FromInput { input: i, axis: j }` | Comes from input `i`'s `j`-th axis |
| `Dynamic` | `Inferred` | Compute from total element count |
| Evaluatable expression | `Fixed(evaluated)` | Folded to concrete at lower time |

### `AiOp → ShapeSpec`

| AiOp family | ShapeSpec |
|-------------|-----------|
| Unary elementwise, norms | `SameAs(0)` |
| Binary elementwise | `Broadcast(0, 1)` |
| Where / ternary | `BroadcastAll` |
| Reductions (ReduceSum, etc.) | `DropLastDim(0)` |
| Gather | `Dims([FromInput{0,0..}, Fixed(embed_dim)])` |
| Embed | `Dims([Inferred, Fixed(dim)])` |
| MatMul / BatchMatMul | `Custom` → `resolve_matmul` |
| Reshape | `Custom` → `resolve_reshape` (uses shape-value bytes) |
| Expand | `Custom` / `BroadcastAll` (uses shape-value input) |
| Concat | `Custom` → `resolve_concat` |
| Conv / Pool | `Custom` |

**Expand example**: `[1, 1, hidden]` → `[batch, seq, hidden]`.
Without this plan: emit `[0, 0, 2048]`, guess at runtime from buffer size.
With this plan: record `Custom` + `shape_value_input=Some(1)`. Runtime reads the
shape-value tensor bytes from `BufferArena`, calls `resolve_reshape()` → `[batch, seq, hidden]`.

---

## Data Structures

### `ShapeContextGraph` (new archive section)

```rust
/// Compile-time shape projection map. Embedded in the .holo archive.
pub struct ShapeContextGraph {
    /// Fully concrete shapes known at compile time (constants, weights).
    pub seeds: Vec<ShapeSeed>,
    /// Per-node projections in topological order.
    pub projections: Vec<ShapeProjectionEntry>,
}

pub struct ShapeSeed {
    pub node_id: NodeId,
    pub shape: Vec<usize>,  // fully concrete
}

pub struct ShapeProjectionEntry {
    pub node_id: NodeId,
    /// Input node IDs that contribute to this op's output shape.
    pub input_node_ids: Vec<NodeId>,
    /// How to compute output shape from those inputs.
    pub spec: ShapeSpecRepr,
    /// If set, input[n] carries shape-value bytes (Reshape, Expand, Pad, etc.)
    pub shape_value_input: Option<u8>,
}

/// Serializable mirror of hologram's ShapeSpec (runtime-only, not serializable).
pub enum ShapeSpecRepr {
    SameAs(u8),
    Broadcast(u8, u8),
    BroadcastAll,
    DropLastDim(u8),
    Dims(Vec<ShapeDimRepr>),
    Custom,
}

pub enum ShapeDimRepr {
    Fixed(u32),
    FromInput { input: u8, axis: i8 },
    Inferred,
}
```

---

## Implementation Steps

### Step 1 — `AiOp → ShapeSpecRepr` translator
**New file**: `crates/hologram-ai-common/src/lower/shape_spec_bridge.rs`

```rust
pub fn ai_op_to_shape_spec(
    op: &AiOp,
    inputs: &[TensorId],
    output_tid: TensorId,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> (ShapeSpecRepr, Option<u8>)  // (spec, shape_value_input)
```

Use `OpCategory` (from `crates/hologram-ai-common/src/ir/op.rs`) for structural
classification. For symbolic dims in `Dims(...)` entries, walk input `TensorInfo`
shapes to identify which input and axis each `DimExpr::Var` comes from.

Reference: `infer_output_shapes()` in `crates/hologram-ai-common/src/opt/shape_prop.rs`
for the per-op shape rules.

### Step 2 — Build `ShapeContextGraph` during lowering
**Modified file**: `crates/hologram-ai-common/src/lower/builder.rs`

After the existing lowering loop:
1. Walk `AiNode`s in topological order
2. For each node:
   - If output shape is fully concrete → emit `ShapeSeed`
   - Otherwise → call `ai_op_to_shape_spec()` → emit `ShapeProjectionEntry`
3. Collect into `ShapeContextGraph`; add to `ExecContext`

Seed entries also come from `DataPropagation`'s `known_i64_values` (shape-computation
tensors fully evaluated at compile time).

### Step 3 — Archive section
**Modified file**: `crates/hologram-ai-common/src/exec_context.rs`

Add `shape_context: ShapeContextGraph` as a serializable section alongside the
existing `ShapeRecipeSection`.

### Step 4 — Runtime `walk_shape_context()`
**New function in**: `crates/hologram-ai-common/src/exec_context.rs`

```rust
pub fn walk_shape_context(
    ctx_graph: &ShapeContextGraph,
    arena: &BufferArena,        // shape-value tensor bytes for Reshape/Expand
    runtime_inputs: &[(NodeId, Vec<usize>)],  // user-supplied input shapes
    shape_map: &mut ShapeMap,   // hologram's runtime shape store
)
```

Algorithm:
1. Seed `ShapeMap` from `ctx_graph.seeds`
2. Inject `runtime_inputs`
3. For each `ShapeProjectionEntry` (topological order — already sorted):
   - Collect `input_shapes` from `ShapeMap`
   - If `shape_value_input` is set: read bytes from `arena`
   - Build `ShapeContext { input_shapes, shape_tensor_bytes, ... }`
   - Convert `entry.spec` → `ShapeSpec`; call `resolve_float_shape()` (hologram)
   - Store in `ShapeMap[entry.node_id]`

This single walk fully populates `ShapeMap` before any dispatch.

### Step 5 — Retire `ParamRecipe` for shape-resolved dims
**Modified file**: `crates/hologram-ai-common/src/lower/strategy.rs`

Once `ShapeContextGraph` covers output shape projection, `ParamRecipe` is no longer
needed for dims that `walk_shape_context()` resolves. Keep recipes only for true kernel
scalar params where the struct field drives execution (not output shape).

**What hologram already resolves** (no change needed):
- `RmsNorm { size: 0 }` → last dim of input ✓ (`resolve_dynamic_sizes`)
- `Softmax { size }` → overrides even non-zero when actual shape differs ✓
- `MatMul { k }` → `infer_matmul_k()` from buffer sizes ✓

### Step 6 — Extend hologram `resolve_dynamic_sizes()` for remaining kernel params
**File**: `hologram/crates/hologram-exec/src/eval/executor.rs`

Extend to cover ops whose struct fields still read directly without inference:

```rust
// Attention: infer head_dim from Q shape when head_dim == 0
FloatOp::Attention { head_dim: 0, .. } => {
    // head_dim = input[0].shape[-1] / num_q_heads
}

// Embed: infer dim from embedding table shape when dim == 0
FloatOp::Embed { dim: 0, quant } => {
    // dim = weight_tensor.shape[-1]
}

// Concat: infer size_a/size_b from input shapes when == 0
FloatOp::Concat { size_a: 0, size_b: 0, dtype } => {
    // Resolve from ShapeMap input shapes
}
```

This completes elimination of `ParamRecipe` — when all 0-sentinels are covered by
either `walk_shape_context()` (output shapes) or extended `resolve_dynamic_sizes()`
(kernel scalar params), no explicit recipe entries are needed.

---

## Files to Modify

| Action | File |
|--------|------|
| New | `crates/hologram-ai-common/src/lower/shape_spec_bridge.rs` |
| Modified | `crates/hologram-ai-common/src/exec_context.rs` (add `ShapeContextGraph`, `walk_shape_context`) |
| Modified | `crates/hologram-ai-common/src/lower/builder.rs` (emit projections) |
| Modified | `crates/hologram-ai-common/src/lower/strategy.rs` (retire dim-only recipes) |
| Modified | `hologram/crates/hologram-exec/src/eval/executor.rs` (extend `resolve_dynamic_sizes`) |

## Reuse

- `OpCategory` in `crates/hologram-ai-common/src/ir/op.rs` — drives SameAs/Broadcast/DropLastDim without per-op cases
- `infer_output_shapes()` in `crates/hologram-ai-common/src/opt/shape_prop.rs` — reference for Custom op rules
- `resolve_float_shape()` in `hologram/crates/hologram-exec/src/eval/shape_resolve.rs` — call directly in `walk_shape_context()`
- `correct_stale_shape()` in `hologram/crates/hologram-exec/src/eval/shape_resolve.rs` — stale sentinel correction
- `DataPropagation::known_i64_values` in `crates/hologram-ai-common/src/opt/data_prop.rs` — seed `ShapeSeed` entries
- `ShapeSpec::inferred_by_fixed()` / `ShapeSpec::inferred_1d()` in `hologram/crates/hologram-core/src/op/shape_spec.rs`

---

## End State

After this plan:

1. `hologram-ai compile(model)` runs with fully symbolic shapes → valid `.holo` archive
2. Archive contains: `FloatOp` nodes (0-sentinels) + `ShapeContextGraph` (projection map)
3. Runtime: supply input shapes → `walk_shape_context()` projects shapes through entire
   DAG in one topological pass → `ShapeMap` fully populated
4. Dispatch: `resolve_dynamic_sizes()` (extended) patches remaining 0-sentinel kernel
   params from `ShapeMap`
5. Same compiled archive executes for any `seq_len`, `batch` — shape-polymorphic

---

## Verification

1. `cargo test -p hologram-ai-common` — unit tests for `ai_op_to_shape_spec()` and
   `walk_shape_context()`
2. `cargo test -p hologram-ai-conformance` — exec conformance against ORT intermediates
3. End-to-end: `cargo test -p hologram-ai --features e2e -- tinyllama` with variable
   seq_len inputs (1, 7, 128, 512)
4. Assert: `ShapeMap` after `walk_shape_context()` matches ORT intermediate shapes at
   every node
5. Assert: no `ParamRecipe::DimVar` or `ParamRecipe::RuntimeInferred` entries remain
   in final archive

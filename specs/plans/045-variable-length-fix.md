# Plan 045: Variable-Length Execution Fix

**Status:** Open
**Created:** 2026-04-02
**Branch:** `feat/cpu-inference-perf`

## Problem

Models must be compiled with `--seq-len N` matching the exact prompt token count.
If compiled at seq=32 but run with 24 tokens, output is garbage because:
- Reshape/Expand ops use compiled dimensions (32)
- Softmax/RmsNorm use inferred dimensions from buffer length (24)
- This inconsistency corrupts shapes mid-graph

## Root Cause

`resolve_size()` in `float_dispatch/mod.rs` falls back to `n_floats` when
`n_floats % compiled_size != 0`. But some ops (Reshape, Expand) embed the
compiled seq_len in their parameters and don't use `resolve_size()` at all.

## Fix Strategy

### Option A: 0-sentinels for all seq-dependent dims (recommended)

During lowering, emit `0` for ALL dimensions that depend on sequence length:
- `FloatOp::MatMul { m: 0, k, n }` — m is seq-dependent
- `FloatOp::Reshape` target shape — seq dimension is 0
- `FloatOp::Softmax { size: 0 }` — when size is seq-dependent

The executor resolves 0 → infer from buffer length. All ops use the same
inference, so dimensions stay consistent.

**Changes:**
- `hologram-ai-common/src/lower/strategy.rs` — track which dims are seq-dependent
  from the `DimVarTable`, emit 0 instead of concrete values
- `hologram-exec/src/float_dispatch/mod.rs` — improve `resolve_size()` heuristics
- `hologram-exec/src/tape.rs` — ensure all kernels handle 0-sentinel dims

### Option B: Shape context projection (existing infrastructure)

The `ShapeContextGraph` already maps compiled shapes to runtime shapes.
Wire it into `execute_direct` so every instruction's output gets correct
shape metadata. The old `execute_inner` used `shape_overrides` — port this
to the new single-path executor.

## Status Update (2026-04-05)

### What Works Now
- **prompt <= compiled seq_len**: ShapeContextGraph + KV cache produces
  correct output when compiled at max context length (e.g., seq=2048) and
  prompted with any shorter sequence. Shape overrides flow through
  `input_metas` in `execute_direct`.
- **Compile at model context length**: Default compilation uses the model's
  full `context_length` (2048 for TinyLlama). Any prompt up to that length
  works with variable-length execution. This is the recommended path.

### What Was Attempted and Failed
- **Option A partial (Dynamic dims):** Setting seq-dependent dims to
  `Dim::Dynamic` after optimization breaks lowering — `concrete_last_dim`
  returns None for ALL seq dims, causing Softmax/RmsNorm/MatMul to get
  size=0 which the runtime can't resolve correctly.
- **Shape tensor i64 zeroing:** Zeroing `known_i64_values` at seq positions
  requires following Reshape data flow to map target tensor axes to shape
  tensor element indices. The axis-based `seq_dim_positions` set doesn't
  map to element indices of 1-D shape constants.

### Infrastructure Built (ready for use)
- `concretize_all_dims` now returns `seq_dim_positions: HashSet<(TensorId, usize)>`
  identifying which tensor dims are seq-dependent
- `retain_live_nodes()` on ShapeContextGraph prunes dead entries after fusion
- ShapeContextGraph wired into HoloRunner.execute/execute_with_kv
- `execute_direct` populates input_metas from shape_overrides
- `execute_tape_with_kv_shapes_cached` combines all three: KV + shapes + cache

### Remaining Work (true 0-sentinel lowering)
The correct fix requires:
1. **Find Reshape nodes** in the AiGraph whose shape tensor contains
   seq-dependent values (by tracing Shape→Gather→Concat→Reshape chains)
2. **Zero the specific elements** in the shape tensor constant's i64 values
   that correspond to seq-dependent positions
3. **Handle Expand** similarly — zero seq dims in target_shape
4. **Ensure runtime Reshape** handles 0 elements by inferring from total
   buffer element count and the non-zero target dims

This is a graph-analysis problem in the compiler, not a lowering strategy
change. The DeferredStrategy already handles 0-sentinels correctly for
MatMul/Softmax/RmsNorm — only Reshape/Expand need the targeted fix.

## Implementation: Shape Tensor Zeroing Pass

### Why shape metadata overrides aren't enough

Shape overrides (`input_metas` from `shape_overrides`) tell the *next* op
what shape a buffer has. But they can't fix the computation *inside* an op.
When MatMul has `baked_n=24` and the buffer has 36 elements, the MatMul
reads 24 elements and produces wrong results. The op parameter must itself
be 0 so the runtime infers it from the buffer.

For MatMul/Softmax/RmsNorm, the DeferredStrategy already emits 0 when dims
are symbolic. The issue: after `concretize_all_dims`, all dims are concrete
so DeferredStrategy sees concrete values and bakes them.

### The fix: post-concretization pass on the AiGraph

Add a new pass after `post_concretization_repair` but before `lower()`.
This pass runs on the AiGraph (not the hologram Graph) and zeros
seq-dependent values in shape tensor constants.

**File:** `hologram-ai/crates/hologram-ai/src/compiler.rs`
(new function, called from compile paths)

```rust
fn zero_reshape_seq_dims(
    graph: &mut AiGraph,
    seq_dim_positions: &HashSet<(TensorId, usize)>,
) {
    // 1. Find all Reshape/Flatten nodes.
    // 2. For each, get the output TensorId and check which axes are seq-dependent
    //    using seq_dim_positions.
    // 3. Get known_i64_values from the OUTPUT tensor's tensor_info (DataProp
    //    stores resolved values per-consumer on the output, not the shape input).
    // 4. Zero the i64 values at seq-dependent axis positions.
    // 5. Also update the constant param bytes if the shape tensor was materialized.
}
```

### Key data flow (from DataPropagation)

For a Reshape `[1, seq, 2048] → [1, seq, 32, 64]`:
- Output tensor has `known_i64_values = [Some(1), Some(24), Some(32), Some(64)]`
- Element index 1 corresponds to the seq axis
- `seq_dim_positions` has `(output_tid, axis=1)` — axis 1 is seq-dependent
- So zero `known_i64_values[1]` → `Some(0)`

The mapping is: `seq_dim_positions.(tid, axis)` maps directly to
`tensor_info[tid].known_i64_values[axis]` when the tid IS the Reshape
output tensor. This works because DataProp stores resolved values on the
output tensor, indexed by axis.

### What about shape tensor constants?

`infer_reshape_shape()` in `builder.rs` checks the OUTPUT tensor's
`known_i64_values` first (priority 1). If we zero seq dims there, it
will read the zeroed values and emit the correct shape with 0-sentinels.

If the output has no `known_i64_values`, it falls back to the shape
tensor constant param (priority 2). We should also zero seq dims in
constant params for completeness, but the output path is sufficient for
LLM Reshape patterns where DataProp has populated the values.

### What about Expand?

`Expand { ndim, target_shape }` is lowered by `resolve_op()` in
`strategy.rs` (lines 940-994). It reads target_shape from the input
tensor's `known_i64_values`. Same fix: zero seq-dependent i64 values
on the Expand's shape input tensor.

### What about MatMul m?

MatMul m is already 0-sentinel when the dim is symbolic. After
concretization it becomes concrete (e.g., 24). The fix: in the
`zero_reshape_seq_dims` pass, also scan MatMul/Gemm nodes and mark
their seq-dependent shape dims for zeroing. But MatMul's m comes from
the input tensor's shape (e.g., shape[0] or shape[1]), not from
`known_i64_values`. The lowering reads it via `concrete_last_dim()` or
`dim_at_expr()`.

For MatMul, the simplest fix: set the seq-dependent axis of the MatMul
INPUT tensor's shape to `Dim::Dynamic` (just that one dim, not all).
Then `concrete_last_dim()` returns None for that dim → m becomes 0.
The other dims (k, n) stay concrete because they're hidden_dim/head_dim.

This is safe because:
- Only the seq axis becomes Dynamic
- `post_concretization_repair` has already finished
- The Dynamic dim won't propagate (we're about to enter lowering)
- DeferredStrategy handles the 0 correctly

### Summary of changes

1. **New function `zero_seq_dims_for_lowering()`** in compiler.rs:
   - For Reshape/Flatten output tensors: zero `known_i64_values[axis]`
     where `(tid, axis) ∈ seq_dim_positions`
   - For Expand shape input tensors: same treatment
   - For activation tensors feeding MatMul/Softmax/RmsNorm: set
     seq-dependent `Dim` to `Dynamic` in tensor_info shape

2. **Call after `post_concretization_repair`, before `lower()`**

3. **Remove prompt-length guard** in `resolve_seq_mode()` — all prompts
   work with 0-sentinel ops

### Critical files

| File | Change |
|------|--------|
| `hologram-ai/src/compiler.rs` | New `zero_seq_dims_for_lowering()` function |
| `hologram-ai/src/commands/run_cmd.rs` | Remove prompt guard |
| `hologram-ai-common/src/lower/builder.rs` | Verify `infer_reshape_shape` handles 0 |

### Why this works where previous attempts failed

Previous attempts:
- **mark_seq_dims_dynamic**: Set ALL tensor dims to Dynamic → broke ALL
  ops (Softmax, RmsNorm, MatMul all got size=0, runtime couldn't resolve)
- **zero_seq_i64_values**: Zeroed by axis index on ALL tensors → wrong
  because shape tensor element indices ≠ target tensor axis indices

This approach:
- Only zeros `known_i64_values` on Reshape/Expand OUTPUT/input tensors
  (targeted, not global)
- Only sets `Dim::Dynamic` on the seq axis of MatMul/norm INPUT tensors
  (one dim, not all)
- Other dims (batch=1, hidden=2048, heads=32, head_dim=64) stay concrete
- DeferredStrategy naturally handles the mix of concrete + 0-sentinel

## Testing

- Compile TinyLlama at seq=24, run with 10-token prompt → correct output
- Compile TinyLlama at seq=24, run with 36-token prompt → correct output
- Compile at seq=128, run with 10 tokens → correct output
- KV cache decode still works (seq=1)
- Non-LLM models (BERT, ResNet) unaffected
- Verify: `HOLOGRAM_PROFILE=1` shows same ops firing as seq=2048 model

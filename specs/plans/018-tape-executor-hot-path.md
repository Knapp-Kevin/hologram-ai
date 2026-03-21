# Plan 018: Tape Executor Wiring + Execution Hot-Path Optimizations

## Context

Decode speed is the P2 priority on the current sprint. The hologram executor's per-node dispatch loop in `dispatch_level()` has three categories of overhead that can be eliminated by pre-computing at graph-load time:

1. **Per-node op matching** — a large `match node.op { ... }` block runs for every node every inference call
2. **Per-node shape gathering** — `propagate_level_shapes()` + `input_shapes` construction runs for every node, even elementwise ops that just pass through input[0]'s shape
3. **Per-node elem_size computation** — `compute_output_elem_size()` does HashMap lookups + `FloatOp::output_dtype()` per node

The tape infrastructure (`Tape`, `Instruction`, `BoxedKernel`) already exists in `hologram-exec/src/tape.rs` but is **not wired into any execution path**. Similarly, `WeightCache` and `dispatch_float_into` exist but are unused.

All execution code lives in the hologram base crate. hologram-ai (compiler-only) just calls the APIs from `HoloRunner`.

---

## Implementation

### Task A: SameAs(0) shape propagation fast path (hologram)

**Standalone improvement — can ship immediately.**

In `propagate_level_shapes()` (`hologram-exec/src/eval/shape_propagate.rs`), add an early-exit for `SameAs(0)` ops before the expensive `input_elem_counts` + `resolve_float_shape()` path:

```rust
if matches!(fop.output_shape_spec(), ShapeSpec::SameAs(0)) {
    if let Some(src_id) = node.inputs.first()... {
        if let Some(shape) = shape_map.get(src_id) {
            shape_map.insert(node_id, shape.to_vec());
            continue;
        }
    }
}
```

Eliminates: per-input `compiled_dtypes.get()` + `arena.get()` + `resolve_float_shape()` for ~60-70% of nodes (all unary elementwise, norms, softmax, etc.).

**Files:** `hologram/crates/hologram-exec/src/eval/shape_propagate.rs`

### Task B: Skip input_shapes gathering for elementwise ops (hologram)

In `dispatch_level()` (`hologram-exec/src/eval/executor.rs` ~line 1362), the `input_shapes: Vec<Vec<usize>>` is built for every node. For unary elementwise ops, shapes are never used by the kernel. For binary elementwise, shapes are only needed when broadcasting (different input sizes).

Add fast path:
- Unary elementwise/bytebool: skip `input_shapes` entirely, pass empty vec
- Binary elementwise: only build shapes when `input_refs[0].len() != input_refs[1].len()`

Eliminates: `shape_map.get()` + `compiled_dtypes.get()` + `resolve_compiled_shape()` per input for the majority of nodes.

**Files:** `hologram/crates/hologram-exec/src/eval/executor.rs` (in `dispatch_level()`)

### Task C: TapeBuilder — pre-compute kernel + elem_size (hologram)

New module `hologram-exec/src/tape_builder.rs`:

```rust
pub fn build_tape(sg: &SerializedGraph, schedule: &ExecutionSchedule) -> ExecResult<Tape>
```

For each node in schedule order:
1. Resolve `BoxedKernel` from the `FloatOp` (bakes op params into closure)
2. Pre-compute `output_elem_size` from `FloatOp::output_dtype()` + compiled dtypes (done once, not per-dispatch)
3. Record `input_indices` from `InputSource::Node(id).index()`
4. Tag `needs_shapes: bool` from `fop.output_shape_spec()` and `fop.category()`

Extend `Instruction` (or use `BoxedInstruction`) with:
- `needs_shapes: bool`
- `shape_spec` (for cheap output shape resolution)
- `compiled_shape: Option<SmallVec<[usize; 4]>>`

**Shape-aware vs simple split:** Most ops (elementwise, norms, softmax) are `Simple` — kernel gets flat `&[u8]` inputs, shape is resolved from `ShapeSpec`. Shape-aware ops (Reshape, Shape, MatMul, Gather, Concat, Transpose) need input shapes passed to the kernel.

**Files:**
- New: `hologram/crates/hologram-exec/src/tape_builder.rs`
- Modify: `hologram/crates/hologram-exec/src/tape.rs` (extend Instruction, add `execute_with_shapes()`)
- Modify: `hologram/crates/hologram-exec/src/lib.rs` (re-export)

### Task D: Wire tape into public API (hologram)

Add to `hologram-exec/src/mmap/mod.rs`:

```rust
pub fn build_tape_from_plan(plan: &LoadedPlan) -> ExecResult<Tape>
pub fn execute_tape(tape: &Tape, plan: &LoadedPlan, inputs: &GraphInputs) -> ExecResult<GraphOutputs>
pub fn execute_tape_with_kv_state(...) -> ExecResult<GraphOutputs>
```

KV cache handling: check for `KvWrite`/`KvRead` ops in a pre-level scan (same pattern as current `execute_core_with_kv`), handle those separately, then run the tape for remaining nodes.

Keep `execute_plan()` unchanged — tape path is opt-in.

**Files:**
- `hologram/crates/hologram-exec/src/mmap/mod.rs`
- `hologram/crates/hologram-exec/src/lib.rs`
- `hologram/src/lib.rs` (facade re-exports)

### Task E: Wire tape from HoloRunner (hologram-ai)

Add tape fields to `HoloRunner`:
```rust
tape: Option<hologram::Tape>,
decode_tape: Option<hologram::Tape>,
```

Build tapes in `HoloRunner::from_bytes()` with `.ok()` fallback. Use tape in `execute()` / `execute_with_kv()` when available, fall back to `execute_plan()` otherwise.

**Files:** `hologram-ai/crates/hologram-ai/src/compiler.rs`

### Task F: Wire WeightCache into execution (hologram)

The `WeightCache` in `hologram-exec/src/kv/weight_cache.rs` exists but isn't used. Wire it into the tape executor so quantized weights are deserialized once at first use, not per-dispatch.

Pass `&mut WeightCache` into the tape execution context. LUT-GEMM instructions capture `ConstantId` and look up cached weights.

**Files:**
- `hologram/crates/hologram-exec/src/tape.rs` (add `WeightCache` to execution context)
- `hologram/crates/hologram-exec/src/kv/weight_cache.rs` (no changes needed — already complete)

---

## Ordering

```
Phase 1 (independent, immediate value):
  Task A: SameAs(0) fast path in shape propagation
  Task B: Skip input_shapes for elementwise ops

Phase 2 (tape infrastructure):
  Task C: TapeBuilder (pre-compute kernel + elem_size)
  Task D: Wire tape into hologram public API

Phase 3 (integration):
  Task E: Wire tape from HoloRunner in hologram-ai
  Task F: Wire WeightCache into tape executor
```

Tasks A and B can ship independently and immediately. Tasks C-D are the core tape work in hologram. Task E wires it from hologram-ai. Task F is additive.

---

## Verification

1. Run existing conformance tests through both paths: `cargo test -p hologram-ai-conformance`
2. Run TinyLlama end-to-end generation with tape executor enabled
3. Profile with `--features profile` to measure per-node dispatch time reduction
4. Compare decode latency before/after (target: 0.7s/token -> <0.1s/token with all P2 tasks)

---

## Key Files

| File | Repo | Role |
|------|------|------|
| `crates/hologram-exec/src/tape.rs` | hologram | Tape types, executor loop |
| `crates/hologram-exec/src/eval/executor.rs` | hologram | Current dispatch loop (reference + fast paths) |
| `crates/hologram-exec/src/eval/shape_propagate.rs` | hologram | Shape propagation (fast path target) |
| `crates/hologram-exec/src/float_dispatch/mod.rs` | hologram | `dispatch_float_ctx`, `dispatch_float_into` |
| `crates/hologram-exec/src/kv/weight_cache.rs` | hologram | WeightCache (ready, needs wiring) |
| `crates/hologram-exec/src/mmap/mod.rs` | hologram | Public API surface for tape functions |
| `crates/hologram-ai/src/compiler.rs` | hologram-ai | HoloRunner (tape wiring point) |

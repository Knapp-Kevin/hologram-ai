# Plan 013: Clean KV Cache & Decode Architecture

## Context

KV cache and decode mode currently work (TinyLlama produces `\n2 + 2 is ` correctly
for 7 tokens before Q4_0 noise causes degeneration), but the implementation is a
stack of band-aids:

1. **Metadata erasure**: GGUF importer knows n_kv_heads/head_dim → `AiOp::KvSlotWrite`
   discards them → lowering emits `FloatOp::KvWrite{n_kv_heads:0, head_dim:0}` →
   runtime reverse-engineers via `infer_kv_params()` scanning the compiled graph
2. **Synthetic position injection**: Executor intercepts RoPE nodes when
   `kv.write_pos() > 0` and injects a fake 4-byte tensor via `NodeId + 100_000`
3. **Branching prefill/decode**: KvWrite handler has `if previous.is_empty()` branch
4. **Shape reconstruction**: KV output shape computed from stride = n_kv_heads * head_dim,
   falls back to flat shape when those are 0

This plan replaces all of that with a clean forward-flowing architecture.

**Design principle**: Metadata flows forward (import → IR → lowered graph → archive →
runtime), never backward. No reverse-engineering. No synthetic graph nodes.

---

## Step 1: Extend AiOp::KvSlotWrite/KvSlotRead with architecture params

**Files:**
- `hologram-ai-common/src/ir/op.rs` — add `n_kv_heads: u32, head_dim: u32` fields

```rust
KvSlotWrite { layer: usize, is_key: bool, n_kv_heads: u32, head_dim: u32 },
KvSlotRead  { layer: usize, n_kv_heads: u32, head_dim: u32 },
```

This triggers compiler errors at every construction site, ensuring we fix them all.

---

## Step 2: Populate at construction sites

**GGUF path** (`hologram-ai-gguf/src/arch/llama.rs` ~lines 103-114):
- `params.head_count_kv` and `params.embedding_length / params.head_count` are in scope
- Pass them to `AiOp::KvSlotWrite { n_kv_heads: params.head_count_kv, head_dim: computed }`

**ONNX path** (`hologram-ai-common/src/opt/kv_slot_injection.rs`):
- The pass iterates `GroupedQueryAttention` nodes which have `num_kv_heads` and `head_dim`
- Extract and pass to `AiOp::KvSlotWrite`

---

## Step 3: Fix lowering — forward metadata instead of zeros

**File:** `hologram-ai-common/src/lower/strategy.rs` (lines 653-671)

```rust
AiOp::KvSlotWrite { layer, is_key, n_kv_heads, head_dim } => {
    (FloatOp::KvWrite {
        layer: *layer as u32,
        n_kv_heads: *n_kv_heads,   // was: 0
        head_dim: *head_dim,       // was: 0
        is_key: *is_key,
    }, vec![])
}
```

Same for `KvSlotRead`.

---

## Step 4: Extend ModelMetaSection with KV params

**File:** `hologram/hologram-archive/src/section/model_meta.rs`

Add `n_layers: u32`, `n_kv_heads: u32`, `head_dim: u32` to `ModelMetaSection`.
Non-LLM models set all three to 0.

**Compiler side** (`hologram-ai/src/compiler.rs` + `cli.rs`):
- Add `n_kv_heads` and `head_dim` to `ModelMetadata`
- Populate from `extract_metadata()` (already reads from `graph.metadata`)
- Write into `ModelMetaSection` during archive creation

---

## Step 5: Delete `infer_kv_params()` — read from archive metadata

**File:** `hologram-ai/src/commands/run_cmd.rs`

Replace the 40-line `infer_kv_params()` function (which scans compiled graph nodes
to recover erased metadata) with a direct read from `ModelMetaSection`:

```rust
let n_layers = meta.n_layers;
let n_kv_heads = meta.n_kv_heads;
let head_dim = meta.head_dim;
```

---

## Step 6: Add ExecutionContext to eliminate synthetic position hack

**File:** `hologram/hologram-exec/src/eval/executor.rs`

```rust
pub struct ExecutionContext {
    pub position_offset: u32,  // Current token position for RoPE
}
```

- Created at start of each `execute_with_kv_state()` call from `kv.write_pos()`
- Passed to `dispatch_float(op, inputs, ctx)` for all nodes (not just RoPE)
- RoPE reads `ctx.position_offset` instead of a synthetic input[1] tensor
- Non-KV execution passes `None` — zero overhead

**File:** `hologram/hologram-exec/src/float_dispatch.rs`

Add `ctx: Option<&ExecutionContext>` parameter to `dispatch_float`. Only RoPE reads it:

```rust
FloatOp::RotaryEmbedding { dim, base, n_heads } => {
    let start_pos = ctx.map(|c| c.position_offset as usize).unwrap_or(0);
    dispatch_rope(inputs, *dim, *base, *n_heads, start_pos)
}
```

**Delete:** The entire `RotaryEmbedding { .. } if kv.write_pos() > 0` match arm and
the `NodeId + 100_000` synthetic tensor injection.

---

## Step 7: Unified prefill/decode path via `read_through()`

**File:** `hologram/hologram-exec/src/kv_cache.rs`

Add methods that include pending (just-written, pre-advance) data:

```rust
pub fn read_k_through(&self, layer: u32, pending_seq: usize) -> &[f32] {
    let stride = self.n_kv_heads as usize * self.head_dim as usize;
    let end = (self.write_pos + pending_seq) * stride;
    &self.k_buffers[layer as usize][..end]
}
```

**File:** `hologram/hologram-exec/src/eval/executor.rs`

Replace the branching KvWrite handler with:

```rust
// Write to cache
if *is_key { kv.write_layer(*layer, floats, &[]); }
else { kv.write_layer(*layer, &[], floats); }

// Output: full cache including just-written data (unified path)
let full = if *is_key {
    kv.read_k_through(*layer, seq)
} else {
    kv.read_v_through(*layer, seq)
};
arena.insert(node_id, bytemuck::cast_slice(full).to_vec());
shape_map.insert(node_id, vec![
    kv.write_pos() + seq,
    *n_kv_heads as usize,
    *head_dim as usize,
]);
```

No `if previous.is_empty()` branch. Prefill (write_pos=0): reads [0..N].
Decode (write_pos=M): reads [0..M+1]. Same code.

---

## Implementation Order

| # | Scope | Step | Dependency |
|---|-------|------|------------|
| 1 | hologram-ai | Steps 1-3: AiOp fields + lowering | None |
| 2 | hologram | Step 4: ModelMetaSection fields | None (parallel with 1) |
| 3 | hologram-ai | Step 5: Delete infer_kv_params | Steps 1-4 |
| 4 | hologram | Step 6: ExecutionContext | None (parallel with 1-3) |
| 5 | hologram | Step 7: read_through + unified path | Step 6 |
| 6 | hologram | Step 8: Remove clone | Subsumed by Step 7 |

Steps 1+2 and 4 can run in parallel. Total: 3 sequential phases.

---

## Verification

```bash
# After each step, all existing tests must pass:
cd /Users/auser/work/uor/hologram/hologram && cargo test -p hologram-exec --lib
cd /Users/auser/work/uor/hologram/hologram-ai && cargo test -p hologram-ai-common
cd /Users/auser/work/uor/hologram/hologram-ai && cargo test -p hologram-ai --test mini_fixture

# After Step 5: recompile and verify KV params from archive
cargo run -p hologram-ai -- compile --model models/TinyLlama-*.gguf \
  --tokenizer models/TinyLlama-*/tokenizer.json --output /tmp/kv_test
cargo run -p hologram-ai -- info /tmp/kv_test/*.holo  # should show n_kv_heads=4 head_dim=64

# After Step 7: full generation test
cargo run -p hologram-ai -- run /tmp/kv_test/*.holo \
  --prompt "..." --max-tokens 10 --temperature 0

# Full regression
cargo test && cargo clippy -- -D warnings
```

## Critical Files

| Step | File | Change |
|------|------|--------|
| 1 | `hologram-ai-common/src/ir/op.rs` | Add n_kv_heads/head_dim to KvSlotWrite/KvSlotRead |
| 2 | `hologram-ai-gguf/src/arch/llama.rs` | Populate from params |
| 2 | `hologram-ai-common/src/opt/kv_slot_injection.rs` | Populate from GQA |
| 3 | `hologram-ai-common/src/lower/strategy.rs` | Forward metadata (no zeros) |
| 4 | `hologram/hologram-archive/src/section/model_meta.rs` | Add KV fields |
| 4 | `hologram-ai/src/compiler.rs` + `cli.rs` | Populate ModelMetaSection |
| 5 | `hologram-ai/src/commands/run_cmd.rs` | Delete infer_kv_params() |
| 6 | `hologram/hologram-exec/src/eval/executor.rs` | ExecutionContext |
| 6 | `hologram/hologram-exec/src/float_dispatch.rs` | ctx param on dispatch_float |
| 7 | `hologram/hologram-exec/src/kv_cache.rs` | read_k_through/read_v_through |
| 7 | `hologram/hologram-exec/src/eval/executor.rs` | Unified KvWrite handler |

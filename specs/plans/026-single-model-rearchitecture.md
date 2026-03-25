# Rearchitecture: Single-Model LLM with Runtime KV Cache

## Context

After 2-3 days debugging the prefill/decode pipeline split, the root cause is clear: compiling TWO separate models (prefill at seq=2048, decode at seq=1) creates a data layout mismatch at the KV cache boundary that no amount of transpose hacking can fix. The runtime (`dispatch_kv_write`, `resolve_size`) already handles variable-length inputs — the dual-model split is unnecessary complexity.

**The fix**: compile ONE model. Use it for both prefill and decode. The KV cache's `write_pos == 0` check already distinguishes the two modes at runtime. Remove the pipeline split entirely.

## Architecture: Pipeline Is The Only Format

**One mode everywhere.** Every model — LLM, ResNet, BERT, Whisper, multi-component — compiles through `compile_components()` as a pipeline archive. A single-component model is just a pipeline with 1 component.

This means:
- Remove `compile_single_graph()` — dead path
- Remove `compile_llm_pipeline()` — was the dual-model split
- `compile()` always calls `compile_components()` with 1+ ComponentSpecs
- `HoloRunner` always loads a pipeline archive
- No `is_pipeline` checks — everything is a pipeline

### Remove
- `compile_single_graph()` — replaced by `compile_components([single_spec])`
- `compile_llm_pipeline()` — replaced by `compile_components([single_llm_spec])`
- `force_single_graph` flag + CLI `--single-graph`
- `HoloRunner.decode_plan`, `decode_tape`, `_decode_bytes` — no dual-model state
- `is_pipeline` checks — everything is pipeline
- Pipeline vs single-graph detection in `from_storage()`
- `KvLayout` enum — redundant with `heads_first` on the KV ops

### Keep
- `compile_components()` — becomes THE compilation path for all models
- `PipelineWriter` — universal archive format (1 or N components)
- `KvSlotInjection` — still injects KvWrite/KvRead around GQA
- `KvCacheState` — unchanged
- `dispatch_kv_write/read` — already handles prefill/decode via `write_pos`
- `resolve_size()` — runtime shape inference
- `TensorMeta` — runtime metadata
- `compile_multi_onnx()` — for Whisper/SD multi-ONNX pipelines
- `WeightStore` + `build_with_shared_weights()` — weight dedup across components

## Implementation Steps

### Step 1: Unify compilation — everything through `compile_components`

**File**: `crates/hologram-ai/src/compiler.rs`

Replace the `is_llm` branching with a single path:
```rust
// Before:
let archive_bytes = if is_llm && !self.force_single_graph {
    self.compile_llm_pipeline(&ai_graph, &mem_plan, pre_concretized)?
} else {
    self.compile_single_graph(&ai_graph, &mem_plan)?
};

// After:
let archive_bytes = self.compile_components(
    vec![ComponentSpec {
        name: "model".into(),
        role: ComponentRole::Backbone,
        weight_group: "model".into(),
        graph: &ai_graph,
        mem_plan: &mem_plan,
        phase: LowerPhase::Forward,
        weights: collect_weight_bytes(&ai_graph)?.into(),
        ..
    }],
    vec![], // no inter-component connections for single model
)?;
```

Remove `compile_single_graph()` and `compile_llm_pipeline()`.
Remove `force_single_graph` field + `--single-graph` flag.
Remove `pre_concretized` graph (no need to clone for decode).

### Step 2: Simplify HoloRunner — always loads pipeline

**File**: `crates/hologram-ai/src/compiler.rs`

`HoloRunner` always loads a pipeline archive (even for 1-component models):
- Remove `decode_plan`, `decode_tape`, `_decode_bytes`
- Remove `is_pipeline` flag — everything is pipeline
- `from_storage()` always uses `LoadedPipeline` to parse
- For 1-component pipeline, loads the single model's plan + tape

`execute_with_kv()` is trivial:
```rust
pub fn execute_with_kv(&self, inputs, kv_state) {
    hologram::execute_tape_with_kv(&self.tape, &self.plan, inputs, kv_state)
}
```

### Step 3: Simplify run_cmd.rs

**File**: `crates/hologram-ai/src/commands/run_cmd.rs`

- Remove `runner.is_pipeline()` check — always check `model_meta.n_layers > 0` for KV cache
- Remove padding mode logic — variable-length inputs work with `resolve_size`
- Position IDs: `[write_pos]` for decode (already correct)

### Step 4: Remove KvLayout and heads_first from KV ops

Since there's only one model with one layout:
- Remove `KvLayout` enum from `ir/op.rs`
- Remove `layout` field from `AiOp::KvSlotWrite/KvSlotRead`
- Remove `heads_first` from `FloatOp::KvWrite/KvRead`
- Remove `heads_first` from `TapeKernel::KvWrite/KvRead`
- KvWrite always transposes heads→seq for storage (the data IS heads-first from ONNX fusion)
- KvRead always transposes seq→heads for output

For GGUF (seq-first): add `heads_first` flag back ONLY on the GQA op (already exists). KvSlotInjection reads the flag and decides whether to transpose. This is simpler than a separate `KvLayout` type.

Wait — actually keep `heads_first` on the KV ops. It's clean, explicit, and handles both ONNX and GGUF. Just remove `KvLayout` (redundant with `heads_first`).

### Step 5: Clean up conformance tests

- Remove `force_single_graph` from all tests
- Simplify `tinyllama_decode_conformance` — compile ONE model, run prefill + decode
- Remove probe sub-model tests that were debugging artifacts

### Step 6: Verify

1. `cargo test` — all tests pass
2. `tinyllama_logit_conformance` — prefill matches ORT
3. `tinyllama_decode_conformance` — decode matches ORT
4. CLI: `hologram-ai run model.holo --prompt "What is the capital of France?"` → coherent English
5. BERT and ResNet E2E tests — still pass
6. `cargo clippy -- -D warnings` — clean

## Files Modified

| File | Change |
|------|--------|
| `compiler.rs` | Remove `compile_llm_pipeline`, `force_single_graph`, simplify `HoloRunner` |
| `cli.rs` | Remove `--single-graph` flag |
| `run_cmd.rs` | Remove `is_pipeline` check, simplify KV cache setup |
| `ir/op.rs` | Remove `KvLayout` enum, remove `layout` from KvSlot ops |
| `kv_slot_injection.rs` | Use `heads_first` from GQA directly (already does) |
| `lower/strategy.rs` | Remove `layout` → `heads_first` mapping |
| `exec_conformance.rs` | Remove `force_single_graph`, simplify decode test |

## What This Achieves

- **ONE format (pipeline), ONE compilation path, ONE execution mode** — for ALL models
- **Removes ~400+ lines** of dual-model split code, `is_pipeline` checks, dual-tape dispatch
- **Eliminates the entire class of layout mismatch bugs** — no more prefill/decode data handoff
- **TensorMeta + resolve_size** handle variable-length inputs at runtime
- **KvWrite/KvRead** handle prefill vs decode via `write_pos == 0` (already works)
- **No more "coincidental" flat layout** — data is genuinely heads-first throughout
- **Multi-component models** (Whisper, SD) work through the SAME `compile_components` path
- **Simpler mental model**: one model, one tape, variable-length inputs, KV cache for autoregressive

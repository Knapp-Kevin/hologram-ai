# Plan 073: Compilation Speed Optimization

## Context

Prior P4 work (Plan 017/020) addressed low-hanging fruit: LTO, cached topo_order,
early convergence, avoid double LLM compilation. The remaining bottleneck is the
**LLM pipeline** which compiles 3 independent graphs (prefill/decode/verify)
entirely sequentially. Each goes through concretize -> repair -> validate ->
memory plan -> lower -> compile. Secondary opportunities exist in weight
quantization parallelism and opt pass skipping.

## Phase 1: Instrumentation (do first)

Add `tracing::info_span!` around each major pipeline stage so we can measure
where time actually goes. No logic changes — zero regression risk.

**Files:**
- `crates/hologram-ai/src/compiler.rs` — spans around:
  - Import (~L283)
  - Optimization pipeline (~L335)
  - Weight collection (~L376)
  - Each LLM graph prep (concretize+repair+validate+plan: ~L407/427/446)
  - Each `compile_one_component` call (~L484/489/494)
  - Inside `compile_one_component`: spans for `lower()`, `hologram::compile()`,
    and archive assembly
- `crates/hologram-ai-common/src/opt/pipeline.rs` — span per pass in `run()` (L181-183)
- `crates/hologram-ai-common/src/lower/builder.rs` — spans around weight
  registration loop and node lowering loop

## Phase 2: Parallelize LLM graph compilation (highest impact)

The 3 LLM graphs are fully independent after `ai_graph.clone()`. They share
no mutable state.

**Step 2a** — Extract repeated concretize -> repair -> validate -> plan into:
```rust
struct PreparedComponent {
    graph: AiGraph,
    mem_plan: MemoryPlan,
    seq_dim_positions: Option<HashSet<(TensorId, usize)>>,
}

fn prepare_component(
    ai_graph: AiGraph,
    seq_len: Option<u64>,
    name: &str,
    track_seq_dims: bool,
) -> anyhow::Result<PreparedComponent>
```

**Step 2b** — Run all 3 preparations in parallel via `std::thread::scope`:
```rust
std::thread::scope(|s| {
    let prefill = s.spawn(|| prepare_component(ai_graph, seq_override, "prefill", true));
    let decode  = s.spawn(|| prepare_component(decode_graph, Some(1), "decode", false));
    let verify  = s.spawn(|| prepare_component(verify_graph, Some(8), "verify", false));
});
```

Use `std::thread::scope` (not rayon) to avoid work-stealing contention with
inner parallelism in `hologram::compile`.

**Step 2c** — Run `compile_one_component` for all 3 in parallel the same way.

**Thread safety:** `compile_one_component` calls `lower()` (pure, takes `&AiGraph`)
and `hologram::compile()` (takes owned `Graph`). Both are stateless. Weight index
is passed by reference to prefill only; decode/verify pass `None`. Safe to parallelize.

**File:** `crates/hologram-ai/src/compiler.rs` lines 398-623

**Expected speedup:** ~2.5-3x on the LLM compilation path (wall clock =
max(prefill, decode, verify) instead of sum).

## Phase 3: Pass skip predicates

Add optional `should_run` to the `Pass` trait. Fusion passes can cheaply check
whether the graph contains matching op patterns before doing a full traversal.

**Files:**
- `crates/hologram-ai-common/src/opt/pipeline.rs` — trait + check in `run()`
- Individual pass files — implement predicates

```rust
pub trait Pass: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, graph: AiGraph) -> anyhow::Result<AiGraph>;
    fn should_run(&self, _graph: &AiGraph) -> bool { true }
}
```

Candidates: `SwiGluFusion` (no Sigmoid+Mul), `LayerNormFusion` (no ReduceMean),
`AttentionFusion` (no MatMul), `KvSlotInjection` (no GroupedQueryAttention),
`PatchPruneInjection` (no Conv2d).

**Expected speedup:** ~5-10% on optimization time (more for generic/ViT pipelines
where LLM-specific passes are definitely no-ops).

## Phase 4: Parallel weight quantization

During lowering, Q4 quantization processes weights one at a time. Pre-quantize
eligible weights in parallel before the main lowering loop using bounded chunks.

**File:** `crates/hologram-ai-common/src/lower/builder.rs`

Before the node lowering loop (~L347), batch-quantize with `rayon::par_iter`
in chunks of 8 (controls peak memory to ~240 MB for 30 MB average weights):

```rust
for chunk in q4_eligible_vec.chunks(8) {
    let results: Vec<_> = chunk.par_iter()
        .filter_map(|&tid| { /* quantize */ })
        .collect();
    for (tid, bytes) in results {
        early_quant_bytes.insert(tid, bytes);
    }
}
```

**Expected speedup:** ~10-20% on Q4 compilation.

## Phase 5: Arc-shared param data (reduces clone cost)

Change `AiParam::Inline { data: Vec<u8>, .. }` to `Arc<Vec<u8>>`. Makes
`ai_graph.clone()` near-free for data payload. Mmap params are already metadata.

**Files:**
- `crates/hologram-ai-common/src/ir/param.rs` — enum definition
- ONNX importer callsites constructing `AiParam::Inline`
- `crates/hologram-ai-common/src/lower/builder.rs` `param_bytes_owned()` — `.to_vec()`

Mechanical refactor. Benefits Phase 2 by making the two `ai_graph.clone()` cheaper.

## Implementation order

```
Phase 1 (instrument)         <- do first, no dependencies
    |
Phase 2 (parallel graphs)   <- biggest win
    |
Phase 3 (skip predicates)   <- guided by Phase 1 data
Phase 4 (parallel quant)    <- independent
Phase 5 (Arc params)        <- helps Phase 2
```

## Verification

1. `cargo test` — all existing tests pass
2. `cargo clippy -- -D warnings` — no new warnings
3. Compile TinyLlama ONNX with `RUST_LOG=info` and compare tracing span
   durations before/after each phase
4. Verify identical `.holo` archive output (byte-for-byte) before/after
   parallelization changes
5. `cargo bench --bench inference` — no runtime regression

## Expected total speedup

**LLM compilation: ~3-4x faster** (Phase 2 dominates).
Non-LLM compilation: ~10-15% faster (Phases 3-5).

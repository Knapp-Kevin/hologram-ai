# Plan 017 — Performance Optimization

## Context

The SPRINT.md lists performance tasks (P1-P3) under "Short Term: make it fast". hologram-ai is a compiler-only crate — it can't change runtime kernels (those live in hologram base crate), but it can:
1. Produce better-optimized compiled graphs (fused attention, KV cache)
2. Compile faster (reduce compilation time)
3. Enable users to compile at optimal settings (smaller seq, release builds)

The compilation pipeline currently runs **~40 optimization passes** per compilation (11 MVP + 27 aggressive fixpoint + 2 post). For LLM models, the decode graph triggers a **full re-import and re-compilation**, doubling total time.

---

## Phase 1: Quick Wins (compiler settings)

### 1.1 Release profile with LTO
**File:** `Cargo.toml`
**Effort: S | Impact: HIGH**

Add release profile — currently only `[profile.dev] opt-level = 2` exists. No release profile means Cargo defaults (no LTO, 16 codegen-units).

```toml
[profile.release]
lto = "thin"
codegen-units = 1
```

`codegen-units = 1` alone yields 10-20% for compute-bound workloads. `thin` LTO adds cross-crate inlining. This benefits both compilation speed (compiler runs faster) and execution speed (hologram base kernels link better).

### 1.2 Early convergence detection in fixpoint loop
**File:** `crates/hologram-ai/src/compiler.rs:214`
**Effort: S | Impact: MEDIUM**

The aggressive pipeline runs exactly 3 iterations. Add a convergence check — count remaining `Dynamic`/`Var` dims after each iteration. Break early when count stops decreasing. Saves up to 9 pass invocations when model converges in 2 iterations.

```rust
let mut prev_dynamic_count = usize::MAX;
for pass_num in 0..3 {
    ai_graph = aggressive_pipeline.run(ai_graph)?;
    let dynamic_count = count_non_concrete_dims(&ai_graph);
    if dynamic_count >= prev_dynamic_count {
        break; // converged
    }
    prev_dynamic_count = dynamic_count;
    // clear known_i64_values...
}
```

### 1.3 Extract shared `post_concretization_repair` function
**File:** `crates/hologram-ai/src/compiler.rs`
**Effort: S | Impact: LOW (enables 1.2 cleanly)**

The aggressive pipeline + fixpoint loop is duplicated in `compile()` (line 181) and `compile_with_debug_info()` (line ~513). Extract to a shared function so convergence detection applies everywhere.

---

## Phase 2: Compilation Pipeline Speedups

### 2.1 Cache `topo_order` on AiGraph
**File:** `crates/hologram-ai-common/src/ir/graph.rs:192`
**Effort: M | Impact: MEDIUM**

`topo_order()` builds 3 HashMaps per call. Called ~40 times per compilation. Add a cached field:

```rust
pub struct AiGraph {
    // ...existing...
    cached_topo: RefCell<Option<Vec<NodeId>>>,
}
```

- Read-only passes (ShapeProp, DataProp, ConstEval) reuse the cache
- Passes that modify nodes (DeadNodeElim, ConstFold, AttentionFusion) invalidate via `invalidate_caches()`

### 2.2 Vec-indexed lookups instead of HashMap
**Files:** `crates/hologram-ai-common/src/ir/graph.rs`, all pass files
**Effort: M | Impact: MEDIUM**

Every pass independently builds `HashMap<u32, usize>` for node lookup. Since `NodeId = u32` and max IDs are ~2000 (TinyLlama), use `Vec<Option<usize>>` indexed by NodeId. Add helper methods on AiGraph:

```rust
fn node_index_map(&self) -> Vec<Option<usize>>  // NodeId → index in nodes vec
fn producer_map(&self) -> Vec<Option<NodeId>>     // TensorId → producing NodeId
```

Cache alongside topo_order.

### 2.3 Avoid double compilation for LLM decode graph
**File:** `crates/hologram-ai/src/compiler.rs:856`
**Effort: L | Impact: HIGH for LLM compile time**

Currently `compile_llm_pipeline` re-imports from disk and runs the full pipeline again with `seq_len_override: Some(1)`. Clone the AiGraph after MVP optimization (before concretization), then concretize twice:

```rust
let ai_graph_mvp = pipeline.run(ai_graph)?;
let prefill = concretize_and_lower(ai_graph_mvp.clone(), seq_len)?;
let decode  = concretize_and_lower(ai_graph_mvp, 1)?;
```

Saves: ONNX re-parse + 11 MVP passes. For TinyLlama, this is ~50% of total LLM compile time.

---

## Phase 3: Execution Speed via Better Code Generation

### 3.1 Criterion benchmark harness
**Files:** `Cargo.toml`, new `crates/hologram-ai-common/benches/`
**Effort: M | Impact: Enabling (prerequisite for measuring all other work)**

Add criterion benchmarks for:
- `topo_order()` on 1156-node graph
- Full MVP pipeline on a small synthetic graph
- Aggressive pipeline single iteration
- `lower()` on a post-optimization graph

### 3.2 Re-enable AttentionFusion with conformance gate (SPRINT P2)
**File:** `crates/hologram-ai-common/src/opt/attention_fusion.rs`
**Effort: L | Impact: HIGH**

Fused attention (single GroupedQueryAttention op) is much faster than 4+ individual ops (MatMul, Softmax, Add, Transpose). Four known bugs to fix:

1. **K^T routing** — `find_pre_transpose` stops at Mul nodes on the K path. Fix: trace through Mul to find both the scale and the Transpose.
2. **Double-scaling** — scale on K path not detected, gets applied twice. Fix: detect scale Mul on K path and pass it to the fused kernel.
3. **Output shape** — produces `[1,1,5,2048]` instead of `[1,32,5,64]`. Fix: reshape output to `[batch, num_heads, seq, head_dim]`.
4. **GQA head count** — V uses expanded 32 heads but kernel expects 4-head GQA. Fix: detect whether K/V are pre-expanded; if so, set `num_kv_heads = num_heads`.

**Strategy:** Write conformance test fixtures first (SDPA variants from SPRINT P2), then fix each bug, then add a conformance gate that reverts fusion if output diverges.

### 3.3 KV cache for autoregressive generation (SPRINT P3)
**File:** `crates/hologram-ai-common/src/opt/kv_slot_injection.rs`
**Effort: L | Impact: HIGH (depends on 3.2)**

Infrastructure exists: `KvSlotInjection` pass, `LowerPhase::Prefill/Decode`, `compile_llm_pipeline`, `MemoryPlanner`. Re-enable once AttentionFusion works:

1. Re-enable `KvSlotInjection` in MVP pipeline (already present at pipeline.rs:49)
2. Verify prefill writes K/V to cache
3. Verify decode reads cached K/V and processes only 1 new token
4. Test multi-token generation for coherent output

---

## Execution Order

| # | Task | Effort | Impact | Depends On |
|---|------|--------|--------|------------|
| 1 | 1.1 Release profile + LTO | S | HIGH | — |
| 2 | 1.3 Extract shared repair function | S | LOW | — |
| 3 | 1.2 Early convergence detection | S | MEDIUM | #2 |
| 4 | 2.1 Cache topo_order | M | MEDIUM | — |
| 5 | 2.2 Vec-indexed lookups | M | MEDIUM | #4 |
| 6 | 3.1 Criterion benchmarks | M | Enabling | — |
| 7 | 2.3 Avoid double LLM compilation | L | HIGH | #2 |
| 8 | 3.2 Re-enable AttentionFusion | L | HIGH | #6 |
| 9 | 3.3 KV cache | L | HIGH | #8 |

Items 1-3 are a quick first PR. Items 4-5 are a second PR. Item 7 is a standalone PR. Items 8-9 are the heavy lifts from SPRINT P2/P3.

---

## Verification

- **1.1**: `cargo build --release` succeeds; binary size and `hologram-ai compile` wall time decrease
- **1.2-1.3**: Add `tracing::info!` for iteration count; verify TinyLlama converges in ≤2 iterations
- **2.1-2.2**: Criterion benchmarks show improvement on `topo_order` and pass runtimes
- **2.3**: `cargo test` passes; LLM compilation time measured before/after
- **3.1**: `cargo bench` runs and produces reports
- **3.2**: Conformance tests pass for all 4 SDPA variants; fused TinyLlama matches ORT
- **3.3**: Multi-token generation produces coherent English text

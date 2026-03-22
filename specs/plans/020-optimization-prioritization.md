# Plan 020: Optimization Task Prioritization

## Context

The sprint has four remaining optimization areas (P2d, P3, P4, P5) with ~15
individual tasks. This plan assesses what has been done, what remains, and
provides a priority ordering based on impact, effort, and dependencies.

Design principle: hologram-ai is a compiler only (ADR-0016). All runtime
kernels live in hologram base. Tasks that require new FloatOps or kernel
changes are cross-repo and noted as such.

---

## Status Assessment

### P2d: Remaining decode optimizations — MOSTLY DONE

| Task | Status | Repo | Notes |
|------|--------|------|-------|
| `dispatch_float_into` — buffer reuse | **DONE** | hologram base | Wired into tape executor via `BoxedInstruction::FloatInto` |
| `WeightCache` — cache deserialized quantized weights | **DONE** | hologram base | `TapeContext.weight_cache` caches across dispatches |
| Level-aware tape execution for KV decode | Design only | hologram base | Split tape around KvWrite/KvRead per level |

**Key constraint:** f32 ONNX decode at 13.6 tok/s is near memory bandwidth
ceiling (4.1 GB weights x ~60 GB/s DDR = ~15 tok/s theoretical max). Further
speedup requires weight quantization (GGUF models).

### P3: Compiler fusion passes — MOSTLY DONE

| Task | Status | Repo | Notes |
|------|--------|------|-------|
| SwiGLU fusion | **DONE** | hologram-ai | `swiglu_fusion.rs` wired into MVP pipeline |
| Add+RMSNorm residual fusion | **DONE** | Both | Pass + lowering in hologram-ai; kernel + dispatch in hologram base |
| QK-Norm + RoPE + KV-Store fusion | Design only | Cross-repo | Depends on stable tape executor + hologram base Attention op extension |

### P4: Compilation speed — DONE

| Task | Status | Repo | Notes |
|------|--------|------|-------|
| Release profile with LTO | **DONE** | hologram-ai | `[profile.release]` with `lto="thin"`, `codegen-units=1` |
| Extract `post_concretization_repair` | **DONE** | hologram-ai | Was duplicated 3x, now a single function |
| Early convergence detection | **DONE** | hologram-ai | Breaks when dynamic dim count stops decreasing |
| Cache `topo_order` | **DONE** | hologram-ai | `RefCell<Option<Vec<NodeId>>>` with invalidation |
| Avoid double LLM compilation | **DONE** | hologram-ai | Clone pre-optimized graph, re-concretize at seq=1 |

### P5: Variable-length prefill — BLOCKED

| Task | Status | Repo | Notes |
|------|--------|------|-------|
| Wire ShapeContextGraph into execute() | Implemented but disabled | hologram-ai | SeqMode::Variable exists but disabled |
| SeqMode::Variable | Disabled | hologram-ai | Most recent commit disabled it |
| Hologram executor baked param resolution | **Blocker** | hologram base | FloatOp params (m/k/n, size) baked at compile time |

**Blocker:** hologram executor bakes FloatOp params at compile time. When
runtime buffer sizes differ from compiled values, results are wrong. Unblocking
requires hologram base to resolve baked params from runtime buffer sizes.

---

## Priority Order

### Tier 1: Quick wins — DONE

1. ~~Release profile with LTO~~ (P4) — DONE
2. ~~Extract shared `post_concretization_repair`~~ (P4) — DONE
3. ~~Early convergence detection in fixpoint loop~~ (P4) — DONE

### Tier 2: Compilation speed — DONE

4. ~~Cache `topo_order` on AiGraph~~ (P4) — DONE
5. ~~Avoid double LLM compilation~~ (P4) — DONE

### Tier 3: Cross-repo — DONE

6. ~~Wire `dispatch_float_into`~~ (P2d) — **DONE** in hologram base.
   Wired into tape executor via `BoxedInstruction::FloatInto`.

7. ~~Wire `WeightCache` into tape executor~~ (P2d) — **DONE** in hologram base.
   `TapeContext.weight_cache` caches deserialized quantized weights.

8. ~~Add+RMSNorm residual fusion~~ (P3) — **DONE** end-to-end.
   hologram-ai: pass + lowering + pipeline. hologram base: kernel + dispatch.

### Tier 4: Blocked / deferred

9. **Level-aware tape execution** (P2d) — tape executor maturity needed
10. **QK-Norm + RoPE + KV-Store fusion** (P3) — design first, cross-repo
11. **Variable-length prefill** (P5) — hologram executor baked param blocker

---

## Execution Plan

**Phase A (Tier 1):** DONE. Release profile, fixpoint dedup, convergence
detection.

**Phase B (Tier 2):** DONE. Cached topo_order + avoid double LLM compilation.

**Phase C (Tier 3):** DONE. dispatch_float_into and WeightCache already
wired in hologram base tape executor. AddRmsNorm kernel already implemented.

**Phase D (Tier 4):** Blocked until prerequisites are met.

---

## Verification

- `cargo build --release` succeeds (Tier 1)
- `cargo test` passes after each change
- `cargo clippy -- -D warnings` clean
- LLM compilation time measured before/after Tier 2 changes
- Conformance tests pass after any fusion changes

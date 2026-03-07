# hologram-ai: Risk Register

---

## Format

Each risk includes:
- **Description** — what the risk is
- **Impact** — how bad if it materializes (High / Medium / Low)
- **Likelihood** — how likely (High / Medium / Low)
- **Mitigation** — what we do about it
- **Phase** — MVP or later

---

## R-01: Operator Coverage Gaps

**Description:** The ONNX op set is vast (200+ operators). The LLaMA/GGUF
architecture map covers a specific subset. Models from other architectures
may use ops that aren't in the dispatch table, producing `AiOp::Opaque` nodes
that block lowering.

**Impact:** High — any opaque node causes a hard lowering failure for the affected model.

**Likelihood:** High — expected for edge-case ONNX models and newer architectures.

**Mitigation:**
- `AiOp::Opaque` is a named, explicit type; failures are clear
- Coverage gap tracking: run importer over HuggingFace ONNX model zoo, record missing ops
- Priority-order gap closure by model popularity
- `hologram-ai-validate` prints coverage report before test
- Accept opaque-node models as "unsupported" with clear error message, not panic

**Phase:** MVP (partially) — full coverage is Phase 2+

---

## R-02: Quantization Complexity

**Description:** GGUF supports 20+ quantization schemes. Many have subtly
different block layouts, scale formats, and alignment requirements. Incorrect
dequant logic produces numerically wrong outputs that may not be obviously
wrong (tokens look plausible but are wrong).

**Impact:** High — silent numerical errors are worse than crashes.

**Likelihood:** High — dequant bugs are common in new implementations.

**Mitigation:**
- Unit tests for every quant scheme with precomputed reference values (from GGML source)
- Validate dequant output against llama.cpp `--debug-dump-quants` mode
- Use `hologram-ai-validate` for golden token comparison on all quant types before shipping
- Priority order: Q4_0 → Q8_0 → Q4_K_M → Q5_K_M → Q6_K → remainder

**Phase:** MVP and ongoing

---

## R-03: Backend Kernel Capability Mismatch

**Description:** `hologram-ai-lower` assumes certain hologram kernels exist (e.g.
`qgemm_q4_0_f32`, `gqa_fused_f32`, `rope_f32`). If these don't exist in the
target `hologram` version, lowering fails or falls back to slower decomposed paths.

**Impact:** Medium — fallback paths work but may be 2–10x slower.

**Likelihood:** Medium — hologram is under active development; kernel availability unknown.

**Mitigation:**
- Define `BackendCapabilities` struct queried at lowering time
- Every kernel has a software fallback path
- Maintain a hologram kernel availability table in `specs/`
- Sync with hologram team before Phase 3 (quantized kernel integration)
- CPU backend is all-software, guaranteed to work without kernel assumptions

**Phase:** Phase 2–3 (CPU MVP is unaffected)

---

## R-04: Memory Planning Uncertainty

**Description:** Tensor liveness analysis and buffer aliasing depend on
understanding which ops can reuse buffers. Incorrect liveness → double-write
or use-after-free of aliased buffers. Incorrect sizing → OOM during inference.

**Impact:** High — memory bugs are hard to debug, especially on device.

**Likelihood:** Medium — liveness analysis is well-understood but complex.

**Mitigation:**
- Use conservative (no-alias) memory planning for MVP
- Introduce aliasing incrementally with explicit tests per aliasing decision
- Validate memory plan total bytes against reference (GGUF file size + activation estimate)
- `MemoryPlanner` has a `conservative: bool` flag; default true for MVP

**Phase:** MVP (conservative), Phase 2 (optimized)

---

## R-05: Dynamic Shape Complexity

**Description:** `seq_len` in LLMs is dynamic (varies per request). The
hologram `ExecutionPlan` may not support dynamic-shape nodes, requiring either
a fixed-shape plan per sequence length or runtime dispatch.

**Impact:** High — without dynamic shapes, KV-cache and variable-length prompts
are extremely awkward.

**Likelihood:** Medium — depends on hologram's plan design.

**Mitigation:**
- Use `Dim::Symbolic("seq_len")` throughout IR to track the problem early
- For MVP: fix seq_len at compile time (e.g. max_seq_len) as simplification
- Phase 2: coordinate with hologram team on dynamic shape plan support
- Fallback: compile N plans for common seq lengths (chunked attention)

**Phase:** MVP avoids it; Phase 2 requires resolution

---

## R-06: LLM Runtime Semantics Drift

**Description:** New LLM architectures (Mamba, RWKV, SSM variants) use
execution patterns (recurrent state, selective scan) that don't map cleanly
to the transformer-centric `AiOp` set. Over time the op set may become
architecture-biased.

**Impact:** Medium — limits the range of models we can support.

**Likelihood:** Medium — architecture diversity is increasing.

**Mitigation:**
- Keep `AiOp` extensible with escape hatches (`Opaque`, custom ops)
- Design the architecture recognizer registry to be easily extended
- Do not over-specialize `AiGraph` for transformer-specific patterns in MVP
- Review op set every 6 months against active model landscape

**Phase:** Phase 3+

---

## R-07: Tokenizer Integration Risk

**Description:** `hologram-ai-stream` requires a tokenizer to encode prompts and
decode tokens. Tokenizer implementations (BPE, SentencePiece, tiktoken) are
complex. Bundling one introduces a large dependency; using an external one
requires a stable interface.

**Impact:** Low — tokenizer is logically separate from inference pipeline.

**Likelihood:** Low — `Box<dyn Tokenizer>` approach avoids bundling.

**Mitigation:**
- `Tokenizer` trait is a thin interface: `encode`, `decode`, `eos_token_id`
- Ship zero bundled tokenizers in `hologram-ai`
- Provide example integration with `tokenizers` crate (HuggingFace)
- GGUF metadata includes tokenizer vocab; provide a minimal BPE implementation
  in `hologram-ai-stream` as an optional feature for standalone use

**Phase:** Phase 2 (needed for streaming demo)

---

## R-08: Portability Risk (Windows / WASM)

**Description:** Some hologram backends or platform-specific ops may not
compile or work on all targets. SIMD intrinsics differ across `x86_64` and
`aarch64`. WASM has no threading model.

**Impact:** Medium — reduces deployable targets.

**Likelihood:** Medium — WASM and Windows are lower-priority targets.

**Mitigation:**
- Mark platform-specific code with `#[cfg(target_arch = ...)]` guards
- Pure-Rust software fallbacks for all SIMD-accelerated ops
- WASM target: importer and IR crates compile without issues; backends are feature-gated
- CI: compile-only check for WASM target from Phase 2 onward

**Phase:** Phase 2 (Windows), Phase 3 (WASM)

---

## R-09: Performance Validation Risk

**Description:** `hologram-ai` must meet acceptable performance thresholds
to be a credible alternative to llama.cpp. Without benchmarking infrastructure,
regressions go undetected.

**Impact:** Medium — performance parity is a long-term goal; MVP is correctness-first.

**Likelihood:** Medium — performance is rarely free in a new compiler pipeline.

**Mitigation:**
- Define tokens/sec benchmark harness in Phase 2
- Establish baseline vs llama.cpp at Phase 2 completion
- Treat performance as a first-class metric from Phase 3 onward
- Track memory footprint as a second metric (critical for edge deployment)
- Do not micro-optimize in MVP; establish the correct architecture first

**Phase:** Phase 2 (baseline), Phase 3 (optimization target)

---

## R-10: hologram API Instability

**Description:** `hologram` is under active development. If
`ExecutionPlan`, `ExecutionBackend`, `MemoryRegion`, or `ArtifactReference`
change their APIs, `hologram-ai` compilation breaks.

**Impact:** High — blocks all progress on `hologram-ai` during hologram API churn.

**Likelihood:** Medium — early-stage project, APIs are not yet stable.

**Mitigation:**
- Pin hologram dependency to a specific revision for MVP development
- Define a minimal surface of hologram types that `hologram-ai` needs
  (document in [architecture.md](architecture.md) section 2)
- Coordinate with hologram team on API stability signals
- Introduce a hologram abstraction shim in `hologram-ai-lower` if needed to
  buffer against API changes

**Phase:** MVP and ongoing

---

## Risk Summary Table

| ID | Risk | Impact | Likelihood | Phase |
|----|------|--------|------------|-------|
| R-01 | Operator coverage gaps | High | High | MVP+ |
| R-02 | Quantization complexity | High | High | MVP+ |
| R-03 | Backend kernel mismatch | Medium | Medium | Phase 2–3 |
| R-04 | Memory planning bugs | High | Medium | MVP+ |
| R-05 | Dynamic shape complexity | High | Medium | Phase 2 |
| R-06 | LLM semantics drift | Medium | Medium | Phase 3+ |
| R-07 | Tokenizer integration | Low | Low | Phase 2 |
| R-08 | Portability (WASM/Win) | Medium | Medium | Phase 2–3 |
| R-09 | Performance validation | Medium | Medium | Phase 2–3 |
| R-10 | hologram API instability | High | Medium | MVP+ |

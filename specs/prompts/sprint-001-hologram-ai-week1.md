# Sprint-001: hologram-ai Week 1 — Foundations

## Objective

Scaffold the `hologram-ai` workspace and implement the core IR and GGUF importer
sufficient to run a single forward pass on a LLaMA-family model.

## Goals

- Canonical `AiGraph` IR is fully defined with all core types
- GGUF parser handles LLaMA architecture with Q4_0 quantization
- Conservative memory planner produces valid `MemoryPlan`
- Lowering table covers the LLaMA op set
- Single forward pass compiles and runs on CPU without panicking
- Output tensor has correct shape

## Inputs

- Architecture docs: `specs/projects/hologram-ai/`
- Bootstrap prompt: `specs/prompts/hologram-ai/02-repo-bootstrap.md`
- ADR-0002 through ADR-0006

## Deliverables

1. `hologram-ai/` workspace with all 14 crate stubs created
2. `hologram-ai-ir` — full type definitions, `AiGraph::validate()`, unit tests
3. `hologram-ai-quant` — `QuantScheme`, `QuantDescriptor`, `dequant_q4_0`, `dequant_q8_0`
4. `hologram-ai-gguf` — binary parser, `LlamaArch` recognizer, `import_gguf()`
5. `hologram-ai-opt` — `OptPipeline`, `ConstantFolding`, `DeadNodeElimination`
6. `hologram-ai-mem` — `MemoryPlanner` conservative mode
7. `hologram-ai-lower` — `lower()` covering LLaMA op subset
8. `hologram-ai-session` — `ModelCompiler::compile()`, `InferenceSession::run()`
9. `CLAUDE.md` at repo root
10. `tests/fixtures/gguf/tiny-llama-q4_0.gguf` (synthetic, committed)
11. Integration test: import → single forward pass → shape check

## Tasks

### Day 1–2: Crate scaffolding + hologram-ai-ir

- [ ] Create workspace Cargo.toml with all members
- [ ] `cargo new --lib crates/hologram-ai-ir` (and all others)
- [ ] Implement `DType`, `Shape`, `Dim`, `QuantScheme`, `QuantDescriptor`
- [ ] Implement `AiOp` enum (complete, even if most variants are stubs)
- [ ] Implement `AiNode`, `AiParam`, `TensorInfo`, `AiGraph`
- [ ] Implement `AiGraph::validate()`
- [ ] Unit tests for graph construction and validation
- [ ] `cargo test -p hologram-ai-ir` passes

### Day 2–3: hologram-ai-quant + hologram-ai-gguf

- [ ] Implement `QuantDescriptor` with all GGUF quant types
- [ ] Implement `dequant_q4_0` matching ggml-quants.h reference
- [ ] Implement `dequant_q8_0`
- [ ] Unit test: `dequant_q4_0` against precomputed reference values
- [ ] Implement GGUF binary parser (header, KV, tensor index)
- [ ] Implement `GgufMetadata` struct
- [ ] Implement `LlamaArch::build_graph()`
- [ ] Implement `import_gguf(path, opts) -> Result<AiGraph>`
- [ ] Test: import a synthetic GGUF fixture, check node count and op types
- [ ] `cargo test -p hologram-ai-gguf` passes

### Day 3: hologram-ai-opt + hologram-ai-mem

- [ ] Implement `Pass` trait
- [ ] Implement `OptPipeline::run()`
- [ ] Implement `ConstantFolding` pass
- [ ] Implement `DeadNodeElimination` pass
- [ ] Implement `MemoryPlanner::plan()` — conservative (no-alias)
- [ ] `MemoryPlan` tracks total_weight_bytes and total_activation_bytes
- [ ] Tests for both passes and the planner
- [ ] `cargo test -p hologram-ai-opt` and `cargo test -p hologram-ai-mem` pass

### Day 4: hologram-ai-lower + hologram-ai-session

- [ ] Implement `lower()` with op dispatch for LLaMA ops
- [ ] Buffer binding: `MemoryPlan::BufferAlloc` → `hologram::MemoryRegion`
- [ ] Implement `ModelCompiler` (orchestrates the full compile pipeline)
- [ ] Implement `InferenceSession::new()` and `run()`
- [ ] `cargo test -p hologram-ai-lower` and `cargo test -p hologram-ai-session` pass

### Day 5: Integration + CI

- [ ] Write integration test: full pipeline from GGUF → run → shape check
- [ ] Generate and commit `tests/fixtures/gguf/tiny-llama-q4_0.gguf`
- [ ] Create `CLAUDE.md`
- [ ] Set up `.github/workflows/ci.yml`
- [ ] `cargo test --workspace` passes on both macOS and Linux
- [ ] `cargo clippy --workspace -- -D warnings` clean

## Exit Criteria

- [ ] `hologram-ai-gguf` imports the synthetic fixture without error
- [ ] Integration test: output tensor shape is `[1, vocab_size]` (or `[seq_len, vocab_size]`)
- [ ] All workspace crate stubs compile
- [ ] CI pipeline passes on ubuntu-latest and macos-latest
- [ ] `CLAUDE.md` documents the architecture and non-negotiables

## Risks

- `hologram` crate API may not match expected interface → use `todo!()` stubs
  in `hologram-ai-lower` and resolve via hologram team sync
- Synthetic GGUF fixture generation requires Python gguf library → document
  as optional, manually verify fixture format

## Notes

The single-pass inference does not need to produce numerically correct outputs
in Week 1 — only the shape and dtype must be correct. Numerical validation
against llama.cpp is a Week 4 milestone.

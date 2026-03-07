# Sprint-004: hologram-ai Week 4 — Validation + First Demos

## Objective

Implement the validation harness and achieve numerical correctness against
reference runtimes. Produce the first public demos: ONNX encoder inference
and GGUF LLM generation.

## Goals

- `hologram-ai-validate` framework is complete and produces structured reports
- ONNX MatMul/Relu/LayerNorm outputs match ONNX Runtime within f32 tolerance
- GGUF TinyLlama greedy top-1 token matches llama.cpp reference
- `hologram-ai-cli validate` command works end-to-end
- All unit tests pass, CI green on both platforms
- MVP is complete per ADR-0006 exit criteria

## Inputs

- Validation prompt: `specs/prompts/hologram-ai/05-validation-harness.md`
- Week 3 deliverables (KV-cache + streaming working)

## Deliverables

1. `hologram-ai-validate` — `ValidationSuite`, `ValidationReport`, `compare_tensors()`
2. `hologram-ai-validate` — ORT subprocess integration
3. `hologram-ai-validate` — llama.cpp subprocess integration
4. `hologram-ai-cli` — `validate` subcommand
5. Reference integration tests (tagged `#[ignore]`)
6. Golden tensor tests for committed fixtures
7. CI: nightly reference test workflow
8. `specs/sprints/sprint-004-hologram-ai-week4.md` marked complete (this file)
9. `hologram-architecture` docs updated: research report, ecosystem page

## Tasks

### Day 1: Validation foundation

- [ ] Implement `Tolerance` with `f32_default()`, `f16_default()`, `quantized()`
- [ ] Implement `compare_tensors()` — max_abs_err, mean_abs_err, cosine_sim
- [ ] Implement `TensorComparison`, `ValidationReport`
- [ ] Unit test `compare_tensors()` for known cases (identical, near-identical, divergent)

### Day 2: ONNX validation

- [ ] Implement `ValidationSuite::run_hologram_onnx()`
- [ ] Implement ORT subprocess invocation (serialize inputs → python → read outputs)
- [ ] Implement `ValidationSuite::validate_onnx()`
- [ ] Integration test (ignore): ONNX MatMul vs ORT → passes with f32 tolerance
- [ ] Test without ORT: `ValidationSuite` skips gracefully when ORT not available

### Day 3: GGUF validation

- [ ] Implement `ValidationSuite::run_hologram_greedy()`
- [ ] Implement llama.cpp subprocess invocation
- [ ] Implement `ValidationSuite::validate_gguf_greedy_token()`
- [ ] Integration test (ignore): TinyLlama 1.1B Q4_0 greedy token 1 matches llama.cpp
- [ ] Document required environment variables: `LLAMACPP_BIN`, `ORT_PYTHON`

### Day 4: CLI validate + golden tests

- [ ] Implement `hologram-ai-cli validate` subcommand
- [ ] Support `--report output.json` for CI integration
- [ ] Implement golden tensor test runner
- [ ] Write golden tests for committed fixtures (shape + dtype only for tiny models)
- [ ] Verify: `hologram-ai validate --onnx tests/fixtures/onnx/matmul-f32.onnx` prints PASS

### Day 5: Demo, CI, documentation

- [ ] Demo script: download TinyLlama 1.1B Q4_0, run `hologram-ai generate`, record output
- [ ] Demo script: run `hologram-ai validate --gguf tinyllama.gguf` and show PASS
- [ ] Add nightly CI workflow for reference tests
- [ ] Update `docs/src/content/docs/ecosystem/hologram-ai.md` in `hologram-architecture`
- [ ] Final `cargo test --workspace` passes
- [ ] Final `cargo clippy --workspace -- -D warnings` passes

## Exit Criteria (MVP Complete per ADR-0006)

- [ ] `hologram-ai-gguf` imports TinyLlama 1.1B Q4_0 without error
- [ ] Single forward pass produces `[1, vocab_size]` logits tensor (correct dtype F32)
- [ ] Top-1 greedy token matches llama.cpp reference on prompt "The capital of France is"
- [ ] ONNX MatMul outputs match ORT within `max_abs_err < 1e-5`
- [ ] All unit tests pass: `cargo test --workspace`
- [ ] CI green on ubuntu-latest and macos-latest
- [ ] `hologram-ai validate` CLI command produces structured JSON report

## Post-Sprint: Phase 2 Kickoff

After Week 4 exit criteria are met, create the Phase 2 sprint backlog:
- Sprint-005: KV-cache validation (multi-turn correctness vs llama.cpp)
- Sprint-006: Larger model support (7B, mmap loading)
- Sprint-007: Metal backend bring-up
- Sprint-008: Benchmark harness

Reference: `specs/projects/hologram-ai/roadmap.md` Phase 2.

# Plan 010: GGUF Generation Quality Fix

## Goal

Fix TinyLlama GGUF autoregressive generation to produce coherent English text.
The model compiles correctly and the prefill logits are correct, but subsequent
tokens degenerate into incoherent spaceless word streams.

## Root Cause

Two compounding issues in the generation loop (`run_cmd.rs`):

1. **Repetition penalty applied to prompt tokens.** The `argmax_with_repetition_penalty`
   function penalized ALL tokens in the window (including prompt tokens). For a 40-token
   prompt, every prompt token was suppressed, biasing toward non-space-prefixed variants.

2. **Greedy argmax with Q4_0 quantization noise.** Pure argmax amplifies small logit
   errors from 4-bit quantization into degenerate loops. Temperature-scaled top-k
   sampling absorbs this noise.

3. **Unnecessary padding to 2048.** With ShapeContextGraph available, the model can
   handle variable seq_len natively. Padding wastes ~50x computation and introduces
   embedding noise from padding token 0.

## Changes

### Generation loop rewrite (`crates/hologram-ai/src/commands/run_cmd.rs`)

- **`SeqMode` enum**: Replaces ad-hoc padding logic with `FixedPad(n)` vs `Variable { max_ctx }`.
  `has_shape_context() → Variable`, otherwise `FixedPad`. Single layer-header extraction.

- **`sample_next_token()`**: Replaces `argmax_with_repetition_penalty`. Supports:
  - Temperature-scaled top-k sampling (default: temp=0.7, top_k=40)
  - Repetition penalty on **generated tokens only** (not prompt tokens)
  - Greedy mode with `--temperature 0`

- **`--temperature`, `--top-k`, `--verbose` CLI args**: Production controls for generation.

- **Per-step diagnostics**: `--verbose` prints logit stats, top-5 tokens at every step.
  Always printed at step 0 for debugging.

- **`extract_logits_at_pos()`**: Clean extraction of logits at target position.

### Compiler (`crates/hologram-ai/src/compiler.rs`)

- Added `HoloRunner::has_shape_context()` method.

## Verification

```bash
# Quick check: compiles
cargo check -p hologram-ai

# E2E: GGUF produces English with spaces
cargo test -p hologram-ai --features e2e -- tinyllama_gguf_runs --nocapture

# Full conformance: no regressions
ORT_STRATEGY=system cargo test -p hologram-ai-conformance --features conformance

# Lint
cargo clippy -p hologram-ai -- -D warnings
```

## Status

- [x] Rewrite `run_cmd.rs` — SeqMode, sampling, penalty fix, diagnostics
- [x] Add `has_shape_context()` to HoloRunner
- [ ] Run TinyLlama GGUF and verify coherent output
- [ ] Add RoPE+GQA chain conformance fixture
- [ ] Update E2E test assertions

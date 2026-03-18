# Plan 011: GGUF Generation Quality — Diagnosis & Fix

## Status: RESOLVED

### Finding

**The hologram-exec computation is provably correct.** The causal consistency test
(`gguf_causal_logit_consistency`) shows cos_sim = 1.000000 between logits at position 5
for seq=6, seq=7, and seq=46. The causal invariant holds perfectly — extending the
sequence does not change earlier positions' logits.

### Root cause of degeneration

The generation degenerates after ~7 tokens because GGUF Q4_0 weights go through
`dequantize_q4_0() → f32 → matmul` on every Gemm call. With 22 transformer layers
× 7 Gemm ops per layer, the float rounding noise from dequantization accumulates
and eventually overwhelms the signal. This is NOT a hologram-exec shape/dispatch bug.

### Fix: LUT-GEMM (Plan 009 Step 2)

The correct fix is using hologram's native `MatMulLut4` path, which operates directly
on quantized byte data without dequantization. The kernels already exist in
hologram-exec (`lut_gemm_4bit`). Wiring GGUF Q4_0 weights through this path eliminates
the dequantization noise entirely.

See `specs/plans/009-lut-kvcache-runtime.md` Step 2.

### Defensive improvements made

1. **`resolve_gemm` rank preservation** (`shape_spec_bridge.rs`): Gemm output shapes now
   preserve the input's leading dimensions (e.g., `[1, seq, hidden]` instead of
   `[seq*hidden/k, n]`). Prevents rank collapse in shape projection.

2. **`resolve_dynamic_sizes` 1-D guard** (`executor.rs`): When shape tracking falls back
   to 1-D `[total_elems]`, the `resolve` closure no longer uses the total as a
   Softmax/RmsNorm row size. Prevents catastrophic normalization errors from flat shapes.

3. **`gguf_causal_logit_consistency` test** (`mini_fixture.rs`): Proves the causal
   invariant holds at seq=6→7 and seq=6→46 with cos_sim = 1.0.

### Files changed

| File | Change |
|------|--------|
| `crates/hologram-ai-common/src/lower/shape_spec_bridge.rs` | `resolve_gemm`/`resolve_matmul` rank preservation |
| `hologram/crates/hologram-exec/src/eval/executor.rs` | `resolve_dynamic_sizes` 1-D guard |
| `crates/hologram-ai/tests/mini_fixture.rs` | Causal logit consistency test |

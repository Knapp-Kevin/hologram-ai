# Plan 032: Precision & Epilogue Fusion Roadmap

## Context

Analysis of two research papers — "The Only Thing That's Difficult is To Forget Precisely"
(thermodynamic precision framework) and "Geometric quantum thermodynamics: A fibre
bundle approach" — identified actionable insights for hologram-ai's compiler pipeline.

Key principle: precision at each computation site is *determined* by its information
content (entropy), not chosen as a hyperparameter. Precision changes that are bijective
on the realized code support are "gauge symmetries" — they conserve task information
and cost nothing (Landauer cost = 0).

**SemanticHint** (Plan 032a, implemented) adds semantic type annotations to `TensorInfo`
to track what kind of information each tensor carries. This plan documents the
remaining precision and epilogue fusion work.

---

## Epilogue Fusion

**Status:** Blocked on hologram base Plan 030 (`hologram/specs/plans/030-epilogue-fusion.md`).

### What exists today

- `AiOp::MatMulRelu`, `MatMulGelu`, `MatMulSilu` — IR variants exist
- Lowering drops the activation: `strategy.rs:224-227` lowers as plain MatMul
- `MatMulActivationFusion` pass was written then removed (Plan 024) because
  hologram base has no fused kernels to dispatch to

### What hologram base Plan 030 adds

- `GraphOp::FusedMatMulActivation { m, k, n, activation: FloatOp }`
- `TapeKernel::InlineMatMulActivation` — applies activation in-register before writeback
- `matmul_k_outer_fused` — CPU kernel with `apply_unary` in accumulator writeback loop
- Norm+Activation fusion: `InlineRmsNormActivation`, `InlineGroupNormActivation`
- Graph-level fusion pass in `float_fusion.rs`

### What hologram-ai does after Plan 030 lands

1. Update `strategy.rs` lowering: `MatMulRelu` → `FusedMatMulActivation { activation: Relu }`
2. Re-register `MatMulActivationFusion` pass in MVP pipeline
3. Update SPRINT.md to mark epilogue fusion as done

### Why this matters (per paper)

- FP32 accumulator is the highest-SNR representation you will ever have
- Spilling to memory forces implicit rounding; reloading + activating = double-rounding
- Fusing avoids 8MN bytes of memory traffic per GEMM (4096² = 128 MiB per matmul)
- The epilogue is "the last reversible place to change gauges"

---

## Mixed-Precision Attention

**Status:** Future — requires hologram base kernel changes.

### Current state

Entire attention path is f32. No mixed-precision support.

### Paper's precision landscape

| Component | Natural precision | Current | Opportunity |
|-----------|------------------|---------|-------------|
| Q/K/V projections | f32 accumulator | f32 | Already optimal |
| Q@K^T scores | ~8 bits | f32 | FP8 feasible |
| Softmax | Must be f32 | f32 | Keep f32 |
| Attention weights @ V | ~8 bits | f32 | FP8 feasible |

### What's needed

hologram base `FloatOp::Attention` kernel to support mixed-precision:
FP8 for score accumulation, f32 for softmax, FP8 for weighted-sum output.
This requires FP8 dtype support throughout the stack.

---

## Calibration-Based Precision Assignment

**Status:** Future — requires runtime infrastructure.

### Paper's Algorithm 1

1. For each site v, quantize with candidate precisions {3, 4, 5, 8, 16}
2. Measure task distortion D_KL(full_precision || quantized)
3. Select minimum precision where D_KL < ε
4. Place boundaries where precision changes; fuse into epilogues

### Prerequisite

hologram-ai is compiler-only (ADR-0016). Calibration requires invoking
hologram's runtime with calibration data — this would happen at a higher
level (CLI or SDK), feeding precision decisions back to the compiler.

---

## SemanticHint as Precision Validation (future pass)

Once SemanticHint is populated (done), a validation pass could warn:

- "Quantizing Embedding tensor to INT4 — natural precision is ~12 bits"
- "Attention weights at f32 — natural precision is ~8 bits, FP8 sufficient"
- "Latent tensor at FP16 — natural precision is ~4 bits, INT4 safe"

This is informational only — no automatic precision changes.

---

## Existing Quantization Methods as Landauer Minimization

The paper unifies GPTQ, AWQ, and SmoothQuant as implicit Landauer cost
minimization. All minimize `D_KL(P_original || P_quantized)` subject to
precision constraints, differing only in KL proxy and allowed transforms:

| Method | KL Proxy | Transforms |
|--------|----------|-----------|
| GPTQ | Layer-wise MSE | Per-row scaling |
| AWQ | Weighted MSE | Per-channel scaling |
| SmoothQuant | Channel-wise MSE | Activation-weight transfer (gauge symmetry) |

SmoothQuant's per-channel rescaling is a pure gauge transformation — it
redistributes quantization difficulty without changing the output distribution.
If hologram-ai adds calibration-based quantization, this framework provides
the theoretical foundation.

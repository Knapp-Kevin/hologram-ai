# Plan 075: Fix Attention Scale for Split-Scale ONNX Pattern

## Context

When Qwen2-0.5B is compiled and run with a multi-token prompt (M>1), the
attention output diverges from ORT. Single-token (M=1) is correct because
softmax of a scalar is always 1.0 regardless of scale.

**Root cause:** PyTorch's `scaled_dot_product_attention` splits the scale
factor `1/√d` across Q and K as separate Mul ops:

```
Q_scaled = Q_proj × (1/√8)     # Mul on Q
K_scaled = K_proj × (1/√8)     # Mul on K
scores = Q_scaled @ K_scaled^T  # Combined: Q @ K^T × (1/8) = Q @ K^T / √64
```

This is mathematically equivalent to `Q @ K^T / √d` but the ONNX graph has
two separate Mul nodes instead of one after the scores MatMul.

`AttentionFusion.find_pre_transpose_with_scale()` traces backward through K
and absorbs the K-side Mul scale (0.3536). It sets `effective_scale = 0.3536`
on the GQA op. But:

1. The Q-side Mul remains in the graph as a separate instruction
2. KvSlotInjection inserts KvWrite AFTER the K-side scale Mul, so the
   K entering the kernel may or may not be prescaled depending on whether
   `find_pre_transpose_with_scale` returned a tensor before or after the Mul
3. The kernel applies `effective_scale` as `alpha` in the BLAS sgemm

The combined effect can be scale³ instead of the correct scale².

## Diagnostic Evidence

With `HOLOGRAM_DUMP_DIR`:
- Embedding: exact match with ORT
- Post-embed RMSNorm: 1.19e-6 diff (match)
- Q projection + bias: 8.58e-6 diff (match)
- **Attention output: 1.80e-1 diff (DIVERGED)**
- Position 0 of attention output: 1.19e-7 diff (match — M=1 equivalent)
- Positions 1-4: diverged (different softmax distributions)

Q is scaled by 0.3536 (instruction 38, Mul). K from KvWrite already contains
the RoPE-applied K (instruction 39). The attention kernel receives both as
inputs and applies `scale = 0.3536` on top.

## Investigation Plan

### Phase 1: Map the exact data flow

For the first transformer layer of Qwen2, trace the complete Q/K/V path from
projection to attention kernel input. Document which tensor IDs flow through
which instructions.

Key questions to answer:
1. Does `find_pre_transpose_with_scale` return a K tensor BEFORE or AFTER the
   scale Mul? (Check the tensor ID it returns vs the Mul output ID)
2. Does `KvSlotInjection` place KvWrite on the pre-scale or post-scale K?
3. Does the KvWrite output (which feeds the Attention kernel) contain scaled
   or unscaled K values?
4. Is the Q-side Mul node part of the fused chain (removed) or separate
   (still executed)?

Tool: Add tracing to `find_pre_transpose_with_scale` and `KvSlotInjection`
that logs tensor IDs and names at each step. Run on Qwen2 and capture.

### Phase 2: Determine the correct fix

Based on Phase 1 findings, one of these fixes is needed:

**Option A: Absorb both Q and K scale Muls into the chain.**
When the SDPA chain detects a Mul(scores_matmul, constant) pattern but
ALSO finds that Q and/or K have prescale Muls, accumulate all scales and
remove all Mul nodes. Set `effective_scale = product_of_all_scales`.

Downside: requires tracing Q backward (currently only K is traced).

**Option B: Don't absorb K-side scale; let both execute as separate ops.**
Change `find_pre_transpose_with_scale` to NOT absorb the Mul. Return the
K tensor after scaling. Set `effective_scale = 1.0` (or just chain.scale).
The Q and K Mul ops remain as separate instructions, and the kernel uses
`scale = 1.0` (no additional scaling).

Downside: loses a minor optimization (two Mul ops instead of fused scale).
But correctness > performance.

**Option C: Detect split-scale pattern explicitly.**
In `match_sdpa_chain`, before looking for a post-MatMul scale Mul, check
if the Q input has a prescale Mul. If so, AND if K also has one (detected
by `find_pre_transpose_with_scale`), compute `effective_scale = q_scale * k_scale`
and absorb both Mul nodes into the chain.

This is the cleanest fix but most complex.

**Recommended: Option B** for immediate correctness, then Option C as an
optimization in a follow-up.

### Phase 3: Implement the fix

**Option B implementation:**

File: `crates/hologram-ai-common/src/opt/attention_fusion.rs`

In `find_pre_transpose_with_scale`, when a Mul(scalar) is encountered:
- Still trace through it (continue backward to find the Transpose)
- But DON'T accumulate the scale
- Return `(pre_transpose_tid, None)` — no scale absorbed

This means:
- K input to GQA = the original K (after scale Mul, before Transpose)
- `effective_scale = None * chain.scale = chain.scale`
- For Qwen2: chain.scale = 1.0 (no post-MatMul Mul found), so effective_scale = 1.0
- Kernel applies scale = 1/√(head_dim) = 1/√64 = 0.125

But this would give `(Q × 0.3536) @ (K × 0.3536)^T × 0.125 = Q @ K^T × 0.0156`,
which is also wrong (scale^2 × 1/√d instead of just scale^2).

So Option B alone doesn't work. The real fix is:

**Option D: Detect that Q has a prescale Mul and account for it.**

In `match_sdpa_chain` or in the fusing logic:
1. Trace Q backward looking for a Mul(scalar) before the Q@K^T MatMul
2. If found: `q_prescale = scalar_value`, absorb the Mul into the chain
3. Trace K backward: `k_prescale = scalar_value` (existing logic)
4. `effective_scale = q_prescale × k_prescale × chain.scale`
5. Both Q and K Mul nodes are in the chain (marked for removal)
6. Kernel receives unscaled Q and unscaled K, applies effective_scale

For Qwen2: `0.3536 × 0.3536 × 1.0 = 0.125 = 1/√64`. Correct.

### Phase 4: Add conformance test

Create a test in `hologram-ai-conformance` or `qwen2_e2e.rs`:

```rust
#[test]
fn qwen2_m5_prefill_matches_ort() {
    // Compile Qwen2-0.5B, run with 5-token prompt
    // Compare last-position logits top-5 against ORT reference
    // ORT top-5: [12095, 32671, 2130, 65892, 1304]
    // Tolerance: top-1 must match, top-5 must overlap by ≥ 3
}
```

### Phase 5: Verify no regression

- Run TinyLlama e2e (LLaMA doesn't use split-scale, so this should be unaffected)
- Run BERT e2e (no attention fusion for encoder-only)
- Run Qwen2 M=1 and M=5

## Key Files

| File | What to change |
|------|---------------|
| `hologram-ai-common/src/opt/attention_fusion.rs` | Trace Q backward for prescale Mul; accumulate into effective_scale |
| `hologram-ai-common/src/opt/attention_fusion.rs:match_sdpa_chain()` | Capture Q-side prescale as part of the chain |
| `hologram-ai-common/src/opt/attention_fusion.rs:find_pre_transpose_with_scale()` | Already handles K-side; may need similar for Q |
| `hologram-ai/tests/qwen2_e2e.rs` | Add M=5 logit comparison test |

## Dependencies

- None on hologram base — the fix is entirely in hologram-ai's attention fusion pass
- The attention kernel's `scale` parameter is correct; the issue is what value the compiler puts there

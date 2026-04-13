# Plan 063 — ViT Patch Pruning (PixelPrune)

**Status:** Phase 1-3 complete  
**Created:** 2026-04-10  
**References:** [PixelPrune paper (arXiv 2604.00886)](https://arxiv.org/abs/2604.00886), [OPPO-Mente-Lab/PixelPrune](https://github.com/OPPO-Mente-Lab/PixelPrune)

## Problem

Vision Transformers process every patch of an input image regardless of content.
For document, GUI, and screenshot images, 30–80% of patches are pixel-identical
to their spatial neighbors (white margins, solid backgrounds, toolbars).  ViT
self-attention is O(N²) in patch count, so redundant patches dominate both
compute and memory:

- A 4096×4096 image at patch_size=16 produces **65,536 tokens**.
  Attention alone: 65536² × head_dim ≈ 17 TFLOP per layer.
- Pruning to 50% → 32,768 tokens → **4× less attention compute, 2× less KV memory**.

PixelPrune demonstrates that a trivial O(N) pixel-space comparison — before the
ViT encoder ever runs — achieves these savings with ≤0.2 points accuracy loss
and zero retraining.

## Recommendation

**Budget-capped static compilation with runtime pre-processing.**

The compiler emits a ViT graph compiled for a fixed `max_kept_patches` budget.
A separate `PatchPrune` runtime kernel runs *before* the compiled graph as a
pre-processing step, selecting the most informative patches up to the budget.
Position IDs are a runtime input (reusing `PositionIdsInjection`), so pruned
patches retain their correct spatial coordinates.

This avoids the variable-length execution dependency entirely.  Shapes through
the ViT are static, all existing fusion passes apply unchanged, and the memory
ceiling is known at compile time.

### Why not the alternatives

| Approach | Problem |
|---|---|
| **Variable-length (Option B)** | Blocked by existing variable-length execution bug. Requires every downstream op to handle data-dependent shapes. High complexity for marginal gain over budget-cap. |
| **Mask-based (Option A)** | Allocates full KV cache and computes full attention — just zeros out scores for pruned tokens. Saves nothing meaningful. |
| **Multi-graph (Option C)** | 2× archive size. Runtime graph-selection adds latency. Doesn't adapt to actual content. |
| **Budget-cap (recommended)** | Static shapes, bounded memory, graceful degradation. Standard approach in TensorRT, CoreML, ONNX Runtime. |

### How budget-cap works

1. User specifies `patch_budget_ratio: f32` in `CompileOptions` (default: 0.5).
2. Compiler computes `max_kept = ceil(grid_h × grid_w × ratio)`.
3. ViT graph is compiled with `seq_len = max_kept` (static).
4. At runtime, PatchPrune kernel selects up to `max_kept` patches:
   - If `N_kept < max_kept`: right-pad with zeros + set attention mask.
   - If `N_kept > max_kept`: keep the top-K by prediction error (largest
     `max_dist(patch, predicted)` = most informative). Graceful degradation.
   - If `N_kept == max_kept`: perfect fit, zero waste.
5. Position IDs input carries the (row, col) → flat_position mapping for
   each kept patch. 2D RoPE / absolute position embeddings stay correct.

For typical document/GUI images at ratio=0.5, the "pad" case dominates and
wastes ~5–15% of budget. The "overflow" case is rare (only photographic images
with near-zero redundancy).

## Architecture

### Compile-time (hologram-ai)

```
                     CompileOptions { patch_budget_ratio: 0.5 }
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│  PatchPruneInjection pass (runs early, before fusion)       │
│                                                             │
│  1. Detect patch-embed Conv2d (graph input → Conv2d with    │
│     kernel matching patch_size, stride == kernel_size)       │
│                                                             │
│  2. Compute grid dims:                                      │
│     grid_h = H / patch_size,  grid_w = W / patch_size       │
│     max_kept = ceil(grid_h × grid_w × ratio)                │
│                                                             │
│  3. Rewrite graph:                                          │
│     a) Add new graph input: `kept_indices` [max_kept, 2]    │
│     b) Add new graph input: `attention_mask` [1, max_kept]  │
│     c) Insert Gather after patch_embed Reshape to select     │
│        only `kept_indices` rows from [N_patches, embed_dim] │
│     d) Replace fixed position embedding Add with             │
│        GatherElements from pos_embed table using             │
│        kept_indices                                          │
│     e) All downstream shapes: seq_len = max_kept (static)   │
│                                                             │
│  4. Inject `patch_budget` into ModelMetaSection so the       │
│     runtime knows the budget without re-deriving it.         │
│                                                             │
│  5. Re-run ShapePropagation to update all downstream shapes. │
└─────────────────────────────────────────────────────────────┘
```

Key insight: **no new AiOp variants needed for the compiled graph**.  The
pruning rewrite uses only existing ops (Gather, GatherElements) with a new
graph input (`kept_indices`).  The actual pruning logic lives entirely in the
runtime kernel that *produces* `kept_indices` before the compiled graph runs.

### Runtime (hologram base)

```
┌──────────────────────────────────────────────────────────────┐
│  PatchPrune pre-processing kernel                            │
│  (runs BEFORE the compiled ViT graph, not inside it)         │
│                                                              │
│  Input:  raw_pixels [1, 3, H, W]                             │
│  Output: kept_indices [max_kept, 2]                          │
│          attention_mask [1, max_kept]                         │
│                                                              │
│  Algorithm (Pred-2D, ~30 lines):                             │
│  1. Divide image into (grid_h × grid_w) blocks of           │
│     (patch_size × patch_size × 3) pixels each.               │
│  2. Raster-scan. For each block (r, c):                      │
│     - Predict from causal neighbor via median-edge rule:     │
│       if dist(diag, upper) < dist(diag, left): use left     │
│       else: use upper                                        │
│     - If max_pixel_diff(block, predicted) > tau: keep it     │
│  3. If |kept| > max_kept: sort by prediction error desc,     │
│     take top-K.                                              │
│  4. If |kept| < max_kept: pad indices with (0,0),            │
│     set mask[padded] = 0.                                    │
│  5. Convert (row, col) pairs to flat position IDs for        │
│     the model's position encoding scheme.                    │
│                                                              │
│  Complexity: O(grid_h × grid_w × patch_dim) — negligible     │
│  vs ViT attention. Vectorizable on GPU (pixel equality).     │
└──────────────────────────────────────────────────────────────┘
```

### Attention mask integration

The compiled ViT graph's `MultiHeadAttention` / `GroupedQueryAttention` ops
already support an optional attention mask input.  Padded positions get
`mask = 0` (or `-inf` in logit space), so they contribute zero to the output.
The mask costs `O(max_kept)` memory — negligible.

For FlashAttention-style kernels, the mask enables **variable-length packing**:
the kernel can skip padded positions entirely (no wasted compute).  This is
already standard in FlashAttention-2's `cu_seqlens` interface.

## Phases

### Phase 1 — PatchPruneInjection compiler pass

**Repo:** hologram-ai  
**Scope:** New opt pass + CompileOptions field + tests

1. Add `patch_budget_ratio: Option<f32>` to `CompileOptions`.
2. Create `crates/hologram-ai-common/src/opt/patch_prune.rs`:
   - Pattern-match: find graph input → Conv2d where `kernel_shape == strides`
     and input has 4D image shape `[N, C, H, W]` with `C ∈ {1, 3, 4}`.
   - Verify downstream: Conv2d → Reshape → Add(·, const_pos_embed).
   - Compute `grid_h`, `grid_w`, `max_kept`.
   - Add `kept_indices` and `attention_mask` as new graph inputs.
   - Insert Gather to index into the patch sequence.
   - Rewrite position embedding from static Add to GatherElements.
   - Store `patch_budget` in metadata.
3. Register in `OptPipeline::vit()` (new pipeline for vision models).
4. Wire into `ModelCompiler` — auto-detect ViT topology and use `vit()`
   pipeline when `patch_budget_ratio.is_some()`.

**Tests:**
- Unit: construct a minimal ViT-shaped AiGraph (Conv2d → Reshape → Add →
  MultiHeadAttention × 2 → output), run the pass, assert graph structure.
- Assert `kept_indices` and `attention_mask` are new graph inputs.
- Assert all downstream seq dims == `max_kept`.
- Assert position embedding is gathered, not broadcast-added.

### Phase 2 — PatchPrune runtime kernel

**Repo:** hologram (base)  
**Scope:** New `FloatOp::PatchPrune` + kernel implementation

1. Add `FloatOp::PatchPrune { tau: f32, grid_h: u32, grid_w: u32, patch_dim: u32, max_kept: u32 }`
   (appended to end of enum).
2. Implement kernel: Pred-2D scan + budget-cap + padding.
3. Platform priority: native (macOS Accelerate vDSP for vectorized pixel diff)
   → WASM (scalar fallback) → x86_64.

**Tests:**
- Lossless (tau=0): all-white image → keeps only anchor → mask mostly zero.
- Lossless (tau=0): checkerboard → keeps all patches → full mask.
- Budget overflow: natural image → verify top-K selection by error magnitude.
- Round-trip: verify that kept patches + indices can reconstruct the original
  patch ordering (lossless guarantee).

### Phase 3 — Pipeline integration

**Repo:** hologram-ai + hologram  
**Scope:** End-to-end ViT compilation with pruning

1. Lower `PatchPrune` metadata from AiGraph into the archive's entrypoint
   so the runtime knows to run the pre-processing kernel.
2. Add `PatchPrunePreprocessor` to the execution pipeline — runs before the
   compiled graph, feeds `kept_indices` and `attention_mask` as inputs.
3. Wire `CompileOptions::patch_budget_ratio` through CLI:
   `hologram-ai compile --patch-budget 0.5 model.onnx`.

**Tests:**
- End-to-end: compile a small ViT (e.g., ViT-Tiny from ONNX model zoo),
  run with a synthetic image, verify output shape and reasonable values.
- Compare output (budget=1.0, no pruning) vs ONNX Runtime reference —
  should be identical (no pruning applied when budget=1.0).
- Compare output (budget=0.5, white-margin document image) vs reference —
  should be within 0.01 cosine similarity.

### Phase 4 (future) — Adaptive budget + ViT-specific fusions

- **Adaptive budget**: instead of fixed ratio, use a two-pass approach:
  1. Quick scan pass (O(N)) to count how many patches survive at tau.
  2. If count fits within a pre-compiled graph, use it directly.
  3. If not, fall back to the budget-capped graph.
  This requires compiling at 2–3 budget levels (25%, 50%, 75%) — small archive
  overhead for large compute savings on high-redundancy inputs.

- **ViT attention fusion**: fuse LayerNorm → QKV projection → masked
  MultiHeadAttention → residual Add into a single fused kernel, similar to
  the LLM deep-decode fusions (Plan 054). The mask integration makes this
  particularly effective since the kernel can skip masked positions in the
  inner loop.

## Dependencies

| Dependency | Status | Blocks |
|---|---|---|
| `FloatOp` in hologram base | Exists | Phase 2 kernel |
| Attention mask support in compiled graphs | Partial (FlashAttentionHint exists) | Phase 1 — need mask as graph input |
| `PositionIdsInjection` pass | Exists | Phase 1 reuses the pattern |
| `ModelMetaSection` | Exists | Phase 1 metadata storage |
| Variable-length execution | NOT required | Budget-cap avoids this entirely |

## Memory impact estimate

For a ViT-L/14 at 336×336 (CLIP in LLaVA):
- Patches: 576 (24×24 grid)
- At budget=0.5: `max_kept = 288`
- Attention per layer: 576² → 288² = **4× reduction**
- KV cache per layer: 576 × 2 × head_dim → 288 × 2 × head_dim = **2× reduction**
- 24 layers total: ~50% memory reduction for the ViT encoder

For a ViT processing 4096×4096 document images (Qwen3-VL style):
- Patches: 65,536
- At budget=0.5: `max_kept = 32,768`
- Attention: 65536² → 32768² = **4× reduction** (17 TFLOP → 4.3 TFLOP per layer)
- This is the difference between "runs" and "OOM" on edge devices.

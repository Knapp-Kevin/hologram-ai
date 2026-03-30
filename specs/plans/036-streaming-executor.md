# Plan 036: Streaming Executor for Resource-Constrained Execution

## Status: Active

## Context

The SD VAE decoder (198MB weights, 448 nodes) uses 51GB RSS at 512×512 resolution.
The weights are only 198MB (zero-copy via mmap) — the 51GB is entirely from
**activation buffers** accumulating in the arena. Each Conv2d at 512×512 with
512 channels produces a 512MB output tensor, and the VAE's residual connections
keep 6-8 such activations alive simultaneously.

This is unacceptable for WASM, Raspberry Pi, or any resource-constrained target.
The weight index infrastructure (layer groups, mmap prefetch/release) exists but
isn't driving execution. The executor loads all constants upfront and keeps all
activations alive until level-boundary eviction.

## Root Cause Analysis

**Current architecture (all in hologram base):**
1. `seed_arena` (mmap/mod.rs:198) — seeds ALL constants at startup
2. Each Conv2d allocates `n × oc × spatial_out` output buffer (~512MB at full res)
3. Arena keeps all activations until `consumer_count` reaches 0
4. Eviction runs at level boundaries — skip connections delay freeing by many levels
5. No layer-level weight loading/unloading

**Memory breakdown for VAE at 512×512:**
- Borrowed constants (zero-copy): 0 MB
- Single Conv2d output: 512 MB
- 6-8 simultaneous activations (skip connections): 3-4 GB
- Arena never releases between layers: accumulates to 51 GB

## Design: Layer-Streaming Executor

### Principle
Process the model **one layer group at a time**. Load only that group's weights,
compute its outputs, free its inputs, then move to the next group. Peak memory
bounded by: `max(layer_weights) + max(layer_activations)`, NOT `sum(all_activations)`.

### Phase 1: Lazy Constant Seeding (hologram base)

**Current:** `seed_arena` iterates ALL `GraphOp::Constant` nodes and inserts
borrowed references at startup.

**Change:** Skip constant seeding entirely. Instead, resolve constants on-demand
during instruction dispatch. When an instruction reads an input that's a constant
node, the executor fetches the weight slice from the mmap at that moment.

**Files:**
- `crates/hologram-exec/src/mmap/mod.rs` — modify `seed_arena` to only seed
  graph inputs (not constants). Add a fallback in `execute_inner`'s input gathering
  that resolves Deferred constants from the weight blob on first access.
- `crates/hologram-exec/src/buffer/arena.rs` — add `get_or_seed_constant()`
  that checks if a node is populated, and if not, checks the constant store.

**Memory impact:** Constants are only paged in when first read. Combined with
`release_range`, weights for completed layers can be released back to the OS.

### Phase 2: Per-Instruction Eviction (hologram base)

**Current:** Eviction runs at level boundaries (end of each level in the
`execute_inner` loop). A node consumed in level 5 but also needed in level 50
stays alive for 45 levels.

**Change:** Decrement consumer counts and evict **per-instruction**, not per-level.
After each instruction executes, immediately check its inputs' consumer counts
and evict any that reach 0.

**Files:**
- `crates/hologram-exec/src/tape.rs` — move the eviction logic from after the
  level loop to after each instruction's swap_insert.
  The consumer count decrement happens right after the instruction reads its
  inputs (not after the whole level).

**Memory impact:** Activations are freed as soon as their last consumer runs,
not at the next level boundary. For linear chains (no skip connections), only
1-2 activations are live at any time.

### Phase 3: Mmap Weight Prefetch/Release per Layer Group (hologram base)

**Current:** `level_weight_ranges` computes per-level byte ranges and prefetches
the next level's weights via `prefetch_read` (cache-line prefetch). No
`release_range` is called.

**Change:** Use the `WeightIndex` layer groups to issue `prefetch_range` for the
NEXT group's weights and `release_range` for the PREVIOUS group's weights at
layer-group boundaries.

**Implementation:**
1. At tape build time, compute a `layer_group_boundaries: Vec<usize>` mapping
   level indices to layer group transitions (using WeightIndex group names).
2. In `execute_inner`, at each level boundary, check if we've crossed a layer
   group boundary. If so:
   - Call `release_range` for the previous group's weight byte range
   - Call `prefetch_range` for the next group's weight byte range

**Files:**
- `crates/hologram-exec/src/tape.rs` — add `layer_group_boundaries` to `EnumTape`,
  compute during `compute_level_weight_ranges`
- `crates/hologram-exec/src/mmap/mod.rs` — pass `HoloLoader` (or a prefetch
  callback) to `execute_with_eviction` for mmap advice calls

**Memory impact:** OS can reclaim weight pages for completed layers. On Linux,
`MADV_DONTNEED` immediately frees physical pages. On macOS, `MADV_FREE_REUSABLE`
marks pages as reclaimable under memory pressure.

### Phase 4: Conv2d Output Streaming (hologram base)

**Current:** `conv2d_core` allocates `vec![0.0f32; n * oc * spatial_out]` —
the full output tensor. At 512×512 with 512 channels: 512MB.

**Change:** If the next consumer is an elementwise op (Add, ReLU, etc.), the
Conv2d can write directly into the consumer's output buffer or process
row-by-row without materializing the full tensor. This is a larger change
that requires the executor to "fuse" consecutive ops.

**Simpler alternative:** Accept the per-layer output size but ensure eviction
frees it immediately after the single consumer runs (Phase 2 handles this).
For skip connections, this doesn't help — but it bounds non-skip activations.

**Files:**
- `crates/hologram-exec/src/float_dispatch/conv.rs` — no change needed if
  Phase 2 eviction is aggressive enough
- Future: graph-level operator fusion to eliminate intermediate tensors

## Execution Order

1. **Phase 2 first** (per-instruction eviction) — biggest impact, simplest change
2. **Phase 1** (lazy constants) — removes unnecessary constant seeding overhead
3. **Phase 3** (mmap prefetch/release) — reduces physical memory via OS hints
4. **Phase 4** (conv output streaming) — only if Phases 1-3 don't suffice

## Expected Memory After Phases 1-2

For the VAE decoder at 512×512:
- Weights: ~198MB (paged in on demand from mmap, released per layer)
- Peak activation: ~1-1.5GB (2-3 simultaneous Conv2d outputs for residual blocks)
- Total: ~1.5-2GB (down from 51GB)

For WASM/RPi (lower resolution, e.g., 256×256):
- Peak activation: ~200-400MB
- Total: ~400-600MB

## Files to Modify (hologram base)

| File | Phase | Change |
|------|-------|--------|
| `crates/hologram-exec/src/mmap/mod.rs` | 1,3 | Lazy seed, prefetch callback |
| `crates/hologram-exec/src/tape.rs` | 2,3 | Per-instruction eviction, group boundaries |
| `crates/hologram-exec/src/buffer/arena.rs` | 1 | `get_or_seed_constant()` fallback |
| `crates/hologram-exec/src/tape_builder.rs` | 3 | Layer group boundary computation |

## Verification

1. VAE decoder at 512×512: RSS < 2GB (down from 51GB)
2. VAE decoder produces correct [1,3,512,512] output with all finite values
3. Existing tests pass (mini_fixture, streaming_conformance, etc.)
4. UNet at full resolution: RSS < 4GB

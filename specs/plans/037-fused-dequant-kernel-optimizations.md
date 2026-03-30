# Plan 037: Fused Dequant-MatMul & Kernel Optimizations

**Status:** Complete
**Created:** 2026-03-30
**Scope:** hologram-exec (hologram base), hologram-archive
**Inspiration:** [justine.lol/matmul](https://justine.lol/matmul/) — llamafile matmul optimizations

## Motivation

Q4_0 Psumbook kernel is **30x slower than f32 BLAS** (SPRINT.md P7). The root cause: the
quantized path in `dispatch_gemm` fully dequantizes the weight matrix to f32 via `decode_weights`
before running `matmul_k_outer`, doubling memory bandwidth.

Key insight (llamafile): fuse dequantization with matrix multiplication so weights are
dequantized per-block in registers, never materializing the full f32 matrix.

## What Already Exists (verified)

- **CPU matmul** (`float_dispatch/matmul.rs`): Goto/BLIS-style with KC=256 L2 cache blocking,
  B-panel packing, const-generic `micro_kernel<MR=4, NR=8>`, rayon parallel M-tile distribution,
  Accelerate BLAS on macOS. Well-optimized for f32.
- **SIMD precedent**: `hologram-core/src/view/simd.rs` has AVX2/SSE4.2/NEON intrinsics using
  compile-time `#[cfg(target_arch, target_feature)]`.
- **LUT-GEMM**: Separate algorithm (psumbook accumulation). Fiber kernel (16-pass radix),
  tiled kernel (4-column), scalar fallback.
- **HoloArchive**: Page-aligns sections (4KB) but not individual tensors.

## Phase 1: Fused Q4_0 Dequant-MatMul — DONE

**Goal:** Eliminate the full-matrix dequantization that causes 30x slowness.

**File:** `hologram-exec/src/float_dispatch/matmul.rs`

- [x] `matmul_dequant_q4_0()` — KC-blocked, MR×NR tiled, dequant-into-pack B panels
- [x] `dequant_pack_q4_0_panel()` — dequant 18-byte Q4_0 blocks → packed f32 panel
- [x] `dispatch_gemm()` fast path for `quant_b == 1` → calls fused kernel
- [x] Rayon parallel M-tile path (same threshold as f32)
- [x] 3 bit-exact conformance tests (basic, m=1, large prefill)

## Phase 1b: Fused Q6_K Dequant-MatMul — DONE

**Goal:** Extend fused dequant to Q6_K (210-byte super-blocks → 256 values).

**File:** `hologram-exec/src/float_dispatch/matmul.rs`

- [x] `matmul_dequant_q6_k()` — same tiling structure, Q6_K on-the-fly dequant
- [x] `dequant_pack_q6_k_panel()` — 6-bit signed integer dequant into packed f32
- [x] `dequant_q6_k_value()` — per-value dequant from 210-byte super-block
- [x] `dispatch_gemm()` fast path for `quant_b == 3` → calls fused kernel
- [x] Rayon parallel M-tile path
- [x] 3 bit-exact conformance tests (basic m=5, m=1, large prefill m=32)
- [x] `cast::dequantize_q6_k` made `pub(super)` for test access

## Phase 2: Adaptive Micro-Kernel Selection for Remainders

**Goal:** Eliminate scalar remainder paths in `matmul_k_outer`.

**File:** `hologram-exec/src/float_dispatch/matmul.rs`

**Current weakness:**
- Remainder columns (n%8): MR×4 tile added for first 4 columns (line 750), but last 0-3
  columns are still scalar
- Remainder rows (m%4): scalar k-outer per row, no MR blocking

**Approach:** `m_remainder_tiled()` helper uses `micro_kernel_packed` at smaller MR:
- [x] `micro_kernel_packed::<2, NR>` for m_rem >= 2 (process pairs)
- [x] `micro_kernel_packed::<1, NR>` for m_rem == 1 (NR-wide vectorization)
- [x] Scalar fallback for corner N-remainder with small MR
- [x] Wired into both parallel and sequential paths of `matmul_k_outer`

## Phase 3: Explicit SIMD Micro-Kernels — DONE

**Goal:** Replace autovectorized `micro_kernel` with hand-tuned SIMD intrinsics.
Applies to ALL platforms (including macOS for the fused dequant paths which bypass BLAS).

**File:** `hologram-exec/src/float_dispatch/matmul.rs`

- [x] **NEON (aarch64):** `micro_kernel_packed_neon` + `micro_kernel_strided_neon`
  — 8 `float32x4` accumulators (2 per row × 4 rows), `vfmaq_f32` fused multiply-add
- [x] **AVX2+FMA (x86_64):** `micro_kernel_packed_avx2` + `micro_kernel_strided_avx2`
  — 4 `__m256` accumulators (one per row), `_mm256_fmadd_ps`
- [x] Wired into `micro_kernel` and `micro_kernel_packed` via compile-time dispatch
  (`#[cfg(target_arch)]` for NEON, runtime `is_x86_feature_detected!` for AVX2)
- [x] Benefits ALL matmul paths: f32 `matmul_k_outer`, fused Q4_0, fused Q6_K
- [x] 384 tests pass with NEON active on aarch64, bit-exact with reference

## Phase 4: SIMD Psumbook Dot Product — DONE (pre-existing)

**File:** `hologram-exec/src/lut_gemm/psumbook.rs`

Already implemented before this plan:
- [x] `dot_neon_256()` — 4×`vfmaq_f32` unrolled, 16 f32s per iteration
- [x] `dot_avx2_256()` — 4×`_mm256_fmadd_ps` unrolled, 32 f32s per iteration
- [x] Wired into `Psumbook8::dot()` via compile-time `#[cfg]`
- [x] 13 tests in psumbook.rs

## Phase 5: Page-Aligned Tensors in HoloArchive — DONE

**Goal:** Enable zero-copy GPU weight loading.

**Files:** `hologram-archive/src/format/header.rs`, `writer/holo_writer.rs`, `weight/index.rs`

Infrastructure was already in place; added missing reader method and tests:
- [x] `FLAG_TENSOR_PAGE_ALIGNED = 1 << 3` flag constant (pre-existing)
- [x] `HoloWriter::tensor_page_aligned(bool)` builder method (pre-existing)
- [x] `page_align_weight_blob(blob, index)` helper (pre-existing)
- [x] `HoloHeader::is_tensor_page_aligned()` reader convenience method (NEW)
- [x] `tensor_page_aligned_flag` test — flag roundtrip (NEW)
- [x] `page_align_offsets_are_4096_aligned` test — verifies alignment + data integrity (NEW)
- [x] `page_align_empty_index` test — edge case (NEW)

**Follow-up:** Metal backend switches from `new_buffer_with_data()` (copies) to
`newBuffer(bytesNoCopy:)` for page-aligned tensors.

## Phase 6: Per-Op Profiling Instrumentation — DONE

**File:** `hologram-exec/src/float_dispatch/mod.rs`

- [x] `#[cfg(feature = "profile")]` tracing span on `dispatch_float_into` with op category
- [x] Feature gate `profile` already exists in `hologram-exec/Cargo.toml`

## Phase 7: Threading Refinement — DONE

**File:** `hologram-exec/src/float_dispatch/matmul.rs`

- [x] Static duty partitioning via `with_min_len(duty)` where
  `duty = (m_tiles + n_threads - 1) / n_threads`
- [x] Applied to all 3 parallel paths: f32 `matmul_k_outer`, fused Q4_0, fused Q6_K

## Execution Order (all phases complete)

```
Phase 1  (fused Q4_0 dequant)    ✅
Phase 1b (fused Q6_K dequant)    ✅
Phase 2  (adaptive remainders)   ✅
Phase 3  (SIMD micro-kernels)    ✅
Phase 4  (SIMD psumbook)         ✅ (pre-existing)
Phase 5  (page-aligned tensors)  ✅
Phase 6  (profiling)             ✅
Phase 7  (threading)             ✅
```

## Key Files

| File | Phase | Change |
|------|-------|--------|
| `hologram-exec/src/float_dispatch/matmul.rs` | 1, 1b, 2, 3, 7 | Fused dequant + remainder + SIMD + threading |
| `hologram-exec/src/float_dispatch/cast.rs` | 1, 1b | Q4_0/Q6_K block format reference, `pub(super)` visibility |
| `hologram-exec/src/float_dispatch/mod.rs` | 6 | Profile feature-gated tracing span |
| `hologram-exec/src/lut_gemm/psumbook.rs` | 4 | SIMD dot product methods |
| `hologram-archive/src/format/header.rs` | 5 | New flag constant |
| `hologram-archive/src/writer/holo_writer.rs` | 5 | Page-align tensor writing |
| `hologram-archive/src/weight/index.rs` | 5 | Aligned offset builder |

## Verification

1. Q4_0 fused kernel: 3 bit-exact tests vs `decode_weights → matmul_k_outer` ✅
2. Q6_K fused kernel: 3 bit-exact tests vs `dequantize_q6_k → matmul_k_outer` ✅
3. SIMD micro-kernels: 384 hologram-exec tests pass with NEON active ✅
4. Page-aligned tensors: 3 new tests (flag, alignment, empty) ✅
5. `cargo test -p hologram-exec -p hologram-archive` — 534 tests, 0 failures ✅
6. `cargo clippy -p hologram-exec -p hologram-archive -- -D warnings` ✅
7. `cargo clippy -- -D warnings` (hologram-ai) ✅
8. Benchmark: TinyLlama Q4_0 inference speed (target: 30x → <5x vs f32 BLAS) — pending
9. Benchmark: f32 decode tok/s regression check (currently 39.1 tok/s) — pending

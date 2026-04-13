# Plan 064: Streaming Compilation for 2 GB RSS Limit

**Status:** Ready for implementation
**Created:** 2026-04-13
**Scope:** hologram-ai-common (lowering), hologram-ai (compiler), hologram base (archive writer)
**Branch:** `feat/streaming-compilation`

## Context

SDXL UNet (10 GB weights, 30K+ nodes) compilation peaks at 50 GB RSS, far exceeding the 2 GB limit. The root causes are:

1. **`early_quant_bytes` HashMap**: Accumulates ALL Q4 quantized weights (~2.5 GB) — entries are never removed after consumption
2. **Graph constant store**: Q4 bytes are `.clone()`d into `ConstantData::Bytes` AND kept in the HashMap — double memory
3. **`collect_weight_bytes`**: Allocates a single `Vec<u8>` for all non-Q4 weights (~500 MB-1 GB)
4. **Archive assembly**: `vec![0u8; total_size]` materializes the entire .holo in memory

Import (~200 MB metadata) and optimization passes (~200 MB, metadata-only) are already within budget. Only lowering and archive writing need changes.

## Phase 1: Fix the Q4 Accumulation Leak

**File:** `crates/hologram-ai-common/src/lower/builder.rs`

Replace `early_quant_bytes.get(&wt)` with `early_quant_bytes.remove(&wt)` at all 4 call sites. Gives owned data (eliminating `.clone()`) and frees the entry immediately.

**Impact:** HashMap drops from ~2.5 GB (all weights) to ~30 MB (one weight at a time). Peak: ~3 GB.

## Phase 2: Spill Q4 Constants to Temp File

**Goal:** Q4 bytes never live in the Graph's constant store in memory.

**`builder.rs`**: Add `q4_spill: Option<tempfile::NamedTempFile>` to lowering context. Write Q4 bytes to the spill file immediately after quantization. Use `ConstantData::Deferred` instead of `ConstantData::Bytes`. Remove `early_quant_bytes` HashMap entirely.

**`compiler.rs`**: Wire the spill file through `compile_one_component`. Resolve `Deferred` constants from the spill file during archive assembly.

**Peak RSS after Phase 2:** ~480 MB.

## Phase 3: Streaming Archive Write

**`hologram-archive/src/writer/holo_writer.rs`**: Add `build_to_file(path)` alongside `build() -> Vec<u8>`. Writes header + graph + sections via buffered file I/O. Streams weight section from source files + spill file directly to output.

**`compiler.rs`**: Use file-based path for models with total weight bytes > 256 MB.

## Phase 4: OS Page Cache Advise

**`builder.rs` — `param_bytes_owned()`**: After reading a weight, call `posix_fadvise(FADV_DONTNEED)` (Linux) or `fcntl(F_NOCACHE)` (macOS) to prevent OS page cache inflation.

## Runtime Performance

- Phase 1: **Faster** — eliminates 6 GB of `.clone()` allocations
- Phase 2: **~5% slower** for large models (temp file I/O), negligible for small models
- Phase 3: **Neutral to faster** — streaming write avoids Vec reallocation pressure
- Phase 4: **Zero overhead** — single syscall per weight read

## Verification

```bash
# Phase 1: peak RSS < 5 GB for SDXL UNet
# Phase 2: peak RSS < 1 GB
# Phase 3: peak RSS < 800 MB
# Regression: TinyLlama ≥ 40 tok/s
# Functional: SDXL .holo loads and executes
```

# Plan 035: Runtime Cleanup — hologram-ai integration with hologram base runtime updates

## Status: Active

## Context

hologram base (commit `1e48540`) shipped major runtime improvements for SD
pipeline execution. hologram-ai needs corresponding cleanup and SPRINT
documentation updates.

### hologram base changes (committed)

| Feature | Impact |
|---------|--------|
| Conv2d BLAS sgemm integration | VAE decoder: 494s → 0.9s (526x) |
| Liveness-based arena eviction | UNet RSS: 31GB → <1GB |
| Parallel float matmul (row + batch) | Rayon parallelism for non-BLAS platforms (WASM) |
| Weight index + mmap prefetch/release | Layer-granular madvise for weight paging |
| Prewarm guard (>2GB skip) | Prevents OOM on large models |
| Alignment safety (safe_cast_f32) | Handles misaligned borrowed buffers |
| binary_broadcast div-by-zero fix | Prevents panic on empty inputs |
| Pipeline archive auto-detection (load_auto) | Transparent pipeline unwrap |

### hologram-ai changes (this plan)

| Change | File |
|--------|------|
| Remove debug eprintln! | `lower/strategy.rs` |
| Clean test diagnostics | `sd_unet_e2e.rs`, `sd_vae_e2e.rs` |
| Resize scales shape fix | `opt/shape_prop.rs` |
| Weight alignment padding | `compiler.rs` |
| `AiParam::as_f32_slice()` | `ir/param.rs` |
| `load_auto` migration | `bert_e2e.rs`, `resnet50_e2e.rs` |
| New VAE decoder test | `sd_vae_e2e.rs` |
| SPRINT.md update | `specs/SPRINT.md` |

## Verification

```
cargo clippy -p hologram-ai-common -- -D warnings
cargo test -p hologram-ai --features e2e --release
```

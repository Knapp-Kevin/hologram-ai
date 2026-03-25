# Plan 027: Stable Diffusion Support

## Context

hologram-ai supports text/embedding models (TinyLlama, BERT, ResNet) but has no image-generation model support. Stable Diffusion is the next target — it exercises cross-attention, GroupNorm, multi-component pipelines, and image output. The multi-component compilation infrastructure (Plan 021) is already complete, and `ModelKind::ImageGen` already exists in hologram base. The main gaps are: a missing GroupNorm kernel, output type detection, and runtime image output.

**Branch:** `feat/stable-diffusion` (from `feat/single-model-rearchitecture`)

---

## Phase 1: GroupNorm Lowering

**Goal:** Make GroupNorm compilable. SD UNet uses GroupNorm in every residual block — this is the critical blocker.

**Status:** `AiOp::GroupNorm` exists in IR, ONNX importer maps `"GroupNormalization"` to it, shape spec handles it. BUT: no lowering in `strategy.rs` (catch-all returns `None`) and no `FloatOp::GroupNorm` in hologram base.

### hologram base (separate PR)
- Add `FloatOp::GroupNorm { num_groups: u32, epsilon: u32 }`
- Implement dispatch kernel (reshape to [N, G, C/G, ...], normalize per-group, apply scale+bias)
- Follow `InstanceNorm` pattern for ShapeSpec, input_count, name

### hologram-ai
- Add lowering in [strategy.rs](crates/hologram-ai-common/src/lower/strategy.rs) ~line 650:
  ```rust
  AiOp::GroupNorm { num_groups, epsilon } => {
      let eps_bits = epsilon.to_bits();
      (FloatOp::GroupNorm { num_groups: *num_groups, epsilon: eps_bits }, vec![])
  }
  ```

### Verification
- Unit test: AiGraph with GroupNorm → lower → compile → execute → compare reference
- `cargo test && cargo clippy -- -D warnings`

---

## Phase 2: Single-Component UNet Compilation (MVP)

**Goal:** `hologram-ai compile -m unet.onnx -o out/` succeeds for SD v1.5 UNet.

### Steps
1. Download SD v1.5 UNet ONNX from HuggingFace
2. Compile and fix failures. Expected issues:
   - **GroupNorm** (Phase 1)
   - **Cross-attention**: UNet has self-attention AND cross-attention (Q from latent, K/V from text encoder). `AttentionFusion` may not recognize cross-attention. The UNet compiles via `OptProfile::Generic` which skips attention fusion — this is correct for now.
   - **Resize** mode "nearest" for upsampling — lowering exists, verify kernel
   - **SiLU**: already mapped as `AiOp::Silu` and lowered
3. Add conformance test verifying compilation + output shape [1, 4, 64, 64] + all finite values

### Key files
- [compiler.rs](crates/hologram-ai/src/compiler.rs) — main compile path
- [strategy.rs](crates/hologram-ai-common/src/lower/strategy.rs) — any new lowering
- New: `crates/hologram-ai-conformance/tests/sd_unet_conformance.rs`

### Verification
- `hologram-ai compile -m unet/model.onnx -o out/`
- Conformance test: compile + execute with dummy inputs, verify output shape and finiteness

---

## Phase 3: Output Type System + ModelKind Detection

**Goal:** Models get the correct `ModelKind` in archive metadata. `ModelKind::ImageGen` already exists in hologram base.

### Changes to [cli.rs](crates/hologram-ai/src/cli.rs) lines 117-135
Add `kind` field to manifest TOML and detection heuristic:

```toml
kind = "image-gen"  # maps to ModelKind::ImageGen
```

- Add optional `kind: Option<String>` to `Manifest` struct
- Map `"text-llm"` → `TextLlm`, `"image-gen"` → `ImageGen`, etc.
- For single-model compilation, keep existing `is_llm` heuristic
- Thread detected kind into `ModelMetaSection` construction

### Verification
- SD manifest compiles with `kind: ImageGen` in archive metadata
- Existing TinyLlama → `TextLlm`, ResNet/BERT → `Generic` (unchanged)

---

## Phase 4: Full 3-Component SD Pipeline Archive

**Goal:** `hologram-ai compile --manifest sd.toml -o out/` produces a pipeline archive with text encoder + UNet + VAE decoder.

### Manifest (`sd_v15.toml`)
```toml
kind = "image-gen"

[[component]]
name = "text_encoder"
path = "text_encoder/model.onnx"
role = "encoder"
weight_group = "clip"

[[component]]
name = "unet"
path = "unet/model.onnx"
role = "backbone"
weight_group = "unet"

[[component]]
name = "vae_decoder"
path = "vae_decoder/model.onnx"
role = "decoder"
weight_group = "vae"

[[connection]]
from = "text_encoder:last_hidden_state"
to = "unet:encoder_hidden_states"

[[connection]]
from = "unet:sample"
to = "vae_decoder:latent_sample"
```

### Steps
- `compile_multi_onnx()` already handles generic multi-ONNX — no new compiler code expected
- Text encoder (CLIP) is a transformer but doesn't need KV cache — `OptProfile::Generic` is correct
- Verify all 3 components compile, archive has correct `MetaSection`

### Key files
- [compiler.rs](crates/hologram-ai/src/compiler.rs) lines 611+ — `compile_multi_onnx`
- [cli.rs](crates/hologram-ai/src/cli.rs) lines 276+ — `parse_manifest`

### Verification
- `hologram-ai compile --manifest sd_v15.toml -o out/`
- `hologram-ai info out/sd.holo` shows 3 components with correct roles
- Archive size is sum of components (no unintended duplication — different weight groups)

---

## Phase 5: Runtime Image Output + Denoising Demo

**Goal:** `hologram-ai run sd.holo --prompt "a photo of a cat" --output cat.png` generates an image.

**Compiler-only note:** This is CLI demonstration code, analogous to the autoregressive generation loop already in `run_cmd.rs`. The denoising scheduler is ~30 lines of pure math.

### Changes

1. **Multi-component runner** — extend `HoloRunner` to support named component execution:
   ```rust
   runner.execute_component("text_encoder", &inputs)?
   runner.execute_component("unet", &inputs)?
   runner.execute_component("vae_decoder", &inputs)?
   ```

2. **Denoising loop** in [run_cmd.rs](crates/hologram-ai/src/commands/run_cmd.rs):
   - Tokenize prompt → execute text_encoder → text embeddings
   - Init random latent [1, 4, 64, 64]
   - For t in scheduler timesteps (e.g., 20 Euler-a steps): predict noise → update latent
   - Execute vae_decoder → image tensor [1, 3, 512, 512]

3. **Image output** — add `--output` flag, save [1,3,H,W] f32 tensor as PNG
   - Add `image` crate as optional dependency (`image-output` feature)
   - Clamp [0,1], convert to u8 RGB, save

4. **Minimal scheduler** — Euler-a or DDIM (~30 lines)

### Key files
- [run_cmd.rs](crates/hologram-ai/src/commands/run_cmd.rs) — SD generation path
- [compiler.rs](crates/hologram-ai/src/compiler.rs) — HoloRunner multi-component loading

### Verification
- `hologram-ai run out/sd.holo --prompt "a photo of a cat" --output cat.png`
- Output is valid PNG, not all zeros or noise
- Existing `--prompt` text generation still works

---

## Dependency Graph

```
Phase 1 (GroupNorm) ──── Phase 2 (UNet MVP)
        │                       │
Phase 3 (Output types) ────── Phase 4 (Full pipeline) ── Phase 5 (Runtime demo)
```

Phases 1 and 3 are independent. Phase 2 needs Phase 1. Phase 4 needs 2+3. Phase 5 needs 4.

---

## SPRINT.md Updates

Under **Medium Term: Multi-model support → Any ONNX model**:
- Update SD UNet line item with progress
- Add sub-items for each phase

Under **Long Term: Production readiness → Performance**:
- Update `[ ] Multi-modal output trait` line to reference Phase 5

---

## Critical Files

| File | Change |
|------|--------|
| `specs/plans/027-stable-diffusion.md` | This plan |
| `specs/SPRINT.md` | Update SD status, add phase tracking |
| `crates/hologram-ai-common/src/lower/strategy.rs` | GroupNorm lowering |
| `crates/hologram-ai/src/cli.rs` | Manifest `kind` field, ModelKind detection |
| `crates/hologram-ai/src/commands/run_cmd.rs` | SD generation loop, image output |
| `crates/hologram-ai/src/compiler.rs` | HoloRunner multi-component execution |
| New: `crates/hologram-ai-conformance/tests/sd_unet_conformance.rs` | UNet conformance |
| hologram base (separate repo) | `FloatOp::GroupNorm` + dispatch kernel |

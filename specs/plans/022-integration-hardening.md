# Plan 022: Integration Hardening + KvExecutor Deprecation + SD UNet Support

## Context

The hologram base crate completed Phases 8-9 (EnumTape executor, multi-backend GPU). The handoff spec (Section 8) deprecates `KvExecutor` and mandates `build_tape_from_plan` → `execute_tape` for all work. hologram-ai still has 7 call sites using the legacy `execute_plan` path (17-140x slower).

SPRINT.md line 83-88 incorrectly claims PreAttentionFusion works end-to-end — hologram base's `dispatch_attention()` ignores `qk_norm`/`rope`/`rope_base` fields (both dispatch sites use `..`). This must be corrected and the pass gated.

**Goals**:
1. Migrate all execution paths from `execute_plan` to EnumTape
2. Gate pre-attention fusion until hologram base implements fused qk_norm/rope kernel
3. Add `--manifest` flag to existing `compile` command for multi-component models
4. Stable Diffusion UNet — next ONNX model target

### Legacy `execute_plan` Usage (must migrate)

| File | Line | API | Context |
|------|------|-----|---------|
| `compiler.rs` | 1604-1617 | `execute_tape` with `execute_plan` fallback | `InferenceSession::execute()` |
| `compiler.rs` | 1659 | `execute_plan_with_kv_state` | `execute_with_kv()` — LLM decode inner loop |
| `compiler.rs` | 1776 | `execute_plan` | `run_with_shape_context()` helper |
| `exec_conformance.rs` | 46, 2270 | `execute_plan` | Conformance tests |
| `exec_conformance.rs` | 1315 | `execute_plan_with_intermediates` | Debug conformance |
| `exec_conformance.rs` | 1476 | `execute_plan_with_intermediates_and_shape_hints` | Shape-aware debug |
| `resnet50_e2e.rs` | 83 | `execute_plan` | ResNet E2E test |

---

## Phase 1: KvExecutor → EnumTape Migration

### 1.1 `InferenceSession` struct — make tape non-optional

**File**: `crates/hologram-ai/src/compiler.rs`

- Change `tape: Option<EnumTape>` → `tape: EnumTape`
- Change `_decode_tape: Option<EnumTape>` → `decode_tape: Option<EnumTape>` (only present for pipeline archives)
- In `from_bytes()`: propagate error from `build_tape_from_plan()` instead of `.ok()`

### 1.2 `InferenceSession::execute()` — remove fallback

**File**: `crates/hologram-ai/src/compiler.rs` (lines 1601-1618)

```rust
pub fn execute(&self, inputs: &hologram::GraphInputs) -> anyhow::Result<hologram::GraphOutputs> {
    hologram::execute_tape(&self.tape, &self.plan, inputs)
        .map_err(|e| anyhow::anyhow!("{e}"))
}
```

### 1.3 `execute_with_kv()` — migrate to `execute_tape_with_kv`

**File**: `crates/hologram-ai/src/compiler.rs` (lines 1648-1661)

Confirmed: `hologram::execute_tape_with_kv` exists (`lib.rs:58`), `TapeContext::with_kv_cache()` exists (`tape.rs:217`).

```rust
pub fn execute_with_kv(
    &self,
    inputs: &hologram::GraphInputs,
    kv_state: &mut hologram::KvCacheState,
) -> anyhow::Result<hologram::GraphOutputs> {
    let (tape, plan) = if kv_state.write_pos() > 0 {
        (self.decode_tape.as_ref().expect("pipeline archive must have decode tape"),
         self.decode_plan.as_ref().expect("pipeline archive must have decode plan"))
    } else {
        (&self.tape, &self.plan)
    };
    hologram::execute_tape_with_kv(tape, plan, inputs, kv_state)
        .map_err(|e| anyhow::anyhow!("{e}"))
}
```

### 1.4 `run_with_shape_context()` — migrate to tape

**File**: `crates/hologram-ai/src/compiler.rs` (lines 1771-1778)

```rust
pub fn run_with_shape_context(archive: &HoloArchive, inputs: &hologram::GraphInputs) -> anyhow::Result<hologram::GraphOutputs> {
    let plan = hologram::load_from_bytes(&archive.bytes)?;
    let tape = hologram::build_tape_from_plan(&plan)?;
    hologram::execute_tape(&tape, &plan, inputs)
        .map_err(|e| anyhow::anyhow!("{e}"))
}
```

### 1.5 Conformance tests — migrate to tape

**File**: `crates/hologram-ai-conformance/tests/exec_conformance.rs`

- Lines 46, 2270: `execute_plan` → `build_tape_from_plan` + `execute_tape`
- Lines 1315, 1476: **Keep on legacy** — `execute_plan_with_intermediates` has no tape equivalent yet (handoff spec Section 8 lists "add intermediate capture to EnumTape" as prerequisite)

### 1.6 ResNet E2E test — migrate to tape

**File**: `crates/hologram-ai/tests/resnet50_e2e.rs` (line 83)

Same pattern: `build_tape_from_plan` + `execute_tape`.

### 1.7 Update documentation

Update KvExecutor references in 16 spec/doc files. Mark as deprecated, point to EnumTape.

### Verification
- `cargo test` — all tests pass
- `cargo clippy -- -D warnings`
- Grep for `execute_plan` — only intermediate-capture sites remain

---

## Phase 2: Pre-Attention Fusion — Gate Until Kernel Support

### 2.1 Remove `PreAttentionFusion` from `mvp()` pipeline

**File**: `crates/hologram-ai-common/src/opt/pipeline.rs`

Remove `PreAttentionFusion` from the pass list in `OptPipeline::mvp()`. Add comment:
```rust
// TODO(hologram-base): re-enable PreAttentionFusion when dispatch_attention
// handles qk_norm/rope flags (currently ignored via `..` destructuring)
```

### 2.2 Commit staged + untracked changes

Files to commit (forward-compatible, pass is gated):
- `crates/hologram-ai-common/src/ir/op.rs` — 3 new GQA fields
- `crates/hologram-ai-common/src/opt/mod.rs` — module declaration + re-export
- `crates/hologram-ai-common/src/opt/pipeline.rs` — `generic()` method (PreAttentionFusion removed from `mvp()`)
- `crates/hologram-ai-common/src/opt/pre_attention_fusion.rs` — pass + 4 unit tests

### 2.3 Add lowering unit test for GQA with flags

**File**: `crates/hologram-ai-common/src/lower/strategy.rs`

Test that `AiOp::GroupedQueryAttention { qk_norm: true, rope: true, rope_base: 10000.0, .. }` produces `FloatOp::Attention` with matching flags.

### Verification
- `cargo test` — 4 unit tests pass, TinyLlama ONNX still correct (fusion doesn't fire)
- `cargo clippy -- -D warnings`

---

## Phase 3: Multi-Component CLI — Extend `compile` with `--manifest`

The CLI has exactly 3 core commands (`compile`, `info`, `download`). Rather than adding a new subcommand, extend the existing `compile` command with a `--manifest` flag. The two modes are mutually exclusive:

```
# Single model (existing)
hologram-ai compile -m model.onnx -o out/

# Multi-component (new — via manifest)
hologram-ai compile --manifest pipeline.toml -o out/
```

### 3.1 Add `--manifest` to `Compile` command

**File**: `crates/hologram-ai/src/cli.rs`

Add `manifest: Option<PathBuf>` arg to `Compile`. When present, parse the TOML manifest into `ModelSource::MultiOnnx`. When absent, use `--model` as today. Make `--model` and `--manifest` mutually exclusive via clap group.

Manifest format:
```toml
[[component]]
name = "encoder"
path = "encoder.onnx"
role = "encoder"
weight_group = "shared"

[[component]]
name = "decoder"
path = "decoder.onnx"
role = "decoder"
weight_group = "shared"

[[connection]]
from = "encoder:hidden_states"
to = "decoder:encoder_hidden_states"
```

### 3.2 E2E test for multi-component

**File**: New `crates/hologram-ai/tests/multi_component_e2e.rs`

Build two synthetic ONNX models, compile via `ModelSource::MultiOnnx`, verify archive loads with correct `MetaSection` and weight dedup.

### Verification
- `cargo test multi_component` passes
- `hologram-ai compile --manifest test.toml -o out/` succeeds

---

## Phase 4: Stable Diffusion UNet Support

**Goal**: `hologram-ai compile -m unet.onnx -o out/` compiles and executes correctly.

Stable Diffusion UNet is the next ONNX model target per SPRINT.md. It exercises vision + attention + cross-attention — a superset of what TinyLlama (causal LLM) and ResNet-50 (pure vision) cover.

### 4.1 Obtain Stable Diffusion UNet ONNX model

Export via `optimum-cli export onnx --model stabilityai/stable-diffusion-2-1 --task stable-diffusion` or download the UNet component from a pre-exported SD pipeline. The UNet is the most complex component (~800+ nodes).

### 4.2 Compile and fix failures

SD UNet uses patterns not yet E2E tested:
- **Cross-attention** — Q from latent, K/V from text encoder (different sequence lengths). Verify `AttentionFusion` handles cross-attention (`num_q_heads` may differ from context)
- **Self-attention** (bidirectional, non-causal) — `causal: false`
- **GroupNorm** — may need new `AiOp::GroupNorm` or decomposition in `OpDecomposition` pass
- **Conv2d + GroupNorm + SiLU** — common ResBlock pattern, verify all lower correctly
- **Upsample / Resize** — bilinear interpolation, verify `FloatOp::Resize` dispatch
- **Timestep embedding** — sinusoidal positional encoding, likely decomposes to trig ops
- **No KV cache** — single forward pass, use `OptProfile::Generic`
- **3 inputs**: `sample` (latent), `timestep` (scalar), `encoder_hidden_states` (text conditioning)

### 4.3 SD UNet conformance test

**File**: New in `crates/hologram-ai-conformance/tests/`

1. Load UNet ONNX
2. Compile via `ModelCompiler::default().compile(ModelSource::OnnxPath(...))`
3. Create dummy inputs: `sample=[1,4,64,64]`, `timestep=[1]`, `encoder_hidden_states=[1,77,1024]`
4. Run through ORT for reference outputs
5. Run through hologram tape executor
6. Compare noise prediction output within tolerance

### 4.4 SD UNet E2E test

**File**: New `crates/hologram-ai/tests/sd_unet_e2e.rs`

Compile UNet, execute with random latent + timestep + text embedding, verify output shape `[1, 4, 64, 64]` and no NaN/Inf.

### Verification
- `cargo test sd_unet` passes
- `cargo clippy -- -D warnings`

---

## SPRINT.md Updates

### Fix PreAttentionFusion claim (P3, lines 83-88)

Change `[x]` to `[ ]`:
```
- [ ] QK-Norm + RoPE pre-attention fusion — `PreAttentionFusion` pass implemented
  with 4 unit tests. **Gated**: hologram base `dispatch_attention()` ignores
  `qk_norm`/`rope` flags. Pass removed from `mvp()` pipeline until kernel support
  lands. AiOp fields and lowering are forward-compatible.
```

### Update P2c (lines 60-62) after migration

```
- [x] Wire tape executor from `InferenceSession` — build tape at load time,
  use `execute_tape` / `execute_tape_with_kv` for all execution. No fallback.
```

### Update line 213

Change `execute_plan` → `execute_tape`.

### Add Plan 022 section under "Active"

```
### KvExecutor → EnumTape Migration (Plan 022)
- [ ] Remove `execute_plan` fallback from `InferenceSession::execute()`
- [ ] Migrate `execute_with_kv()` to `execute_tape_with_kv`
- [ ] Migrate `run_with_shape_context()` to tape
- [ ] Migrate conformance + E2E tests to tape
- [ ] Update spec/doc KvExecutor references
```

---

## Sequencing

```
Phase 1 (KvExecutor migration)     ─┐
Phase 2 (Gate pre-attention fusion) ─┤──> SPRINT.md updates
Phase 3 (CLI --manifest)            ─┘
                                     ↓
                               Phase 4 (SD UNet)
```

Phases 1-3 are independent and can run in parallel. Phase 4 depends on Phase 1 (SD UNet tests use tape executor).

---

## Critical Files

| File | Changes |
|------|---------|
| `crates/hologram-ai/src/compiler.rs` | Make tape non-optional, remove fallback, migrate KV + helper |
| `crates/hologram-ai-conformance/tests/exec_conformance.rs` | Migrate 2 of 4 `execute_plan` calls |
| `crates/hologram-ai/tests/resnet50_e2e.rs` | Migrate to tape |
| `crates/hologram-ai-common/src/opt/pipeline.rs` | Remove `PreAttentionFusion` from `mvp()` |
| `crates/hologram-ai-common/src/opt/pre_attention_fusion.rs` | Commit (gated) |
| `crates/hologram-ai-common/src/ir/op.rs` | Commit (forward-compatible fields) |
| `crates/hologram-ai/src/cli.rs` | Add `--manifest` flag to `compile` |
| `specs/SPRINT.md` | Fix claims, add Plan 022, update refs |

## Resolved Questions

1. **`execute_tape` KV support**: `hologram::execute_tape_with_kv` exists (`lib.rs:58`), `TapeContext::with_kv_cache()` exists (`tape.rs:217`). Unblocked.
2. **`FloatOp::Attention` qk_norm/rope**: NOT implemented. Both dispatch sites use `..`. Pass must be gated.
3. **SPRINT.md accuracy**: Line 83-88 falsely claims PreAttentionFusion works end-to-end. Must be corrected.
4. **CLI design**: Single `compile` command with `--model` / `--manifest` mutual exclusion. No new subcommands.

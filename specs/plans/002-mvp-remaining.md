# Plan 002: MVP Remaining Work

**Status:** Planning
**Date:** 2026-03-07
**ADRs:** ADR-0002, ADR-0005, ADR-0006, ADR-0007, ADR-0012, ADR-0015, ADR-0016
**Roadmap ref:** MVP (Weeks 1–4) exit criteria

---

## Design Principle

**hologram-ai is a compiler only.** It contains zero runtime code. All execution
kernels, custom handlers, and float ops belong in the hologram base crate.
hologram-ai parses, optimizes, lowers, and writes archives. That's it.

CLI commands: `compile`, `info`, `download`. No `run`, no `validate`.
(`hologram run` handles execution; validation is done by compiling then
running via hologram.)

---

## Gap Analysis

| Gap | Current State | Required |
|-----|---------------|----------|
| Runtime code in hologram-ai | `custom_ops.rs` has ~60 f32 handlers | Delete — hologram needs native float ops |
| `Run` CLI command | Exists as facade to `hologram run` | Delete — users call `hologram run` directly |
| Multi-graph lowering | Single graph; `_kv_layout` unused | `LowerPhase` enum, prefill/decode graphs |
| KV-cache ops | No `KvSlotWrite`/`KvSlotRead` in `AiOp` | KV read/write ops in IR |
| KV-cache layout | `KvCacheLayout::none()` always | `MemoryPlanner` computes from arch params |
| Pipeline archive | Single flat archive via `HoloWriter` | `PipelineWriter` bundles sub-archives |
| LayerHeader | Not emitted | Named `lm.prefill`/`lm.decode` with tensor ports |
| LLM meta section | Not emitted | `SECTION_LLM_META` (0x0011) |
| Tokenizer section | Not emitted | `SECTION_TOKENIZER` (0x1001) |
| Tokenizer archive packing | No `archive.rs` | ConstantStore pack/unpack |
| ConstantFolding | No-op stub | Fold identity chains on constants |
| Lowering dispatch | Maps to `GraphOp::Custom` | Map to native `GraphOp::Float(FloatOp)` |

---

## Blocked on hologram base crate

The following items **cannot be completed** until hologram adds native float ops.
See `specs/plans/hologram-types-needed.md` for the full change request.

- **Native float tensor ops** — `FloatAdd`, `FloatMatMul`, `FloatSoftmax`,
  `FloatRmsNorm`, `FloatAttention`, etc. Without these, lowering must use
  `GraphOp::Custom` and ship runtime handlers.
- **Shape metadata on graph edges** — hologram graphs have no per-edge
  shape/dtype. Lowering currently bakes shapes into closure captures.
- **`LlmMetaSection`**, **`TokenizerSectionData`** — spec says these types
  live in hologram. Workaround: local `EmbeddableSection` implementations.

---

## Work Items (hologram-ai side)

### 1. Delete Runtime Code

**Goal:** Remove all custom op handlers. hologram-ai ships zero runtime code.

**Changes:**
- Delete `crates/hologram-ai-common/src/lower/custom_ops.rs`
- Remove `CustomOpRegistry` from `LoweringOutput`
- Remove `Run` command from CLI
- Update `lower/dispatch.rs` to map `AiOp` → native `GraphOp` variants
  (blocked on hologram adding float ops; can stub with `GraphOp::Custom` for now)

**Files:**
- `crates/hologram-ai-common/src/lower/custom_ops.rs` (delete)
- `crates/hologram-ai-common/src/lower/builder.rs`
- `crates/hologram-ai-common/src/lower/dispatch.rs`
- `crates/hologram-ai/src/cli.rs`

### 2. KV-Cache Ops in IR

**Goal:** Add KV-cache read/write operations to `AiOp`.

**Changes:**
- Add `AiOp::KvSlotWrite { layer: usize }` — writes K/V to cache
- Add `AiOp::KvSlotRead { layer: usize }` — reads cached K/V
- GGUF arch builders emit these ops in attention blocks
- Shape propagation rules for KvSlotWrite/KvSlotRead

**Files:**
- `crates/hologram-ai-common/src/ir/op.rs`
- `crates/hologram-ai-common/src/opt/shape_prop.rs`
- `crates/hologram-ai-gguf/src/arch/llama.rs`

### 3. KV-Cache Layout Computation

**Goal:** `MemoryPlanner` computes real `KvCacheLayout` from arch params.

**Changes:**
- Read n_layers, n_kv_heads, head_dim, max_seq_len, dtype from `AiGraph` metadata
- Compute `total_bytes = n_layers × 2 × n_kv_heads × head_dim × max_seq_len × dtype_size`
- Return populated `KvCacheLayout`

**Files:**
- `crates/hologram-ai-common/src/mem/planner.rs`

### 4. Multi-Graph Lowering

**Goal:** Lower `AiGraph` twice with `LowerPhase` for prefill + decode.

**Changes:**
- Add `LowerPhase` enum: `Prefill`, `Decode`, `DecodeBucket(u64)`
- Prefill: `input_ids [batch, seq_len]`, `kv_cache [n_bytes]` in/out, `logits` out
- Decode: `input_ids [batch, 1]`, `present_len [] u32`, `kv_cache [n_bytes]` in/out, `logits` out
- `LoweringOutput` includes `layer_name` and `layer_descriptor` with `TensorPort` entries

**Files:**
- `crates/hologram-ai-common/src/lower/builder.rs`
- `crates/hologram-ai-common/src/lower/dispatch.rs`

### 5. Pipeline Archive Construction

**Goal:** Bundle prefill + decode sub-archives via `PipelineWriter`.

**Changes:**
- `ModelCompiler::compile()` detects LLMs (has `arch` metadata)
- Lower twice → `hologram::compile()` twice → `PipelineWriter` bundles them
- Each sub-archive gets `LayerHeader` with named layer + tensor ports
- Non-LLM models: single archive with `"model.forward"` layer

**Files:**
- `crates/hologram-ai/src/compiler.rs`

### 6. LLM Meta Section

**Goal:** Embed `SECTION_LLM_META` (0x0011) in each sub-archive.

**Changes:**
- Define local `LlmMetaSection` implementing `EmbeddableSection`
  (using `SECTION_CUSTOM_BASE + 0x11` until hologram adds the type)
- Contains: `KvCacheLayout`, model type, prefill/decode layer IDs
- Migrate to `hologram::LlmMetaSection` when available

**Files:**
- `crates/hologram-ai-common/src/sections/llm_meta.rs` (new)
- `crates/hologram-ai/src/compiler.rs`

### 7. Tokenizer Section

**Goal:** Embed `SECTION_TOKENIZER` (0x1001) from GGUF metadata.

**Changes:**
- Add `archive.rs` with ConstantStore pack/unpack for vocab/merges/scores
- Define local `TokenizerSectionData` implementing `EmbeddableSection`
- Compiler extracts tokenizer from GGUF → packs → embeds in prefill sub-archive

**Files:**
- `crates/hologram-ai-tokenizer/src/archive.rs` (new)
- `crates/hologram-ai/src/compiler.rs`

### 8. ConstantFolding

**Goal:** Replace the no-op stub with actual folding.

**Changes:**
- Fold `Identity` nodes whose input is `Constant`
- Fold `Reshape` of constant tensors
- Remove dead constant nodes

**Files:**
- `crates/hologram-ai-common/src/opt/constant_fold.rs`

### 9. CLI Cleanup

**Goal:** CLI has exactly three commands: `compile`, `info`, `download`.

**Changes:**
- Delete `Command::Run` variant and all associated code
- Keep `Command::Info` (delegates to `hologram inspect` for .holo, prints
  metadata for .onnx/.gguf)
- Keep `Command::Compile` (import → optimize → lower → write archive)
- Keep `Command::Download` (HuggingFace acquisition)

**Files:**
- `crates/hologram-ai/src/cli.rs`

---

## Execution Order

```
1. CLI cleanup (delete Run) + delete custom_ops.rs
     ↓
2. KV-cache ops in IR
     ↓
3. KV-cache layout computation
     ↓
4. Multi-graph lowering (LowerPhase)
     ↓
5. Pipeline archive construction
     ↓
6. LLM meta section ──────┐
                           │ (parallel)
7. Tokenizer section ─────┘
     ↓
8. ConstantFolding
```

Steps 1–3 and 8 are unblocked.
Steps 4–7 depend on hologram adding native float ops for the lowering to
emit correct native `GraphOp` variants (can stub with `Custom` placeholder).

---

## MVP Exit Criteria (from roadmap.md)

- [ ] `hologram-ai compile tinyllama.gguf` produces a valid `.holo` pipeline archive
- [ ] Archive `LayerHeader` declares `lm.prefill` and `lm.decode` with correct tensor ports
- [ ] `SECTION_LLM_META` reports correct `KvCacheLayout` for TinyLlama 1.1B
- [ ] `KvExecutor` yields logits of correct shape from compiled archive
- [ ] hologram-ai contains zero runtime code (no custom op handlers)
- [ ] All unit tests pass

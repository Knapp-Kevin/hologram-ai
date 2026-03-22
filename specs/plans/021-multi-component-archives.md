# Plan: Multi-Component Archive Support (CALM-ready)

## Context

The CALM paper (Continuous Autoregressive Language Models) demonstrates that next-generation efficient LLMs will be **multi-component architectures**: autoencoder + transformer backbone + generative head. Each component is independently compilable but connected at inference time.

Our existing `PipelineWriter` already bundles N named sub-archives — it's the right foundation. But the compiler pipeline hardcodes the 2-component LLM pattern (prefill/decode), duplicates weights across sub-archives, and has no metadata for component relationships.

**Goal**: Generalize the compilation pipeline so any N-component model can be compiled into a single archive with shared weights and component relationship metadata. This unlocks:
- **CALM** (autoencoder + backbone + generative head)
- **Encoder-decoder** models (Whisper, T5, mBART)
- **Stable Diffusion** (VAE encoder + UNet + VAE decoder)
- **MoE** (router + N experts)
- Any future multi-component ONNX model

**Answer to "do subgraphs support this?"**: No — `AiGraph.subgraphs` is for control flow (If/Loop/Scan) within a single model. `PipelineWriter` is the correct abstraction for multi-component models, and it already works. We need to generalize the compiler's use of it.

### Invariants (what does NOT change)

- **`.holo` remains the sole compilation target.** `PipelineWriter` already produces `.holo` pipeline archives (that's how LLM prefill/decode work today). This plan generalizes how the compiler constructs them — the output format is structurally identical.
- **hologram-ai remains compiler-only (ADR-0016).** No runtime code is added. Metadata sections are purely descriptive bytes in the archive. Zero runtime performance impact.
- **No new archive format versions.** All changes are additive sections and generalized compiler logic.
- **Runtime weight access stays zero-indirection.** Deduplicated weights are resolved at archive-load time (offset rewriting), not at inference time. During execution, weight pointers are direct — same as today.
- **No duplicative abstractions.** Phase 2 *generalizes* `MetaSection` rather than adding a parallel section. The LLM pipeline becomes a specialization of the general component model.

### hologram-base alignment

Recent hologram-base changes confirm this plan's architecture:
- **PipelineWriter** — production-ready, supports N named sub-archives with CRC32 checksums. No modifications needed.
- **Tape executor** — 140x faster than KvExecutor. Each component gets its own tape, built once. Multi-component inference = execute N tapes in sequence.
- **Metal GPU backend** — 16 kernels, zero-copy unified memory. Multi-component models get GPU acceleration automatically via `BackendSelector::Auto`.
- **Per-component KV-cache** — each `TapeContext` carries isolated `KvCacheState`. Components without attention simply don't allocate KV buffers.

---

## Phase 1: Generic N-Component Compilation

**What**: Extract the repeated lower→compile→assemble pattern into a reusable function, then build a generic `compile_components()` that takes N specs and produces a pipeline archive.

### 1.1 Extract `compile_one_component()` helper

The logic at [compiler.rs:472-496](crates/hologram-ai/src/compiler.rs#L472-L496) (prefill) and [compiler.rs:506-528](crates/hologram-ai/src/compiler.rs#L506-L528) (decode) is nearly identical. Extract into:

```rust
/// Compile a single AiGraph into a sub-archive ready for PipelineWriter.
fn compile_one_component(
    ai_graph: &AiGraph,
    kv_layout: &KvCacheLayout,
    opts: &LoweringOptions,
    phase: &LowerPhase,
    extra_weights: Option<Vec<u8>>,
) -> anyhow::Result<Vec<u8>>
```

**File**: [compiler.rs](crates/hologram-ai/src/compiler.rs)

### 1.2 Add `LowerPhase::Named(String)`

Extend the enum at [lower/mod.rs:17-36](crates/hologram-ai-common/src/lower/mod.rs#L17-L36):

```rust
pub enum LowerPhase {
    Prefill,
    Decode,
    Forward,
    Named(String),  // NEW: arbitrary component name
}

impl LowerPhase {
    pub fn layer_name(&self) -> &str {  // &str, not &'static str
        match self {
            Self::Prefill => "lm.prefill",
            Self::Decode => "lm.decode",
            Self::Forward => "model.forward",
            Self::Named(name) => name.as_str(),
        }
    }
}
```

**File**: [lower/mod.rs](crates/hologram-ai-common/src/lower/mod.rs)

### 1.3 Add `OptProfile` for per-component pass selection

```rust
pub enum OptProfile {
    Llm,         // full MVP pipeline (attention fusion, KV injection)
    Generic,     // shape/data propagation + constant folding only
}
```

LLM components use `Llm`, non-transformer components (autoencoders, heads) use `Generic` — avoids running irrelevant passes like `KvSlotInjection` on components without attention.

### 1.4 Expose `MemoryPlan::empty()`

Components without attention don't need KV-cache. Add:
```rust
impl MemoryPlan {
    pub fn empty() -> Self { /* no KV-cache layout */ }
}
```

### 1.5 New `ComponentSpec` type and `compile_components()`

```rust
pub struct ComponentSpec<'a> {
    pub name: String,
    pub opt_profile: OptProfile,
    pub graph: &'a AiGraph,
    pub mem_plan: &'a MemoryPlan,
    pub phase: LowerPhase,
    pub weights: Option<Vec<u8>>,
}

fn compile_components(
    &self,
    specs: Vec<ComponentSpec<'_>>,
) -> anyhow::Result<Vec<u8>>
```

Iterates specs, runs the appropriate `OptPipeline` per `opt_profile`, calls `compile_one_component` for each, bundles via `PipelineWriter`.

**File**: [compiler.rs](crates/hologram-ai/src/compiler.rs)

### 1.6 Refactor `compile_llm_pipeline` to delegate

The existing method builds its two `AiGraph` variants (prefill at full seq, decode at seq=1), constructs two `ComponentSpec` entries, and calls `compile_components`. Pure refactor — identical output.

### Verification
- Existing `tinyllama_e2e` tests pass unchanged
- New unit test: 3 trivial AiGraph instances → `compile_components` → verify pipeline archive has 3 named entries

---

## Phase 2: Generalize Model Metadata

**What**: Rename `LlmMetaSection` → `MetaSection` and generalize it to describe N components and their relationships. The LLM pipeline becomes a specialization — no new parallel section, no duplication.

### 2.1 Rename `LlmMetaSection` → `MetaSection` and generalize

Rename and extend in [sections/llm_meta.rs](crates/hologram-ai-common/src/sections/llm_meta.rs) (rename file to `meta.rs`, same section ID):

```rust
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct MetaSection {
    /// Components in this pipeline archive.
    pub components: Vec<ComponentDescriptor>,
    /// Data flow between components (output port → input port).
    pub connections: Vec<ComponentConnection>,
}

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ComponentDescriptor {
    pub name: String,           // pipeline key, e.g. "lm.prefill", "ae.encoder"
    pub role: ComponentRole,
    pub weight_group: String,   // components sharing this value share weights (Phase 3)
}

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum ComponentRole {
    Prefill,
    Decode,
    Encoder,
    Decoder,
    Backbone,
    GenerativeHead,
    Forward,
    Custom(String),
}

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ComponentConnection {
    pub from_component: String,
    pub from_output: String,
    pub to_component: String,
    pub to_input: String,
}
```

This replaces `MetaSection` — the LLM case is just 2 descriptors (Prefill + Decode) with 1 connection (shared KV-cache). No separate section type needed.

### 2.2 Extend `ComponentSpec` with role and weight_group

```rust
pub struct ComponentSpec<'a> {
    pub name: String,
    pub role: ComponentRole,       // NEW
    pub weight_group: String,      // NEW — for Phase 3
    pub graph: &'a AiGraph,
    pub mem_plan: &'a MemoryPlan,
    pub phase: LowerPhase,
    pub weights: Option<Vec<u8>>,
}
```

### 2.3 Embed section in `compile_components`

After assembling all sub-archives, build `MetaSection` from the specs and embed it in the pipeline archive wrapper.

### 2.4 LLM pipeline uses MetaSection

`compile_llm_pipeline` sets `role: Prefill` / `role: Decode`, `weight_group: "lm"` for both, and adds a connection for shared KV-cache state. Same section, same code path.

### Verification
- Round-trip test: serialize → deserialize `MetaSection` via rkyv
- Integration: compile LLM pipeline, load archive, extract section, verify 2 descriptors + 1 connection
- Existing tests pass (section format migration is transparent)

---

## Phase 3: Weight Deduplication (hologram-base primitive)

**What**: Components sharing the same source weights (like CALM's encoder/decoder, or LLM's prefill/decode) should not duplicate weight bytes in the archive. Deduplication should be a **hologram-base primitive** following the `hologram-compression` pattern — not ad-hoc logic in the compiler.

### Design principle: follow `hologram-compression`

`hologram-compression` is a self-contained algebraic primitive: `compress(data, mode) → CompressedBlock`, `decompress(block) → data`. It's consumed by `hologram-archive` in the write/read paths transparently. Weight deduplication should follow the same pattern:

- A **hologram-base primitive** (e.g., in `hologram-archive` or a new `hologram-dedup` crate) that provides content-addressable weight block storage
- The **compiler** calls the primitive to register weight blocks and get back references
- The **archive format** stores each unique block once, with a reference table
- The **loader** resolves references transparently on read

### 3.1 Content-addressable weight store (hologram-base)

New primitive in `hologram-archive` (or sibling crate), following `hologram-compression` patterns:

```rust
/// Content-addressable weight storage for pipeline archives.
/// Each unique weight blob is stored once; sub-archives reference by hash.
pub struct WeightStore {
    /// hash → (offset, len) in the deduplicated blob
    index: HashMap<[u8; 32], (u64, u64)>,
    /// Deduplicated weight bytes (concatenated unique blocks)
    blob: Vec<u8>,
}

impl WeightStore {
    /// Register a weight block. Returns a reference ID.
    /// If the block already exists (by content hash), returns the existing ref.
    pub fn insert(&mut self, data: &[u8]) -> WeightRef { ... }

    /// Build the final blob + index for embedding in an archive.
    pub fn build(self) -> (Vec<u8>, WeightIndex) { ... }
}
```

This composes with `hologram-compression`: blocks are compressed *then* deduplicated (or vice versa — the store is agnostic to content).

**File**: hologram base repo — new module in `hologram-archive`

### 3.2 New archive section `SECTION_WEIGHT_DEDUP`

A new section type in `hologram-archive` that maps sub-archive weight references to offsets in a shared weight blob stored at the pipeline level:

```rust
pub const SECTION_WEIGHT_DEDUP: u32 = 4;  // next after SECTION_PIPELINE

pub struct WeightDedup {
    /// Per sub-archive: list of (weight_ref_hash, offset, len) in shared blob
    pub entries: Vec<WeightDedupEntry>,
}
```

### 3.3 Compiler integration

In `compile_components`, instead of cloning weight bytes per sub-archive:
1. Collect all weight blobs into a `WeightStore`
2. Sub-archives that share a `weight_group` get deduplicated automatically (identical content → same ref)
3. Embed the shared blob at the pipeline level with `SECTION_WEIGHT_DEDUP`

The compiler calls the primitive — it doesn't implement dedup logic itself.

### 3.4 Record weight sharing in `MetaSection`

Add to descriptors:

```rust
pub struct ComponentDescriptor {
    // ...existing fields...
    pub weight_source: Option<String>,  // if Some, use weights from named component
}
```

This is a higher-level hint (which component "owns" the canonical weights). The content-addressable store handles the actual dedup.

### Verification
- Compile LLM pipeline, verify archive size is ~50% smaller (weights not duplicated)
- Unit test: 3 components in weight_group "shared", `WeightStore` contains 1 unique block
- Verify `hologram-compression` + `WeightStore` compose correctly (compressed blocks are deduplicated)

---

## Phase 4: Generic Multi-ONNX Compilation

**What**: A generic `ModelSource::MultiOnnx` that imports N ONNX files as components and compiles them via `compile_components`. No per-architecture graph builders — the ONNX graph IS the architecture. Works for CALM, Whisper, Stable Diffusion, BERT, anything.

**Why no `calm.rs`/`bert.rs`**: `arch/llama.rs` exists only because GGUF is a weight format (no graph) — we must construct the graph. ONNX models carry their own graph. Per-architecture builders for ONNX models would overfit and not scale.

### 4.1 `ModelSource::MultiOnnx`

```rust
pub enum ModelSource {
    // ...existing...
    /// Multiple ONNX files forming a multi-component model.
    MultiOnnx {
        components: Vec<ComponentInput>,
    },
}

pub struct ComponentInput {
    pub name: String,              // pipeline key, e.g. "ae.encoder"
    pub path: PathBuf,             // ONNX file
    pub role: ComponentRole,       // Encoder, Decoder, Backbone, etc.
    pub weight_group: String,      // components sharing this share weights
}
```

### 4.2 `compile_multi_onnx()`

```rust
fn compile_multi_onnx(
    &self,
    components: Vec<ComponentInput>,
    connections: Vec<ComponentConnection>,
) -> anyhow::Result<Vec<u8>>
```

For each component:
1. Import ONNX via `hologram_ai_onnx::import_onnx_path`
2. Run `OptPipeline::mvp()` independently
3. Concretize shapes
4. Build `ComponentSpec`

Then call `compile_components()`. Fully generic — zero model-specific code.

### 4.3 Inter-component shape validation

After importing all components, validate that connected ports have compatible tensor shapes:

```rust
fn validate_connections(
    components: &[(String, &AiGraph)],
    connections: &[ComponentConnection],
) -> Vec<ValidationWarning>
```

Emit compile warnings (not errors) if shapes can't be statically checked (some dims may be symbolic).

### 4.4 CLI support

```
hologram-ai compile \
    --component ae.encoder:encoder.onnx:Encoder:autoencoder \
    --component backbone:backbone.onnx:Backbone:backbone \
    --component gen.head:head.onnx:GenerativeHead:backbone \
    --component ae.decoder:decoder.onnx:Decoder:autoencoder \
    -o model.holo
```

Format: `name:path:role:weight_group`. Works for any multi-component model — CALM, Whisper, SD.

### 4.5 Optional: convenience presets

Thin wrappers that expand to `ComponentInput` lists (no architecture-specific compilation code):

```rust
/// Convenience: expand a CALM model spec into component inputs.
pub fn calm_preset(encoder: PathBuf, backbone: PathBuf, head: PathBuf) -> Vec<ComponentInput> {
    vec![
        ComponentInput { name: "ae.encoder".into(), path: encoder, role: Encoder, weight_group: "autoencoder".into() },
        ComponentInput { name: "backbone".into(), path: backbone, role: Backbone, weight_group: "backbone".into() },
        ComponentInput { name: "gen.head".into(), path: head, role: GenerativeHead, weight_group: "backbone".into() },
        // decoder reuses encoder ONNX with shared weight_group
    ]
}
```

### Verification
- Integration: compile 3 small ONNX files → multi-component pipeline archive
- Verify `MetaSection` has correct descriptors and connections
- Verify shared `weight_group` components are deduplicated
- **Real model test**: Compile Whisper (encoder-decoder) from HuggingFace ONNX as a 2-component pipeline — validates the generic path with a real model before CALM exists
- Verify `validate_connections` catches shape mismatches between connected components

---

## Dependency Graph

```
Phase 1 (Generic compilation) ─┬─ Phase 2 (Component metadata)
                                ├─ Phase 3 (Weight dedup — hologram-base PR)
                                └─ Phase 4 (CALM) ← requires 1 + 2 + 3
```

Phases 2 and 3 are independent of each other and can be developed in parallel after Phase 1.
Phase 3 requires a hologram-base PR (WeightStore primitive + SECTION_WEIGHT_DEDUP).

---

## Critical Files

| File | Repo | Changes |
|------|------|---------|
| [compiler.rs](crates/hologram-ai/src/compiler.rs) | hologram-ai | ComponentSpec, compile_components(), compile_one_component() |
| [lower/mod.rs](crates/hologram-ai-common/src/lower/mod.rs) | hologram-ai | LowerPhase::Named variant |
| [sections/llm_meta.rs → meta.rs](crates/hologram-ai-common/src/sections/llm_meta.rs) | hologram-ai | Rename to MetaSection, add ComponentDescriptor, ComponentRole, ComponentConnection |
| [sections/mod.rs](crates/hologram-ai-common/src/sections/mod.rs) | hologram-ai | Rename `pub mod llm_meta` → `pub mod meta` |
| New: weight store module in `hologram-archive` | hologram (base) | WeightStore, WeightRef, SECTION_WEIGHT_DEDUP |
| [pipeline_writer.rs](hologram-archive/src/writer/pipeline_writer.rs) | hologram (base) | Integrate WeightStore into pipeline assembly |
| [compiler.rs](crates/hologram-ai/src/compiler.rs) | hologram-ai | ModelSource::MultiOnnx, ComponentInput, compile_multi_onnx() |

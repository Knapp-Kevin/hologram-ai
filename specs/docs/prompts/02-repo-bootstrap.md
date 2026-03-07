# Prompt: hologram-ai Repo Bootstrap

## Purpose

This prompt bootstraps the `hologram-ai` implementation repository from scratch.
It is suitable for Cursor, Claude Code, or Codex.

Run this prompt in the **empty `hologram-ai` directory** (not in `hologram-architecture`).

---

## Context

You are setting up the `hologram-ai` Rust workspace.

`hologram-ai` is an inference-first AI model compiler and runtime for the
Hologram ecosystem. It imports ONNX, GGUF, and GGML model formats and lowers
them to a `hologram::Graph + hologram::ExecutionSchedule` for execution via
the `hologram::KvExecutor`.

The architecture decisions are documented in `../hologram-architecture/specs/projects/hologram-ai/`.

You must implement exactly the crate layout defined in `crate-layout.md`.
All design decisions are fixed; this prompt is about generating the code, not
re-designing the system.

---

## Step 1: Workspace Cargo.toml

Create `Cargo.toml` at the workspace root:

```toml
[workspace]
resolver = "2"
members = [
    "crates/hologram-ai-quant",
    "crates/hologram-ai-common",
    "crates/hologram-ai-onnx",
    "crates/hologram-ai-gguf",
    "crates/hologram-ai-ggml",
    "crates/hologram-ai",
]

[workspace.package]
version    = "0.1.0"
edition    = "2021"
license    = "MIT OR Apache-2.0"
repository = "https://github.com/uor-framework/hologram-ai"

[workspace.dependencies]
# hologram runtime — root crate re-exports everything (Graph, KvExecutor, etc.)
hologram = { path = "../hologram" }

# workspace-internal (6-crate layout — see crate-layout.md)
hologram-ai-quant   = { path = "crates/hologram-ai-quant" }
hologram-ai-common  = { path = "crates/hologram-ai-common" }
hologram-ai-onnx    = { path = "crates/hologram-ai-onnx" }
hologram-ai-gguf    = { path = "crates/hologram-ai-gguf" }
hologram-ai-ggml    = { path = "crates/hologram-ai-ggml" }
# hologram-ai (root facade) has no workspace alias — import direct

# third-party
bytes        = "1"
half         = { version = "2", features = ["std"] }
smallvec     = { version = "1", features = ["union"] }
thiserror    = "2"
anyhow       = "1"
tracing      = "0.1"
serde        = { version = "1", features = ["derive"] }
serde_json   = "1"
futures      = "0.3"
async-stream = "0.3"
approx       = "0.5"
clap         = { version = "4", features = ["derive"] }
memmap2      = "0.9"
prost        = "0.13"
```

---

## Step 2: Create All Crates

For each crate below, run `cargo new --lib crates/<name>` and then implement
the initial stub as described.

### `crates/hologram-ai-quant`

Foundational quantization crate. No dependency on hologram or AI IR types.
Implement fully in Week 1 before hologram-ai-common.

`src/lib.rs` must expose:

```rust
mod scheme;
mod descriptor;
mod blocks;
mod dequant;
mod quant;

pub use scheme::QuantScheme;
pub use descriptor::QuantDescriptor;
pub use blocks::{Q4_0Block, Q4_1Block, Q5_0Block, Q8_0Block};
pub use dequant::{dequant_q4_0, dequant_q8_0, dequant_tensor};
pub use quant::quant_tensor;
```

Block layouts must match `ggml-quants.h` exactly:
- `Q4_0Block`: `d: f16` (scale), `qs: [u8; 16]` (32 nibbles)
- `Q8_0Block`: `d: f16` (scale), `qs: [i8; 32]`

Software dequant functions:
```rust
pub fn dequant_q4_0(block: &Q4_0Block) -> [f32; 32]
pub fn dequant_q8_0(block: &Q8_0Block) -> [f32; 32]
```

**Cargo.toml:**
```toml
[package]
name = "hologram-ai-quant"
version.workspace = true
edition.workspace = true

[dependencies]
half     = { workspace = true }
smallvec = { workspace = true }
```

---

### `crates/hologram-ai-common`

The compiler core. Implement fully in Week 1 after hologram-ai-quant.

Contains: IR types, optimization passes, memory planner, and lowering.

`src/lib.rs` must expose:

```rust
// IR
pub mod ir;
pub use ir::{
    AiGraph, AiNode, AiOp, AiParam, TensorInfo, Shape, Dim, DType,
    NodeId, TensorId, ImportWarning,
};

// Re-export quant types used in TensorInfo
pub use hologram_ai_quant::{QuantDescriptor, QuantScheme};

// Optimization
pub mod opt;
pub use opt::{OptPipeline, ConstantFolding, DeadNodeElimination, ShapePropagation};

// Memory planner
pub mod mem;
pub use mem::{MemoryPlanner, MemoryPlan};

// Lowering
pub mod lower;
pub use lower::{lower, LoweringOutput, LoweringOptions};
```

`LoweringOutput` is the output of lowering:

```rust
pub struct LoweringOutput {
    pub graph:    hologram::Graph,
    pub schedule: hologram::ExecutionSchedule,
    pub registry: hologram::CustomOpRegistry,
}

pub fn lower(
    graph:    &AiGraph,
    mem_plan: &MemoryPlan,
    opts:     &LoweringOptions,
) -> Result<LoweringOutput>
```

Key `AiGraph` invariants to enforce:
- All `TensorId` refs in nodes exist in `tensor_info`
- All `TensorId` refs in `params` exist in `tensor_info`
- Graph is a DAG (no cycles)
- All `AiParam::Inline` bytes are non-empty

Provide a `validate()` method on `AiGraph` that checks these invariants and
returns `Vec<ValidationError>`.

**Op dispatch table** — `lower()` maps `AiOp` variants to `hologram::GraphOp`:

| AiOp | GraphOp | Notes |
|------|---------|-------|
| `MatMul` (Q4_0 weights) | `GraphOp::MatMulLut4(ConstantId)` | weights in `ConstantStore` |
| `MatMul` (Q8_0 weights) | `GraphOp::MatMulLut8(ConstantId)` | weights in `ConstantStore` |
| `Gelu`, `Relu`, `Silu`, `Tanh` | `GraphOp::Lut(LutOp::…)` | O(1) LUT |
| `Add`, `Mul`, `Neg`, etc. | `GraphOp::Prim(PrimOp::…)` | byte-domain prim |
| `MultiHeadAttention`, `GroupedQueryAttention` | `GraphOp::Custom { id, arity: 3 }` | via `CustomOpRegistry` |
| `RmsNorm`, `LayerNorm` | `GraphOp::Custom { id, arity: 2 }` | via `CustomOpRegistry` |
| `Dequantize` | `GraphOp::Custom { id, arity: 1 }` | explicit per ADR-0004 |
| Weight constants | `GraphOp::Constant(ConstantId)` | native `GraphOp` |

Register custom op handlers in `CustomOpRegistry` at the start of `lower()`.

**Buffer binding** — `MemoryPlan::BufferAlloc` maps to `hologram::BufferArena`
slices provided at session run time.

**Weight storage:**
- Small tensors: `ConstantData::Bytes(bytes)` stored inline in `ConstantStore`
- Large GGUF tensors: `ConstantData::Deferred { … }` + `HoloLoader` for mmap loading

**Cargo.toml:**
```toml
[package]
name = "hologram-ai-common"
version.workspace = true
edition.workspace = true

[dependencies]
hologram-ai-quant = { workspace = true }
hologram          = { workspace = true }
bytes             = { workspace = true }
thiserror         = { workspace = true }
tracing           = { workspace = true }
serde             = { workspace = true }
```

---

### `crates/hologram-ai-onnx`

ONNX model importer.

Public API:
```rust
pub fn import_onnx(bytes: &[u8], opts: OnnxImportOptions) -> Result<AiGraph, ImportError>
pub fn import_onnx_path(path: &Path, opts: OnnxImportOptions) -> Result<AiGraph, ImportError>
```

Parse protobuf using `prost`-generated bindings. No dependency on ONNX Runtime C library.
Supports ONNX opset 13–21.

**Cargo.toml:**
```toml
[package]
name = "hologram-ai-onnx"
version.workspace = true
edition.workspace = true

[dependencies]
hologram-ai-common = { workspace = true }
bytes  = { workspace = true }
prost  = { workspace = true }
```

---

### `crates/hologram-ai-gguf`

GGUF file importer (v1, v2, v3).

Public API:
```rust
pub fn import_gguf(path: &Path, opts: GgufImportOptions) -> Result<AiGraph, ImportError>
pub fn import_gguf_bytes(bytes: &[u8], opts: GgufImportOptions) -> Result<AiGraph, ImportError>
```

Parse header, metadata KV, tensor index. Implement `LlamaArch` recognizer.
Map GGUF quant types to `QuantDescriptor`.

Built-in architecture recognizers: `LlamaArch`, `MistralArch`, `PhiArch`,
`QwenArch`, `GemmaArch`, `Phi3Arch`.

**Cargo.toml:**
```toml
[package]
name = "hologram-ai-gguf"
version.workspace = true
edition.workspace = true

[dependencies]
hologram-ai-common = { workspace = true }
hologram-ai-quant  = { workspace = true }
bytes   = { workspace = true }
memmap2 = { workspace = true }
```

---

### `crates/hologram-ai-ggml`

GGML v1 checkpoint importer (pre-GGUF legacy format).

Public API:
```rust
pub fn import_ggml(path: &Path, opts: GgmlImportOptions) -> Result<AiGraph, ImportError>
```

Hardcoded topology for supported model families (llama v1 format).
This is a migration utility; GGUF is the primary ongoing format.

**Cargo.toml:**
```toml
[package]
name = "hologram-ai-ggml"
version.workspace = true
edition.workspace = true

[dependencies]
hologram-ai-common = { workspace = true }
hologram-ai-quant  = { workspace = true }
bytes = { workspace = true }
```

---

### `crates/hologram-ai`

The single public entry point. Contains session management, streaming token
generation, validation harness, and CLI. Consumers only need this crate.

**Session API:**
```rust
pub struct InferenceSession { ... }
impl InferenceSession {
    pub fn run(&mut self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>>
    pub fn generate(&mut self, tokens: &[u32], opts: &GenerateOptions) -> Result<Vec<u32>>
    pub fn reset_cache(&mut self)
}

pub struct ModelCompiler;
impl ModelCompiler {
    pub fn compile(source: ModelSource, opts: CompileOptions) -> Result<CompiledModel>
}

pub enum ModelSource {
    OnnxBytes(Bytes),
    OnnxPath(PathBuf),
    GgufPath(PathBuf),
    GgmlPath(PathBuf),
    AiGraph(AiGraph),
}
```

`ModelCompiler::compile()` drives the full pipeline:
```
ModelSource → import → AiGraph → OptPipeline → MemoryPlan → lower() → CompiledModel
```

`CompiledModel` holds shared (read-only after compilation) state:
```rust
pub struct CompiledModel {
    pub graph:    Arc<hologram::Graph>,
    pub schedule: Arc<hologram::ExecutionSchedule>,
    pub registry: Arc<hologram::CustomOpRegistry>,
    pub executor: Arc<hologram::KvExecutor>,
}
```

`InferenceSession` holds per-session mutable state:
- `compiled: Arc<CompiledModel>` — shared, read-only
- `kv_cache: hologram::BufferArena` — per-session KV-cache buffers

**Streaming:**
```rust
pub fn stream_tokens(
    session: InferenceSession,
    tokenizer: Box<dyn Tokenizer>,
    prompt: &str,
    opts: GenerateOptions,
) -> TokenStream

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
    fn eos_token_id(&self) -> u32;
}
```

**Public re-exports:**
```rust
pub use hologram_ai_common::{AiGraph, AiOp, TensorInfo, DType};
pub use hologram_ai_quant::{QuantDescriptor, QuantScheme};
pub use crate::session::{InferenceSession, ModelCompiler, CompiledModel};
pub use crate::stream::{TokenStream, Token, Tokenizer, GenerateOptions};
pub use hologram_ai_onnx::import_onnx;
pub use hologram_ai_gguf::import_gguf;
pub use hologram_ai_ggml::import_ggml;
pub use crate::validate::ValidationSuite;
```

**Cargo.toml:**
```toml
[package]
name = "hologram-ai"
version.workspace = true
edition.workspace = true

[dependencies]
hologram            = { workspace = true }
hologram-ai-quant   = { workspace = true }
hologram-ai-common  = { workspace = true }
hologram-ai-onnx    = { workspace = true }
hologram-ai-gguf    = { workspace = true }
hologram-ai-ggml    = { workspace = true }
futures      = { workspace = true }
async-stream = { workspace = true }
clap         = { workspace = true }
anyhow       = { workspace = true }
tracing      = { workspace = true }
serde        = { workspace = true }
serde_json   = { workspace = true }
```

---

## Step 3: CLAUDE.md

Create `CLAUDE.md` at the repo root:

```markdown
# hologram-ai

Rust workspace implementing an AI model compiler and runtime for Hologram.

## Architecture

See `../hologram-architecture/specs/projects/hologram-ai/` for all design docs.

Key ADRs:
- ADR-0002: Canonical AI IR (`AiGraph`) is the single internal representation
- ADR-0003: Format-specific logic is fully contained within importer crates
- ADR-0004: Quantization is first-class; dequantize is explicit in the IR
- ADR-0005: InferenceSession owns Graph + KV-cache; hologram owns execution
- ADR-0006: MVP = GGUF + CPU + single forward pass
- ADR-0007: hologram-ai-lower targets hologram::Graph + hologram::KvExecutor directly

## Crate hierarchy

hologram-ai-quant (quant schemes, block layouts, dequant — no IR deps)
  ← hologram-ai-common (IR + opt passes + mem planner + lowering)
      ← hologram-ai-onnx
      ← hologram-ai-gguf
      ← hologram-ai-ggml
  ← hologram-ai (session + stream + validate + CLI)

## Commands

cargo test --workspace                              # run all tests
cargo test -p hologram-ai-quant                    # test quant primitives
cargo test -p hologram-ai-common                   # test core compiler
cargo test -p hologram-ai-gguf                     # test GGUF importer
cargo run -p hologram-ai -- generate <model.gguf> "<prompt>"

## Non-negotiables

- Never reference hologram subcrates directly — import only via `hologram` root crate
- Never call out to ONNX Runtime, llama.cpp, or any C library at runtime
- Reference runtimes are for validation only (#[ignore] tests)
- Never strip quantization at import time
- All format-specific types stay inside their importer crate
- No AI concepts (attention, tokens, KV-cache) leak into hologram crates
```

---

## Step 4: Test Fixtures

Create `tests/fixtures/` directory structure.

Generate synthetic fixtures via `scripts/gen-fixtures.py`:

```python
# gen-fixtures.py — generates minimal synthetic GGUF for testing
# Uses gguf-py (pip install gguf) for writing GGUF files
```

Commit the following minimal fixtures:
- `tests/fixtures/gguf/tiny-llama-q4_0.gguf` — 2 layers, 64 hidden, Q4_0 weights
- `tests/fixtures/golden/tiny-llama-q4_0/input.json` — `{"input_ids": [1, 2, 3]}`
- `tests/fixtures/golden/tiny-llama-q4_0/output_logits_shape.json` — `[1, 3, 32000]`

The golden fixture does not commit exact logit values for the tiny model —
only shape and dtype. Full numerical golden tests use the real TinyLlama 1.1B
model (downloaded, not committed).

---

## Step 5: CI

Create `.github/workflows/ci.yml`:

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --workspace
      - run: cargo clippy --workspace -- -D warnings
      - run: cargo fmt --check
```

---

## Implementation Order (Week 1)

Implement in this order to ensure the pipeline compiles end-to-end:

1. `hologram-ai-common/src/ir` — full IR type definitions (no logic yet)
2. `hologram-ai-common/src/quant` — `QuantScheme`, `QuantDescriptor`, `dequant_q4_0`, `dequant_q8_0`
3. `hologram-ai-gguf` — parser + `LlamaArch` recognizer + `import_gguf()`
4. `hologram-ai-common/src/opt` — `OptPipeline` skeleton + `ConstantFolding`
5. `hologram-ai-common/src/mem` — `MemoryPlanner` conservative mode
6. `hologram-ai-common/src/lower` — `lower()` with Q4_0 + LLaMA op subset
7. `hologram-ai/src/session` — `ModelCompiler::compile()` + `InferenceSession::run()`
8. Integration test: import → run → check output shape

---

## Verification Prompt (run after each crate)

After implementing each crate, run:
```
cargo test -p <crate-name>
cargo clippy -p <crate-name> -- -D warnings
```

After implementing the full pipeline:
```
cargo test --test integration
```

If the integration test passes (output tensor has correct shape), the MVP pipeline
is working. Proceed to Week 2 tasks.

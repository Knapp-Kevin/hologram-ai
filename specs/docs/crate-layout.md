# hologram-ai: Crate Layout

---

## Workspace Structure

```
hologram-ai/
├── Cargo.toml                     # workspace root
├── CLAUDE.md                      # agent instructions for this repo
├── README.md
├── crates/
│   ├── hologram-ai-ir/                # canonical AI model IR
│   ├── hologram-ai-onnx/              # ONNX importer
│   ├── hologram-ai-gguf/              # GGUF importer
│   ├── hologram-ai-ggml/              # GGML checkpoint importer
│   ├── hologram-ai-opt/               # optimization passes on AiGraph
│   ├── hologram-ai-quant/             # quantization types and dequant ops
│   ├── hologram-ai-mem/               # memory planning (liveness, aliasing)
│   ├── hologram-ai-lower/             # lowering: AiGraph → hologram ExecutionPlan
│   ├── hologram-ai-session/           # inference session lifecycle + KV-cache
│   ├── hologram-ai-stream/            # streaming token generation (autoregressive)
│   ├── hologram-ai-backend/           # backend portability adapters
│   ├── hologram-ai-validate/          # reference runtime validation harness
│   ├── hologram-ai-cli/               # CLI tool (inspect, run, validate models)
│   └── hologram-ai/                   # top-level public facade
├── tests/
│   ├── fixtures/
│   │   ├── onnx/                  # committed ONNX test models (tiny/synthetic)
│   │   ├── gguf/                  # committed GGUF test models (tiny/synthetic)
│   │   └── golden/                # golden tensor outputs for regression
│   └── integration/               # cross-crate integration tests
└── scripts/
    ├── download-test-models.sh    # optional: fetch larger models for full tests
    └── gen-fixtures.py            # generate synthetic test fixtures
```

---

## Crate Responsibilities

### `hologram-ai-ir`

**The single most important crate.** All other crates depend on it.

Defines:
- `AiGraph` — the complete model graph
- `AiNode` — a single operation node
- `AiOp` — the operation enum (all supported ops)
- `AiParam` — stored parameters (inline or lazy via `ArtifactReference`)
- `TensorInfo` — dtype, shape, quant descriptor for a tensor value
- `Shape` — symbolic or concrete tensor dimensions
- `DType` — data type enum (F32, F16, BF16, INT8, INT4, etc.)
- `QuantDescriptor` — quantization scheme metadata
- `NodeId`, `TensorId` — graph identifiers

**Does not depend on** any other `hologram-ai-*` crate.
**Depends on** `hologram-types`, `hologram-artifacts` (for `ArtifactReference`).

---

### `hologram-ai-onnx`

ONNX model importer.

Exposes:
```rust
pub fn import_onnx(bytes: &[u8], opts: OnnxImportOptions) -> Result<AiGraph>
pub fn import_onnx_path(path: &Path, opts: OnnxImportOptions) -> Result<AiGraph>
```

Internals:
- `onnx.proto3` parsed via `prost`-generated types (no C dependency)
- `shape_infer` module: forward shape propagation over ONNX graph
- `op_map` module: maps `op_type` string → `AiOp`
- `opaque_node` fallback for unsupported ops
- `DataResolver` trait for external weight files

Supports ONNX opset 13–21.

**Depends on** `hologram-ai-ir`, `prost`, `bytes`.

---

### `hologram-ai-gguf`

GGUF file importer (all versions: v1, v2, v3).

Exposes:
```rust
pub fn import_gguf(path: &Path, opts: GgufImportOptions) -> Result<AiGraph>
pub fn import_gguf_bytes(bytes: &[u8], opts: GgufImportOptions) -> Result<AiGraph>
```

Internals:
- `parser` module: hand-written binary parser, no external parse deps
- `metadata` module: GGUF KV metadata → `GgufMetadata` struct
- `tensor_index` module: build `TensorIndex` from GGUF tensor list
- `arch_registry` module: `ArchRecognizer` trait + built-in recognizers
- `quant_map` module: GGUF quant types → `QuantDescriptor`

Built-in architecture recognizers: `LlamaArch`, `MistralArch`, `PhiArch`,
`QwenArch`, `GemmaArch`, `Phi3Arch`.

**Depends on** `hologram-ai-ir`, `hologram-ai-quant`, `bytes`, `memmap2`.

---

### `hologram-ai-ggml`

GGML v1 checkpoint importer (pre-GGUF legacy format).

Exposes:
```rust
pub fn import_ggml(path: &Path, opts: GgmlImportOptions) -> Result<AiGraph>
```

Internals: hand-written parser, hardcoded topology for supported families.
Lower priority than GGUF; primarily a migration path for legacy weights.

**Depends on** `hologram-ai-ir`, `hologram-ai-quant`, `bytes`.

---

### `hologram-ai-opt`

Optimization pass pipeline over `AiGraph`.

Exposes:
```rust
pub struct OptPipeline { ... }
impl OptPipeline {
    pub fn default() -> Self
    pub fn fast() -> Self
    pub fn custom(passes: Vec<Box<dyn Pass>>) -> Self
    pub fn run(&self, graph: AiGraph) -> Result<AiGraph>
}

pub trait Pass: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, graph: AiGraph) -> Result<AiGraph>;
}
```

Built-in passes:
- `ConstantFolding` — fold constant expressions to `AiParam`
- `DeadNodeElimination` — remove unreachable nodes
- `ShapePropagation` — propagate concrete shapes
- `AttentionFusion` — fuse QKV + mask + softmax → `MultiHeadAttention`
- `FfnFusion` — fuse gate + up + silu → `FusedSwiGLU`
- `QuantMatMulFusion` — fuse `Dequantize → MatMul` → `QuantizedMatMul`
- `LayerNormFusion` — fuse `Add → Norm` patterns
- `ReshapeElimination` — remove identity reshape/transpose pairs
- `CastElimination` — remove redundant casts

**Depends on** `hologram-ai-ir`, `hologram-ai-quant`.

---

### `hologram-ai-quant`

Quantization type system and dequantization utilities.

Defines:
- `QuantScheme` — all supported quantization schemes
- `QuantDescriptor` — scheme + block metadata
- `dequant_tensor()` — software dequantization for CPU fallback
- `quant_tensor()` — software quantization for test fixtures

Used by all crates that need to create or inspect quantized tensors.

**Depends on** `hologram-ai-ir` (for `DType`), `half` (for f16/bf16).

---

### `hologram-ai-mem`

Memory planner for AI model graphs.

Exposes:
```rust
pub struct MemoryPlanner { ... }
impl MemoryPlanner {
    pub fn new(opts: MemPlanOptions) -> Self
    pub fn plan(&self, graph: &AiGraph) -> Result<MemoryPlan>
}

pub struct MemoryPlan {
    pub buffers: Vec<BufferAlloc>,
    pub tensor_buffer_map: HashMap<TensorId, BufferId>,
    pub kv_cache_layout: Option<KvCacheLayout>,
    pub total_weight_bytes: u64,
    pub total_activation_bytes: u64,
}
```

**Depends on** `hologram-ai-ir`, `hologram-memory` (for buffer types).

---

### `hologram-ai-lower`

Lowers `AiGraph + MemoryPlan` to `hologram::ExecutionPlan`.

Exposes:
```rust
pub fn lower(
    graph: &AiGraph,
    mem_plan: &MemoryPlan,
    opts: &LoweringOptions,
) -> Result<ExecutionPlan>
```

Internals:
- `topo_sort` — topological ordering with memory constraints
- `op_dispatch` — `AiOp` → hologram node mapping table
- `buffer_bind` — bind `MemoryPlan` buffers to `MemoryRegion`
- `param_pack` — pack `AiParam::Lazy` → `ArtifactReference`

**Depends on** `hologram-ai-ir`, `hologram-ai-mem`, `hologram-ai-quant`,
`hologram-planner`, `hologram-memory`, `hologram-artifacts`.

---

### `hologram-ai-session`

Inference session lifecycle.

Exposes:
```rust
pub struct InferenceSession { ... }
impl InferenceSession {
    pub fn new(plan: ExecutionPlan, device: Arc<dyn ExecutionBackend>, opts: SessionOptions) -> Result<Self>
    pub fn run(&mut self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>>
    pub fn generate(&mut self, tokens: &[u32], opts: &GenerateOptions) -> Result<Vec<u32>>
    pub fn reset_cache(&mut self)
}

pub struct ModelCompiler { ... }
impl ModelCompiler {
    pub fn compile(source: ModelSource, opts: CompileOptions) -> Result<CompiledModel>
}
```

`ModelCompiler` runs the full pipeline: import → optimize → plan → lower → wrap in `CompiledModel`.
`CompiledModel` creates `InferenceSession` instances on demand.

**Depends on** `hologram-ai-lower`, `hologram-ai-opt`, `hologram-ai-mem`, `hologram-ai-onnx`,
`hologram-ai-gguf`, `hologram-ai-ggml`, `hologram-runtime`.

---

### `hologram-ai-stream`

Autoregressive streaming token generation.

Exposes:
```rust
pub struct TokenStream { ... }
impl Stream for TokenStream {
    type Item = Result<Token>;
}

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

Sampling strategies: `GreedySampler`, `TopKSampler`, `TopPSampler`,
`TemperatureSampler`, `MinPSampler`.

**Depends on** `hologram-ai-session`, `futures`.

---

### `hologram-ai-backend`

Backend portability adapters and feature flags.

Crate is split into sub-features:

```toml
[features]
default = ["cpu"]
cpu     = ["hologram-cpu"]
metal   = ["hologram-metal"]
cuda    = ["hologram-cuda"]
```

Exports `BackendFactory::create(config: &BackendConfig) -> Arc<dyn ExecutionBackend>`.

**Depends on** `hologram-runtime`, optionally `hologram-cpu`, `hologram-metal`, `hologram-cuda`.

---

### `hologram-ai-validate`

Reference runtime validation harness.

Exposes:
```rust
pub struct ValidationSuite { ... }
impl ValidationSuite {
    pub fn run_onnx_comparison(&self, model: &Path, input: Tensor) -> ValidationReport
    pub fn run_gguf_comparison(&self, model: &Path, prompt: &str) -> ValidationReport
    pub fn compare_tensors(a: &Tensor, b: &Tensor, tol: Tolerance) -> TensorComparison
}
```

Can run reference runtimes as subprocesses (ONNX Runtime CLI, llama.cpp CLI)
or via optional feature-gated C library bindings.

**Depends on** `hologram-ai-session`, `hologram-ai-onnx`, `hologram-ai-gguf`, `approx`.

---

### `hologram-ai-cli`

CLI tool for model inspection, running, and validation.

Commands:
```
hologram-ai inspect <model>          # print model structure and metadata
hologram-ai run <model> --input ...  # run single inference
hologram-ai generate <model> <prompt># autoregressive generation
hologram-ai validate <model>         # compare against reference runtime
hologram-ai lower <model> --emit-plan# dump the lowered ExecutionPlan
```

**Depends on** `hologram-ai`, `clap`.

---

### `hologram-ai`

Public facade crate. Re-exports the most commonly needed types.

```rust
pub use holo_ai_ir::{AiGraph, AiOp, TensorInfo, DType, QuantDescriptor};
pub use holo_ai_session::{InferenceSession, ModelCompiler, CompiledModel};
pub use holo_ai_stream::{TokenStream, Token, Tokenizer, GenerateOptions};
pub use holo_ai_onnx::import_onnx;
pub use holo_ai_gguf::import_gguf;
pub use holo_ai_ggml::import_ggml;
pub use holo_ai_validate::ValidationSuite;
```

---

## Crate Dependency Matrix

```
hologram-ai-ir           (no hologram-ai-* deps)
hologram-ai-quant        → hologram-ai-ir
hologram-ai-onnx         → hologram-ai-ir
hologram-ai-gguf         → hologram-ai-ir, hologram-ai-quant
hologram-ai-ggml         → hologram-ai-ir, hologram-ai-quant
hologram-ai-opt          → hologram-ai-ir, hologram-ai-quant
hologram-ai-mem          → hologram-ai-ir
hologram-ai-lower        → hologram-ai-ir, hologram-ai-mem, hologram-ai-quant
hologram-ai-session      → hologram-ai-lower, hologram-ai-opt, hologram-ai-mem,
                        hologram-ai-onnx, hologram-ai-gguf, hologram-ai-ggml
hologram-ai-stream       → hologram-ai-session
hologram-ai-backend      → hologram-runtime
hologram-ai-validate     → hologram-ai-session, hologram-ai-onnx, hologram-ai-gguf
hologram-ai-cli          → hologram-ai
hologram-ai              → hologram-ai-session, hologram-ai-stream, hologram-ai-onnx,
                        hologram-ai-gguf, hologram-ai-ggml, hologram-ai-validate
```

---

## Naming Rationale

The `hologram-ai-` prefix (not `hologram-ai-`) is used for internal crates to:
- Keep published crate names short
- Match the workspace package naming convention in `hologram` (`hologram-*`)
- Distinguish the published workspace crates from the repo name

The top-level facade is `hologram-ai` (publishable) and `hologram-ai` is the repo name.

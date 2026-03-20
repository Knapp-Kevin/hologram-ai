# Research Report: hologram-ai Full Architecture

- Date: 2026-03-06
- Status: Accepted as planning baseline
- Author: Architecture

---

## 1. Purpose

This report defines the full architecture for `hologram-ai`: a Rust-first,
inference-first consumer repository that imports ONNX, GGUF, and GGML models,
lowers them to the Hologram-native execution graph, and runs them via the
`hologram` runtime. It serves as the design baseline from which ADRs, plans,
sprints, and the repo-bootstrap prompt are derived.

---

## 2. Position in the Ecosystem

```
┌─────────────────────────────────────────┐
│           hologram-ai                   │
│                                         │
│  importers → hologram-ai-ir → lowering      │
│  → hologram ExecutionPlan               │
│  → inference session + KV-cache         │
│  → streaming token generation           │
└─────────────┬───────────────────────────┘
              │ depends on
              ▼
┌─────────────────────────────────────────┐
│           hologram                      │
│  graph, execution plan, memory, runtime │
└─────────────────────────────────────────┘
```

`hologram-ai` is a strict **consumer** of `hologram`. It adds:
- AI-specific model importers
- Canonical AI model IR
- Optimization and lowering passes
- Inference session lifecycle
- KV-cache, quantization, and streaming token generation

`hologram` remains sandbox-agnostic and AI-agnostic. No AI concepts leak into it.

---

## 3. Core Design Principles

1. **One canonical IR.** All three format importers (ONNX, GGUF, GGML) emit the
   same `AiGraph`. No format-specific execution paths reach the runtime.

2. **Quantization is first-class.** Quant descriptors are carried through IR,
   optimizations, lowering, and memory planning. They are never stripped or
   silently upcast at import time.

3. **Lowering is explicit.** The `hologram-ai-lower` crate maps `AiGraph` to
   `hologram::ExecutionPlan`. The boundary is a typed, versioned interface.

4. **Memory planning is pre-runtime.** Tensor lifetimes, buffer aliasing, and
   KV-cache sizing are resolved at compile time, not lazily during inference.

5. **Sessions own their state.** `InferenceSession` owns the execution plan,
   KV-cache, and device binding. Multiple concurrent sessions are safe.

6. **Backend portability via traits.** The `ExecutionBackend` trait from
   `hologram` is the only required runtime coupling. CPU, GPU, and future
   Hologram-native accelerators are interchangeable.

7. **Validation by reference.** `hologram-ai-validate` compares outputs against
   reference runtimes (ONNX Runtime, llama.cpp) for correctness testing.

---

## 4. Crate Layout

```
hologram-ai/
├── Cargo.toml                     # workspace root
├── CLAUDE.md                      # agent instructions for this repo
├── crates/
│   ├── hologram-ai-ir/                # canonical AI model IR
│   ├── hologram-ai-onnx/              # ONNX importer
│   ├── hologram-ai-gguf/              # GGUF importer
│   ├── hologram-ai-ggml/              # GGML checkpoint importer
│   ├── hologram-ai-opt/               # optimization passes on hologram-ai-ir
│   ├── hologram-ai-quant/             # quantization descriptors and dequant ops
│   ├── hologram-ai-mem/               # memory planning (lifetimes, aliasing)
│   ├── hologram-ai-lower/             # lowering: hologram-ai-ir → hologram ExecutionPlan
│   ├── hologram-ai-session/           # inference session lifecycle + KV-cache
│   ├── hologram-ai-stream/            # streaming token generation (autoregressive)
│   ├── hologram-ai-backend/           # backend portability adapters
│   ├── hologram-ai-validate/          # reference runtime validation harness
│   └── hologram-ai/                   # top-level public facade
└── tests/
    ├── onnx/                      # ONNX model fixtures
    ├── gguf/                      # GGUF model fixtures
    └── golden/                    # golden output files for validation
```

### Dependency Graph (internal crates)

```
hologram-ai (facade)
├── hologram-ai-session
│   ├── hologram-ai-lower
│   │   ├── hologram-ai-ir
│   │   ├── hologram-ai-opt
│   │   │   └── hologram-ai-ir
│   │   ├── hologram-ai-quant
│   │   │   └── hologram-ai-ir
│   │   └── hologram-ai-mem
│   │       └── hologram-ai-ir
│   └── hologram-ai-backend
├── hologram-ai-stream
│   └── hologram-ai-session
├── hologram-ai-onnx  → hologram-ai-ir
├── hologram-ai-gguf  → hologram-ai-ir
└── hologram-ai-ggml  → hologram-ai-ir
```

All crates depend on `hologram` types but do not depend on each other
laterally, except through the dependency graph above.

---

## 5. Canonical AI Model IR (`hologram-ai-ir`)

### Design

The IR is a typed directed acyclic graph of operations over typed tensors.

```rust
/// A complete compiled AI model graph.
pub struct AiGraph {
    pub name: String,
    pub nodes: Vec<AiNode>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub params: HashMap<TensorId, AiParam>,
    pub tensor_info: HashMap<TensorId, TensorInfo>,
}

/// A single operation node.
pub struct AiNode {
    pub id: NodeId,
    pub op: AiOp,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub attrs: NodeAttrs,
}

/// Tensor type, shape, and quantization descriptor.
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Shape,          // symbolic or concrete dims
    pub quant: QuantDescriptor,
}

/// A stored parameter (weight, bias, embedding table).
pub struct AiParam {
    pub info: TensorInfo,
    pub storage: ParamStorage,
}

pub enum ParamStorage {
    Inline(Bytes),
    Lazy(ArtifactReference),   // hologram artifact reference
}
```

### Operations

`AiOp` covers the operation set required to express transformer-class models
plus basic feedforward networks. Core operations:

```
Tensor math:   MatMul, BatchMatMul, Einsum
Activation:    Relu, Gelu, GeluApprox, Silu, Tanh, Sigmoid, Softmax
Normalization: LayerNorm, RmsNorm, GroupNorm, BatchNorm
Attention:     MultiHeadAttention, GroupedQueryAttention, FlashAttentionHint
Shape:         Reshape, Transpose, Concat, Split, Slice, Gather, Scatter
Elem-wise:     Add, Sub, Mul, Div, Pow, Sqrt, Exp, Log, Neg, Abs, Clamp
Reduction:     ReduceSum, ReduceMean, ReduceMax, ArgMax
Quant:         Quantize, Dequantize, QuantizedMatMul
Control:       Identity, Cast, Constant
```

The `FlashAttentionHint` op is a logical marker. The lowering pass decides
whether to emit a fused attention kernel or decompose it.

### Shape Representation

```rust
pub enum Dim {
    Concrete(u64),
    Symbolic(String),         // e.g. "batch", "seq_len"
    Dynamic,
}

pub type Shape = SmallVec<[Dim; 6]>;
```

Symbolic dims allow shape inference to propagate through the graph without
requiring concrete batch sizes at import time.

---

## 6. Format Importers

### 6.1 ONNX Importer (`hologram-ai-onnx`)

**Input:** ONNX protobuf bytes (`.onnx` file or byte slice)

**Strategy:**
- Parse via `prost`-generated types from `onnx.proto3`
- Walk the `GraphProto` node list
- Map each `NodeProto.op_type` to an `AiOp`
- Emit `TensorProto` weights as `AiParam::Inline` or `AiParam::Lazy` depending on size threshold
- Run shape inference via `hologram-ai-onnx::shape_infer`
- Unknown ops map to `AiOp::Opaque` with the raw protobuf preserved

**Key decisions:**
- Import opset 13–21 (cover the practical model ecosystem as of 2026)
- `ConstantOfShape`, `Shape`, `Gather` on axes → fold into `AiParam` at import time
- External data files (ONNX large model format) resolved via a `DataResolver` trait
- No runtime dependency on `onnxruntime` C library

```rust
pub fn import_onnx(bytes: &[u8], opts: OnnxImportOptions) -> Result<AiGraph>
pub fn import_onnx_path(path: &Path, opts: OnnxImportOptions) -> Result<AiGraph>
```

### 6.2 GGUF Importer (`hologram-ai-gguf`)

**Input:** GGUF v1/v2/v3 file (llama.cpp format)

**Strategy:**
- Parse GGUF header, metadata kv, and tensor index via hand-written parser
- Metadata KV → `AiGraph::metadata` (arch, context length, embedding dim, etc.)
- Reconstruct model graph from architecture metadata (not from an explicit graph stored in GGUF - GGUF stores weights, not graph topology)
- Tensor data stored as `AiParam::Lazy(ArtifactReference)` backed by mmap
- Quant types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, etc.) → `QuantDescriptor`
- Graph topology is inferred from arch metadata (llama, mistral, phi, qwen, etc.)

```rust
pub fn import_gguf(path: &Path, opts: GgufImportOptions) -> Result<AiGraph>
pub fn import_gguf_bytes(bytes: &[u8], opts: GgufImportOptions) -> Result<AiGraph>
```

**Architecture recognizer registry:**

```rust
pub trait ArchRecognizer: Send + Sync {
    fn arch_name(&self) -> &str;
    fn matches(&self, metadata: &GgufMetadata) -> bool;
    fn build_graph(&self, metadata: &GgufMetadata, tensors: &TensorIndex) -> Result<AiGraph>;
}
```

Built-in recognizers: `LlamaArch`, `MistralArch`, `PhiArch`, `QwenArch`, `GemmaArch`.
Extensible via `GgufImportOptions::extra_recognizers`.

### 6.3 GGML Importer (`hologram-ai-ggml`)

**Input:** Original GGML checkpoint format (`.bin` files, pre-GGUF era)

**Strategy:**
- Parse file magic + version header
- Extract hyperparameters, vocab, and weight tensors
- Map to `AiGraph` using hardcoded topology for supported model families
- Less extensible than GGUF due to format limitations; supports llama v1 format
- Primarily a migration path from legacy checkpoints

```rust
pub fn import_ggml(path: &Path, opts: GgmlImportOptions) -> Result<AiGraph>
```

---

## 7. Quantization Representation (`hologram-ai-quant`)

Quantization descriptors are carried on every `TensorInfo`:

```rust
pub struct QuantDescriptor {
    pub scheme: QuantScheme,
}

pub enum QuantScheme {
    None,                          // f32 / f16 / bf16 native
    // GGUF k-quant families
    Q4_0, Q4_1,
    Q5_0, Q5_1,
    Q8_0,
    Q2_K, Q3_K_S, Q3_K_M, Q3_K_L,
    Q4_K_S, Q4_K_M,
    Q5_K_S, Q5_K_M,
    Q6_K,
    IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS,
    // ONNX / standard quant
    Int8Sym { scale: f32 },
    Int8Asym { scale: f32, zero_point: i32 },
    Int4Block { block_size: u32 },
    Float8E4M3,
    Float8E5M2,
    // Compute-time quant
    Dynamic { target: Box<QuantScheme> },
}
```

Operations on quantized tensors:

```rust
// In hologram-ai-ir AiOp:
AiOp::Quantize { scheme: QuantScheme }
AiOp::Dequantize
AiOp::QuantizedMatMul { lhs_scheme: QuantScheme, rhs_scheme: QuantScheme }
```

The optimization pass (`hologram-ai-opt`) may fuse `Dequantize → MatMul` into
`QuantizedMatMul` when a backend supports it.

---

## 8. Optimization Passes (`hologram-ai-opt`)

Passes run on `AiGraph` before lowering. Each pass is a pure `Fn(&AiGraph) -> AiGraph`.

### Pass registry

```
constant_folding          Fold Constant + Shape ops into inline params
dead_node_elimination     Remove unreachable nodes
shape_propagation         Propagate concrete shapes through graph
attention_fusion          Fuse QKV matmuls + attention mask + softmax → MultiHeadAttention
ffn_fusion                Fuse gate + up-proj + SiLU → FusedSwiGLU
quant_matmul_fusion       Fuse Dequantize → MatMul → Quantize into QuantizedMatMul
layer_norm_fusion         Fuse Add → Layernorm patterns
reshape_elimination       Remove no-op Reshape/Transpose pairs
cast_elimination          Remove redundant Cast ops
```

Pass pipeline is configurable:

```rust
pub struct OptPipeline {
    passes: Vec<Box<dyn Pass>>,
}

impl OptPipeline {
    pub fn default() -> Self  // standard pass order
    pub fn fast() -> Self     // only critical passes
    pub fn run(&self, graph: AiGraph) -> Result<AiGraph>
}
```

---

## 9. Memory Planning (`hologram-ai-mem`)

Memory planning resolves buffer allocation before lowering to the execution plan.

### Steps

1. **Tensor lifetime analysis** — for each tensor, determine the first and last
   node that reads/writes it (liveness intervals)
2. **Buffer alias analysis** — identify tensors that can share buffers
   (non-overlapping lifetimes, compatible dtype + alignment)
3. **In-place op detection** — identify ops that can overwrite their input
   buffer (e.g. activation functions, in-place add)
4. **KV-cache pre-allocation** — given max_seq_len and model config, compute
   required KV buffer size per layer and total
5. **Weight layout decisions** — interleaved vs. planar for quantized weights
6. **Alignment annotation** — annotate each buffer with required alignment
   (16B for SIMD, 128B for potential DMA, etc.)

### Output

```rust
pub struct MemoryPlan {
    pub buffers: Vec<BufferAlloc>,
    pub tensor_buffer_map: HashMap<TensorId, BufferId>,
    pub kv_cache_layout: Option<KvCacheLayout>,
    pub total_weight_bytes: u64,
    pub total_activation_bytes: u64,
}

pub struct KvCacheLayout {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub dtype: DType,
    pub bytes_per_layer: u64,
    pub total_bytes: u64,
}
```

---

## 10. Lowering (`hologram-ai-lower`)

Lowering transforms `AiGraph + MemoryPlan` into `hologram::ExecutionPlan`.

### Lowering pipeline

```
AiGraph
  + MemoryPlan
  + LoweringOptions
        │
        ▼
   [node ordering]   topological sort with memory constraints
        │
   [op dispatch]     map AiOp → hologram execution nodes
        │
   [buffer binding]  bind MemoryPlan buffers to hologram MemoryRegion
        │
   [param packing]   AiParam::Lazy → ArtifactReference (hologram artifact system)
        │
        ▼
hologram::ExecutionPlan
```

### Op dispatch table (partial)

| AiOp | hologram node |
|------|--------------|
| MatMul | `gemm_f32` / `gemm_f16` / `gemm_q4` |
| QuantizedMatMul | `qgemm_{lhs}_{rhs}` |
| MultiHeadAttention | `mha_fused` or decomposed GEMM sequence |
| GroupedQueryAttention | `gqa_fused` or decomposed |
| LayerNorm | `layer_norm_f32` / `rms_norm_f32` |
| Softmax | `softmax_f32` |
| Gelu | `gelu_f32` / `gelu_approx_f32` |
| Silu | `silu_f32` |
| Reshape | memory-only, no compute node |
| Cast | `cast_{src}_{dst}` |
| Quantize / Dequantize | `quantize_{scheme}` / `dequantize_{scheme}` |

### Public API

```rust
pub fn lower(
    graph: &AiGraph,
    mem_plan: &MemoryPlan,
    opts: &LoweringOptions,
) -> Result<ExecutionPlan>
```

---

## 11. Inference Session (`hologram-ai-session`)

### Session lifecycle

```
ModelSource (path / bytes)
   │
   ├─ import_*() → AiGraph
   │
   ├─ OptPipeline::run() → AiGraph (optimized)
   │
   ├─ MemoryPlanner::plan() → MemoryPlan
   │
   ├─ lower() → ExecutionPlan
   │
   └─ InferenceSession::new(plan, device, opts)
            │
            ▼
      session.run(inputs) → outputs        (single-shot)
      session.generate(prompt, opts)       (autoregressive)
      session.stream(prompt, opts)         (streaming token iterator)
```

### Types

```rust
pub struct InferenceSession {
    plan: Arc<ExecutionPlan>,
    kv_cache: Option<KvCache>,
    device: Arc<dyn ExecutionBackend>,
    opts: SessionOptions,
}

pub struct SessionOptions {
    pub max_seq_len: Option<usize>,
    pub kv_cache_dtype: DType,
    pub threads: Option<usize>,
    pub seed: Option<u64>,
}

pub struct KvCache {
    pub layers: Vec<KvLayer>,
    pub layout: KvCacheLayout,
    pub present_len: usize,    // tokens currently in cache
}

pub struct KvLayer {
    pub k: BufferView,         // hologram BufferView
    pub v: BufferView,
}
```

### Session.run() (single inference)

```rust
impl InferenceSession {
    pub fn run(
        &mut self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>>
}
```

---

## 12. Streaming Token Generation (`hologram-ai-stream`)

Streaming autoregressive decoding loop:

```rust
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub stop_sequences: Vec<Vec<u32>>,
    pub seed: Option<u64>,
}

pub struct TokenStream {
    session: InferenceSession,
    tokenizer: Box<dyn Tokenizer>,
    context: Vec<u32>,            // current token context
    opts: GenerateOptions,
    step: usize,
}

impl Stream for TokenStream {
    type Item = Result<Token>;
    // polls session.run() for next token, samples, appends to context
}

pub struct Token {
    pub id: u32,
    pub text: String,
    pub logprob: Option<f32>,
    pub is_stop: bool,
}
```

Sampling strategies live in `hologram-ai-stream::sampler`:
- `GreedySampler`
- `TopKSampler`
- `TopPSampler` (nucleus)
- `TemperatureSampler`
- `MinPSampler`

Prefill (prompt encoding) and decode (token generation) phases are explicitly
separate within the session to allow different batching strategies.

---

## 13. Backend Portability (`hologram-ai-backend`)

`hologram-ai-backend` provides thin adapter crates that bind `hologram`'s
`ExecutionBackend` trait to concrete execution targets.

```
hologram-ai-backend-cpu    pure Rust + SIMD, via `hologram-cpu` backend
hologram-ai-backend-metal  Apple Metal, via `hologram-metal` backend (future)
hologram-ai-backend-cuda   CUDA, via `hologram-cuda` backend (future)
```

The backends are feature-gated:

```toml
[features]
default  = ["cpu"]
cpu      = ["hologram-cpu"]
metal    = ["hologram-metal"]
cuda     = ["hologram-cuda"]
```

`InferenceSession` takes `Arc<dyn ExecutionBackend>` and is backend-agnostic.

---

## 14. Validation Strategy (`hologram-ai-validate`)

### Reference runtime comparison

Two validation approaches:

**Numerical validation:**
- Run same model with same input through `hologram-ai` and reference runtime
- Compute max absolute error, mean absolute error, cosine similarity
- Assert tolerances (f32: MAE < 1e-5, f16: MAE < 1e-3)
- For quantized: MAE < quantization noise floor

**Token generation validation:**
- For language models: given same prompt, assert top-k tokens match reference
- Accept small rank differences due to floating-point non-determinism
- Hard fail on semantic divergence (different predicted next token for greedy)

### Reference runtimes

| Format | Reference |
|--------|-----------|
| ONNX | ONNX Runtime (via `ort` crate or subprocess) |
| GGUF | llama.cpp CLI (subprocess) |
| Both | HuggingFace Transformers (Python subprocess, optional) |

### Test model fixtures

Small models committed or downloaded at test time:
- `tests/onnx/gpt2-small.onnx` (~500MB, optional CI download)
- `tests/onnx/bert-base.onnx` (~400MB, optional)
- `tests/gguf/tinyllama-1.1b-q4.gguf` (~700MB, optional)
- `tests/onnx/matmul-f32.onnx` (tiny synthetic, committed)
- `tests/gguf/tiny-q4_0.gguf` (minimal synthetic, generated by fixture script)

---

## 15. hologram ↔ hologram-ai Contract

### What hologram-ai consumes from hologram

| Type | Source crate |
|------|-------------|
| `ExecutionPlan` | `hologram-planner` |
| `ExecutionBackend` trait | `hologram-runtime` |
| `MemoryRegion` | `hologram-memory` |
| `BufferView` | `hologram-memory` |
| `ArtifactReference` | `hologram-artifacts` |
| `NodeId`, `TensorId` (reused types) | `hologram-types` |

### What hologram-ai does NOT touch

- Sandbox (process/WASM/microVM) — belongs to `hologram-sandbox`
- Graph planner internals — `hologram-planner` is a black box to `hologram-ai`
- Execution lifecycle beyond submitting an `ExecutionPlan` and waiting for outputs

---

## 16. MVP Scope

The MVP delivers a working end-to-end path for a single model family.

### MVP constraints

- One model: TinyLlama 1.1B (GGUF Q4_0)
- CPU backend only
- No streaming (single-pass run())
- No KV-cache (single prompt, no multi-turn)
- No ONNX importer (deferred to Week 2)
- No validation harness (deferred to Week 3)

### MVP deliverables

1. `hologram-ai-ir` with core ops and quant descriptor
2. `hologram-ai-gguf` importing TinyLlama Q4_0
3. `hologram-ai-opt` with constant folding + attention fusion
4. `hologram-ai-mem` with tensor lifetime analysis
5. `hologram-ai-lower` mapping to `hologram::ExecutionPlan`
6. `hologram-ai-session` with `run()` single-pass
7. Integration test: load TinyLlama → run one forward pass → check output shape

---

## 17. Open Questions

1. **hologram kernel availability** — which GEMM/attention kernels exist in
   `hologram` today? Lowering completeness depends on this.
2. **ArtifactReference mmap semantics** — does `hologram-artifacts` support
   zero-copy mmap of large weight files? Required for GGUF lazy loading.
3. **Tokenizer scope** — should `hologram-ai-stream` bundle a tokenizer or accept
   an external `Box<dyn Tokenizer>`? Recommend external to avoid dependency bloat.
4. **Multi-GPU sharding** — out of scope for MVP but the MemoryPlan and session
   model should not preclude it.
5. **LoRA / adapter layers** — GGUF supports adapter metadata. Deferred post-MVP.

---

## 18. Dependency Choices

| Need | Crate | Notes |
|------|-------|-------|
| ONNX protobuf parse | `prost` + generated | no C deps |
| GGUF file parse | hand-written | simple format, no deps |
| GGML parse | hand-written | legacy format |
| Async streams | `futures` + `async-stream` | for TokenStream |
| Numeric types | `half` | f16/bf16 |
| Tensor storage | `bytes` | zero-copy byte slices |
| Testing | `approx` | float equality assertions |
| Tracing | `tracing` | structured diagnostics |
| Serialization | `serde` + `serde_json` | session config, test fixtures |

---

## 19. Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          hologram-ai                                │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                         │
│  │hologram-ai-  │  │hologram-ai-  │  │hologram-ai-  │  ← importers            │
│  │onnx      │  │gguf      │  │ggml      │                         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                         │
│       └──────────────┴──────────────┘                               │
│                      │ AiGraph                                      │
│               ┌──────▼───────┐                                      │
│               │ hologram-ai-ir   │  ← canonical IR                     │
│               └──────┬───────┘                                      │
│                      │                                              │
│               ┌──────▼───────┐                                      │
│               │ hologram-ai-opt  │  ← optimization passes              │
│               └──────┬───────┘                                      │
│                      │                                              │
│         ┌────────────┼────────────┐                                 │
│         │            │            │                                 │
│  ┌──────▼──┐  ┌──────▼──┐  ┌─────▼──────┐                         │
│  │hologram-ai- │  │hologram-ai- │  │hologram-ai-    │                         │
│  │quant    │  │mem      │  │lower       │  ← compile phase        │
│  └─────────┘  └─────────┘  └─────┬──────┘                         │
│                                   │ ExecutionPlan                   │
│                            ┌──────▼───────┐                        │
│                            │hologram-ai-      │  ← runtime phase       │
│                            │session       │                        │
│                            └──────┬───────┘                        │
│                                   │                                 │
│                            ┌──────▼───────┐                        │
│                            │hologram-ai-      │  ← generation          │
│                            │stream        │                        │
│                            └──────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
                      │ depends on
                      ▼
              ┌────────────────┐
              │   hologram     │
              │ (runtime core) │
              └────────────────┘
```

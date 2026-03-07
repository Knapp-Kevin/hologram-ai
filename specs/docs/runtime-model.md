# hologram-ai: Runtime Model

---

## Overview

The runtime model defines how compiled models are loaded, how inference sessions
are created and managed, how weights are loaded lazily, and how autoregressive
token generation is driven.

---

## Model Compilation

`ModelCompiler` drives the full compile pipeline:

```rust
pub struct ModelCompiler {
    opt_pipeline: OptPipeline,
    mem_planner: MemoryPlanner,
    lower_opts: LoweringOptions,
    device: Arc<dyn ExecutionBackend>,
}

impl ModelCompiler {
    pub fn compile(&self, source: ModelSource) -> Result<CompiledModel>
}

pub enum ModelSource {
    OnnxBytes(Bytes),
    OnnxPath(PathBuf),
    GgufPath(PathBuf),
    GgmlPath(PathBuf),
    AiGraph(AiGraph),    // already-imported graph
}
```

**Compile pipeline steps:**

```
ModelSource
  → import_*(...)                  → AiGraph (raw)
  → opt_pipeline.run(graph)        → AiGraph (optimized)
  → mem_planner.plan(&graph)       → MemoryPlan
  → lower(&graph, &plan, &opts)    → ExecutionPlan
  → CompiledModel { plan, layout } → ready for sessions
```

---

## `CompiledModel`

A `CompiledModel` is a reusable, shareable compiled artifact:

```rust
pub struct CompiledModel {
    plan: Arc<ExecutionPlan>,
    kv_layout: Option<KvCacheLayout>,
    input_metadata: Vec<TensorMeta>,
    output_metadata: Vec<TensorMeta>,
    metadata: ModelMetadata,    // arch info, context len, vocab size, etc.
}

impl CompiledModel {
    pub fn session(&self, opts: SessionOptions) -> Result<InferenceSession>
    pub fn metadata(&self) -> &ModelMetadata
}
```

Multiple `InferenceSession` instances can be created from a single
`CompiledModel` (they share the plan; each has its own KV-cache).

---

## Weight Loading

Weights are loaded via `ArtifactReference` from the hologram artifact system.

### Strategies

**Eager (full load):** All weights materialized at `CompiledModel` creation.
Preferred for small models or when the full model fits in RAM.

**Lazy (mmap):** Weights are memory-mapped from file. Pages are faulted in on
first access. Preferred for large GGUF models (70B+).

`ParamStorage` in `AiParam` controls this:
```rust
pub enum ParamStorage {
    Inline(Bytes),                    // fully in memory
    Lazy(ArtifactReference),          // via hologram artifact system (mmap etc.)
}
```

---

## Inference Session

```rust
pub struct InferenceSession {
    model: Arc<CompiledModel>,
    kv_cache: Option<KvCache>,
    device: Arc<dyn ExecutionBackend>,
    opts: SessionOptions,
}

pub struct SessionOptions {
    pub max_seq_len: Option<usize>,   // cap for KV-cache allocation
    pub kv_cache_dtype: DType,        // f16 recommended for memory efficiency
    pub threads: Option<usize>,       // CPU thread count
    pub seed: Option<u64>,            // for sampling reproducibility
}
```

### `session.run()` — Single Forward Pass

```rust
impl InferenceSession {
    pub fn run(
        &mut self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>>
}
```

Steps:
1. Validate input tensor shapes against `CompiledModel::input_metadata`
2. If KV-cache is present, inject cache pointers and current offset as plan inputs
3. Submit `ExecutionPlan` to `ExecutionBackend`
4. Wait for completion
5. Extract output tensors from backend
6. If KV-cache present, update `kv_cache.present_len`
7. Return output tensors

---

## KV-Cache

### Design

The KV-cache stores the key and value tensors from all attention layers for
all previously processed tokens. This avoids recomputing them on each decode step.

```rust
pub struct KvCache {
    pub layers: Vec<KvLayer>,
    pub layout: KvCacheLayout,
    pub present_len: usize,         // tokens currently in cache
    pub max_seq_len: usize,
}

pub struct KvLayer {
    pub k: Arc<BufferView>,         // [max_seq_len, n_kv_heads, head_dim]
    pub v: Arc<BufferView>,         // [max_seq_len, n_kv_heads, head_dim]
}

pub struct KvCacheLayout {
    pub layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub dtype: DType,
    pub bytes_per_layer: u64,
    pub total_bytes: u64,
}
```

### KV-cache in the execution plan

Two special node types in the lowered plan:

`KvSlotWrite(layer, seq_offset)` — writes the current token's K/V projections
into the cache at position `seq_offset`.

`KvSlotRead(layer, seq_len)` — reads all cached K/V up to `seq_len` for attention computation.

The session passes `present_len` as a runtime input to the plan on each call.

### Prefill vs. decode modes

**Prefill (prompt phase):**
- Process all prompt tokens in one forward pass (batch = 1, seq = prompt_len)
- Write all K/V to cache at once
- More efficient per-token than decode phase

**Decode (generation phase):**
- Process one token per forward pass
- Read full cache + write one new slot
- Autoregressive loop

The `InferenceSession` switches between these modes automatically based on
whether `present_len == 0` (prefill) or `> 0` (decode).

---

## Token Generation

### `session.generate()` — Blocking generation

```rust
impl InferenceSession {
    pub fn generate(
        &mut self,
        token_ids: &[u32],           // prompt tokens
        opts: &GenerateOptions,
    ) -> Result<Vec<u32>>            // generated tokens (not including prompt)
}
```

### `stream_tokens()` — Async streaming

```rust
pub fn stream_tokens(
    session: InferenceSession,
    tokenizer: Box<dyn Tokenizer>,
    prompt: &str,
    opts: GenerateOptions,
) -> TokenStream

impl Stream for TokenStream {
    type Item = Result<Token>;
}
```

### `GenerateOptions`

```rust
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub temperature: f32,        // 1.0 = no scaling
    pub top_p: f32,              // 1.0 = disabled
    pub top_k: usize,            // 0 = disabled
    pub min_p: f32,              // 0.0 = disabled
    pub repetition_penalty: f32, // 1.0 = disabled
    pub stop_sequences: Vec<Vec<u32>>,
    pub seed: Option<u64>,
}
```

### Sampling pipeline

```
logits (f32 vector of vocab_size)
  → apply temperature scaling
  → apply repetition penalty (if enabled)
  → top-k filter (if k > 0)
  → top-p filter (if p < 1.0)
  → min-p filter (if p > 0.0)
  → softmax
  → sample from distribution (or argmax for greedy)
  → check stop sequences
  → emit Token
```

---

## Session Lifecycle

```
CompiledModel::session(opts)          → InferenceSession (fresh, empty cache)
  │
  ├── run(inputs)                     → single forward pass
  │
  ├── generate(prompt_tokens, opts)   → blocking full generation
  │
  ├── stream_tokens(...)              → async Token stream
  │
  ├── reset_cache()                   → clear KV-cache, reuse session
  │
  └── drop                           → buffers released
```

Multiple concurrent sessions from the same `CompiledModel`:
- Each has independent KV-cache
- Shared `Arc<ExecutionPlan>` — plan is read-only
- Shared `Arc<dyn ExecutionBackend>` — backend must be thread-safe

---

## Execution Backend Dispatch

```rust
pub trait ExecutionBackend: Send + Sync {
    fn submit(&self, plan: &ExecutionPlan, inputs: &TensorMap) -> Result<TensorMap>;
    fn capabilities(&self) -> BackendCapabilities;
    fn device_name(&self) -> &str;
}

pub struct BackendCapabilities {
    pub has_qgemm_q4_0: bool,
    pub has_qgemm_q8_0: bool,
    pub has_flash_attention: bool,
    pub has_fused_mha: bool,
    pub max_threads: usize,
    pub supports_f16: bool,
    pub supports_bf16: bool,
}
```

The lowering pass queries `BackendCapabilities` to select optimal node mappings.
The session queries it to configure the execution strategy.

---

## Open Questions (runtime)

1. **`hologram::ExecutionPlan` dynamic shape support** — how does the current
   hologram planner handle `seq_len` as a runtime variable? This directly affects
   prefill implementation.

2. **`ArtifactReference` mmap semantics** — does `hologram-artifacts` guarantee
   that mmap'd files remain valid across multiple plan invocations? Required for
   lazy weight loading.

3. **Backend thread-safety contract** — is `ExecutionBackend::submit` safe to
   call from multiple threads concurrently? Required for concurrent sessions.

4. **KV-cache buffer ownership** — should KV-cache buffers be owned by the session
   or allocated from a pool managed by the backend? Pool allocation enables
   better memory reuse for concurrent sessions.

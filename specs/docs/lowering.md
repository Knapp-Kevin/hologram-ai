# hologram-ai: Lowering Design

---

## Overview

Lowering is the translation from `AiGraph` (AI-semantic IR) to
`hologram::ExecutionPlan` (Hologram execution IR).

This is the **critical boundary** between the AI-level compiler and the
Hologram runtime. After lowering, hologram-ai code no longer runs.
The `ExecutionPlan` is owned by `hologram`.

---

## Canonical AI IR (`hologram-ai-ir`)

### `AiGraph`

```rust
pub struct AiGraph {
    pub name: String,
    pub nodes: Vec<AiNode>,           // topologically sorted
    pub inputs: Vec<TensorId>,        // graph-level inputs (e.g. token ids)
    pub outputs: Vec<TensorId>,       // graph-level outputs (e.g. logits)
    pub params: HashMap<TensorId, AiParam>,
    pub tensor_info: HashMap<TensorId, TensorInfo>,
    pub metadata: HashMap<String, MetaValue>,  // arch config, rope params, etc.
    pub warnings: Vec<ImportWarning>,
}
```

### `AiNode`

```rust
pub struct AiNode {
    pub id: NodeId,
    pub op: AiOp,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub attrs: NodeAttrs,
}
```

### `AiOp` — Complete Enum

```rust
pub enum AiOp {
    // Core linear algebra
    MatMul,
    BatchMatMul,
    Gemm { alpha: f32, beta: f32, trans_a: bool, trans_b: bool },
    Einsum { equation: String },

    // Activations
    Relu, Gelu, GeluApprox, Silu, Tanh, Sigmoid,
    Softmax { axis: i64 },
    LogSoftmax { axis: i64 },

    // Normalization
    LayerNorm { axis: i64, epsilon: f32 },
    RmsNorm { epsilon: f32 },
    GroupNorm { num_groups: u32, epsilon: f32 },
    BatchNorm { epsilon: f32, momentum: f32, training: bool },

    // High-level attention (semantic ops, pre-fusion)
    MultiHeadAttention {
        num_heads: u32,
        head_dim: u32,
        scale: Option<f32>,
        causal: bool,
    },
    GroupedQueryAttention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: Option<f32>,
        causal: bool,
    },
    FlashAttentionHint,    // lowering pass decides if flash attn is available

    // Positional encoding
    RotaryEmbedding { base: f32, dim: u32 },
    AlibiSlope,

    // Shape manipulation
    Reshape { allow_zero: bool },
    Transpose { perm: Vec<u32> },
    Concat { axis: i64 },
    Split { axis: i64, sizes: Vec<u64> },
    Slice { axes: Vec<i64>, starts: Vec<i64>, ends: Vec<i64>, steps: Vec<i64> },
    Gather { axis: i64 },
    GatherElements { axis: i64 },
    Scatter { axis: i64, reduce: ScatterReduce },
    Unsqueeze { axes: Vec<i64> },
    Squeeze { axes: Vec<i64> },
    Expand,
    Tile { repeats: Vec<u64> },

    // Elementwise binary
    Add, Sub, Mul, Div, Pow, Mod,
    Min, Max,
    And, Or, Xor, Not,
    Equal, Less, LessOrEqual, Greater, GreaterOrEqual,

    // Elementwise unary
    Abs, Neg, Sqrt, Exp, Log, Sign, Floor, Ceil, Round, Clip,
    Erf, Reciprocal,

    // Reductions
    ReduceSum { axes: Vec<i64>, keepdims: bool },
    ReduceMean { axes: Vec<i64>, keepdims: bool },
    ReduceMax { axes: Vec<i64>, keepdims: bool },
    ReduceMin { axes: Vec<i64>, keepdims: bool },
    ArgMax { axis: i64, keepdims: bool },
    ArgMin { axis: i64, keepdims: bool },

    // Embeddings
    Embed,                 // token_ids → embedding vectors
    CausalMask,            // generate causal attention mask

    // Quantization (explicit in IR)
    Quantize { scheme: QuantScheme },
    Dequantize,
    QuantizedMatMul { lhs_scheme: QuantScheme, rhs_scheme: QuantScheme },

    // Fused ops (produced by optimization passes)
    FusedSwiGLU,           // gate × up → silu(gate) × up
    FusedLayerNormResidual, // x + residual → layernorm

    // Type / control
    Cast { to: DType },
    Constant { value: AiParam },
    Identity,

    // Fallback for unsupported ops
    Opaque { op_type: String, raw_attrs: Vec<u8> },
}
```

### `TensorInfo`

```rust
pub struct TensorInfo {
    pub logical_dtype: DType,    // arithmetic dtype (what math sees it as)
    pub storage_dtype: DType,    // storage dtype (how it's packed on disk/memory)
    pub shape: Shape,
    pub quant: QuantDescriptor,
}
```

### `Shape`

```rust
pub type Shape = SmallVec<[Dim; 6]>;

pub enum Dim {
    Concrete(u64),
    Symbolic(String),    // e.g. "batch", "seq_len", "n_tokens"
    Dynamic,             // unknown, resolved at runtime
}
```

---

## Optimization Passes (pre-lowering)

Passes must run before lowering. Minimum required passes:

| Pass | Required for lowering |
|------|-----------------------|
| `ConstantFolding` | Yes — eliminates shape ops that confuse lowering |
| `ShapePropagation` | Yes — lowering needs shapes for buffer sizing |
| `DeadNodeElimination` | Yes — prevents emitting unreachable plan nodes |
| `AttentionFusion` | Recommended — enables fused MHA kernel |
| `QuantMatMulFusion` | Recommended — enables quantized GEMM |

---

## Lowering Pipeline

```
AiGraph (optimized, shape-propagated)
        +
MemoryPlan
        │
   ┌────▼────────────────────────┐
   │ 1. Topological Sort         │  (with memory constraints)
   └────┬────────────────────────┘
        │ ordered node list
   ┌────▼────────────────────────┐
   │ 2. Op Dispatch              │  AiOp → hologram node type
   └────┬────────────────────────┘
        │ node list with hologram types
   ┌────▼────────────────────────┐
   │ 3. Buffer Binding           │  MemoryPlan buffers → MemoryRegion
   └────┬────────────────────────┘
        │
   ┌────▼────────────────────────┐
   │ 4. Param Packing            │  AiParam → ArtifactReference
   └────┬────────────────────────┘
        │
   ┌────▼────────────────────────┐
   │ 5. Plan Assembly            │  → hologram::ExecutionPlan
   └─────────────────────────────┘
```

---

## Op Dispatch Table

| `AiOp` | Hologram node | Fallback |
|--------|---------------|---------|
| `MatMul` (f32) | `gemm_f32` | — |
| `MatMul` (f16) | `gemm_f16` | cast → `gemm_f32` |
| `MatMul` (bf16) | `gemm_bf16` | cast → `gemm_f32` |
| `QuantizedMatMul(Q4_0, f32)` | `qgemm_q4_0_f32` | `dequant → gemm_f32` |
| `QuantizedMatMul(Q8_0, f32)` | `qgemm_q8_0_f32` | `dequant → gemm_f32` |
| `MultiHeadAttention` | `mha_fused_f32` | decomposed GEMM sequence |
| `GroupedQueryAttention` | `gqa_fused_f32` | decomposed |
| `FlashAttentionHint` | `flash_attn_f16` (if available) | `mha_fused_f32` |
| `RmsNorm` | `rms_norm_f32` | — |
| `LayerNorm` | `layer_norm_f32` | — |
| `Softmax` | `softmax_f32` | — |
| `Gelu` | `gelu_f32` | — |
| `GeluApprox` | `gelu_approx_f32` | — |
| `Silu` | `silu_f32` | — |
| `RotaryEmbedding` | `rope_f32` | decomposed sincos |
| `Embed` | `embed_f32` | — |
| `Add` | `add_f32` / `add_broadcast_f32` | — |
| `Reshape` | memory alias only | — |
| `Transpose` | `transpose_f32` | — |
| `Cast` | `cast_{src}_{dst}` | — |
| `Quantize` | `quantize_{scheme}` | — |
| `Dequantize` | `dequantize_{scheme}` | — |
| `Concat` | `concat_f32` | — |
| `FusedSwiGLU` | `swiglu_f32` | decomposed |
| `Opaque` | **lowering error** | — |

The dispatch table is a registry, not a match statement. Backend capability
queries happen at lowering time: if a fused kernel isn't available, the
fallback sequence is emitted instead.

---

## Quantization Handling in Lowering

Quantized weights stored as `AiParam::Lazy` remain in their quantized storage
format. The lowering pass decides the dequantization strategy:

**Strategy A: Eager dequant at plan start**
- Insert `dequant_{scheme}` nodes at the plan boundary
- All compute runs in f32/f16
- Chosen when: backend has no quantized GEMM kernels

**Strategy B: Fused quant kernels**
- `QuantizedMatMul` → `qgemm_{lhs}_{rhs}`
- Dequantization is kernel-internal
- Chosen when: backend declares `HAS_QGEMM` capability

**Strategy C: Mixed**
- Per-op decision based on backend capability query
- Default for CPU backend (has Q4_0 and Q8_0 GEMM but not all quant types)

The `LoweringOptions::quant_strategy` field selects the strategy or leaves it
to auto-detection.

---

## Shape Handling in Lowering

**Concrete shapes** → buffer sizes computed at lowering time.

**Symbolic shapes** (e.g. `seq_len = Dim::Symbolic`) → lowering emits
shape-parametric plan nodes. `hologram::ExecutionPlan` must support
dynamic shapes (TBD with hologram team — see open questions).

**Dynamic shapes** → lowering emits placeholder nodes with runtime
size-calculation ops. Higher cost; avoided where possible.

---

## KV-Cache Lowering

KV-cache buffers are allocated via `MemoryPlan::kv_cache_layout`.

At lowering time:
- A `KvCacheSlotWrite` node is emitted after each attention layer's K and V projections
- A `KvCacheSlotRead` node is emitted at the beginning of each attention computation
- The cache buffer is bound as a persistent `MemoryRegion` across plan invocations

The `InferenceSession` manages the cache offset counter and passes it as an
input to the plan on each invocation.

---

## Lowering Errors

Lowering is strict. It fails if:
- Any `AiOp::Opaque` node is encountered
- A required hologram kernel is not available on the target backend
- Shape information is missing for buffer allocation (except for dynamic dims)
- Memory plan is inconsistent with graph (tensor IDs don't match)

Lowering warnings (non-fatal):
- A requested fused kernel is unavailable, falling back to decomposed sequence
- A symbolic dim was not resolved (will require runtime shape dispatch)

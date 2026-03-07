# hologram-ai: Full Architecture

---

## 1. System Purpose

`hologram-ai` is a compiler and runtime integration layer. Given a foreign AI
model artifact, it produces an executable `hologram::ExecutionPlan` and manages
the inference session lifecycle on top of the Hologram execution backend.

The canonical internal flow:

```
Foreign model artifact
        │
   ┌────▼─────────────────────────┐
   │   Format Importer            │  (hologram-ai-onnx / hologram-ai-gguf / hologram-ai-ggml)
   └────┬─────────────────────────┘
        │ AiGraph  ←──── canonical AI IR (hologram-ai-ir)
   ┌────▼─────────────────────────┐
   │   Optimization Passes        │  (hologram-ai-opt)
   └────┬─────────────────────────┘
        │ AiGraph (optimized)
   ┌────▼─────────────────────────┐
   │   Memory Planner             │  (hologram-ai-mem)
   └────┬─────────────────────────┘
        │ MemoryPlan
   ┌────▼─────────────────────────┐
   │   Lowering                   │  (hologram-ai-lower)
   └────┬─────────────────────────┘
        │ hologram::ExecutionPlan
   ┌────▼─────────────────────────┐
   │   Inference Session          │  (hologram-ai-session)
   │   + KV-Cache                 │
   └────┬─────────────────────────┘
        │ Token / Tensor output
   ┌────▼─────────────────────────┐
   │   Streaming Decoder          │  (hologram-ai-stream)
   └──────────────────────────────┘
```

---

## 2. System Boundaries

### hologram-ai owns

- All AI model format parsing and interpretation
- Canonical `AiGraph` IR and all optimization passes on it
- Quantization descriptors and quant-aware lowering
- Memory planning for AI workloads
- Lowering from `AiGraph` to `hologram::ExecutionPlan`
- Inference session lifecycle (`InferenceSession`)
- KV-cache layout, allocation, and update logic
- Autoregressive token generation loop
- Streaming output interface
- Validation harness against reference runtimes

### hologram owns (hologram-ai consumes these)

- `ExecutionPlan` — the execution contract
- `ExecutionBackend` trait — the execution abstraction
- `MemoryRegion` and `BufferView` — buffer addressing
- `ArtifactReference` — lazy loading of large blobs
- Backend implementations (CPU, Metal, CUDA, etc.)

### hologram-ai does NOT own

- Actual kernel implementations (GEMM, attention, etc.)
- Process/WASM/microVM sandbox isolation
- Network transport
- Tokenizer implementations (accepted as `Box<dyn Tokenizer>`)

---

## 3. Major Layers

### Layer 1: Format Importers

Three importers, one per format. Each is a standalone crate.

| Crate | Input | Output |
|-------|-------|--------|
| `hologram-ai-onnx` | ONNX protobuf bytes | `AiGraph` |
| `hologram-ai-gguf` | GGUF v1/v2/v3 file | `AiGraph` |
| `hologram-ai-ggml` | GGML checkpoint file | `AiGraph` |

**Key constraint:** Format-specific logic must not escape the importer boundary.
After `import_*()` returns an `AiGraph`, no downstream code knows or cares
which format the model came from.

### Layer 2: Canonical AI IR (`hologram-ai-ir`)

`AiGraph` is the single representation all importers target and all downstream
passes consume. It is a typed DAG of `AiNode` operations over `AiTensor` values.

Quantization descriptors are embedded in `TensorInfo`. They are never stripped.

See [lowering.md](lowering.md) for the IR specification.

### Layer 3: Optimization Passes (`hologram-ai-opt`)

Pure graph-to-graph transformations. Each pass is a stateless function:
`fn pass(graph: AiGraph) -> Result<AiGraph>`.

Passes operate on the semantic `AiGraph` level, not on `hologram` graph nodes.

### Layer 4: Memory Planner (`hologram-ai-mem`)

Takes an optimized `AiGraph` and produces a `MemoryPlan`:
- Tensor liveness intervals
- Buffer alias candidates
- In-place op detection
- KV-cache sizing
- Weight layout decisions
- Alignment annotations

### Layer 5: Lowering (`hologram-ai-lower`)

Maps `AiGraph + MemoryPlan` to `hologram::ExecutionPlan`.
This is the boundary between AI-semantic code and Hologram-native code.

After lowering, `hologram-ai` treats the plan as opaque and submits it to
the `ExecutionBackend`.

### Layer 6: Inference Session (`hologram-ai-session`)

Manages:
- Compiled `ExecutionPlan`
- KV-cache buffers
- Device binding
- Single-pass `run()` and multi-step `generate()` APIs

### Layer 7: Streaming Decoder (`hologram-ai-stream`)

Wraps `InferenceSession` in an autoregressive loop. Implements `Stream<Item = Token>`.

---

## 4. Canonical Model Representation

**Recommendation: yes, define an AI-specific IR above raw Hologram graph IR.**

Rationale: foreign formats carry semantic structure (multi-head attention,
rope embeddings, norm layers, MLP blocks) that is expensive to reconstruct
from raw arithmetic ops. Preserving this structure through the optimization
phase enables high-value fusions (attention fusion, FFN fusion, norm fusion)
before lowering to Hologram primitives. Fusing at the Hologram graph level
would require pattern-matching over much lower-level ops.

The `AiGraph` IR preserves semantic structure until the lowering boundary,
then maps cleanly to `hologram::ExecutionPlan` nodes.

---

## 5. Semantic Structure in the IR

The following structures survive into `AiGraph` before lowering:

| Structure | IR representation |
|-----------|------------------|
| Multi-head attention | `AiOp::MultiHeadAttention` |
| Grouped query attention | `AiOp::GroupedQueryAttention` |
| Flash attention hint | `AiOp::FlashAttentionHint` |
| RMS normalization | `AiOp::RmsNorm` |
| Layer normalization | `AiOp::LayerNorm` |
| SwiGLU / SiLU gate | `AiOp::FusedSwiGLU` (post-fusion) |
| Rotary embeddings | `AiOp::RotaryEmbedding` |
| Embedding lookup | `AiOp::Embed` |
| Causal attention mask | represented as `AiOp::CausalMask` |

These high-level ops allow the lowering pass to select optimal Hologram kernel
bindings (e.g. fused MHA, flash attention if supported).

---

## 6. Quantization

Quantization is first-class throughout the pipeline.

**Logical dtype** vs **storage dtype** are distinct in `TensorInfo`:

```rust
pub struct TensorInfo {
    pub logical_dtype: DType,   // F32 — what arithmetic sees it as
    pub storage_dtype: DType,   // Q4_0 — how bits are stored
    pub quant: QuantDescriptor, // scale/zp/block metadata
    pub shape: Shape,
}
```

Dequantization is **explicit in the IR** as `AiOp::Dequantize`. The
`hologram-ai-opt` pass may fuse `Dequantize → MatMul` into `AiOp::QuantizedMatMul`
when a backend supports the fused kernel.

This keeps the IR honest and lets backends declare their quant kernel
capabilities rather than assuming them.

---

## 7. Shape and DType Propagation

Shape propagation runs as a required optimization pass before lowering.

Symbolic dimensions (`batch_size`, `seq_len`) are preserved through the graph.
Concrete dimensions are folded to constants.

Shape inference failures at import time produce `Dim::Dynamic` annotations.
The lowering pass must handle dynamic shapes via runtime dispatch.

DType propagation:
- `Dequantize` outputs widen to the widest dtype operand needs (usually f32 or f16)
- `Cast` ops are inserted by the lowering pass where dtypes mismatch
- The planner annotates each node's input/output dtypes before lowering

---

## 8. Backend Matrix

### MVP backend

**CPU only** — pure Rust reference implementation via `hologram-cpu`.

Rationale: maximizes portability for validation, avoids GPU toolchain overhead
during the compiler pipeline bring-up phase, covers all test platforms.

### Phase 2 backends

**Metal** (Apple Silicon) — highest priority after CPU, covers the primary
development hardware for the team.

### Phase 3 backends

**CUDA** and **WebGPU** — after Metal proves the backend abstraction works.

### Backend portability

`InferenceSession` takes `Arc<dyn ExecutionBackend>`. No backend-specific
code in `hologram-ai-session` or anywhere above `hologram-ai-lower`. Backend swap
is a one-line change at session construction.

---

## 9. Portability

| Target | Priority | Notes |
|--------|----------|-------|
| `aarch64-apple-darwin` (M-series) | P0 | primary dev hardware |
| `x86_64-unknown-linux-gnu` | P0 | CI and server targets |
| `x86_64-apple-darwin` | P1 | Intel Mac |
| `x86_64-pc-windows-msvc` | P2 | Windows server |
| `wasm32-wasi` | P3 | no SIMD-heavy backends; pure IR + lowering only |

---

## 10. Dataflow Summary

```
                     ┌──────────────────┐
                     │  Model artifact  │
                     │  (.onnx / .gguf  │
                     │   / .bin)        │
                     └────────┬─────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Format Importer    │  hologram-ai-{onnx,gguf,ggml}
                   └──────────┬──────────┘
                              │ AiGraph
                   ┌──────────▼──────────┐
                   │  Optimization       │  hologram-ai-opt
                   │  Passes             │  (fusion, folding, shape prop)
                   └──────────┬──────────┘
                              │ AiGraph (optimized)
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼─────┐  ┌──────▼─────┐  ┌─────▼──────┐
     │ hologram-ai-quant│  │ hologram-ai-mem│  │(shape/dtype│
     │ (quant descs)│  │ (memory    │  │  validated)│
     └──────────────┘  │  plan)     │  └────────────┘
                       └──────┬─────┘
                              │ AiGraph + MemoryPlan
                   ┌──────────▼──────────┐
                   │  Lowering           │  hologram-ai-lower
                   └──────────┬──────────┘
                              │ hologram::ExecutionPlan
                   ┌──────────▼──────────┐
                   │  Inference Session  │  hologram-ai-session
                   │  + KV-Cache         │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Streaming Decoder  │  hologram-ai-stream
                   └──────────┬──────────┘
                              │ Token stream
                   ┌──────────▼──────────┐
                   │  Application        │
                   └─────────────────────┘
```

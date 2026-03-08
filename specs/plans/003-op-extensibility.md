# Op Extensibility: Novel Operations and WASM Kernels

## Status: Future (Phase 3+)

## Context

As of Sprint 2, all AI ops are native `FloatOp` variants in hologram base crate.
Archives are fully self-describing — `hologram run` works without any
`CustomOpRegistry`. This is the correct foundation, but raises the question:
how do users add truly novel operations without waiting for a hologram release?

## Current Design

```
AiOp (hologram-ai IR)
  → dispatch() → GraphOp::Float(FloatOp::Variant)
    → serialized into .holo archive via rkyv
      → hologram run → float_dispatch::dispatch_float() → kernel
```

Every `FloatOp` variant has a kernel in `hologram-exec/src/float_dispatch.rs`.
New ops require changes to both `hologram-core` (enum) and `hologram-exec`
(kernel), then a hologram release.

## Strategy: Three Tiers

### Tier 1: Op Decomposition (Primary — Available Now)

Most "new" ops are compositions of existing primitives. Compilers should
decompose them at lowering time rather than adding new `FloatOp` variants.

**Example:** A hypothetical `Mish` activation `x * tanh(softplus(x))`:
```
AiOp::Mish → lower as:
  t1 = FloatOp::Exp(x)           // exp(x)
  t2 = FloatOp::Log(1 + t1)      // softplus = ln(1 + exp(x))
  t3 = FloatOp::Tanh(t2)         // tanh(softplus(x))
  out = FloatOp::Mul(x, t3)      // x * tanh(softplus(x))
```

**When to use:** Any op that can be expressed as a DAG of existing `FloatOp`
primitives with acceptable performance. This covers ~90% of new op requests.

**Advantages:**
- No hologram changes needed
- Archives remain portable across hologram versions
- Optimizer can fuse/simplify the decomposed subgraph

### Tier 2: Serializable Op Descriptors (Future — hologram change needed)

For ops that cannot be efficiently decomposed (e.g., novel attention variants
with custom masking, specialized quantization schemes), add a
`GraphOp::Descriptor` variant that carries a self-describing op specification:

```rust
// In hologram-core
pub enum GraphOp {
    // ... existing variants ...
    /// Self-describing op with embedded kernel specification.
    Descriptor(OpDescriptor),
}

pub struct OpDescriptor {
    /// Unique op name (e.g., "com.example.sliding_window_attention")
    pub name: String,
    /// Serialized parameters (MessagePack, CBOR, or custom binary)
    pub params: Vec<u8>,
    /// Input/output arity
    pub arity: u8,
    /// Kernel source (see KernelSpec)
    pub kernel: KernelSpec,
}

pub enum KernelSpec {
    /// Decompose into a subgraph of native ops at execution time
    Decomposition(Vec<(GraphOp, Vec<u32>)>),
    /// WASM kernel (see Tier 3)
    Wasm(WasmKernel),
}
```

**When to use:** Truly novel operations that don't decompose well into
existing primitives. Requires hologram to support `OpDescriptor` execution.

**Advantages:**
- Self-describing: archives carry their own op definitions
- Forward-compatible: old hologram versions can skip/error gracefully
- No hologram release needed for new ops (once the framework is in place)

### Tier 3: WASM Kernels (Future — hologram change needed)

For maximum extensibility, allow ops to carry their own kernel implementations
as embedded WASM modules:

```rust
pub struct WasmKernel {
    /// Compiled WASM module bytes
    pub module: Vec<u8>,
    /// Entry point function name
    pub entry: String,
    /// ABI version for input/output buffer layout
    pub abi_version: u32,
}
```

The executor would:
1. Instantiate the WASM module (via `wasmtime` or `wasmer`)
2. Map input byte buffers into WASM linear memory
3. Call the entry point
4. Extract output bytes

**When to use:** Custom kernels that need arbitrary computation (novel
quantization schemes, domain-specific operations, research prototypes).

**Advantages:**
- True extensibility: users ship arbitrary kernels in archives
- Sandboxed execution: WASM provides memory safety guarantees
- Cross-platform: WASM modules run everywhere hologram runs

**Considerations:**
- Performance overhead: WASM → native call boundary, no SIMD (unless WASM SIMD)
- Archive size: WASM modules add bytes to each archive
- Security: WASM is sandboxed but we'd want resource limits (memory, CPU)
- Dependency: adds `wasmtime`/`wasmer` to hologram-exec

## Implementation Roadmap

| Phase | Item | Blocked On |
|-------|------|------------|
| Now   | Op decomposition in hologram-ai lowering | Nothing — available today |
| 3     | `OpDescriptor` variant in `GraphOp` | hologram base crate change |
| 3     | Decomposition-based `KernelSpec` execution | hologram-exec change |
| 4+    | WASM kernel support | `wasmtime` integration, ABI design |
| 4+    | WASM SIMD for performance-critical kernels | WASM SIMD proposal |

## Decision

- **Primary strategy:** Op decomposition (Tier 1). All new ops in hologram-ai
  should first attempt decomposition into existing `FloatOp` primitives.
- **When decomposition is insufficient:** Propose the op as a native `FloatOp`
  variant in hologram base crate (requires hologram release).
- **Future:** Implement `OpDescriptor` (Tier 2) when the first op arrives
  that genuinely cannot be decomposed and needs cross-version portability.
- **Later future:** WASM kernels (Tier 3) for research/prototyping use cases.

## References

- ADR-0016: hologram-ai is a compiler only
- `hologram-core/src/op/float_op.rs`: Native FloatOp enum (55 variants)
- `hologram-exec/src/float_dispatch.rs`: Kernel implementations
- `hologram-ai-common/src/lower/dispatch.rs`: AiOp → FloatOp dispatch

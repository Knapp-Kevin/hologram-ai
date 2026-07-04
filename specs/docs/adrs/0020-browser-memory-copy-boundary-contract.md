# ADR-0020: Browser Memory and Copy Boundary Contract

**Status:** Proposed
**Date:** 2026-07-04
**Relates to:** ADR-0017, ADR-0019, CONFORMANCE classes ZM, ZA, PV, NS

## Context

The browser path stores model artifacts in OPFS, but several hot paths still copy full artifacts into JS or WASM memory:

- ONNX files are read from OPFS into `ArrayBuffer` before compile.
- `.holo` archives are read from OPFS into `Uint8Array` before generation.
- WASM entry points copy byte slices into Rust-owned vectors in several compile/load paths.
- `HoloRunner::from_bytes(holo.to_vec())` makes full-archive loading a memory boundary.

This means OPFS currently provides persistence, not zero-copy runtime execution.

The current implementation can still be valuable, but claims around not using CPU or RAM must be avoided. The browser path uses CPU through WASM and uses JS/WASM heap memory.

## Decision

All browser memory and copy boundaries must be explicitly instrumented, documented, and reduced over time.

The project should define a browser memory contract around four phases:

1. Model acquisition
2. Compile
3. Archive load
4. Generation

Each phase must identify:

- source location: network, OPFS, JS heap, WASM memory, device memory
- copies made
- peak expected memory
- whether the phase is streaming, chunked, or whole-buffer
- whether failure is controlled when memory is insufficient

## Required work

### 1. Instrument copy boundaries

Add timing and memory markers around:

- fetch response streaming
- OPFS writes
- OPFS reads
- `arrayBuffer()` calls
- WASM initialization
- calls to `compile`, `compile_onnx_with_data`, `compile_safetensors_streamed`, `describe`, `run`, and `generate`
- `HoloRunner::from_bytes` browser call sites

### 2. Add memory telemetry helpers

Create a browser-safe telemetry module that reports what is available per browser:

- `performance.memory` where exposed
- `navigator.deviceMemory` where exposed
- WASM memory size if available through the generated bindings or wrapper
- archive byte length
- largest tensor byte length
- OPFS file sizes

Missing metrics should be reported as `null`, not fabricated.

### 3. Classify each path

Each path should be classified as one of:

- `streamed`
- `chunked`
- `whole-buffer`
- `copied-to-wasm`
- `device-resident`

The initial expected classification is:

| Path | Expected classification today |
|---|---|
| Safetensors download to OPFS | streamed, with per-tensor buffering |
| ONNX download to OPFS | streamed to OPFS, then whole-buffer compile |
| `.holo` generation load | whole-buffer, copied to WASM |
| WASM generation | CPU WASM, heap-backed |
| WebGPU execution | not implemented |

## Acceptance criteria

- A memory/copy-boundary report exists for the browser path.
- Benchmark JSON from ADR-0019 includes phase-level memory and copy information.
- The browser UI or logs can expose when a model path is whole-buffer rather than streamed.
- The codebase does not describe OPFS storage as zero-RAM or zero-copy execution unless the specific path actually proves that property.
- Oversized model failures report the specific phase that exceeded the budget.

## Non-goals

- This ADR does not require WebGPU implementation.
- This ADR does not require eliminating all copies immediately.
- This ADR does not change native runtime behavior.

## Consequences

Positive:

- The project can distinguish real low-memory improvements from storage convenience.
- Browser memory failures become debuggable.
- Performance claims become more honest and more useful.

Negative:

- Some current paths will be visibly classified as whole-buffer.
- This may force product language to become less magical and more accurate.

That is acceptable. Accuracy is cheaper than debugging user tabs that vanish like they owed Chrome money.
# ADR-0023: Device-Native Browser Execution Path

**Status:** Proposed
**Date:** 2026-07-04
**Relates to:** ADR-0017, ADR-0020, Plan 067, CONFORMANCE classes ZM, ZA, PV, NS

## Context

The current browser path executes through WASM. That means it uses CPU execution and WASM linear memory.

This is compatible with static hosting and useful for portability, but it does not satisfy the stronger product goal of minimizing CPU and system RAM use during inference.

To approach that goal, browser execution must move toward device-native buffers and compute, most likely through WebGPU first and possibly WebNN where available.

The existing design direction in Plan 067 already points toward a `ComputeMemory` and `ComputeBackend` abstraction with `WebGpuMemory` and `WebGpuBackend`. This ADR adopts that direction for the browser path.

## Decision

WASM remains the portable baseline execution path.

WebGPU becomes the target browser acceleration path for models and kernels that can be executed device-natively.

The browser runtime should eventually support three execution modes:

| Mode | Description | Expected use |
|---|---|---|
| `wasm-cpu` | current portable CPU/WASM path | baseline and fallback |
| `webgpu-partial` | selected kernels execute on WebGPU, controlled fallback allowed only during development | experimental acceleration |
| `webgpu-device-native` | full graph executes with device-resident buffers and no mid-graph CPU fallback | target performance path |

No production performance claim should rely on `webgpu-partial` unless fallback counts and transfer costs are reported.

## Required architecture

### 1. Explicit backend selection

The browser app should expose the selected backend in diagnostics:

- `wasm-cpu`
- `webgpu-partial`
- `webgpu-device-native`
- `unsupported`

Backend selection should be deterministic and logged.

### 2. Device-memory abstraction

The browser path should align with a future `ComputeMemory` abstraction:

- allocate device buffers
- upload weights once
- keep intermediates on device
- perform one final readback when needed
- avoid CPU fallback during claimed device-native execution

### 3. Kernel coverage accounting

Every model run on WebGPU must report:

- total kernel count
- WebGPU kernel count
- CPU fallback kernel count
- device uploads
- device readbacks
- synchronization count

Any CPU fallback during generation must be visible.

### 4. No hidden CPU fallback in readiness claims

A run may be useful with fallback, but it is not device-native.

The readiness language must distinguish:

- functional fallback
- accelerated partial execution
- full device-native execution

## Acceptance criteria

- Browser diagnostics report the active backend.
- WebGPU availability is detected and reported without breaking WASM fallback.
- A backend capability report exists before attempting WebGPU execution.
- Any WebGPU prototype reports kernel coverage and fallback count.
- A production-ready WebGPU path requires zero mid-graph CPU fallback for supported models.
- Benchmark output from ADR-0019 includes backend mode and fallback counts.

## Non-goals

- This ADR does not implement WebGPU kernels directly.
- This ADR does not remove WASM CPU fallback.
- This ADR does not require WebNN support before WebGPU work begins.

## Consequences

Positive:

- The project has a credible path toward reducing CPU use.
- Device-native performance claims become measurable.
- CPU fallback stops hiding inside happy-path demos.

Negative:

- WebGPU introduces browser-specific complexity.
- Kernel coverage becomes a hard gating problem.
- Some models may remain WASM-only until coverage improves.

That is acceptable. Moving compute to the device is not a slogan. It is a backend, a memory model, and a pile of unglamorous accounting.
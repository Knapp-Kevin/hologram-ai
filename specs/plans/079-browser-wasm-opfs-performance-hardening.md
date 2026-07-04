# Plan 079: Browser WASM + OPFS Performance Hardening

**Status:** Proposed
**Created:** 2026-07-04
**Related ADRs:** ADR-0019, ADR-0020, ADR-0021, ADR-0022, ADR-0023, ADR-0024

## Purpose

This plan sequences the work required to evaluate and harden the browser inference path.

The current implementation is credible as a browser-local WASM prototype with OPFS persistence. It is not yet proven as a mature browser inference engine, and it does not currently satisfy any literal `no CPU` or `no RAM` claim.

The goal is to convert the browser path from a promising implementation into a measured, bounded, repeatable system.

## Ground truth

Current browser path:

- uses WASM for client-side compile/run/generate
- stores artifacts in OPFS
- uses workers for download and generation paths
- uses CPU execution through WASM
- uses JS heap and WASM linear memory
- has browser tests with mocked paths
- does not yet have a visible full browser performance matrix
- does not yet have implemented WebGPU device-native execution in this fork

Target path:

- repeatable browser benchmark harness
- explicit memory and copy-boundary telemetry
- OPFS cache integrity and quota management
- first-class browser CI
- clear WebGPU/WebNN decision boundary
- evidence-gated performance claims

## ADR stack

### ADR-0019: Browser Performance Evaluation Contract

Defines the required browser benchmark matrix and metrics.

Deliverables:

- browser benchmark harness
- JSON result format
- tiny fixture CI smoke
- manual real-model benchmark path

### ADR-0020: Browser Memory and Copy Boundary Contract

Defines how to identify and measure JS heap, WASM memory, OPFS reads, whole-buffer loads, and copy boundaries.

Deliverables:

- memory telemetry helpers
- phase-level copy-boundary reporting
- classification of streamed, chunked, whole-buffer, copied-to-WASM, and device-resident paths

### ADR-0021: OPFS Storage and Runtime Boundary

Separates OPFS persistence from runtime execution memory.

Deliverables:

- cache metadata
- integrity checks
- quota reporting
- interrupted-write recovery
- plan for segmented or lazy `.holo` access

### ADR-0022: Browser CI Contract for WASM and Web App Paths

Makes the browser app a first-class CI target.

Deliverables:

- `pnpm wasm` in CI
- `pnpm build` in CI
- browser tests in CI
- benchmark artifact upload when available

### ADR-0023: Device-Native Browser Execution Path

Defines WebGPU as the target browser acceleration path, with WASM CPU retained as fallback.

Deliverables:

- backend diagnostics
- WebGPU capability report
- kernel coverage accounting
- fallback counts
- explicit backend mode in benchmark output

### ADR-0024: Performance Claims and Readiness Gates

Defines evidence levels for performance and readiness claims.

Deliverables:

- docs claim audit
- README wording cleanup
- readiness levels attached to measured paths
- prohibition on unsupported `no CPU` and `no RAM` wording

## Recommended implementation order

1. Add browser CI for WASM and web build.
2. Add benchmark harness with tiny fixture output.
3. Add memory and copy-boundary instrumentation.
4. Add OPFS cache metadata, integrity, and quota reporting.
5. Add real-model manual benchmark path.
6. Audit README and public-facing performance claims.
7. Prototype WebGPU capability detection and backend diagnostics.
8. Begin kernel coverage work only after fallback accounting exists.

## Readiness gates

### Gate A: Browser path is testable

Required:

- CI builds WASM package
- CI builds web app
- CI runs tiny browser fixture
- controlled oversized-model failure is tested

### Gate B: Browser path is measurable

Required:

- benchmark JSON output
- first-token latency
- steady tokens/sec
- OPFS read/write timings
- memory telemetry where available

### Gate C: Browser path is honest

Required:

- docs distinguish OPFS storage from runtime memory
- docs distinguish WASM CPU from device-native execution
- unsupported `no CPU` and `no RAM` claims are removed or qualified

### Gate D: Browser path is accelerable

Required:

- backend mode is reported
- WebGPU support is detected
- kernel coverage is reported
- fallback counts are visible

### Gate E: Browser path is production-claimable

Required:

- repeated browser matrix exists
- real model benchmark exists
- memory ceiling and failure modes are documented
- budget gates prevent regression
- device-native claims include zero mid-graph CPU fallback for supported models

## Non-goals

- Do not block native runtime work.
- Do not remove WASM CPU fallback.
- Do not require large real-model downloads on every CI run.
- Do not treat browser performance as proven until browser measurements exist.

## Success definition

This plan succeeds when the project can answer the following questions with measured data:

- What model sizes work in the browser today?
- Which browser and OS combinations are supported?
- How much memory is used during download, compile, archive load, and generation?
- How fast is first token and steady decode?
- Which paths are whole-buffer versus streamed?
- What does OPFS save, and what does it not save?
- When WebGPU is enabled, how much work stays on device?

If those answers are not measurable, the system is not ready for performance claims. It may still be promising, but promising is where software goes before users arrive and ruin the fantasy.
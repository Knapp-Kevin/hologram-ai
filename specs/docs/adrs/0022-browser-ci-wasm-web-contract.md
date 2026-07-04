# ADR-0022: Browser CI Contract for WASM and Web App Paths

**Status:** Proposed
**Date:** 2026-07-04
**Relates to:** ADR-0017, ADR-0019, CONFORMANCE classes AR, NS, PV

## Context

The Rust workspace has meaningful CI coverage for format, clippy, build, unit tests, structural V&V, portability builds, and ONNX Runtime conformance.

The browser path has package scripts for WASM build, Vite build, and browser tests, but those browser-specific checks are not clearly enforced as first-class CI gates in the visible workflow.

A browser-local inference project cannot rely only on native Rust CI. The web path must prove that the generated WASM package, TypeScript adapter, workers, OPFS paths, and browser tests continue to work together.

## Decision

The browser app must have a first-class CI lane.

The minimum CI checks are:

- build the WASM package with `wasm-pack`
- build the web app with TypeScript and Vite
- run browser tests for tiny fixture paths
- run OPFS mocked or browser-backed smoke tests
- verify oversized model failure behavior
- upload benchmark JSON artifacts when available

## Required CI stages

### 1. WASM package build

Run from `apps/web` or root:

```bash
pnpm install
pnpm wasm
```

This validates that `crates/hologram-ai-wasm` builds for the web target and generates bindings expected by `apps/web/src/holo.ts`.

### 2. Web build

Run:

```bash
pnpm build
```

This validates TypeScript, Vite bundling, worker imports, and static hosting compatibility.

### 3. Browser tests

Run:

```bash
pnpm test
```

At minimum, CI must cover:

- WASM initialization
- OPFS availability or mocked equivalent
- tiny fixture compile
- tiny fixture run
- generation worker handshake
- oversized model controlled failure

### 4. Browser performance smoke

Once ADR-0019 is implemented, CI should run a tiny benchmark and upload the JSON artifact.

The tiny benchmark should be stable enough for regression detection but should not fail on minor timing variance. Hard budgets should be introduced only after baseline data exists.

## Acceptance criteria

- A CI workflow exists for the browser app.
- CI runs `pnpm wasm`, `pnpm build`, and browser tests.
- Tiny fixture browser tests run without live network dependency.
- Oversized model failure behavior is tested.
- CI artifacts include benchmark JSON once the benchmark harness exists.
- Documentation explains how browser CI differs from native Rust V&V.

## Non-goals

- Do not require real Hugging Face model downloads in every PR.
- Do not require large-model browser benchmarks in CI.
- Do not replace native Rust conformance CI.

## Consequences

Positive:

- Browser regressions become visible.
- Generated WASM bindings remain compatible with the TypeScript adapter.
- The browser app is treated as a product surface, not a decorative appendix.

Negative:

- CI becomes slower.
- Browser test flakiness may require stabilization work.

That is acceptable. The browser path is part of the claim. It should endure the indignity of being tested.
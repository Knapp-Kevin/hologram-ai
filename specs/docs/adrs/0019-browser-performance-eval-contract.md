# ADR-0019: Browser Performance Evaluation Contract

**Status:** Proposed
**Date:** 2026-07-04
**Relates to:** ADR-0017, CONFORMANCE class PV, NS

## Context

The browser app currently demonstrates a real Rust/WASM pipeline and OPFS-backed local persistence. That is valuable, but it does not yet prove browser inference performance.

The existing performance evidence is weighted toward native Rust tests, synthetic benchmarks, mocked browser tests, and gated large-model characterization. Those are useful, but they are not a substitute for measuring the actual browser path: download, OPFS persistence, WASM initialization, archive load, first token, steady decode, memory growth, and failure behavior.

The stated product goal should be treated precisely:

- The browser path should minimize server dependency.
- OPFS should reduce repeated download and compile cost.
- Content addressing should reduce duplicate resident data.
- Future device-native execution may move compute away from the CPU.
- The current browser path still uses CPU execution through WASM and uses JS/WASM memory.

## Decision

A browser-local performance contract is required before any claim of browser inference readiness.

The benchmark suite must measure the real web app path, not only native Criterion benches or Rust unit tests.

The minimum benchmark matrix is:

| Model or fixture | Purpose |
|---|---|
| Tiny synthetic MLP or matmul fixture | deterministic plumbing and CI baseline |
| SmolLM2-135M Instruct | smallest real chat model path |
| Qwen2.5-0.5B Instruct | practical small-model stress case |
| Oversized mock model | controlled failure and memory-limit behavior |

The minimum metrics are:

- browser name and version
- operating system
- model id
- source format: ONNX, safetensors, or `.holo`
- model size on disk
- compile duration
- `.holo` archive size
- OPFS write duration
- OPFS read duration
- WASM initialization duration
- archive load duration
- first-token latency
- tokens/sec over 16, 32, and 64 token generation windows
- JS heap before and after, where browser APIs allow it
- WASM memory size before and after, where exposed
- failure mode when the model exceeds browser memory limits

## Implementation requirements

Create a benchmark harness under one of:

- `apps/web/src/perf/`
- `apps/web/bench/`

The harness should be usable from:

- a manual browser benchmark page or panel
- Playwright or Vitest browser tests
- CI for tiny fixtures

Benchmark output must be machine-readable JSON.

Suggested output shape:

```json
{
  "browser": "chromium",
  "model": "SmolLM2-135M-Instruct",
  "format": "safetensors",
  "compile_ms": 0,
  "opfs_write_ms": 0,
  "opfs_read_ms": 0,
  "wasm_init_ms": 0,
  "archive_load_ms": 0,
  "first_token_ms": 0,
  "tokens_per_second_16": 0,
  "tokens_per_second_32": 0,
  "tokens_per_second_64": 0,
  "heap_before_mb": null,
  "heap_after_mb": null,
  "wasm_memory_mb": null,
  "status": "pass"
}
```

## Acceptance criteria

- A browser benchmark harness exists and runs locally.
- At least one tiny fixture benchmark runs in CI.
- At least one real-model benchmark can be run manually without code changes.
- Results are emitted as JSON.
- First-token latency and steady decode throughput are captured separately.
- Oversized-model behavior is measured and reported as a controlled failure.
- Documentation explains how to run the browser benchmark suite.

## Non-goals

- Do not optimize runtime kernels in this ADR.
- Do not claim browser production readiness from native-only benches.
- Do not treat OPFS persistence as proof of low-memory inference.

## Consequences

Positive:

- Browser performance claims become falsifiable.
- Regressions can be caught before they become README folklore.
- Coding agents receive a measurable target instead of vibes dressed as engineering.

Negative:

- This may expose that the current browser path is slower or more memory-hungry than desired.
- Browser-specific failures will need platform-specific handling.

That tradeoff is acceptable. A failed benchmark is better than an attractive falsehood.
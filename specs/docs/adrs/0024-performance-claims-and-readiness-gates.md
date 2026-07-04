# ADR-0024: Performance Claims and Readiness Gates

**Status:** Proposed
**Date:** 2026-07-04
**Relates to:** ADR-0017, ADR-0019, ADR-0020, ADR-0023, CONFORMANCE class PV

## Context

The project has ambitious language around browser-local inference, OPFS, content addressing, zero movement, quantization, and future device-native execution.

Some of those properties are real in specific parts of the stack. Others are future-facing, native-only, synthetic, gated, or not yet demonstrated in the browser app.

To keep the project credible, public and internal performance claims must be tied to explicit evidence.

The most important distinction:

- Current browser execution uses CPU through WASM and uses JS/WASM memory.
- OPFS provides local persistence, not zero-RAM execution.
- Content-addressed reuse can reduce duplicate work or duplicate resident data, but it does not erase the need to materialize the working set.
- Future WebGPU or WebNN execution may move compute and working buffers to device memory, but that is a separate readiness level.

## Decision

Performance claims must be gated by evidence level.

The project will use the following readiness levels:

| Level | Name | Meaning |
|---|---|---|
| L0 | Design | architecture or plan exists, no implementation proof |
| L1 | Build proof | code builds for the target |
| L2 | Functional proof | tiny fixture runs correctly |
| L3 | Browser proof | real browser path runs and is measured |
| L4 | Real-model proof | real model runs with measured latency, throughput, and memory |
| L5 | Production claim | repeatable browser matrix with budget gates and documented limits |

Claims must state their level.

Example:

- Acceptable: `WASM browser path is L2 for tiny fixtures and L3 for measured local browser runs.`
- Acceptable: `OPFS cache is L3 for browser persistence.`
- Not acceptable: `OPFS avoids RAM use.`
- Not acceptable: `Browser inference does not use CPU.`
- Not acceptable: `WebGPU execution is production-ready` unless L5 evidence exists.

## Required claim rules

### 1. Browser-local

Allowed when:

- model artifacts can be acquired or loaded in the browser
- compile or run happens client-side
- no server inference is required for the measured path

Must still disclose:

- CPU/WASM execution if applicable
- memory ceiling
- model-size limits

### 2. OPFS-backed

Allowed when:

- artifacts are persisted in OPFS
- reload avoids network download or recompilation where applicable

Must not imply:

- zero-copy execution
- no RAM use
- native filesystem mmap behavior

### 3. Low-memory

Allowed only when:

- peak JS heap and WASM memory are measured
- copy boundaries are documented
- model size and browser are specified

### 4. Device-native

Allowed only when:

- WebGPU or WebNN backend is active
- kernel fallback count is reported
- no mid-graph CPU fallback exists for production claims
- device upload/readback counts are reported

### 5. Tokens/sec

Allowed only when paired with:

- model id
- browser
- OS
- backend mode
- prompt length
- generation length
- first-token latency
- steady decode window

## Acceptance criteria

- README and docs avoid unsupported `no CPU` or `no RAM` wording.
- Browser benchmark output records readiness level for each measured path.
- Claims in docs are linked to benchmark, test, or conformance evidence.
- WebGPU claims are blocked until backend mode and fallback counts are visible.
- OPFS claims distinguish storage persistence from runtime memory behavior.

## Non-goals

- This ADR does not reduce ambition.
- This ADR does not prevent experimental claims in clearly marked plans.
- This ADR does not require every doc to be rewritten immediately, but public-facing claims should be prioritized.

## Consequences

Positive:

- The project becomes easier to trust.
- Contributors know what evidence is required before broad claims are made.
- Performance language stays aligned with implementation reality.

Negative:

- Some current phrasing may need to become less flashy.
- More measurements are required before claiming readiness.

That is acceptable. Flashy unsupported claims are cheap. Credibility is expensive, which is why so few projects seem to buy any.
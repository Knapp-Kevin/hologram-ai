# ADR-0021: OPFS Storage and Runtime Boundary

**Status:** Proposed
**Date:** 2026-07-04
**Relates to:** ADR-0017, ADR-0019, ADR-0020, CONFORMANCE classes ZM, ZA, PV

## Context

The browser app uses OPFS to persist models, tokenizer assets, tensor chunks, and compiled `.holo` archives. This is a useful browser-native storage strategy.

However, OPFS is not automatically a runtime memory strategy. A file stored in OPFS still has to be read, copied, streamed, mapped, uploaded, or otherwise materialized before execution. Current paths often read artifacts into `Uint8Array` or WASM-owned buffers before compile or generation.

The repo should preserve the distinction between:

- persisted local storage
- streaming acquisition
- runtime execution memory
- device-resident buffers

Confusing these leads to inaccurate claims such as `OPFS means no RAM`. It does not. OPFS means local browser storage. RAM is still used when bytes are read and executed.

## Decision

OPFS is the browser persistence layer, not the runtime execution layer.

The browser runtime must explicitly define when bytes move from OPFS into:

- JS heap
- WASM linear memory
- Rust-owned vectors
- runtime buffer pools
- future WebGPU or WebNN device buffers

A future low-memory path should avoid full-archive reads where possible by splitting archive metadata from tensor payloads and resolving tensor payloads lazily or by address.

## Required design direction

### 1. Separate archive metadata from payload access

The `.holo` browser loader should evolve toward:

- small metadata read
- tensor section manifest
- content-addressed tensor references
- lazy or chunked payload access
- explicit failure when a requested payload cannot fit the current memory budget

### 2. Treat OPFS tensor chunks as first-class artifacts

Safetensors streaming already writes tensor chunks by κ label. This should become part of the runtime design rather than a download-side convenience.

Required metadata:

- original model id
- tensor key
- κ label
- dtype
- shape
- byte length
- source file
- source byte range if known
- integrity hash
- compile timestamp
- archive reference

### 3. Add cache integrity and recovery

OPFS cache must handle partial writes and interrupted downloads.

Required behavior:

- write temp files first
- finalize atomically where browser API allows
- verify expected byte length
- verify κ label after write
- detect missing tokenizer/config companions
- clean abandoned temp artifacts
- recover or redownload corrupt chunks

### 4. Add quota-aware behavior

Browser storage quota is not guaranteed. The app must detect and report:

- estimated quota
- used storage
- model cache footprint
- projected additional footprint before download or compile
- eviction candidates

Use `navigator.storage.estimate()` where available.

## Acceptance criteria

- OPFS cache metadata exists and can be listed from the browser app.
- Cached artifacts record κ label, byte length, model id, and source path.
- Interrupted download or compile does not leave an artifact marked complete.
- Cache integrity can be checked without re-running full inference.
- The app can report storage quota and model cache footprint.
- `.holo` archive loading has an explicit plan for moving away from full-buffer reads.

## Non-goals

- This ADR does not require a complete lazy `.holo` loader immediately.
- This ADR does not require browser mmap, which OPFS does not provide like a native filesystem.
- This ADR does not claim OPFS removes RAM use.

## Consequences

Positive:

- Browser storage becomes reliable enough for repeated local inference.
- The runtime can move toward chunked or address-resolved execution.
- Cache corruption becomes diagnosable instead of spooky.

Negative:

- Additional metadata and recovery logic are required.
- OPFS cannot be treated as a transparent native filesystem.

That is acceptable. Browsers are tiny operating systems in a trench coat. We should treat them accordingly.
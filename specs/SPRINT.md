# Current Sprint — hologram-ai

## Sprint Goal

Enforce compiler-only boundary: remove runtime code, add KV-cache and
multi-graph lowering, produce pipeline archives with named entrypoints.

**Design principle:** hologram-ai is a compiler only (ADR-0016). It ships
zero runtime code. All kernels (GEMM, attention, norms, etc.) belong in
hologram base crate. CLI: `compile`, `info`, `download` — nothing else.

---

## In Progress

- [ ] KV-cache ops: add `AiOp::KvSlotWrite`/`KvSlotRead` to IR
- [ ] KV-cache layout: `MemoryPlanner` computes real `KvCacheLayout` from arch params
- [ ] Multi-graph lowering: `LowerPhase` enum, prefill/decode graph emission
- [ ] Pipeline archive: `PipelineWriter` bundles prefill + decode sub-archives
- [ ] `LayerHeader` with named `lm.prefill`/`lm.decode` + tensor ports
- [ ] LLM meta section: `SECTION_LLM_META` (0x0011) embedding
- [ ] Tokenizer section: `SECTION_TOKENIZER` (0x1001) + `archive.rs` packing
- [ ] ConstantFolding: implement actual folding (currently no-op)

See `specs/plans/002-mvp-remaining.md` for full details.

---

## Done

- [x] Remove `InferenceSession` + structural cleanup (ADR-0016)
- [x] Symbolic shapes: `DimExpr` algebra, `DimVarTable`, `ConstraintStore` (ADR-0015)
- [x] Tokenizer expansion: Unigram (Viterbi), WordPiece, multi-algorithm dispatch (ADR-0012)
- [x] GGUF v2/v3 binary parser + metadata extraction (ADR-0006)
- [x] LlamaArch graph construction from GGUF tensors
- [x] Compiler rework: `HoloArchive` + `CompileStats` replacing `CompiledModel`
- [x] CLI: `inspect_gguf`, compile stats output
- [x] Shape propagation optimization pass (`ShapePropagation`)
- [x] Delete `Run` CLI command — users call `hologram run` directly
- [x] 60 tests passing, zero clippy warnings
- [x] Native `FloatOp` in hologram base crate (55 variants, kernels, dispatch, CLI inspect)
- [x] Lowering emits `GraphOp::Float(FloatOp::...)` for ALL ops (zero custom ops remaining)
- [x] Deleted `custom_ops.rs` — all 446 lines removed, no `CustomHandler` closures
- [x] Removed `CustomOpRegistry` from `LoweringOutput` — lowering is pure native ops
- [x] Op extensibility plan documented (`specs/plans/003-op-extensibility.md`)

See `specs/plans/001-spec-alignment-completed.md` for full details.

---

## Recently Unblocked

- **All ops are native FloatOp** — DONE. `FloatOp` expanded to 55 variants
  covering arithmetic, activations, trig, boolean, comparison, linear algebra,
  normalization, reductions, attention, embedding, dequantization, and structural
  ops. `custom_ops.rs` deleted entirely. `CustomOpRegistry` removed from
  `LoweringOutput`. Archives are fully self-describing; `hologram run` works
  out of the box.

## Still Blocked on hologram base crate

- **Shape metadata on graph edges** — hologram graphs have no per-edge
  shape/dtype, forcing shapes to be baked into closure captures
- **`LlmMetaSection`**, **`TokenizerSectionData`** — spec says these live
  in hologram. Workaround: local `EmbeddableSection` implementations.
- **`KvExecutor::execute_layer()`** — does not exist; manual sub-archive
  extraction required

---

## Notes

- CLI: exactly 3 commands — `compile`, `info`, `download`
- ONNX importer path still works (single-archive, non-pipeline)
- GGUF importer supports `llama`, `mistral`, `codellama`, `tinyllama` arch names
- No backwards compatibility concerns — can break APIs freely
- Future extensibility: op decomposition (now), serializable op descriptors (Phase 3), WASM kernels (Phase 4+). See `specs/plans/003-op-extensibility.md`.

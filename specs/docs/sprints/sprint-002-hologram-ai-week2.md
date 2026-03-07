# Sprint-002: hologram-ai Week 2 — ONNX Importer + More Quant Types

## Objective

Implement the ONNX importer, expand quant coverage to all common GGUF types,
and add shape propagation. By end of week, both ONNX and GGUF models should
be importable and runnable.

## Goals

- ONNX opset 13–17 importer handles BERT and GPT-2 class models
- Quant coverage expanded: Q4_0, Q4_1, Q5_0, Q8_0, Q2_K, Q4_K_M, Q6_K
- Shape propagation pass fills in symbolic dims where possible
- Golden tests pass for ONNX MatMul, LayerNorm, and BERT embedding fixtures
- GGML importer stub (parser only, no arch recognizer yet)

## Inputs

- ONNX prompt: `specs/prompts/hologram-ai/03-onnx-import-and-lowering.md`
- Week 1 deliverables (GGUF + single pass working)

## Deliverables

1. `hologram-ai-onnx` — full importer with op_map, shape_infer, data_resolver
2. Expanded `hologram-ai-quant` — 8 quant types with unit tests
3. `hologram-ai-opt` — `ShapePropagation` pass added
4. ONNX test fixtures: `matmul-f32.onnx`, `relu-f32.onnx`, `layernorm-f32.onnx`
5. `hologram-ai-ggml` — binary parser stub (no arch recognizer)
6. Integration tests: ONNX MatMul fixture runs end-to-end
7. Additional GGUF arch recognizers: `MistralArch`, `PhiArch`

## Tasks

### Day 1: ONNX proto setup

- [ ] Add `prost` dependency and `build.rs` to `hologram-ai-onnx`
- [ ] Download and commit `onnx.proto3`
- [ ] Verify `prost` generates bindings without errors

### Day 2: op_map + shape_infer

- [ ] Implement `op_map.rs` for 25+ core ops
- [ ] Implement `shape_infer.rs` — parse `value_info`, populate `TensorInfo`
- [ ] Handle `ConstantOfShape` and `Shape` ops → `AiOp::Constant` via folding

### Day 3: import.rs + test fixtures

- [ ] Implement `import_onnx()` full pipeline
- [ ] Generate Python fixture creation script
- [ ] Commit `tests/fixtures/onnx/matmul-f32.onnx`
- [ ] Unit test: `import_onnx(matmul_fixture)` produces 1 node of op `MatMul`

### Day 4: Expanded quant + ShapePropagation

- [ ] Implement `dequant_q4_1`, `dequant_q5_0`, `dequant_q5_1`
- [ ] Implement `dequant_q2_k`, `dequant_q4_k` (K-quant block format)
- [ ] Implement `dequant_q6_k`
- [ ] Unit test each against reference values from ggml-quants.h
- [ ] Implement `ShapePropagation` pass in `hologram-ai-opt`
- [ ] Test: shape propagation fills in MatMul output shape from input shapes

### Day 5: Integration + more arch recognizers

- [ ] Integration test: ONNX MatMul → run → compare output to expected zero result
- [ ] Implement `MistralArch` recognizer in `hologram-ai-gguf`
- [ ] Implement `PhiArch` recognizer in `hologram-ai-gguf`
- [ ] `hologram-ai-ggml` parser stub (parses header + tensors, no topology yet)
- [ ] `cargo test --workspace` passes

## Exit Criteria

- [ ] `hologram-ai-onnx` imports `matmul-f32.onnx` and produces valid `AiGraph`
- [ ] ONNX MatMul fixture runs end-to-end and produces `[2, 8]` output tensor
- [ ] All GGUF quant types listed above have unit-tested dequant implementations
- [ ] Mistral 7B GGUF imports without error (arch recognized, graph constructed)
- [ ] Shape propagation fills concrete shapes for ONNX fixtures
- [ ] `cargo clippy --workspace -- -D warnings` clean

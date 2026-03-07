# hologram-ai: Testing Strategy

---

## Guiding Principles

1. **Validate numerically, not structurally.** Tests must check that outputs are
   correct, not just that no panic occurred.

2. **Reference runtimes are the ground truth.** ONNX Runtime and llama.cpp are
   used as oracles for correctness, not as execution backends.

3. **Fixtures first.** Every test has a committed or script-reproducible fixture.
   Tests that require download are marked `#[ignore]` and run separately in CI.

4. **Tolerance policy must be explicit.** Every floating-point comparison
   declares its tolerance and the reason for it.

5. **Layer tests at every boundary.** Unit tests per crate. Integration tests at
   the importer-IR boundary and the IR-session boundary.

---

## Test Taxonomy

### Unit Tests (per-crate)

Each crate has tests in `crates/<crate>/tests/` or inline `#[cfg(test)]` modules.

| Crate | Primary test targets |
|-------|---------------------|
| `hologram-ai-ir` | `AiGraph` construction, serialization, structural validation |
| `hologram-ai-onnx` | op_map completeness, shape inference, specific op behaviors |
| `hologram-ai-gguf` | header parsing, metadata extraction, quant type mapping |
| `hologram-ai-ggml` | header parsing, tensor extraction |
| `hologram-ai-quant` | dequant numerical correctness (Q4_0, Q8_0, Q4_K, etc.) |
| `hologram-ai-opt` | each pass in isolation, pass idempotency |
| `hologram-ai-mem` | liveness intervals, alias detection, KV layout sizing |
| `hologram-ai-lower` | op dispatch table completeness, buffer binding correctness |
| `hologram-ai-session` | session creation, run() input/output validation |
| `hologram-ai-stream` | sampler distributions, stop sequence detection |

---

### Integration Tests (`tests/integration/`)

End-to-end tests that cross crate boundaries:

#### Import → IR round-trips

```rust
#[test]
fn onnx_matmul_import_produces_valid_graph() {
    let graph = import_onnx(include_bytes!("../fixtures/onnx/matmul-f32.onnx"), Default::default()).unwrap();
    assert_eq!(graph.nodes.len(), 1);
    assert!(matches!(graph.nodes[0].op, AiOp::MatMul));
}
```

#### Graph optimization invariants

```rust
#[test]
fn constant_folding_eliminates_const_shape_nodes() {
    let graph = import_onnx(SHAPE_GATHER_MODEL, Default::default()).unwrap();
    let opts = OptPipeline::default().run(graph).unwrap();
    // After constant folding, no Shape/Gather/ConstantOfShape nodes remain
    assert!(!opts.nodes.iter().any(|n| matches!(n.op, AiOp::Opaque { .. })));
}
```

#### Memory plan sanity

```rust
#[test]
fn memory_plan_total_bytes_is_deterministic() {
    let graph = import_gguf_tiny().unwrap();
    let plan1 = MemoryPlanner::new(Default::default()).plan(&graph).unwrap();
    let plan2 = MemoryPlanner::new(Default::default()).plan(&graph).unwrap();
    assert_eq!(plan1.total_weight_bytes, plan2.total_weight_bytes);
}
```

#### Full pipeline smoke test (CPU backend)

```rust
#[test]
fn gguf_tinyllama_single_forward_pass_cpu() {
    // Requires: tests/fixtures/gguf/tinyllama-tiny.gguf (committed synthetic)
    let model = ModelCompiler::default().compile(ModelSource::GgufPath("...")).unwrap();
    let mut session = model.session(Default::default()).unwrap();
    let inputs = hashmap!["input_ids" => Tensor::from(&[1u32, 2, 3, 4][..])];
    let outputs = session.run(inputs).unwrap();
    assert!(outputs["logits"].shape()[1] > 0);
}
```

---

### Importer Fixture Tests

Tiny/synthetic models committed into the repo:

| Fixture | Format | Purpose |
|---------|--------|---------|
| `matmul-f32.onnx` | ONNX | single MatMul node, known inputs/outputs |
| `relu-f32.onnx` | ONNX | single Relu, activation shape test |
| `attention-opset17.onnx` | ONNX | simplified MHA block |
| `tinyllama-tiny.gguf` | GGUF | 2-layer, 32-dim synthetic llama; Q4_0 |
| `phi-tiny.gguf` | GGUF | 2-layer, 32-dim synthetic phi; F16 |

Fixtures are generated via `scripts/gen-fixtures.py` (Python, uses `onnx`
and a custom GGUF writer). Committed at small size (<1MB each).

---

### Golden Tensor Tests

For each fixture, a `.npz` file stores known-good inputs and outputs:

```
tests/golden/
  matmul-f32/
    input.npz
    output.npz
  tinyllama-tiny/
    input_ids.npz
    logits.npz
```

Golden test harness:

```rust
fn run_golden_test(fixture: &str) {
    let (inputs, expected_outputs) = load_golden(fixture);
    let model = compile_fixture(fixture);
    let outputs = model.session(Default::default()).run(inputs).unwrap();
    for (key, expected) in &expected_outputs {
        let actual = &outputs[key];
        assert_tensors_close(actual, expected, Tolerance::f32_default());
    }
}
```

---

### Reference Runtime Comparison Tests

These tests run the same model through `hologram-ai` and a reference runtime
and compare outputs numerically.

They require external tooling and are `#[ignore]` by default.

```bash
# Run reference tests (requires ort and/or llama.cpp installed)
cargo test --test reference -- --ignored
```

#### ONNX reference comparison

```rust
#[test]
#[ignore = "requires ort CLI"]
fn onnx_bert_base_matches_onnxruntime() {
    let input = load_test_input("bert-base");
    let holo_out = run_hologram_ai("bert-base.onnx", &input);
    let ort_out = run_ort_cli("bert-base.onnx", &input);
    assert_tensors_close(&holo_out["logits"], &ort_out["logits"], Tolerance {
        max_abs_err: 1e-5,
        mean_abs_err: 1e-6,
        cosine_sim_min: 0.9999,
    });
}
```

#### GGUF reference comparison (token generation)

```rust
#[test]
#[ignore = "requires llama.cpp CLI + model download"]
fn gguf_tinyllama_top1_token_matches_llamacpp() {
    let prompt = "The capital of France is";
    let holo_token = run_hologram_ai_greedy("tinyllama-1.1b-q4.gguf", prompt, 1);
    let llama_token = run_llamacpp_greedy("tinyllama-1.1b-q4.gguf", prompt, 1);
    assert_eq!(holo_token, llama_token,
        "Greedy top-1 token must match llama.cpp reference");
}
```

---

## Tolerance Policy

### f32 models (ONNX F32, GGUF F32)

```rust
Tolerance {
    max_abs_err: 1e-5,
    mean_abs_err: 1e-6,
    cosine_sim_min: 0.9999,
}
```

### f16 models

```rust
Tolerance {
    max_abs_err: 1e-3,
    mean_abs_err: 1e-4,
    cosine_sim_min: 0.999,
}
```

### Quantized models (Q4_0, Q8_0, etc.)

```rust
Tolerance {
    // tolerance is relative to quantization noise floor
    max_abs_err: quant_noise_floor(scheme) * 2.0,
    mean_abs_err: quant_noise_floor(scheme),
    cosine_sim_min: 0.99,
}
```

For token generation validation: **top-1 greedy token must match** the reference
for the same model, same prompt, same temperature (greedy = 0 temperature).
Top-5 match is a warning, not a failure.

---

## Shape and DType Validators

Every imported `AiGraph` runs through structural validation before leaving the importer:

```rust
pub fn validate_graph(graph: &AiGraph) -> Result<Vec<ValidationWarning>> {
    // 1. All tensor IDs referenced in nodes must exist in tensor_info
    // 2. All input/output tensor IDs must exist
    // 3. No cycles (DAG validation)
    // 4. All params have TensorInfo
    // 5. Quantized params have QuantDescriptor != None
    // 6. No negative or zero-valued concrete dims
}
```

---

## CI Test Matrix

```yaml
# .github/workflows/ci.yml (proposed structure for hologram-ai repo)
jobs:
  unit_tests:
    runs-on: [ubuntu-latest, macos-latest]
    steps:
      - cargo test --workspace --exclude hologram-ai-cli

  integration_tests:
    runs-on: ubuntu-latest
    steps:
      - cargo test --test integration

  golden_tests:
    runs-on: ubuntu-latest
    steps:
      - cargo test --test golden

  reference_tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'   # nightly only
    steps:
      - ./scripts/download-test-models.sh
      - cargo test --test reference -- --ignored
```

---

## Quantization Dequant Unit Tests

The `hologram-ai-quant` crate must test software dequantization against known values:

```rust
#[test]
fn q4_0_dequant_matches_ggml_reference() {
    // Known Q4_0 block: 16 4-bit values + delta
    let block = Q4_0Block { delta: 1.5f16, quants: [0xAB, 0xCD, ...] };
    let actual = dequant_q4_0(&block);
    let expected = [/* precomputed reference values */];
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-3);
    }
}
```

One test per quant scheme, covering:
- Q4_0, Q4_1
- Q5_0, Q5_1
- Q8_0
- Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K
- IQ4_XS

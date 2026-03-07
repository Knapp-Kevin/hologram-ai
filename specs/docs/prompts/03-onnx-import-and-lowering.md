# Prompt: ONNX Import and Lowering

## Purpose

Implement the ONNX importer (`hologram-ai-onnx`) and expand the lowering table
to cover ONNX-sourced `AiGraph` nodes.

Run this prompt after Week 1 MVP is complete (GGUF + single pass working).

---

## Context

You are working in the `hologram-ai` repository.
The `hologram-ai-ir`, `hologram-ai-quant`, `hologram-ai-opt`, `hologram-ai-mem`, and
`hologram-ai-lower` crates are already implemented and passing tests.

Your task is to implement `hologram-ai-onnx` and integrate it with the existing pipeline.

Architecture reference:
- `../hologram-architecture/specs/projects/hologram-ai/import-pipeline.md` (ONNX section)
- `../hologram-architecture/specs/projects/hologram-ai/lowering.md` (op dispatch table)

---

## Task 1: Generate ONNX protobuf bindings

Add `prost` build script to `crates/hologram-ai-onnx/`:

```
crates/hologram-ai-onnx/
├── Cargo.toml
├── build.rs          ← generate from onnx.proto3
├── proto/
│   └── onnx.proto3   ← copy from ONNX repo
└── src/
    ├── lib.rs
    ├── import.rs
    ├── op_map.rs
    ├── shape_infer.rs
    └── data_resolver.rs
```

`build.rs`:
```rust
fn main() {
    prost_build::compile_protos(&["proto/onnx.proto3"], &["proto/"]).unwrap();
}
```

Download `onnx.proto3` from `https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3`
and commit to `proto/`.

---

## Task 2: Implement `op_map.rs`

Map ONNX `op_type` strings to `AiOp`. Implement the following mappings first
(covers BERT and GPT-2 class models):

```rust
pub fn map_op(op_type: &str, attrs: &[AttributeProto]) -> Result<AiOp> {
    match op_type {
        "MatMul" => Ok(AiOp::MatMul),
        "Gemm" => { /* parse alpha, beta, transA, transB attrs */ },
        "Add" => Ok(AiOp::Add),
        "Mul" => Ok(AiOp::Mul),
        "Relu" => Ok(AiOp::Relu),
        "Gelu" => Ok(AiOp::Gelu),
        "Tanh" => Ok(AiOp::Tanh),
        "Sigmoid" => Ok(AiOp::Sigmoid),
        "Softmax" => { /* parse axis attr */ },
        "LayerNormalization" => { /* parse axis, epsilon attrs */ },
        "Reshape" => { /* parse allow_zero attr */ },
        "Transpose" => { /* parse perm attr */ },
        "Gather" => { /* parse axis attr */ },
        "Slice" => { /* parse starts/ends/axes/steps from inputs */ },
        "Concat" => { /* parse axis attr */ },
        "Split" => { /* parse axis, outputs attrs */ },
        "Cast" => { /* parse to attr → DType */ },
        "Unsqueeze" => { /* parse axes */ },
        "Squeeze" => { /* parse axes */ },
        "ReduceMean" => { /* parse axes, keepdims */ },
        "Shape" => Ok(AiOp::Opaque { op_type: "Shape".into(), raw_attrs: vec![] }),
        "Constant" | "ConstantOfShape" => { /* extract value → AiOp::Constant */ },
        // Attention (onnxruntime contrib op)
        "Attention" => { /* parse num_heads */ },
        _ => Ok(AiOp::Opaque { op_type: op_type.to_owned(), raw_attrs: encode_attrs(attrs) }),
    }
}
```

Expand this table incrementally. Every model that fails to lower due to
an `Opaque` node is an opportunity to add a new entry.

---

## Task 3: Implement `shape_infer.rs`

Forward shape propagation from ONNX `value_info` hints.

Start simple: parse `GraphProto.value_info` and `GraphProto.output` for
shape/type info. Use these to populate `TensorInfo` in the imported `AiGraph`.

For tensors without shape info: emit `TensorInfo { shape: vec![Dim::Dynamic] }`.

Run `OptPipeline::default()` including `ShapePropagation` pass after import
to fill in as many shapes as possible from the initializer types.

---

## Task 4: Implement `import.rs`

```rust
pub fn import_onnx(bytes: &[u8], opts: OnnxImportOptions) -> Result<AiGraph, ImportError> {
    let model_proto = onnx::ModelProto::decode(bytes)?;
    let graph_proto = model_proto.graph.ok_or(ImportError::MissingGraph)?;

    // Check opset version
    validate_opset(&model_proto.opset_import)?;

    // Build tensor info from value_info
    let mut tensor_info = build_tensor_info(&graph_proto);

    // Extract initializers (weights)
    let mut params = extract_initializers(&graph_proto, &opts)?;

    // Walk nodes
    let mut nodes = Vec::new();
    for node_proto in &graph_proto.node {
        let op = map_op(&node_proto.op_type, &node_proto.attribute)?;
        nodes.push(AiNode {
            id: NodeId::new(),
            op,
            inputs: node_proto.input.iter().map(|s| TensorId::from(s)).collect(),
            outputs: node_proto.output.iter().map(|s| TensorId::from(s)).collect(),
            attrs: NodeAttrs::default(),
        });
    }

    let graph = AiGraph {
        name: graph_proto.name.clone(),
        nodes,
        inputs: graph_proto.input.iter().map(|i| TensorId::from(&i.name)).collect(),
        outputs: graph_proto.output.iter().map(|o| TensorId::from(&o.name)).collect(),
        params,
        tensor_info,
        metadata: Default::default(),
        warnings: vec![],
    };

    graph.validate()?;
    Ok(graph)
}
```

---

## Task 5: Test fixtures

Commit these tiny ONNX test fixtures:

1. `tests/fixtures/onnx/matmul-f32.onnx` — single MatMul node
2. `tests/fixtures/onnx/relu-f32.onnx` — single Relu node
3. `tests/fixtures/onnx/layernorm-f32.onnx` — LayerNorm node

Generate with Python:
```python
import onnx
from onnx import helper, TensorProto

# matmul-f32.onnx
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 4])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [4, 8])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 8])
node = helper.make_node('MatMul', inputs=['X', 'W'], outputs=['Y'])
graph = helper.make_graph([node], 'matmul', [X, W], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx.save(model, 'tests/fixtures/onnx/matmul-f32.onnx')
```

---

## Task 6: Integration test

```rust
#[test]
fn onnx_matmul_import_and_run_cpu() {
    let bytes = include_bytes!("../fixtures/onnx/matmul-f32.onnx");
    let graph = import_onnx(bytes, Default::default()).unwrap();
    assert_eq!(graph.nodes.len(), 1);
    assert!(matches!(graph.nodes[0].op, AiOp::MatMul));

    let model = ModelCompiler::default().compile(ModelSource::AiGraph(graph)).unwrap();
    let mut session = model.session(Default::default()).unwrap();
    let x = Tensor::zeros(&[2, 4], DType::F32);
    let w = Tensor::zeros(&[4, 8], DType::F32);
    let outputs = session.run(hashmap!["X" => x, "W" => w]).unwrap();
    assert_eq!(outputs["Y"].shape(), &[2, 8]);
}
```

---

## Acceptance Criteria

- `cargo test -p hologram-ai-onnx` passes
- MatMul, Add, Relu, LayerNorm, Softmax, Reshape, Gather, Concat all import correctly
- Integration test passes for MatMul fixture
- No `Opaque` nodes in the above op list after import
- `cargo clippy -p hologram-ai-onnx -- -D warnings` clean

# Plan 031: ONNX Import Parameter Inference

## Context

SD UNet fails at runtime because the first Conv2d outputs the wrong size. Root cause: the ONNX exporter (torch) omits the `kernel_shape` attribute from Conv nodes (it's optional in ONNX — should be inferred from weight tensor). Our importer uses `unwrap_or_default()` → empty vec → shape propagation fails → compiled shape is wrong → runtime produces wrong output → cascades through entire graph.

This is NOT Conv-specific. The pattern affects any op where ONNX attributes are optional and should be inferred from weight/input shapes:
- Conv: `kernel_shape` from weight `[C_out, C_in/g, kH, kW]`
- ConvTranspose: same
- BatchNormalization: already decomposed, but similar pattern
- Pooling: `kernel_shape` sometimes omitted

## Design: Post-Import Parameter Resolution Pass

Rather than fixing individual ops in `op_map.rs`, add a dedicated pass that runs immediately after ONNX import and BEFORE any optimization:

**File:** New `crates/hologram-ai-onnx/src/resolve_op_params.rs`

```rust
pub fn resolve_op_params(graph: &mut AiGraph) {
    for node in &mut graph.nodes {
        match &mut node.op {
            AiOp::Conv { kernel_shape, .. } if kernel_shape.is_empty() => {
                // Infer from weight tensor: input[1] shape = [Co, Ci/g, kH, kW]
                if let Some(weight_shape) = weight_spatial_shape(node, graph, 1) {
                    *kernel_shape = weight_shape;
                }
            }
            AiOp::ConvTranspose { kernel_shape, .. } if kernel_shape.is_empty() => {
                if let Some(weight_shape) = weight_spatial_shape(node, graph, 1) {
                    *kernel_shape = weight_shape;
                }
            }
            // Add more ops as needed
            _ => {}
        }
    }
}
```

The helper `weight_spatial_shape(node, graph, input_idx)` reads the weight param's shape from `tensor_info` and extracts `dims[2:]` (spatial dims).

## Why This Approach

1. **Single responsibility**: The importer handles ONNX parsing, the resolution pass handles inference. No logic duplication.
2. **Extensible**: New ops with optional params are one match arm away.
3. **Testable**: Pure function on AiGraph, easy to unit test.
4. **Non-model-specific**: Fixes ALL models exported by torch, not just SD UNet.

## Verification

- SD UNet Conv output should be `[1,320,64,64]` (5242880 bytes), not `[320]`
- TinyLlama unaffected (no Conv ops)
- ResNet-50 unaffected (kernel_shape already present from opset)
- BERT unaffected (no Conv ops)

## Critical Files

| File | Change |
|------|--------|
| `crates/hologram-ai-onnx/src/resolve_op_params.rs` | NEW — parameter inference |
| `crates/hologram-ai-onnx/src/lib.rs` | Wire after import |
| `crates/hologram-ai-common/src/opt/pipeline.rs` | OR wire as first pass in both pipelines |

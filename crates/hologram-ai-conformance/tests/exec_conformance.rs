//! Execution conformance tests — compile ONNX → hologram, run both, compare.
//!
//! These tests validate the full compile → lower → execute pipeline by
//! comparing hologram executor output against ORT on multi-node ONNX models.
//!
//! Feature-gated behind `conformance` (requires ORT runtime).

#![cfg(feature = "conformance")]

use hologram_ai::{ModelCompiler, ModelSource};
use hologram_ai_conformance::ort_runner::onnx_builder;
use hologram_ai_conformance::ort_runner::runner::{run_onnx_all_outputs, OrtInput};
use hologram_ai_conformance::tolerance::Tolerance;

/// Default tolerance for execution conformance (slightly looser than kernel tests
/// since errors can accumulate across nodes).
fn exec_tol() -> Tolerance {
    Tolerance {
        atol: 1e-4,
        rtol: 1e-3,
    }
}

/// Helper: compile ONNX bytes, execute through hologram, return final output f32s.
fn compile_and_execute(
    model_bytes: &[u8],
    inputs: &[(&str, Vec<usize>, Vec<f32>)],
) -> Vec<f32> {
    let compiler = ModelCompiler::default();
    let (archive, _debug_map) = compiler
        .compile_with_debug_info(ModelSource::OnnxBytes(model_bytes.to_vec()))
        .expect("compilation failed");

    // Build GraphInputs.
    let mut graph_inputs = hologram::GraphInputs::new();
    for (i, (_name, shape, data)) in inputs.iter().enumerate() {
        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        graph_inputs.set_with_shape(i as u32, bytes, shape.clone());
    }

    // Execute.
    let plan = hologram::load_from_bytes(&archive.bytes).expect("loading archive");
    let outputs = hologram::execute_plan(&plan, &graph_inputs).expect("execution failed");

    // Extract first output as f32.
    let (_, out_bytes) = outputs.get(0).expect("no outputs");
    bytemuck::cast_slice::<u8, f32>(out_bytes).to_vec()
}

/// Test: debug map is populated for a compiled model.
#[test]
fn debug_map_populated_for_matmul() {
    let model_bytes = onnx_builder::matmul(2, 4, 3);
    let compiler = ModelCompiler::default();
    let (_archive, debug_map) = compiler
        .compile_with_debug_info(ModelSource::OnnxBytes(model_bytes))
        .expect("compile failed");
    assert!(
        !debug_map.name_to_idx.is_empty(),
        "debug map should not be empty"
    );
}

/// Test: MatMul output matches ORT.
#[test]
fn matmul_matches_ort() {
    let m = 2;
    let k = 4;
    let n = 3;
    let model_bytes = onnx_builder::matmul(m, k, n);

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.1).collect();

    // ORT reference.
    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "A".into(), shape: vec![m, k], data: a_data.clone() },
            OrtInput { name: "B".into(), shape: vec![k, n], data: b_data.clone() },
        ],
    )
    .expect("ORT failed");

    // Hologram.
    let holo_out = compile_and_execute(
        &model_bytes,
        &[("A", vec![m, k], a_data), ("B", vec![k, n], b_data)],
    );

    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        exec_tol(),
    );
    assert!(cmp.passed, "MatMul mismatch: {}", cmp.message);
}

/// Test: Softmax output matches ORT.
#[test]
fn softmax_matches_ort() {
    let rows = 2;
    let size = 8;
    let model_bytes = onnx_builder::softmax(rows, size);

    let input_data: Vec<f32> = (0..rows * size).map(|i| (i as f32) * 0.5 - 3.0).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![OrtInput { name: "input".into(), shape: vec![rows, size], data: input_data.clone() }],
    )
    .expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("input", vec![rows, size], input_data)],
    );

    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        exec_tol(),
    );
    assert!(cmp.passed, "Softmax mismatch: {}", cmp.message);
}

/// Test: RmsNorm composite model (6 nodes) matches ORT.
#[test]
fn rmsnorm_composite_matches_ort() {
    let rows = 2;
    let size = 16;
    let eps = 1e-6;
    let model_bytes = onnx_builder::rms_norm(rows, size, eps);

    let x_data: Vec<f32> = (0..rows * size).map(|i| (i as f32) * 0.1 - 0.8).collect();
    let w_data: Vec<f32> = (0..size).map(|i| 1.0 + (i as f32) * 0.01).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "X".into(), shape: vec![rows, size], data: x_data.clone() },
            OrtInput { name: "Weight".into(), shape: vec![size], data: w_data.clone() },
        ],
    )
    .expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("X", vec![rows, size], x_data), ("Weight", vec![size], w_data)],
    );

    // Composite ops: slightly looser tolerance.
    let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out, &ort_outputs[0].data, tol,
    );
    assert!(cmp.passed, "RmsNorm mismatch: {}", cmp.message);
}

/// Test: Gemm with transB matches ORT.
#[test]
fn gemm_trans_b_matches_ort() {
    let m = 3;
    let k = 4;
    let n = 2;
    let model_bytes = onnx_builder::gemm(m, k, n, 1.0, 1.0, false, true);

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.05).collect();
    let c_data: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "A".into(), shape: vec![m, k], data: a_data.clone() },
            OrtInput { name: "B".into(), shape: vec![n, k], data: b_data.clone() },
            OrtInput { name: "C".into(), shape: vec![m, n], data: c_data.clone() },
        ],
    )
    .expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("A", vec![m, k], a_data), ("B", vec![n, k], b_data), ("C", vec![m, n], c_data)],
    );

    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out, &ort_outputs[0].data, exec_tol(),
    );
    assert!(cmp.passed, "Gemm transB mismatch: {}", cmp.message);
}

/// Test: LayerNorm composite model (9 nodes) matches ORT.
#[test]
fn layernorm_composite_matches_ort() {
    let rows = 2;
    let size = 16;
    let eps = 1e-5;
    let model_bytes = onnx_builder::layer_norm(rows, size, eps);

    let x_data: Vec<f32> = (0..rows * size).map(|i| (i as f32) * 0.1 - 0.8).collect();
    let w_data: Vec<f32> = (0..size).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "X".into(), shape: vec![rows, size], data: x_data.clone() },
            OrtInput { name: "Weight".into(), shape: vec![size], data: w_data.clone() },
            OrtInput { name: "Bias".into(), shape: vec![size], data: b_data.clone() },
        ],
    )
    .expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[
            ("X", vec![rows, size], x_data),
            ("Weight", vec![size], w_data),
            ("Bias", vec![size], b_data),
        ],
    );

    let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out, &ort_outputs[0].data, tol,
    );
    assert!(cmp.passed, "LayerNorm mismatch: {}", cmp.message);
}

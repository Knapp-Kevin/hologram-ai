//! End-to-end ViT patch prune tests — compile, prune, execute.
//!
//! Builds a minimal ViT-shaped ONNX model, compiles with patch pruning,
//! and verifies the full pipeline: PatchPruneInjection pass → archive →
//! HoloRunner preprocessing → Gather-based execution.
//!
//! No external model files or ORT required.
//!
//! Run with:
//!   cargo test -p hologram-ai --test patch_prune_e2e -- --nocapture

use hologram_ai::compiler::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_conformance::ort_runner::onnx_builder::{
    build_multi_node_model, AttrVal, Initializer, Node,
};

/// Build a minimal ViT-shaped ONNX model.
///
/// pixel_values [1, 3, 32, 32] → Conv2d(kernel=16, stride=16) →
///   [1, 8, 2, 2] → Reshape → [1, 4, 8] → Add(pos_embed) → output [1, 4, 8]
///
/// 2×2 grid = 4 patches, embed_dim = 8.
fn build_mini_vit_onnx() -> Vec<u8> {
    let embed_dim = 8usize;
    let patch_size = 16usize;
    let n_patches = 4usize;

    let conv_weight_len = embed_dim * 3 * patch_size * patch_size;
    let conv_weights: Vec<f32> = (0..conv_weight_len)
        .map(|i| ((i as f32) * 0.001).sin() * 0.1)
        .collect();
    let conv_bias = vec![0.0f32; embed_dim];
    let reshape_target = vec![1i64, n_patches as i64, embed_dim as i64];
    let pos_embed: Vec<f32> = (0..(n_patches * embed_dim))
        .map(|i| ((i as f32) * 0.01).cos() * 0.05)
        .collect();

    let nodes = vec![
        Node::with_attrs(
            "Conv",
            &["pixel_values", "conv_weight", "conv_bias"],
            &["conv_out"],
            &[
                (
                    "kernel_shape",
                    AttrVal::Ints(vec![patch_size as i64, patch_size as i64]),
                ),
                (
                    "strides",
                    AttrVal::Ints(vec![patch_size as i64, patch_size as i64]),
                ),
            ],
        ),
        Node::new("Reshape", &["conv_out", "reshape_shape"], &["patches"]),
        Node::new("Add", &["patches", "pos_embed"], &["output"]),
    ];

    let initializers = vec![
        Initializer::float_nd(
            "conv_weight",
            conv_weights,
            vec![embed_dim, 3, patch_size, patch_size],
        ),
        Initializer::float_nd("conv_bias", conv_bias, vec![embed_dim]),
        Initializer::int64_1d("reshape_shape", reshape_target),
        Initializer::float_nd("pos_embed", pos_embed, vec![1, n_patches, embed_dim]),
    ];

    build_multi_node_model(
        &nodes,
        &[("pixel_values", &[1, 3, 32, 32])],
        &[("output", &[1, n_patches, embed_dim])],
        &initializers,
    )
}

/// Compile with pruning disabled → output is full [1, 4, 8].
#[test]
fn no_prune_full_output() {
    let model_bytes = build_mini_vit_onnx();
    let compiler = ModelCompiler {
        patch_budget_ratio: None,
        ..ModelCompiler::default()
    };
    let archive = compiler
        .compile(ModelSource::OnnxBytes(model_bytes))
        .expect("compilation should succeed");

    let pixel_data: Vec<f32> = (0..3072).map(|i| ((i as f32) * 0.001).sin()).collect();
    let mut inputs = hologram::GraphInputs::new();
    inputs.set_with_shape(
        0,
        bytemuck::cast_slice(&pixel_data).to_vec(),
        vec![1, 3, 32, 32],
    );

    let runner = HoloRunner::from_bytes(archive.bytes).expect("loading runner");
    let outputs = runner.execute(&inputs).expect("execution failed");
    let (_, out_bytes) = outputs.get(0).expect("no output");

    let n_floats = out_bytes.len() / 4;
    assert_eq!(n_floats, 32, "expected 32 floats (1×4×8), got {n_floats}");
    eprintln!("no_prune_full_output: PASS ({n_floats} floats)");
}

/// Compile with budget=0.5 → output is [1, 2, 8].
/// PatchPrune preprocessing runs automatically inside execute().
#[test]
fn prune_50pct_reduces_output() {
    let model_bytes = build_mini_vit_onnx();
    let compiler = ModelCompiler {
        patch_budget_ratio: Some(0.5),
        ..ModelCompiler::default()
    };
    let archive = compiler
        .compile(ModelSource::OnnxBytes(model_bytes))
        .expect("compilation should succeed");

    // max_kept = ceil(4 * 0.5) = 2
    let max_kept = 2usize;
    let embed_dim = 8usize;

    // Solid-color image → PatchPrune keeps only anchor + padding.
    let pixel_data = vec![0.5f32; 3072];
    let mut inputs = hologram::GraphInputs::new();
    inputs.set_with_shape(
        0,
        bytemuck::cast_slice(&pixel_data).to_vec(),
        vec![1, 3, 32, 32],
    );

    let runner = HoloRunner::from_bytes(archive.bytes).expect("loading runner");
    assert!(
        runner.has_patch_prune(),
        "runner should have patch pruning enabled"
    );
    let outputs = runner.execute(&inputs).expect("execution failed");
    let (_, out_bytes) = outputs.get(0).expect("no output");

    let n_floats = out_bytes.len() / 4;
    let expected = max_kept * embed_dim;
    assert_eq!(
        n_floats, expected,
        "expected {expected} floats (1×{max_kept}×{embed_dim}), got {n_floats}"
    );
    eprintln!("prune_50pct_reduces_output: PASS ({n_floats} floats, max_kept={max_kept})");
}

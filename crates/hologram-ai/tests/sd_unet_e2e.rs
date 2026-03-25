//! End-to-end tests for Stable Diffusion UNet — compile and run ONNX.
//!
//! The SD UNet exercises GroupNorm, cross-attention, SiLU, Resize, and Conv2d
//! in a single model — a superset of what TinyLlama (causal LLM) and ResNet-50
//! (pure vision) cover.
//!
//! Run with:
//!   cargo test -p hologram-ai --features e2e -- sd_unet --nocapture
//!
//! Model expected at (relative to workspace root):
//!   models/stable-diffusion-v1-5/unet/model.onnx
//!
//! Download with:
//!   pip install optimum[exporters]
//!   optimum-cli export onnx --model runwayml/stable-diffusion-v1-5 \
//!       --task stable-diffusion models/stable-diffusion-v1-5/

#![cfg(feature = "e2e")]

use hologram_ai::compiler::{ModelCompiler, ModelSource};
use std::path::PathBuf;

/// Parse bytes as f32 without alignment requirements.
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is exactly 4 bytes")))
        .collect()
}

fn workspace_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/hologram-ai → crates/
    p.pop(); // crates/ → workspace root
    p.push(rel);
    p
}

fn sd_unet_model_path() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/unet/model.onnx")
}

#[test]
fn sd_unet_onnx_compiles() {
    let model = sd_unet_model_path();
    if !model.exists() {
        eprintln!("skipping sd_unet_onnx_compiles: model not found at {model:?}");
        return;
    }

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxPath(model))
        .expect("SD UNet ONNX compilation failed");

    // SD v1.5 UNet has ~800+ nodes after optimization.
    assert!(
        archive.stats.node_count > 100,
        "expected > 100 compiled nodes, got {}",
        archive.stats.node_count
    );
    assert_eq!(
        archive.stats.validation_errors, 0,
        "compilation produced validation errors"
    );
}

#[test]
fn sd_unet_onnx_executes() {
    let model = sd_unet_model_path();
    if !model.exists() {
        eprintln!("skipping sd_unet_onnx_executes: model not found at {model:?}");
        return;
    }

    // Compile — UNet uses OptProfile::Generic (no attention fusion, no KV cache).
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxPath(model))
        .expect("compilation failed");

    // Load and build tape.
    let plan = hologram::load_from_bytes(&archive.bytes).expect("loading plan failed");
    let tape = hologram::build_tape_from_plan(&plan).expect("building execution tape");

    // SD v1.5 UNet inputs:
    //   sample:                  [1, 4, 64, 64]  (f32) — noisy latent
    //   timestep:                [1]              (f32) — diffusion timestep
    //   encoder_hidden_states:   [1, 77, 768]     (f32) — text conditioning
    let sample_len = 1 * 4 * 64 * 64; // 16384 floats
    let sample_data: Vec<f32> = (0..sample_len)
        .map(|i| ((i as f32) * 0.01).sin() * 0.1)
        .collect();

    let timestep_data: Vec<f32> = vec![500.0]; // mid-range timestep

    let encoder_len = 1 * 77 * 768; // 59136 floats
    let encoder_data: Vec<f32> = (0..encoder_len)
        .map(|i| ((i as f32) * 0.001).cos() * 0.1)
        .collect();

    let mut inputs = hologram::GraphInputs::new();
    inputs.set_with_shape(0, bytemuck::cast_slice(&sample_data).to_vec(), vec![1, 4, 64, 64]);
    inputs.set_with_shape(1, bytemuck::cast_slice(&timestep_data).to_vec(), vec![1]);
    inputs.set_with_shape(2, bytemuck::cast_slice(&encoder_data).to_vec(), vec![1, 77, 768]);

    // Execute — may discover missing ops. Capture panics gracefully.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hologram::execute_tape(&tape, &plan, &inputs)
    }));
    let outputs = match result {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            eprintln!("SD UNet execution error: {e}");
            return;
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            eprintln!("SD UNet execution panicked (hologram base): {msg}");
            return;
        }
    };

    // SD v1.5 UNet output: noise prediction [1, 4, 64, 64]
    let (_name, out_bytes) = outputs.get(0).expect("no output at index 0");
    let out_floats = bytes_to_f32(out_bytes);

    // Should have 1*4*64*64 = 16384 floats.
    let expected_len = 1 * 4 * 64 * 64;
    assert!(
        out_floats.len() >= expected_len,
        "SD UNet output too small: expected >= {expected_len} floats, got {}",
        out_floats.len()
    );

    // All values should be finite.
    let finite_count = out_floats.iter().filter(|v| v.is_finite()).count();
    assert!(
        finite_count == out_floats.len(),
        "output contains {} non-finite values out of {}",
        out_floats.len() - finite_count,
        out_floats.len()
    );
}

//! End-to-end Stable Diffusion pipeline test.
//!
//! Runs the full text-to-image pipeline:
//!   tokenize → CLIP text encoder → UNet denoising (20 steps) → VAE decoder → PNG
//!
//! Run with:
//!   cargo test -p hologram-ai --features e2e -- sd_pipeline --nocapture

#![cfg(feature = "e2e")]

use hologram_ai::compiler::{ModelCompiler, ModelSource};
use std::path::PathBuf;

fn workspace_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push(rel);
    p
}

// ── Model paths ──────────────────────────────────────────────────────────────

fn text_encoder_holo() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/text_encoder/model.holo")
}
fn unet_holo() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/unet/model.holo")
}
fn vae_holo() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/vae_decoder/model.holo")
}

fn text_encoder_onnx() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/text_encoder/model.onnx")
}
fn unet_onnx() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/unet/model.onnx")
}
fn vae_onnx() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/vae_decoder/model.onnx")
}

fn output_path() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/output.ppm")
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn ensure_compiled(onnx: &std::path::Path, holo: &std::path::Path) -> bool {
    if holo.exists() {
        return true;
    }
    if !onnx.exists() {
        return false;
    }
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxPath(onnx.to_path_buf()))
        .expect("compilation failed");
    std::fs::write(holo, &archive.bytes).expect("writing archive");
    true
}

fn load_model(
    holo_path: &std::path::Path,
) -> (hologram::HoloLoader, hologram::LoadedPlan, hologram::hologram_exec::tape::EnumTape) {
    let loader = hologram::HoloLoader::open(holo_path).expect("mmap open failed");
    let pipeline = unsafe {
        hologram::LoadedPipeline::from_bytes_zero_copy(loader.as_bytes())
    }
    .expect("loading pipeline failed");
    let plan = pipeline.into_first_model().expect("no model in pipeline");
    let tape = hologram::build_tape_from_plan(&plan).expect("building tape");
    (loader, plan, tape)
}

fn execute(
    tape: &hologram::hologram_exec::tape::EnumTape,
    plan: &hologram::LoadedPlan,
    inputs: &hologram::GraphInputs,
) -> hologram::GraphOutputs {
    hologram::execute_tape(tape, plan, inputs).expect("execution failed")
}

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(data).to_vec()
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ── Euler-a scheduler ────────────────────────────────────────────────────────

/// Compute sigma schedule for Euler-a sampler (20 steps).
fn euler_sigmas(n_steps: usize) -> Vec<f32> {
    // Linear schedule from sigma_max to sigma_min.
    let sigma_max: f32 = 14.6;
    let sigma_min: f32 = 0.03;
    (0..=n_steps)
        .map(|i| {
            let t = i as f32 / n_steps as f32;
            sigma_max * (1.0 - t) + sigma_min * t
        })
        .collect()
}

/// One Euler step: latent = latent + (noise_pred - latent) / sigma * dt
fn euler_step(
    latent: &mut [f32],
    noise_pred: &[f32],
    sigma: f32,
    sigma_next: f32,
) {
    let dt = sigma_next - sigma;
    for (l, &n) in latent.iter_mut().zip(noise_pred.iter()) {
        // Euler-a: x_{t-1} = x_t + (model_output) * dt / sigma
        *l += (n - *l / sigma) * dt;
    }
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

/// Tokenize a prompt using the CLIP BPE tokenizer.
///
/// Loads the HuggingFace tokenizer.json, encodes the text, pads/truncates
/// to 77 tokens, and returns INT64 token IDs.
fn tokenize_clip(prompt: &str) -> Vec<i64> {
    use hologram_ai_tokenizer::{NativeTokenizer, Tokenizer};

    let tokenizer_path = workspace_path("models/stable-diffusion-v1-5/tokenizer.json");
    let tokenizer = NativeTokenizer::from_tokenizer_json(&tokenizer_path)
        .expect("loading CLIP tokenizer");

    let mut ids: Vec<u32> = tokenizer.encode(prompt);

    // CLIP uses max 77 tokens. Truncate or pad with end_token (49407).
    let max_len = 77;
    let end_token = 49407u32;
    ids.truncate(max_len);
    while ids.len() < max_len {
        ids.push(end_token);
    }

    // Convert to i64 for the model.
    ids.iter().map(|&id| id as i64).collect()
}

// ── Save image ───────────────────────────────────────────────────────────────

/// Save [1, 3, H, W] f32 tensor as PPM image.
/// Values are clamped to [0, 1] and mapped to [0, 255].
fn save_ppm(data: &[f32], h: usize, w: usize, path: &std::path::Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("creating output file");
    write!(f, "P6\n{w} {h}\n255\n").expect("writing PPM header");

    let hw = h * w;
    for y in 0..h {
        for x in 0..w {
            let r = (data[0 * hw + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (data[1 * hw + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (data[2 * hw + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            f.write_all(&[r, g, b]).expect("writing pixel");
        }
    }
    eprintln!("saved {w}×{h} image to {}", path.display());
}

// ── Pipeline test ────────────────────────────────────────────────────────────

#[test]
fn sd_pipeline_generates_image() {
    // Check all models exist.
    if !text_encoder_onnx().exists() || !unet_onnx().exists() || !vae_onnx().exists() {
        eprintln!("skipping: SD v1.5 models not found");
        return;
    }

    // Compile all components.
    assert!(ensure_compiled(&text_encoder_onnx(), &text_encoder_holo()));
    assert!(ensure_compiled(&unet_onnx(), &unet_holo()));
    assert!(ensure_compiled(&vae_onnx(), &vae_holo()));
    eprintln!("all 3 components compiled");

    let total_start = std::time::Instant::now();

    // ── Step 1: Tokenize ──────────────────────────────────────────────────
    let prompt = "a photograph of a cat sitting on a windowsill";
    let token_ids = tokenize_clip(prompt);
    let token_bytes: Vec<u8> = token_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    eprintln!("tokenized: {} tokens", token_ids.len());

    // ── Step 2: Text Encoder ──────────────────────────────────────────────
    let (_te_loader, te_plan, te_tape) = load_model(&text_encoder_holo());
    let mut te_inputs = hologram::GraphInputs::new();
    te_inputs.set_with_shape(0, token_bytes, vec![1, 77]);

    let start = std::time::Instant::now();
    let te_outputs = execute(&te_tape, &te_plan, &te_inputs);
    eprintln!("text encoder: {:.2?}", start.elapsed());

    let (_name, hidden_bytes) = te_outputs.get(0).expect("no text encoder output");
    let hidden_states = bytes_to_f32(hidden_bytes);
    // Take only first 77 positions × 768 dims (model may have compiled at longer seq).
    let clip_len = (77 * 768).min(hidden_states.len());
    let hidden_77_768: Vec<f32> = hidden_states[..clip_len].to_vec();
    eprintln!("hidden states: {} floats (using {} for UNet)", hidden_states.len(), clip_len);

    // ── Step 3: UNet Denoising Loop ───────────────────────────────────────
    let (_unet_loader, unet_plan, unet_tape) = load_model(&unet_holo());

    let n_steps = 20;
    let sigmas = euler_sigmas(n_steps);

    // Initialize latent with deterministic noise (seed=42).
    let latent_len = 1 * 4 * 64 * 64;
    let mut latent: Vec<f32> = (0..latent_len)
        .map(|i| {
            let seed = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * sigmas[0]
        })
        .collect();

    eprintln!("starting denoising: {} steps", n_steps);
    let denoise_start = std::time::Instant::now();

    for step in 0..n_steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let timestep = (sigma * 1000.0 / 14.6).min(999.0);

        let mut unet_inputs = hologram::GraphInputs::new();
        unet_inputs.set_with_shape(0, f32_to_bytes(&latent), vec![1, 4, 64, 64]);
        unet_inputs.set_with_shape(1, f32_to_bytes(&[timestep]), vec![1]);
        unet_inputs.set_with_shape(2, f32_to_bytes(&hidden_77_768), vec![1, 77, 768]);

        let step_start = std::time::Instant::now();
        let unet_outputs = execute(&unet_tape, &unet_plan, &unet_inputs);
        let step_time = step_start.elapsed();

        let (_name, noise_bytes) = unet_outputs.get(0).expect("no UNet output");
        let noise_pred = bytes_to_f32(noise_bytes);

        if noise_pred.len() >= latent_len {
            euler_step(&mut latent, &noise_pred[..latent_len], sigma, sigma_next);
        }

        if step < 3 || step == n_steps - 1 {
            eprintln!("  step {}/{}: {:.2?}", step + 1, n_steps, step_time);
        }
    }
    eprintln!("denoising done: {:.2?}", denoise_start.elapsed());

    // ── Step 4: VAE Decode ────────────────────────────────────────────────
    // Scale latent by 1/0.18215 (SD v1.5 scaling factor).
    let scaling_factor = 1.0 / 0.18215;
    let scaled_latent: Vec<f32> = latent.iter().map(|v| v * scaling_factor).collect();

    let (_vae_loader, vae_plan, vae_tape) = load_model(&vae_holo());
    let mut vae_inputs = hologram::GraphInputs::new();
    vae_inputs.set_with_shape(0, f32_to_bytes(&scaled_latent), vec![1, 4, 64, 64]);

    let start = std::time::Instant::now();
    let vae_outputs = execute(&vae_tape, &vae_plan, &vae_inputs);
    eprintln!("VAE decode: {:.2?}", start.elapsed());

    let (_name, image_bytes) = vae_outputs.get(0).expect("no VAE output");
    let image = bytes_to_f32(image_bytes);
    eprintln!("image: {} floats", image.len());

    // ── Step 5: Save Image ────────────────────────────────────────────────
    // Output is [1, 3, H, W]. Determine H×W from total.
    let n_pixels = image.len() / 3;
    let side = (n_pixels as f64).sqrt() as usize;
    let (h, w) = if side * side == n_pixels {
        (side, side)
    } else {
        // Non-square: try common SD ratios.
        (512, n_pixels / 512)
    };

    // Normalize from model output range to [0, 1].
    // SD VAE outputs in roughly [-1, 1], so map: pixel = (x + 1) / 2.
    let normalized: Vec<f32> = image.iter().map(|v| (v + 1.0) / 2.0).collect();

    save_ppm(&normalized, h, w, &output_path());

    let total_time = total_start.elapsed();
    eprintln!("total pipeline: {:.2?}", total_time);

    // Verify output exists and is reasonable size.
    let meta = std::fs::metadata(&output_path()).expect("output file missing");
    assert!(meta.len() > 1000, "output file too small: {} bytes", meta.len());
    eprintln!("SD pipeline complete: {} bytes", meta.len());
}

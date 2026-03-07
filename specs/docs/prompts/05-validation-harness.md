# Prompt: Validation Harness

## Purpose

Implement `hologram-ai-validate` to compare `hologram-ai` outputs against
reference runtimes (ONNX Runtime and llama.cpp).

Run this prompt after the ONNX importer and GGUF streaming path are working.

---

## Context

The validation harness is the primary mechanism for verifying numerical
correctness. It runs the same model and inputs through both `hologram-ai`
and a reference runtime, then compares outputs using tolerance-aware
tensor comparison.

Architecture reference:
- `../hologram-architecture/specs/projects/hologram-ai/testing-strategy.md`

---

## Task 1: `ValidationSuite` API

```rust
pub struct ValidationSuite {
    pub ort_path: Option<PathBuf>,      // path to onnxruntime CLI or shared lib
    pub llamacpp_path: Option<PathBuf>, // path to llama.cpp main binary
}

pub struct ValidationReport {
    pub model: String,
    pub input_desc: String,
    pub comparisons: Vec<TensorComparison>,
    pub passed: bool,
    pub notes: Vec<String>,
}

pub struct TensorComparison {
    pub name: String,
    pub max_abs_err: f64,
    pub mean_abs_err: f64,
    pub cosine_similarity: f64,
    pub passed: bool,
    pub tolerance: Tolerance,
}

pub struct Tolerance {
    pub max_abs_err: f64,
    pub mean_abs_err: f64,
    pub cosine_sim_min: f64,
}

impl Tolerance {
    pub fn f32_default() -> Self { Tolerance { max_abs_err: 1e-5, mean_abs_err: 1e-6, cosine_sim_min: 0.9999 } }
    pub fn f16_default() -> Self { Tolerance { max_abs_err: 1e-3, mean_abs_err: 1e-4, cosine_sim_min: 0.999 } }
    pub fn quantized(scheme: &QuantScheme) -> Self { /* scheme-dependent */ }
}
```

---

## Task 2: ONNX comparison

Reference runtime: ONNX Runtime via subprocess.

```rust
impl ValidationSuite {
    pub fn validate_onnx(
        &self,
        model_path: &Path,
        inputs: &HashMap<String, Tensor>,
        tolerance: Tolerance,
    ) -> Result<ValidationReport> {
        // 1. Run through hologram-ai
        let holo_outputs = self.run_hologram_onnx(model_path, inputs)?;

        // 2. Run through ORT (subprocess or ort crate)
        let ref_outputs = self.run_ort(model_path, inputs)?;

        // 3. Compare tensor by tensor
        let comparisons = compare_tensor_maps(&holo_outputs, &ref_outputs, &tolerance);

        Ok(ValidationReport {
            model: model_path.display().to_string(),
            input_desc: describe_inputs(inputs),
            comparisons,
            passed: comparisons.iter().all(|c| c.passed),
            notes: vec![],
        })
    }
}
```

**ORT subprocess strategy:** serialize inputs to `.npz`, invoke
`python -m onnxruntime.tools.ort_test_runner`, read outputs from `.npz`.
This avoids linking `onnxruntime.so` in the Rust binary.

**ORT crate strategy (optional feature):** use the `ort` crate for in-process
execution. Feature-gated: `[features] ort = ["dep:ort"]`.

---

## Task 3: GGUF comparison

Reference runtime: llama.cpp `main` binary via subprocess.

```rust
impl ValidationSuite {
    pub fn validate_gguf_greedy_token(
        &self,
        model_path: &Path,
        prompt: &str,
        n_tokens: usize,
    ) -> Result<ValidationReport> {
        // Run hologram-ai greedy generation
        let holo_tokens = self.run_hologram_greedy(model_path, prompt, n_tokens)?;

        // Run llama.cpp greedy generation
        let ref_tokens = self.run_llamacpp_greedy(model_path, prompt, n_tokens)?;

        // Token-level comparison
        let matches: Vec<bool> = holo_tokens.iter()
            .zip(ref_tokens.iter())
            .map(|(h, r)| h == r)
            .collect();

        let passed = matches.iter().all(|&m| m);
        Ok(ValidationReport {
            model: model_path.display().to_string(),
            input_desc: format!("prompt={:?} n_tokens={}", prompt, n_tokens),
            comparisons: vec![TensorComparison {
                name: "greedy_tokens".into(),
                max_abs_err: if passed { 0.0 } else { 1.0 },
                mean_abs_err: matches.iter().filter(|&&m| !m).count() as f64 / n_tokens as f64,
                cosine_similarity: if passed { 1.0 } else { 0.0 },
                passed,
                tolerance: Tolerance { max_abs_err: 0.0, mean_abs_err: 0.0, cosine_sim_min: 1.0 },
            }],
            passed,
            notes: if !passed {
                vec![format!(
                    "Mismatch at token {}: holo={:?} ref={:?}",
                    matches.iter().position(|&m| !m).unwrap(),
                    holo_tokens[matches.iter().position(|&m| !m).unwrap()],
                    ref_tokens[matches.iter().position(|&m| !m).unwrap()],
                )]
            } else { vec![] },
        })
    }
}
```

---

## Task 4: `compare_tensors()` utility

```rust
pub fn compare_tensors(a: &Tensor, b: &Tensor, tol: &Tolerance) -> TensorComparison {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    assert_eq!(a.dtype(), b.dtype(), "dtype mismatch");

    let a_f32 = a.to_f32_vec();
    let b_f32 = b.to_f32_vec();

    let diffs: Vec<f32> = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| (x - y).abs()).collect();
    let max_abs = diffs.iter().cloned().fold(0.0_f32, f32::max) as f64;
    let mean_abs = diffs.iter().sum::<f32>() as f64 / diffs.len() as f64;

    let dot: f32 = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine = (dot / (norm_a * norm_b)) as f64;

    let passed = max_abs <= tol.max_abs_err
        && mean_abs <= tol.mean_abs_err
        && cosine >= tol.cosine_sim_min;

    TensorComparison { name: String::new(), max_abs_err: max_abs, mean_abs_err: mean_abs, cosine_similarity: cosine, passed, tolerance: tol.clone() }
}
```

---

## Task 5: CLI `validate` subcommand

```
hologram-ai validate --onnx model.onnx --input input.json
hologram-ai validate --gguf model.gguf --prompt "Hello" --tokens 10
hologram-ai validate --report report.json <model>
```

The `--report` flag writes a JSON summary of the `ValidationReport`.

---

## Task 6: CI reference tests

Add a `tests/reference/` integration test file:

```rust
// tests/reference/onnx_reference.rs
#[test]
#[ignore = "requires ort CLI in PATH"]
fn bert_base_matches_ort() {
    let suite = ValidationSuite {
        ort_path: which::which("python").ok(),
        ..Default::default()
    };
    let report = suite.validate_onnx(
        Path::new("tests/fixtures/onnx/bert-base-uncased.onnx"),
        &bert_base_test_inputs(),
        Tolerance::f32_default(),
    ).unwrap();
    assert!(report.passed, "ONNX validation failed:\n{:#?}", report);
}

// tests/reference/gguf_reference.rs
#[test]
#[ignore = "requires llama.cpp binary + TinyLlama model"]
fn tinyllama_greedy_matches_llamacpp() {
    let suite = ValidationSuite {
        llamacpp_path: std::env::var("LLAMACPP_BIN").ok().map(PathBuf::from),
        ..Default::default()
    };
    let report = suite.validate_gguf_greedy_token(
        Path::new("tests/fixtures/gguf/tinyllama-1.1b-q4_0.gguf"),
        "The capital of France is",
        5,
    ).unwrap();
    assert!(report.passed, "GGUF validation failed:\n{:#?}", report);
}
```

---

## Acceptance Criteria

- `hologram-ai-validate` crate compiles with no warnings
- `compare_tensors()` unit tests pass for f32 and f16 cases
- `validate_onnx()` works with ORT subprocess when ORT is available
- `hologram-ai validate` CLI command prints pass/fail report
- Reference tests are tagged `#[ignore]` and documented in README
- GitHub Actions CI runs reference tests on a nightly schedule (not every push)

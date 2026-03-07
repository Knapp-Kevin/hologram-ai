# Prompt 07: Download & Convert — HuggingFace Model Acquisition

## Goal

Implement the `hologram-ai download` CLI subcommand that downloads models from
HuggingFace Hub and handles ONNX conversion via a temporary Python virtualenv
when needed.

---

## Prerequisites

- CLI module structure from Prompt 06 in place
- `reqwest`, `indicatif`, `tokio`, `tempfile`, `sha2` dependencies added

---

## Step 1: HuggingFace API Client

### `src/download/hf_api.rs`

Implement a minimal HuggingFace Hub API client.

```rust
use reqwest::Client;
use serde::Deserialize;

pub struct HfClient {
    client: Client,
    token: Option<String>,
}

#[derive(Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub siblings: Vec<FileInfo>,
    pub private: bool,
    pub gated: Option<String>,  // "auto", "manual", or absent
}

#[derive(Deserialize)]
pub struct FileInfo {
    #[serde(rename = "rfilename")]
    pub filename: String,
    pub size: Option<u64>,
    pub lfs: Option<LfsInfo>,
}

#[derive(Deserialize)]
pub struct LfsInfo {
    pub sha256: String,
    pub size: u64,
}

impl HfClient {
    pub fn new(token: Option<String>) -> Self {
        let token = token
            .or_else(|| std::env::var("HF_TOKEN").ok())
            .or_else(|| read_hf_token_file());
        HfClient {
            client: Client::new(),
            token,
        }
    }

    /// Get model metadata and file listing
    pub async fn model_info(&self, model_id: &str) -> Result<ModelInfo> {
        let url = format!("https://huggingface.co/api/models/{model_id}");
        let mut req = self.client.get(&url);
        if let Some(ref token) = self.token {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await?;

        match resp.status().as_u16() {
            401 | 403 => anyhow::bail!(
                "This model requires authentication. Use --token or `huggingface-cli login`"
            ),
            404 => anyhow::bail!("Model not found: {model_id}"),
            s if s >= 400 => anyhow::bail!("HuggingFace API error: {s}"),
            _ => {}
        }

        Ok(resp.json().await?)
    }

    /// Download a file with progress reporting
    pub async fn download_file(
        &self,
        model_id: &str,
        filename: &str,
        revision: &str,
        dest: &Path,
        progress: &ProgressBar,
    ) -> Result<()> {
        let url = format!(
            "https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
        );
        let mut req = self.client.get(&url);
        if let Some(ref token) = self.token {
            req = req.bearer_auth(token);
        }

        let resp = req.send().await?;
        let total = resp.content_length().unwrap_or(0);
        progress.set_length(total);

        let mut file = tokio::fs::File::create(dest).await?;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            progress.inc(chunk.len() as u64);
        }
        progress.finish();
        Ok(())
    }
}

/// Read token from ~/.cache/huggingface/token
fn read_hf_token_file() -> Option<String> {
    let path = dirs::cache_dir()?.join("huggingface/token");
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}
```

---

## Step 2: Format Resolution

### `src/download/mod.rs`

```rust
pub fn resolve_format(
    info: &ModelInfo,
    preferred: Option<DownloadFormat>,
    quantization: Option<&str>,
) -> ResolvedDownload {
    match preferred {
        Some(DownloadFormat::Gguf) => resolve_gguf(info, quantization),
        Some(DownloadFormat::Onnx) => resolve_onnx(info),
        None | Some(DownloadFormat::Auto) => {
            // 1. Try GGUF
            if let Some(resolved) = try_resolve_gguf(info, quantization) {
                return resolved;
            }
            // 2. Try ONNX
            if let Some(resolved) = try_resolve_onnx(info) {
                return resolved;
            }
            // 3. Fall back to conversion
            ResolvedDownload::ConvertToOnnx
        }
    }
}

fn try_resolve_gguf(info: &ModelInfo, quantization: Option<&str>) -> Option<ResolvedDownload> {
    let gguf_files: Vec<_> = info.siblings.iter()
        .filter(|f| f.filename.ends_with(".gguf"))
        .collect();

    if gguf_files.is_empty() {
        return None;
    }

    // If quantization specified, find matching variant
    if let Some(quant) = quantization {
        let quant_upper = quant.to_uppercase();
        let matched = gguf_files.iter()
            .find(|f| f.filename.to_uppercase().contains(&quant_upper));
        if let Some(file) = matched {
            return Some(ResolvedDownload::Gguf { filename: file.filename.clone() });
        }
    }

    // Default: pick smallest GGUF or first one
    let file = gguf_files.first()?;
    Some(ResolvedDownload::Gguf { filename: file.filename.clone() })
}

fn try_resolve_onnx(info: &ModelInfo) -> Option<ResolvedDownload> {
    let onnx_files: Vec<_> = info.siblings.iter()
        .filter(|f| f.filename.ends_with(".onnx"))
        .collect();

    if onnx_files.is_empty() {
        return None;
    }

    // Collect main model + external data files
    let filenames: Vec<String> = onnx_files.iter().map(|f| f.filename.clone()).collect();
    Some(ResolvedDownload::Onnx { filenames })
}
```

---

## Step 3: ONNX Conversion via Python

### `src/download/convert.rs`

```rust
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct ConversionResult {
    pub model_path: PathBuf,
    pub companion_files: Vec<PathBuf>,
}

pub fn convert_to_onnx(
    model_id: &str,
    output_dir: &Path,
    keep_venv: bool,
) -> Result<ConversionResult> {
    // 1. Verify python3 is available
    let python = find_python()?;

    // 2. Create temp directory for virtualenv
    let tmp_dir = tempfile::tempdir()?;
    let venv_path = tmp_dir.path().join("hologram-ai-conv");
    let onnx_output = tmp_dir.path().join("onnx-output");

    eprintln!("Creating Python virtualenv for ONNX conversion...");

    // 3. Create virtualenv
    let status = Command::new(&python)
        .args(["-m", "venv"])
        .arg(&venv_path)
        .status()?;
    if !status.success() {
        anyhow::bail!("Failed to create Python virtualenv");
    }

    // 4. Install dependencies
    let pip = venv_pip_path(&venv_path);
    eprintln!("Installing conversion dependencies (this may take a few minutes)...");
    let status = Command::new(&pip)
        .args(["install", "optimum[exporters]", "transformers", "torch", "onnx"])
        .status()?;
    if !status.success() {
        anyhow::bail!("Failed to install Python dependencies");
    }

    // 5. Run optimum export
    let optimum_cli = venv_bin_path(&venv_path, "optimum-cli");
    eprintln!("Converting {model_id} to ONNX...");
    let status = Command::new(&optimum_cli)
        .args(["export", "onnx", "--model", model_id])
        .arg(&onnx_output)
        .status()?;
    if !status.success() {
        anyhow::bail!("ONNX conversion failed. Check model compatibility with optimum.");
    }

    // 6. Copy output files to destination
    std::fs::create_dir_all(output_dir)?;

    let companion_names = [
        "tokenizer.json",
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ];

    let mut result = ConversionResult {
        model_path: PathBuf::new(),
        companion_files: Vec::new(),
    };

    for entry in std::fs::read_dir(&onnx_output)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let dest = output_dir.join(&name);

        if name.ends_with(".onnx") {
            std::fs::copy(entry.path(), &dest)?;
            result.model_path = dest;
        } else if companion_names.contains(&name.as_str()) {
            std::fs::copy(entry.path(), &dest)?;
            result.companion_files.push(dest);
        }
    }

    // 7. Cleanup
    if !keep_venv {
        // tmp_dir Drop handles cleanup
        eprintln!("Cleaned up conversion virtualenv.");
    } else {
        let venv_path = venv_path.to_string_lossy().to_string();
        // Persist the tmpdir so it doesn't get cleaned up
        std::mem::forget(tmp_dir);
        eprintln!("Virtualenv preserved at: {venv_path}");
    }

    Ok(result)
}

fn find_python() -> Result<PathBuf> {
    for name in ["python3", "python"] {
        if let Ok(path) = which::which(name) {
            // Verify version >= 3.10
            let output = Command::new(&path).args(["--version"]).output()?;
            let version = String::from_utf8_lossy(&output.stdout);
            // Parse and check version...
            return Ok(path);
        }
    }
    anyhow::bail!(
        "python3 required for ONNX conversion. Install Python 3.10+ or use --format gguf"
    )
}

fn venv_pip_path(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("pip")
    } else {
        venv.join("bin").join("pip")
    }
}

fn venv_bin_path(venv: &Path, name: &str) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join(name)
    } else {
        venv.join("bin").join(name)
    }
}
```

---

## Step 4: Download Command Orchestration

### `src/cli/download.rs`

```rust
#[derive(Args)]
pub struct DownloadArgs {
    /// HuggingFace model identifier
    pub model_id: String,

    /// Output directory
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Preferred format
    #[arg(short, long, default_value = "auto")]
    pub format: DownloadFormat,

    /// Git revision on HF Hub
    #[arg(long, default_value = "main")]
    pub revision: String,

    /// Quantization variant (e.g., Q4_0, Q4_K_M)
    #[arg(long)]
    pub quantization: Option<String>,

    /// Keep Python virtualenv after conversion
    #[arg(long)]
    pub keep_venv: bool,

    /// HuggingFace API token
    #[arg(long)]
    pub token: Option<String>,
}

pub fn run(args: DownloadArgs) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_async(args))
}

async fn run_async(args: DownloadArgs) -> anyhow::Result<()> {
    let client = HfClient::new(args.token);

    // 1. Get model info
    eprintln!("Fetching model info for {}...", args.model_id);
    let info = client.model_info(&args.model_id).await?;

    // 2. Resolve output directory
    let model_name = args.model_id.split('/').last().unwrap_or(&args.model_id);
    let output_dir = args.output.unwrap_or_else(|| {
        PathBuf::from("models").join(model_name)
    });
    std::fs::create_dir_all(&output_dir)?;

    // 3. Resolve format
    let resolved = resolve_format(&info, Some(args.format), args.quantization.as_deref());

    match resolved {
        ResolvedDownload::Gguf { filename } => {
            download_gguf(&client, &args.model_id, &args.revision, &filename, &output_dir, &info).await?;
        }
        ResolvedDownload::Onnx { filenames } => {
            download_onnx(&client, &args.model_id, &args.revision, &filenames, &output_dir, &info).await?;
        }
        ResolvedDownload::ConvertToOnnx => {
            convert_to_onnx(&args.model_id, &output_dir, args.keep_venv)?;
        }
    }

    // 4. Print summary
    print_download_summary(&args.model_id, &output_dir)?;
    Ok(())
}
```

---

## Step 5: File Integrity

After downloading each file, verify SHA256 checksum if available from the
HuggingFace API (`LfsInfo.sha256`):

```rust
fn verify_checksum(path: &Path, expected_sha256: &str) -> Result<()> {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)?;
    std::io::copy(&mut file, &mut hasher)?;
    let hash = format!("{:x}", hasher.finalize());
    if hash != expected_sha256 {
        anyhow::bail!(
            "Checksum mismatch for {}: expected {expected_sha256}, got {hash}",
            path.display()
        );
    }
    Ok(())
}
```

---

## Tests

### Unit tests

```rust
#[test]
fn test_resolve_format_prefers_gguf() {
    let info = ModelInfo {
        siblings: vec![
            FileInfo { filename: "model-Q4_0.gguf".into(), .. },
            FileInfo { filename: "model.onnx".into(), .. },
        ],
        ..
    };
    let resolved = resolve_format(&info, None, None);
    assert!(matches!(resolved, ResolvedDownload::Gguf { .. }));
}

#[test]
fn test_resolve_format_falls_back_to_onnx() {
    let info = ModelInfo {
        siblings: vec![
            FileInfo { filename: "model.onnx".into(), .. },
        ],
        ..
    };
    let resolved = resolve_format(&info, None, None);
    assert!(matches!(resolved, ResolvedDownload::Onnx { .. }));
}

#[test]
fn test_resolve_format_triggers_conversion() {
    let info = ModelInfo {
        siblings: vec![
            FileInfo { filename: "pytorch_model.bin".into(), .. },
        ],
        ..
    };
    let resolved = resolve_format(&info, None, None);
    assert!(matches!(resolved, ResolvedDownload::ConvertToOnnx));
}
```

### Integration tests (network-dependent, `#[ignore]`)

```rust
#[test]
#[ignore = "requires network access"]
fn test_download_gguf_model() {
    let tmp = tempfile::tempdir().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_hologram-ai"))
        .args(["download", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"])
        .args(["--format", "gguf"])
        .args(["--quantization", "Q4_0"])
        .args(["--output", tmp.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success());
}
```

---

## Exit Criteria

- [ ] `hologram-ai download meta-llama/Llama-3.2-1B --format gguf` downloads model + tokenizer
- [ ] `hologram-ai download <model> --format onnx` downloads ONNX model + companion files
- [ ] `hologram-ai download <model> --format auto` prefers GGUF, falls back to ONNX, then converts
- [ ] ONNX conversion creates virtualenv, installs deps, runs optimum-cli, copies output
- [ ] `--keep-venv` preserves the virtualenv after conversion
- [ ] `--quantization Q4_K_M` selects correct GGUF variant
- [ ] Missing python3 for conversion produces clear error message
- [ ] Gated model without auth produces clear error message
- [ ] Progress bars show during download
- [ ] SHA256 checksums verified after download
- [ ] All tests pass: `cargo test --workspace`

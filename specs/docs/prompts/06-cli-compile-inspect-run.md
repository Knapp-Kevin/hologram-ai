# Prompt 06: CLI — Compile, Inspect, Run + Benchmarking

## Goal

Implement the `compile`, `inspect`, and `run` CLI subcommands plus the `--stats`
benchmarking system. This restructures the CLI from a single file into a module
directory and adds `.holo` archive support.

---

## Prerequisites

- Sprint-003 complete (generate, streaming, KV-cache working)
- `ModelCompiler::compile()` facade working end-to-end
- `hologram::HoloLoader` and `hologram::HoloWriter` available via hologram crate

---

## Step 1: CLI Module Restructure

Restructure `src/cli.rs` → `src/cli/` module directory.

### `src/cli/mod.rs`

```rust
use clap::{Parser, Subcommand};

mod compile;
mod download;
mod generate;
mod inspect;
mod lower;
mod run;
mod validate;

#[derive(Parser)]
#[command(name = "hologram-ai", version, about = "AI model compiler and inference runtime")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Compile a model into a .holo archive
    Compile(compile::CompileArgs),
    /// Inspect a model or .holo archive
    Inspect(inspect::InspectArgs),
    /// Execute a single forward pass
    Run(run::RunArgs),
    /// Autoregressive text generation
    Generate(generate::GenerateArgs),
    /// Download a model from HuggingFace
    Download(download::DownloadArgs),
    /// Validate model outputs against reference runtimes
    Validate(validate::ValidateArgs),
    /// Lower a model and emit the hologram::Graph
    Lower(lower::LowerArgs),
}
```

### `src/main.rs`

```rust
use clap::Parser;

mod cli;

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    match cli.command {
        cli::Command::Compile(args) => cli::compile::run(args),
        cli::Command::Inspect(args) => cli::inspect::run(args),
        cli::Command::Run(args) => cli::run::run(args),
        cli::Command::Generate(args) => cli::generate::run(args),
        cli::Command::Download(args) => cli::download::run(args),
        cli::Command::Validate(args) => cli::validate::run(args),
        cli::Command::Lower(args) => cli::lower::run(args),
    }
}
```

---

## Step 2: BenchmarkStats System

### `src/stats.rs`

```rust
use std::time::{Duration, Instant};

pub struct BenchmarkStats {
    pub phases: Vec<PhaseStats>,
    pub peak_rss_bytes: u64,
    pub tokens_generated: Option<u32>,
    pub prompt_tokens: Option<u32>,
}

pub struct PhaseStats {
    pub name: String,
    pub duration: Duration,
}

/// RAII phase timer — records elapsed time when dropped
pub struct PhaseTimer {
    name: String,
    start: Instant,
    phases: *mut Vec<PhaseStats>,  // NOTE: use safe alternative in actual impl
}

impl BenchmarkStats {
    pub fn new() -> Self { /* ... */ }

    pub fn start_phase(&mut self, name: &str) -> PhaseTimer { /* ... */ }

    pub fn total_duration(&self) -> Duration {
        self.phases.iter().map(|p| p.duration).sum()
    }

    pub fn tokens_per_second(&self) -> Option<f64> {
        let tokens = self.tokens_generated? as f64;
        let ttft = self.time_to_first_token()?;
        let total = self.total_duration();
        let decode_time = total - ttft;
        if decode_time.is_zero() { return None; }
        Some((tokens - 1.0) / decode_time.as_secs_f64())
    }

    pub fn time_to_first_token(&self) -> Option<Duration> {
        // Sum of import + optimize + lower + compile + first decode
        // Phases named "import", "optimize", "lower", "compile", "prefill"
        let prefill_phases = ["import", "optimize", "lower", "compile", "prefill"];
        let ttft: Duration = self.phases.iter()
            .filter(|p| prefill_phases.contains(&p.name.as_str()))
            .map(|p| p.duration)
            .sum();
        if ttft.is_zero() { None } else { Some(ttft) }
    }

    pub fn prefill_tokens_per_second(&self) -> Option<f64> {
        let prompt_tokens = self.prompt_tokens? as f64;
        let prefill = self.phases.iter().find(|p| p.name == "prefill")?;
        Some(prompt_tokens / prefill.duration.as_secs_f64())
    }

    pub fn display(&self) {
        // Formatted output to stderr
        // See cli.md for example output format
    }
}

/// Get current RSS (resident set size) in bytes
pub fn current_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        // Use mach_task_info
        // mach_task_basic_info.resident_size
        0 // placeholder
    }
    #[cfg(target_os = "linux")]
    {
        // Parse /proc/self/status VmRSS field
        0 // placeholder
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0 // unsupported platform
    }
}
```

---

## Step 3: Compile Command

### `src/cli/compile.rs`

```rust
use clap::Args;
use std::path::PathBuf;

#[derive(Args)]
pub struct CompileArgs {
    /// Path to model file (.gguf, .onnx, .bin)
    pub model: PathBuf,

    /// Output .holo file path
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Force input format
    #[arg(short, long)]
    pub format: Option<InputFormat>,

    /// Quantization lowering strategy
    #[arg(long, default_value = "fused")]
    pub quant_strategy: QuantStrategy,

    /// Maximum sequence length
    #[arg(long, default_value_t = 2048)]
    pub max_seq_len: usize,

    /// Print compilation statistics
    #[arg(long)]
    pub stats: bool,
}

pub fn run(args: CompileArgs) -> anyhow::Result<()> {
    let mut bench = if args.stats { Some(BenchmarkStats::new()) } else { None };

    // 1. Import
    let ai_graph = {
        let _phase = bench.as_mut().map(|b| b.start_phase("import"));
        import_model(&args.model, args.format)?
    };

    // 2. Optimize
    let ai_graph = {
        let _phase = bench.as_mut().map(|b| b.start_phase("optimize"));
        OptPipeline::default().run(ai_graph)?
    };

    // 3. Plan KV-cache
    let kv_layout = KvCachePlanner::plan(&ai_graph)?;

    // 4. Lower
    let lowering_output = {
        let _phase = bench.as_mut().map(|b| b.start_phase("lower"));
        lower(&ai_graph, &kv_layout, &LoweringOptions {
            quant_strategy: args.quant_strategy,
            max_seq_len: args.max_seq_len,
        })?
    };

    // 5. Compile via hologram CLI (or fallback)
    let output_path = args.output.unwrap_or_else(|| {
        args.model.with_extension("holo")
    });

    {
        let _phase = bench.as_mut().map(|b| b.start_phase("compile"));
        compile_via_hologram(&lowering_output.graph, &output_path)?;
    }

    // 6. Print stats
    if let Some(bench) = bench {
        bench.display();
    }

    eprintln!("Compiled: {}", output_path.display());
    Ok(())
}

fn compile_via_hologram(graph: &hologram::Graph, output: &Path) -> anyhow::Result<()> {
    // Try subprocess first
    if let Ok(hologram_bin) = which::which("hologram") {
        let tmp = tempfile::NamedTempFile::new()?;
        let serialized = rkyv::to_bytes::<_, 256>(graph)?;
        std::fs::write(tmp.path(), &serialized)?;

        let status = std::process::Command::new(hologram_bin)
            .args(["compile", "--input"])
            .arg(tmp.path())
            .args(["--output"])
            .arg(output)
            .status()?;

        if !status.success() {
            anyhow::bail!("hologram compile failed with status {}", status);
        }
        return Ok(());
    }

    // Fallback: library call
    eprintln!("warning: `hologram` binary not found on $PATH, using library fallback.");
    eprintln!("Install the hologram CLI for full compilation features.");

    let compilation = hologram::compile(graph.clone())?;
    hologram::HoloWriter::new()
        .schedule(&compilation.schedule)
        .archive(&compilation.archive)
        .write_to_file(output)?;

    Ok(())
}
```

---

## Step 4: Inspect Command

### `src/cli/inspect.rs`

```rust
#[derive(Args)]
pub struct InspectArgs {
    /// Path to .holo, .gguf, .onnx, or .bin file
    pub file: PathBuf,

    /// Output format
    #[arg(long, default_value = "summary")]
    pub format: InspectFormat,
}

#[derive(Clone, ValueEnum)]
pub enum InspectFormat {
    Summary,
    Ops,
    Tensors,
    Metadata,
    Json,
}

pub fn run(args: InspectArgs) -> anyhow::Result<()> {
    let ext = args.file.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "holo" => inspect_holo(&args.file, &args.format),
        "gguf" => inspect_raw(&args.file, InputFormat::Gguf, &args.format),
        "onnx" => inspect_raw(&args.file, InputFormat::Onnx, &args.format),
        "bin"  => inspect_raw(&args.file, InputFormat::Ggml, &args.format),
        _ => anyhow::bail!("Unsupported file extension: .{ext}"),
    }
}

fn inspect_holo(path: &Path, format: &InspectFormat) -> anyhow::Result<()> {
    let loader = hologram::HoloLoader::open(path)?;
    // Extract metadata from the loaded archive
    // Display based on format
    Ok(())
}

fn inspect_raw(path: &Path, fmt: InputFormat, format: &InspectFormat) -> anyhow::Result<()> {
    let graph = import_model(path, Some(fmt))?;
    // Display AiGraph summary based on format
    // summary: model name, arch, params, quant, layers, vocab, context
    // ops: list all AiOp nodes with shapes
    // tensors: list all tensors with dtype info
    // metadata: dump graph.metadata
    // json: serde_json serialize all of the above
    Ok(())
}
```

---

## Step 5: Run Command

### `src/cli/run.rs`

```rust
#[derive(Args)]
pub struct RunArgs {
    /// Path to .holo or model file
    pub model: PathBuf,

    /// Input tensors (JSON or .npz)
    #[arg(short, long)]
    pub input: PathBuf,

    /// Save output tensors to file
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Print execution timing
    #[arg(long)]
    pub stats: bool,

    /// Warmup passes before measured pass
    #[arg(long, default_value_t = 0)]
    pub warmup: usize,
}

pub fn run(args: RunArgs) -> anyhow::Result<()> {
    let mut bench = if args.stats { Some(BenchmarkStats::new()) } else { None };

    // 1. Load or compile model
    let compiled = if is_holo(&args.model) {
        load_holo(&args.model)?
    } else {
        eprintln!("note: compiling model on-the-fly (use `hologram-ai compile` for cached execution)");
        compile_model(&args.model, bench.as_mut())?
    };

    // 2. Create session
    let mut session = compiled.session(SessionOptions::default())?;

    // 3. Load inputs
    let inputs = load_inputs(&args.input)?;

    // 4. Warmup
    for _ in 0..args.warmup {
        let _ = session.run(inputs.clone())?;
        session.reset_cache();
    }

    // 5. Measured pass
    let outputs = {
        let _phase = bench.as_mut().map(|b| b.start_phase("execute"));
        session.run(inputs)?
    };

    // 6. Output
    if let Some(ref output_path) = args.output {
        save_outputs(&outputs, output_path)?;
    }

    // Print output shapes to stdout
    for (name, tensor) in &outputs {
        println!("{name}: {:?} {}", tensor.shape(), tensor.dtype());
    }

    if let Some(bench) = bench {
        bench.display();
    }

    Ok(())
}
```

---

## Step 6: Helper — Model Loading

### Shared utility for detecting and loading `.holo` vs raw formats

```rust
// In src/cli/mod.rs or a shared utility module

pub fn is_holo(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("holo")
}

pub fn import_model(path: &Path, format: Option<InputFormat>) -> Result<AiGraph> {
    let format = format.unwrap_or_else(|| detect_format(path));
    match format {
        InputFormat::Gguf => hologram_ai_gguf::import_gguf(path, Default::default()),
        InputFormat::Onnx => hologram_ai_onnx::import_onnx_path(path, Default::default()),
        InputFormat::Ggml => hologram_ai_ggml::import_ggml(path, Default::default()),
    }
}

pub fn detect_format(path: &Path) -> InputFormat {
    match path.extension().and_then(|e| e.to_str()) {
        Some("gguf") => InputFormat::Gguf,
        Some("onnx") => InputFormat::Onnx,
        Some("bin")  => InputFormat::Ggml,
        _ => InputFormat::Gguf, // default to GGUF
    }
}
```

---

## Step 7: Generate Command Updates

Extend existing generate command to:

1. Accept `.holo` files (skip compilation)
2. Accept `--tokenizer <path>` for explicit tokenizer path
3. Accept `--stats` flag for benchmarking
4. Auto-discover `tokenizer.json` next to model file

---

## Tests

### Unit tests

```rust
#[test]
fn test_benchmark_stats_tokens_per_second() {
    let mut stats = BenchmarkStats::new();
    stats.phases.push(PhaseStats { name: "prefill".into(), duration: Duration::from_millis(200) });
    stats.phases.push(PhaseStats { name: "decode".into(), duration: Duration::from_millis(800) });
    stats.tokens_generated = Some(10);
    stats.prompt_tokens = Some(5);

    let tps = stats.tokens_per_second().unwrap();
    assert!((tps - 11.25).abs() < 0.1); // (10-1) / 0.8 = 11.25
}

#[test]
fn test_detect_format() {
    assert_eq!(detect_format(Path::new("model.gguf")), InputFormat::Gguf);
    assert_eq!(detect_format(Path::new("model.onnx")), InputFormat::Onnx);
    assert_eq!(detect_format(Path::new("model.bin")), InputFormat::Ggml);
}
```

### Integration tests

```rust
#[test]
fn test_compile_command_produces_holo() {
    let output = tempfile::NamedTempFile::new().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_hologram-ai"))
        .args(["compile", "tests/fixtures/gguf/tiny-llama-q4_0.gguf"])
        .args(["--output", output.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(output.path().exists());
}

#[test]
fn test_inspect_gguf_summary() {
    let output = Command::new(env!("CARGO_BIN_EXE_hologram-ai"))
        .args(["inspect", "tests/fixtures/gguf/tiny-llama-q4_0.gguf"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Architecture:"));
    assert!(stdout.contains("Parameters:"));
}
```

---

## Exit Criteria

- [ ] `hologram-ai compile model.gguf` produces a `.holo` file
- [ ] `hologram-ai compile model.gguf --stats` prints phase timings
- [ ] `hologram-ai inspect model.gguf` prints model summary
- [ ] `hologram-ai inspect model.holo` prints archive summary
- [ ] `hologram-ai inspect model.gguf --format json` produces valid JSON
- [ ] `hologram-ai run model.holo --input input.json` produces output tensors
- [ ] `hologram-ai run model.gguf --input input.json` auto-compiles and runs
- [ ] `hologram-ai run model.holo --stats --warmup 3` prints timing after warmup
- [ ] `hologram-ai generate model.holo "Hello" --stats` prints tokens/s
- [ ] `BenchmarkStats::tokens_per_second()` computes correctly
- [ ] All tests pass: `cargo test --workspace`

//! CLI entry point for hologram-ai.

use clap::Parser;
use hologram_ai::download;
use hologram_ai::session::{ModelCompiler, ModelSource, InferenceSession};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "hologram-ai", about = "AI model inference via hologram runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Run inference on a model file.
    Run {
        /// Path to ONNX or GGUF model.
        #[arg(short, long)]
        model: PathBuf,
        /// Input token IDs (comma-separated).
        #[arg(short, long, value_delimiter = ',')]
        tokens: Vec<u32>,
    },
    /// Inspect a `.holo` archive or ONNX model file.
    Info {
        /// Path to a `.holo` or `.onnx` file.
        #[arg(short, long)]
        model: PathBuf,
    },
    /// Compile a model to a `.holo` archive file.
    Compile {
        /// Path to the input model (ONNX or GGUF).
        #[arg(short, long)]
        model: PathBuf,
        /// Output directory for the compiled `.holo` archive.
        #[arg(short, long, value_name = "DIR")]
        output: PathBuf,
    },
    /// Download a model from HuggingFace Hub.
    Download(download::DownloadArgs),
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber_init();
    let cli = Cli::parse();

    match cli.command {
        Command::Run { model, tokens } => {
            let source = model_source_from_path(&model)?;
            let compiled = ModelCompiler::default().compile(source)?;
            let mut sess = InferenceSession::new(Arc::new(compiled));
            let logits = sess.run(&tokens)?;
            println!("logits shape: [{}]", logits.len());
        }
        Command::Info { model } => {
            let ext = model.extension().and_then(|e| e.to_str()).unwrap_or("");
            match ext {
                "holo" => inspect_holo(&model)?,
                "onnx" => inspect_onnx(&model)?,
                other => anyhow::bail!(
                    "info supports .holo and .onnx files, got '.{other}'"
                ),
            }
        }
        Command::Compile { model, output } => {
            let source = model_source_from_path(&model)?;
            let compiled = ModelCompiler::default().compile(source)?;
            if output.exists() && !output.is_dir() {
                anyhow::bail!(
                    "'{}' exists and is not a directory. Remove it or choose a different --output path.",
                    output.display()
                );
            }
            std::fs::create_dir_all(&output)?;
            let stem = model.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
            let holo_path = output.join(format!("{stem}.holo"));
            compiled.save_archive(&holo_path)?;
            println!("wrote {}", holo_path.display());
        }
        Command::Download(args) => {
            download::run(args)?;
        }
    }

    Ok(())
}

// ── Info sub-commands ────────────────────────────────────────────────────────

/// Inspect a compiled `.holo` archive (mirrors `hologram inspect`).
fn inspect_holo(path: &std::path::Path) -> anyhow::Result<()> {
    let data = std::fs::read(path)?;
    let plan = hologram::load_from_bytes(&data)?;
    let h = plan.header();
    let sg = plan.graph();
    let schedule = hologram::hologram_exec::build_schedule(sg)?;

    println!("file:      {:?}", path);
    println!("size:      {} ({})", format_bytes(data.len() as u64), data.len());
    println!("graph:     {}", format_bytes(h.graph_size));
    println!("weights:   {}", format_bytes(h.weights_size));
    println!("sections:  {} ({})", h.section_count, format_bytes(h.section_table_size));
    println!("nodes:     {}", sg.node_count());
    println!("inputs:    [{}]", sg.input_names.join(", "));
    println!("outputs:   [{}]", sg.output_names.join(", "));
    println!("levels:    {}", schedule.levels.len());

    Ok(())
}

/// Inspect an ONNX model file (import + print metadata without compilation).
fn inspect_onnx(path: &std::path::Path) -> anyhow::Result<()> {
    let ai_graph = hologram_ai_onnx::import_onnx_path(path, Default::default())?;

    println!("file:      {:?}", path);
    println!("format:    ONNX");
    println!("nodes:     {}", ai_graph.nodes.len());
    println!("params:    {}", ai_graph.params.len());
    println!("inputs:    {}", ai_graph.inputs.len());
    println!("outputs:   {}", ai_graph.outputs.len());

    // Print model metadata if available.
    use hologram_ai_common::MetaValue;
    for (key, val) in &ai_graph.metadata {
        let s = match val {
            MetaValue::Str(s) => s.clone(),
            MetaValue::Int(i) => i.to_string(),
            MetaValue::Float(f) => format!("{f:.4}"),
            MetaValue::Bool(b) => b.to_string(),
            MetaValue::Ints(v) => format!("{v:?}"),
        };
        println!("{key:<11}{s}");
    }

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn model_source_from_path(path: &std::path::Path) -> anyhow::Result<ModelSource> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "onnx" => Ok(ModelSource::OnnxPath(path.to_owned())),
        "gguf" => Ok(ModelSource::GgufPath(path.to_owned())),
        other  => anyhow::bail!("unsupported model extension: '.{other}'"),
    }
}

/// Format a byte count as a human-readable string (binary units).
fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    match bytes {
        b if b >= GIB => format!("{:.1} GiB", b as f64 / GIB as f64),
        b if b >= MIB => format!("{:.1} MiB", b as f64 / MIB as f64),
        b if b >= KIB => format!("{:.1} KiB", b as f64 / KIB as f64),
        b => format!("{b} B"),
    }
}

fn tracing_subscriber_init() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}

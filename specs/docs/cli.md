# hologram-ai: CLI Specification

---

## Overview

The `hologram-ai` CLI is the user-facing command-line interface for the hologram-ai
compiler and inference runtime. It is a single binary (`hologram-ai`) built from the
facade crate's `src/main.rs` using `clap` with derive macros.

All commands accept `--help` and `--version`. Error output goes to stderr.
Non-zero exit codes indicate failure.

---

## Command Surface

```
hologram-ai <COMMAND>

Commands:
  compile     Compile a model into a .holo archive
  inspect     Inspect a model or .holo archive
  run         Execute a single forward pass
  generate    Autoregressive text generation
  download    Download a model from HuggingFace
  validate    Validate model outputs against reference runtimes
  lower       Lower a model and emit the hologram::Graph
```

---

## `hologram-ai compile`

Compile a foreign model artifact into a `.holo` archive.

```
hologram-ai compile <MODEL> [OPTIONS]

Arguments:
  <MODEL>    Path to model file (.gguf, .onnx, .bin)

Options:
  -o, --output <PATH>           Output .holo file path [default: <MODEL>.holo]
  -f, --format <FORMAT>         Force input format [possible: gguf, onnx, ggml]
      --quant-strategy <STRAT>  Quantization lowering strategy [possible: eager, fused, mixed] [default: fused]
      --max-seq-len <N>         Maximum sequence length for shape concretization [default: 2048]
      --stats                   Print compilation statistics
```

### Pipeline

```
MODEL file
  → import_*()                         AiGraph (raw)
  → opt_pipeline.run()                 AiGraph (optimized)
  → kv_cache_planner.plan()            KvCacheLayout
  → lower()                            hologram::Graph + CustomOpRegistry
  → serialize Graph to tempfile        rkyv SerializedGraph
  → hologram compile <temp> -o <out>   .holo archive (subprocess)
```

### Compilation delegation (see ADR-0009)

The `compile` command delegates the final compilation step to the `hologram` CLI
binary. After import, optimization, and lowering are complete, hologram-ai:

1. Serializes the `hologram::Graph` to a temporary file using `rkyv`
2. Invokes `hologram compile --input <tmpfile> --output <path>.holo` as a subprocess
3. Reports success/failure based on the subprocess exit code

If the `hologram` binary is not found on `$PATH`, hologram-ai falls back to calling
`hologram::compile()` as a library and writing the `.holo` archive via
`hologram::HoloWriter`. A warning is printed when using the fallback path.

### Stats output (with `--stats`)

```
Compilation Statistics
──────────────────────
Import:         142ms   (GGUF v3, LLaMA 1.1B)
Optimization:    89ms   (3 passes: attention, ffn, quant-matmul fusion)
Lowering:        67ms   (1,247 → 3,891 nodes)
Compilation:    312ms   (hologram compile, LUT fusion + CSE + buffer reuse)
──────────────────────
Total:          610ms

Model: TinyLlama 1.1B Q4_0
Parameters: 1,100,048,384 (Q4_0: 98.2%, F32: 1.8%)
Archive size: 637 MB
```

---

## `hologram-ai inspect`

Inspect a model file or compiled `.holo` archive.

```
hologram-ai inspect <FILE> [OPTIONS]

Arguments:
  <FILE>    Path to .holo, .gguf, .onnx, or .bin file

Options:
      --format <FMT>   Output format [possible: summary, ops, tensors, metadata, json] [default: summary]
```

### Input handling

| File extension | Behavior |
|---------------|----------|
| `.holo` | Uses `HoloLoader` to read archive metadata directly |
| `.gguf` | Runs GGUF importer, inspects resulting `AiGraph` |
| `.onnx` | Runs ONNX importer, inspects resulting `AiGraph` |
| `.bin` | Runs GGML importer, inspects resulting `AiGraph` |

### Output formats

**`--format summary`** (default):
```
Model: TinyLlama-1.1B-Chat-v1.0
Architecture: LLaMA
Parameters: 1,100,048,384
Quantization: Q4_0 (4-bit, block size 32)
Layers: 22
Hidden size: 2048
Heads: 32 (KV heads: 4)
Vocab size: 32000
Context length: 2048
Format: GGUF v3
```

**`--format ops`**: Lists all operations with input/output tensor shapes and dtypes.

**`--format tensors`**: Lists all tensors with name, dtype (logical + storage),
shape, and storage type (inline vs deferred/mmap).

**`--format metadata`**: Dumps all key-value metadata from the model file.

**`--format json`**: Machine-readable JSON combining all of the above.

---

## `hologram-ai run`

Execute a single forward pass with explicit tensor inputs.

```
hologram-ai run <MODEL> [OPTIONS]

Arguments:
  <MODEL>    Path to .holo or model file (auto-compiles if not .holo)

Options:
  -i, --input <PATH>     Input tensors (JSON or .npz) [required]
  -o, --output <PATH>    Save output tensors to file (JSON or .npz)
      --stats            Print execution timing
      --warmup <N>       Number of warmup passes before measured pass [default: 0]
```

### Auto-compilation

If `<MODEL>` is not a `.holo` file, the model is compiled on-the-fly using the
same pipeline as `hologram-ai compile`. A note is printed to stderr:

```
note: compiling model on-the-fly (use `hologram-ai compile` for cached execution)
```

### Input format (JSON)

```json
{
  "input_ids": {
    "dtype": "u32",
    "shape": [1, 5],
    "data": [1, 2, 3, 4, 5]
  }
}
```

### Stats output (with `--stats`)

```
Execution Statistics
────────────────────
Compile:      610ms  (on-the-fly; use `compile` to cache)
Warmup:       128ms  (1 pass)
Execution:    142ms  (1 forward pass)
────────────────────
Output shapes:
  logits: [1, 5, 32000] f32
```

---

## `hologram-ai generate`

Autoregressive text generation.

```
hologram-ai generate <MODEL> <PROMPT> [OPTIONS]

Arguments:
  <MODEL>    Path to .holo or model file (auto-compiles if not .holo)
  <PROMPT>   Text prompt for generation

Options:
      --max-tokens <N>       Maximum tokens to generate [default: 128]
      --temperature <F>      Sampling temperature [default: 1.0]
      --top-p <F>            Nucleus sampling threshold [default: 1.0]
      --top-k <N>            Top-k sampling [default: 0]
      --min-p <F>            Min-p sampling threshold [default: 0.0]
      --repetition-penalty <F>  Repetition penalty [default: 1.0]
      --seed <N>             Random seed for reproducibility
      --tokenizer <PATH>     Path to tokenizer.json (HuggingFace tokenizers format)
      --stats                Print generation statistics
```

### Tokenizer resolution

The `--tokenizer` flag specifies a `tokenizer.json` file (HuggingFace tokenizers
format, loaded via the `tokenizers` crate). If `--tokenizer` is not provided:

1. Look for `tokenizer.json` in the same directory as `<MODEL>`
2. If not found, error with: `error: no tokenizer found. Use --tokenizer <path>`

### Stats output (with `--stats`)

```
Generation Statistics
─────────────────────
Tokens generated:     42
Prompt tokens:        12
Time-to-first-token:  245ms
Prefill:              198ms  (60.6 tokens/s)
Decode:               1,847ms  (22.2 tokens/s)
Total:                2,092ms
Peak memory (RSS):    1,247 MB
```

### Token streaming

Tokens are printed to stdout as they are generated (streaming output).
Stats are printed to stderr after generation completes.

---

## `hologram-ai download`

Download a model from HuggingFace Hub.

```
hologram-ai download <MODEL_ID> [OPTIONS]

Arguments:
  <MODEL_ID>    HuggingFace model identifier (e.g., meta-llama/Llama-3.2-1B)

Options:
  -o, --output <DIR>         Output directory [default: ./models/<model-name>]
  -f, --format <FORMAT>      Preferred format [possible: gguf, onnx, auto] [default: auto]
      --revision <REF>       Git revision on HF Hub [default: main]
      --quantization <TYPE>  Quantization variant (e.g., Q4_0, Q4_K_M)
      --keep-venv            Do not delete the Python virtualenv after conversion
      --token <TOKEN>        HuggingFace API token (or set HF_TOKEN env var)
```

### Format resolution (`--format auto`)

```
1. Check HF Hub for GGUF files → download .gguf (preferred)
2. Check HF Hub for ONNX files → download .onnx
3. No pre-built format → convert to ONNX via Python (see below)
```

### GGUF download path

Downloads the GGUF file directly from HuggingFace. If `--quantization` is
specified, selects the matching variant (e.g., `*-Q4_0.gguf`).

Files downloaded:
- `model-Q4_0.gguf` (or matching variant)
- `tokenizer.json` (if present in repo)
- `config.json` (if present in repo)

### ONNX download path

Downloads pre-built ONNX files from HuggingFace.

Files downloaded:
- `model.onnx` (+ any external data files)
- `tokenizer.json`
- `config.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

### ONNX conversion path (see ADR-0010)

When no pre-built GGUF or ONNX is available:

```
1. python3 -m venv <tmpdir>/hologram-ai-conv
2. <venv>/bin/pip install optimum[exporters] transformers torch onnx
3. <venv>/bin/optimum-cli export onnx --model <MODEL_ID> <tmpdir>/onnx-output/
4. Copy output files to --output directory
5. Clean up virtualenv (unless --keep-venv)
```

### Progress reporting

Downloads show progress bars via `indicatif`:

```
Downloading meta-llama/Llama-3.2-1B (GGUF, Q4_0)
  model-Q4_0.gguf  ████████████████████░░░░  78% 512 MB / 657 MB  45 MB/s
  tokenizer.json   ████████████████████████ 100%  1.2 MB
  config.json      ████████████████████████ 100%  842 B
```

### Error handling

| Condition | Behavior |
|-----------|----------|
| Model not found | Error with HF API response |
| Gated model | Error: "This model requires authentication. Use --token or `huggingface-cli login`" |
| No python3 on PATH | Error: "python3 required for ONNX conversion. Install Python 3.10+ or use --format gguf" |
| Network failure | Retry up to 3 times with exponential backoff, then error |
| Disk space | Check available space before download, warn if insufficient |

### Output summary

```
Downloaded: meta-llama/Llama-3.2-1B
Format: GGUF (Q4_0)
Location: ./models/Llama-3.2-1B/
Files:
  model-Q4_0.gguf      657 MB
  tokenizer.json        1.2 MB
  config.json           842 B
```

---

## `hologram-ai validate`

Validate model outputs against reference runtimes. No changes from existing design
(see `specs/docs/prompts/05-validation-harness.md`).

```
hologram-ai validate <MODEL> [OPTIONS]

Options:
      --onnx             Compare against ONNX Runtime
      --gguf             Compare against llama.cpp
      --input <PATH>     Input tensors
      --prompt <TEXT>     Text prompt (for token generation comparison)
      --tokens <N>       Number of tokens to compare
      --report <PATH>    Save validation report as JSON
```

---

## `hologram-ai lower`

Lower a model and emit the `hologram::Graph`. No changes from existing design.

```
hologram-ai lower <MODEL> [OPTIONS]

Options:
      --emit-graph       Print the lowered graph structure
      --emit-registry    Print registered custom ops
      --format <FMT>     Output format [possible: text, json] [default: text]
```

---

## Benchmarking System (`--stats`)

The `--stats` flag is available on `compile`, `run`, and `generate` commands.

### Metrics

| Metric | Commands | Description |
|--------|----------|-------------|
| Import time | compile, run | Time to parse model format into AiGraph |
| Optimization time | compile, run | Time for AI-level fusion passes |
| Lowering time | compile, run | Time for AiGraph → hologram::Graph |
| Compilation time | compile, run | Time for hologram::compile() |
| Total compile time | compile, run | Sum of all compilation phases |
| Execution time | run | Wall time for forward pass |
| Tokens generated | generate | Total output token count |
| Prompt tokens | generate | Input prompt token count |
| Time-to-first-token | generate | Prefill time + first decode step |
| Tokens/s (decode) | generate | `(tokens - 1) / (total_time - ttft)` |
| Prefill tokens/s | generate | `prompt_tokens / prefill_time` |
| Peak memory (RSS) | all | Peak resident set size |
| Parameter count | all | Total with quantization breakdown |

### Implementation

```rust
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

impl BenchmarkStats {
    pub fn tokens_per_second(&self) -> Option<f64>;
    pub fn time_to_first_token(&self) -> Option<Duration>;
    pub fn prefill_tokens_per_second(&self) -> Option<f64>;
    pub fn display(&self);  // formatted terminal output to stderr
}
```

Phase timing wraps each pipeline step with `Instant::now()` / `elapsed()`.

Memory tracking:
- macOS: `mach_task_info` (`mach_task_basic_info.resident_size`)
- Linux: `/proc/self/status` (`VmRSS` field)

Stats output always goes to stderr so it doesn't interfere with model output
on stdout.

---

## CLI Module Structure

```
crates/hologram-ai/src/
├── main.rs              CLI entry point (clap App dispatch)
├── lib.rs               Public facade API (ModelCompiler, etc.)
├── cli/
│   ├── mod.rs           Clap enum + shared utilities
│   ├── compile.rs       compile subcommand
│   ├── inspect.rs       inspect subcommand
│   ├── run.rs           run subcommand
│   ├── generate.rs      generate subcommand
│   ├── download.rs      download subcommand
│   ├── validate.rs      validate subcommand
│   └── lower.rs         lower subcommand
├── session.rs           ModelCompiler, CompiledModel, InferenceSession
├── stream.rs            TokenStream, Tokenizer trait, samplers
├── validate.rs          ValidationSuite, tensor comparison
├── stats.rs             BenchmarkStats, PhaseStats, memory tracking
└── download/
    ├── mod.rs            Download orchestration
    ├── hf_api.rs         HuggingFace API client (reqwest)
    ├── convert.rs        Python virtualenv + optimum-cli conversion
    └── progress.rs       Progress bar utilities (indicatif)
```

---

## Dependencies (additions for CLI)

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
reqwest = { version = "0.12", features = ["json", "stream"] }
indicatif = "0.17"
serde_json = "1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
tempfile = "3"
sha2 = "0.10"        # file integrity checks
dirs = "5"            # default output directory
```

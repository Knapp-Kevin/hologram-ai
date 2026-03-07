# hologram-ai: Roadmap

---

## MVP (Weeks 1–4)

**Goal:** End-to-end single forward pass for a small GGUF decoder-only LLM on CPU.

### Scope

- GGUF importer for LLaMA-family models (Q4_0 quantization)
- `AiGraph` IR with core ops and quant descriptors
- Optimization: constant folding, shape propagation, attention fusion
- Memory planner: tensor liveness + KV-cache layout
- Lowering to `hologram::ExecutionPlan` (CPU backend only)
- `InferenceSession::run()` — single forward pass (no streaming)
- Validation: golden tensor test against committed fixture
- CI: unit tests + integration smoke test

### Exit criteria

- `hologram-ai-gguf` imports TinyLlama 1.1B Q4_0 without error
- Single forward pass produces logits tensor of correct shape and dtype
- Top-1 logit matches llama.cpp reference (greedy, no sampling) on golden prompt
- All unit tests pass on `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`

### Explicitly deferred from MVP

- ONNX importer
- GGML importer
- Streaming token generation
- KV-cache (single-pass only)
- Metal backend
- Multi-turn conversation
- Tokenizer integration

---

## Phase 2 (Weeks 5–10)

**Goal:** Full LLM inference path with streaming, KV-cache, and ONNX support.

### Scope

- KV-cache implementation and multi-turn session management
- Streaming token generation via `TokenStream`
- `hologram-ai-stream` with all sampling strategies
- ONNX importer (opset 13–17, covering BERT + GPT-2 class models)
- GGML importer (migration path for legacy weights)
- Shape propagation for dynamic `seq_len` dimension
- Validation harness: ONNX Runtime and llama.cpp reference comparisons
- CLI: `hologram-ai run`, `hologram-ai generate`, `hologram-ai validate`
- Expanded architecture recognizers: Mistral, Phi, Qwen, Gemma

### Milestones

| Milestone | Deliverable |
|-----------|------------|
| M2.1 | KV-cache: multi-turn TinyLlama conversation works |
| M2.2 | Streaming: `TokenStream` produces tokens with correct stop logic |
| M2.3 | ONNX: BERT base classification passes numerical validation vs ORT |
| M2.4 | ONNX: GPT-2 small text generation matches ORT outputs |
| M2.5 | CLI: `hologram-ai generate` works from command line |

---

## Phase 3 (Weeks 11–18)

**Goal:** Metal backend, quantized kernels, performance validation.

### Scope

- Metal backend integration (`hologram-ai-backend-metal`)
- Quantized GEMM kernels on Metal (Q4_0, Q8_0)
- Flash attention kernel integration (Metal and CPU)
- `hologram-ai-opt`: FFN fusion, layer-norm fusion passes
- Larger model support: 7B/13B models with mmap weight loading
- Performance benchmarking harness
- BF16 support
- Float8 support (experimental)

### Milestones

| Milestone | Deliverable |
|-----------|------------|
| M3.1 | Metal backend: TinyLlama runs on Apple Silicon GPU |
| M3.2 | Quantized GEMM: 2x throughput improvement vs eager dequant |
| M3.3 | 7B model: Mistral 7B Q4_K_M generates 10+ tokens/sec on M2 |
| M3.4 | Benchmark suite published (tokens/sec, memory usage) |

---

## Phase 4 (Future)

**Goal:** Multi-backend portability, larger models, and advanced inference features.

### Items

- CUDA backend (`hologram-ai-backend-cuda`)
- WebGPU backend (`hologram-ai-backend-webgpu`)
- Multi-GPU tensor parallelism
- Speculative decoding
- Continuous batching (for server workloads)
- LoRA / adapter layer support in GGUF
- GGUF v4 format support (as it evolves)
- INT4 block quantization on all backends
- GPTQ / AWQ quantization import
- Vision-language model support (multi-modal inputs)
- Autograd / fine-tuning exploration (separate branch, not MVP concern)

---

## Technical Milestones vs Demo Milestones

### Technical milestones (internal quality gates)

| ID | Description | Phase |
|----|-------------|-------|
| T1 | GGUF parser handles all current quant types | MVP |
| T2 | `AiGraph` validation passes for all committed fixtures | MVP |
| T3 | Lowering table covers all ops in LLaMA graph | MVP |
| T4 | Memory planner deterministic across runs | MVP |
| T5 | KV-cache pointer arithmetic correct for 100+ turns | Phase 2 |
| T6 | ONNX opset 13–17 coverage >90% of ops in test model set | Phase 2 |
| T7 | f32 numerical error < 1e-5 vs ORT on all ONNX test models | Phase 2 |
| T8 | Metal backend passes same golden tests as CPU | Phase 3 |
| T9 | 7B model generates at ≥10 tokens/sec on M2 Pro | Phase 3 |

### Demo milestones (user-visible)

| ID | Description | Phase |
|----|-------------|-------|
| D1 | `hologram-ai generate tinyllama.gguf "Hello"` produces coherent output | MVP |
| D2 | Multi-turn conversation: 10-turn chat with consistent context | Phase 2 |
| D3 | BERT sentiment classification demo via ONNX | Phase 2 |
| D4 | 7B model chat demo on Apple Silicon | Phase 3 |
| D5 | Side-by-side perf comparison with llama.cpp on same hardware | Phase 3 |

---

## Explicit Sequencing Rationale

**GGUF before ONNX** because:
- GGUF is the active LLM ecosystem format
- Decoder-only LLMs are the primary inference workload
- GGUF exercizes quantization early (critical for design validation)
- ONNX adds import complexity (protobuf, opset, external data) that distracts
  from the core compiler pipeline bring-up

**CPU before Metal** because:
- CPU backend is available on all CI machines
- Numerical correctness is easier to debug on CPU
- Metal backend depends on hologram-metal being ready
- Architecture is designed to be backend-agnostic from day one

**KV-cache after single-pass** because:
- KV-cache introduces stateful session complexity
- Single-pass validates the complete lowering pipeline first
- Easier to debug correctness issues without cache state

---

## Deferred Items (explicitly not in any phase above)

- Training / autograd
- Distributed inference (beyond single-machine multi-GPU)
- On-device fine-tuning
- Model compression utilities (post-training quantization, pruning)
- HuggingFace Hub direct download integration
- Safetensors format import
- PyTorch TorchScript import

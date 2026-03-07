# Sprint-003: hologram-ai Week 3 — KV-Cache, Streaming, Fusion Passes

## Objective

Implement the full LLM generation path: KV-cache, multi-turn session,
streaming token generation, and the key optimization passes (attention fusion,
FFN fusion, quantized-matmul fusion).

## Goals

- KV-cache works for LLaMA/Mistral: multi-turn conversation, correct present_len tracking
- `TokenStream` streaming works with GreedySampler and TopPSampler
- Attention fusion pass converts QKV+softmax sequences to `MultiHeadAttention`
- QuantMatMulFusion converts `Dequantize → MatMul` to `QuantizedMatMul`
- `hologram-ai-cli generate` command works from command line
- GGML arch recognizer added (llama v1 format)

## Inputs

- KV-cache prompt: `specs/prompts/hologram-ai/04-gguf-llm-path.md`
- Week 2 deliverables (ONNX + expanded quant + shape prop working)

## Deliverables

1. `hologram-ai-mem` — `KvCacheLayout` computation from `AiGraph` metadata
2. `hologram-ai-lower` — KV-cache slot nodes (`KvSlotWrite`, `KvSlotRead`)
3. `hologram-ai-session` — `run_prefill()`, `run_decode()`, `generate()`, `reset_cache()`
4. `hologram-ai-stream` — `TokenStream`, all samplers, `stream_tokens()`
5. `hologram-ai-opt` — `AttentionFusion`, `FfnFusion`, `QuantMatMulFusion` passes
6. `hologram-ai-ggml` — `LlamaV1Arch` recognizer
7. `hologram-ai-cli` — `generate`, `compile`, `inspect` subcommands
8. `hologram-ai-cli` — `compile` delegates to hologram CLI with library fallback (ADR-0009)
9. `hologram-ai-cli` — `inspect` supports `.holo`, `.gguf`, `.onnx`, `.bin` files
10. Multi-turn integration test
11. Greedy token generation matches expected output for synthetic fixture

## Tasks

### Day 1: KV-cache planner + lowering

- [ ] Implement `MemoryPlanner::compute_kv_layout()` from GGUF metadata
- [ ] Extend `MemoryPlan` with `Option<KvCacheLayout>`
- [ ] Add `KvSlotWrite` and `KvSlotRead` to lowering op dispatch
- [ ] Lower `GroupedQueryAttention` with KV-cache slot nodes
- [ ] Test: `MemoryPlanner` produces correct `KvCacheLayout` for TinyLlama params

### Day 2: Session prefill + decode

- [ ] Implement `InferenceSession::run_prefill()` — injects `present_len=0`
- [ ] Implement `InferenceSession::run_decode()` — injects current `present_len`
- [ ] Implement `InferenceSession::generate()` — prefill + decode loop
- [ ] Implement `InferenceSession::reset_cache()`
- [ ] Test: multi-turn test (2 turns, check `present_len` after each)

### Day 3: hologram-ai-stream

- [ ] Implement `Tokenizer` trait
- [ ] Implement `GreedySampler`, `TopKSampler`, `TopPSampler`, `TemperatureSampler`, `MinPSampler`
- [ ] Implement `sample_token()` pipeline (temperature → top-k → top-p → min-p → sample)
- [ ] Implement `stream_tokens()` using `async-stream`
- [ ] Test: `TokenStream` yields correct number of tokens, stops at EOS
- [ ] Test: greedy sampler always returns argmax

### Day 4: Optimization passes

- [ ] Implement `AttentionFusion` — fuse QKV + scale + mask + softmax → `MultiHeadAttention`
- [ ] Implement `FfnFusion` — fuse gate + up + silu → `FusedSwiGLU`
- [ ] Implement `QuantMatMulFusion` — fuse `Dequantize → MatMul` → `QuantizedMatMul`
- [ ] Test each pass: before/after node count, output equivalence
- [ ] Add all three to `OptPipeline::default()` pass order

### Day 5: CLI + GGML + integration

- [ ] Restructure CLI into `src/cli/` module directory (see Prompt 06)
- [ ] Implement `hologram-ai-cli generate` subcommand using `clap`
- [ ] Implement `hologram-ai-cli compile` subcommand with hologram CLI delegation
- [ ] Implement `hologram-ai-cli inspect` subcommand (summary, ops, tensors, metadata, json formats)
- [ ] Implement `LlamaV1Arch` in `hologram-ai-ggml`
- [ ] Full integration test: GGUF TinyLlama synthetic → `generate(5 tokens)` → correct shape
- [ ] `cargo test --workspace` passes
- [ ] `cargo run -p hologram-ai-cli -- generate synthetic.gguf "Hello"` produces output
- [ ] `cargo run -p hologram-ai-cli -- compile synthetic.gguf -o synthetic.holo` produces archive
- [ ] `cargo run -p hologram-ai-cli -- inspect synthetic.gguf` prints summary

## Exit Criteria

- [ ] `InferenceSession::generate()` produces 5 tokens for a fixed prompt on synthetic model
- [ ] Multi-turn test: 10 turns with growing `present_len`
- [ ] `TokenStream` yields tokens and stops at EOS
- [ ] `AttentionFusion` pass reduces node count for LLaMA graph
- [ ] `hologram-ai-cli generate` prints tokens to stdout
- [ ] `hologram-ai-cli compile` produces `.holo` file
- [ ] `hologram-ai-cli inspect` prints model summary for GGUF and ONNX files
- [ ] All tests pass: `cargo test --workspace`

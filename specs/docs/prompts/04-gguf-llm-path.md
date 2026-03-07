# Prompt: GGUF LLM Full Path (KV-Cache + Streaming)

## Purpose

Extend the `hologram-ai` pipeline from single-pass inference to full
multi-turn LLM generation with KV-cache and streaming token output.

Run this prompt after the ONNX importer is working (Week 3 or later).

---

## Context

At this point:
- `hologram-ai-gguf` imports LLaMA-family GGUF models
- `hologram-ai-session` has a working `run()` single forward pass
- `hologram-ai-opt` and `hologram-ai-lower` handle the core op set

Your task is to implement:
1. KV-cache support in `hologram-ai-session`
2. Multi-turn `generate()` in `InferenceSession`
3. `TokenStream` streaming in `hologram-ai-stream`
4. Prefill / decode phase distinction

Architecture references:
- `../hologram-architecture/specs/projects/hologram-ai/runtime-model.md`

---

## Task 1: KV-Cache in MemoryPlan

Update `hologram-ai-mem` to compute `KvCacheLayout` from `AiGraph` metadata:

```rust
impl MemoryPlanner {
    fn compute_kv_layout(&self, graph: &AiGraph) -> Option<KvCacheLayout> {
        let n_layers = graph.metadata.get("block_count")?.as_u64()? as usize;
        let n_kv_heads = graph.metadata.get("attention.head_count_kv")?.as_u64()? as usize;
        let head_dim = {
            let embd = graph.metadata.get("embedding_length")?.as_u64()? as usize;
            let n_heads = graph.metadata.get("attention.head_count")?.as_u64()? as usize;
            embd / n_heads
        };
        let max_seq_len = self.opts.max_seq_len.unwrap_or(4096);
        let dtype = self.opts.kv_cache_dtype;
        let bytes_per_layer = 2 * n_kv_heads * head_dim * max_seq_len * dtype.bytes();

        Some(KvCacheLayout {
            layers: n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            dtype,
            bytes_per_layer: bytes_per_layer as u64,
            total_bytes: (bytes_per_layer * n_layers) as u64,
        })
    }
}
```

---

## Task 2: KV-Cache Lowering

Update `hologram-ai-lower` to emit KV-cache slot nodes:

For each transformer block in the lowered plan:
1. After K and V projection: `KvSlotWrite { layer, offset_input: "present_len" }`
2. Before attention computation: `KvSlotRead { layer, length_input: "present_len" }`

The `present_len` value is injected as a dynamic plan input by the session.

```rust
// In op_dispatch for GroupedQueryAttention with KV-cache enabled:
fn lower_gqa_with_kvcache(
    node: &AiNode,
    layer_idx: usize,
    kv_buf: &BufferAlloc,
) -> Vec<PlanNode> {
    vec![
        PlanNode::KvSlotWrite { layer: layer_idx, k_src: ..., v_src: ... },
        PlanNode::KvSlotRead  { layer: layer_idx, k_dst: ..., v_dst: ..., len_input: "present_len" },
        PlanNode::Gqa { q: ..., k: ..., v: ..., causal: true },
    ]
}
```

---

## Task 3: Session `generate()` with KV-cache

```rust
impl InferenceSession {
    pub fn generate(
        &mut self,
        token_ids: &[u32],
        opts: &GenerateOptions,
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::new();

        // Prefill phase
        let logits = self.run_prefill(token_ids)?;
        let mut next_token = sample(&logits, opts);
        generated.push(next_token);

        // Decode phase
        for _ in 1..opts.max_new_tokens {
            if self.is_stop_token(next_token, opts) { break; }
            let logits = self.run_decode(next_token)?;
            next_token = sample(&logits, opts);
            generated.push(next_token);
        }

        Ok(generated)
    }

    fn run_prefill(&mut self, tokens: &[u32]) -> Result<Tensor> {
        // present_len starts at 0 for prefill
        let inputs = self.build_prefill_inputs(tokens)?;
        let outputs = self.run_with_kv(inputs, 0)?;
        self.kv_cache.as_mut().map(|c| c.present_len = tokens.len());
        Ok(outputs["logits"].clone())
    }

    fn run_decode(&mut self, token: u32) -> Result<Tensor> {
        let present = self.kv_cache.as_ref().map_or(0, |c| c.present_len);
        let inputs = self.build_decode_inputs(token)?;
        let outputs = self.run_with_kv(inputs, present)?;
        self.kv_cache.as_mut().map(|c| c.present_len += 1);
        Ok(outputs["logits"].clone())
    }
}
```

---

## Task 4: Implement `hologram-ai-stream`

`TokenStream` wraps `InferenceSession` and implements `futures::Stream`:

```rust
use async_stream::try_stream;

pub fn stream_tokens(
    mut session: InferenceSession,
    tokenizer: Box<dyn Tokenizer>,
    prompt: &str,
    opts: GenerateOptions,
) -> impl Stream<Item = Result<Token>> {
    let prompt_ids = tokenizer.encode(prompt);
    try_stream! {
        // Prefill
        session.run_prefill(&prompt_ids)?;

        // Decode loop
        let mut prev_token = *prompt_ids.last().unwrap_or(&0);
        for i in 0..opts.max_new_tokens {
            let logits = session.run_decode(prev_token)?;
            let token_id = sample_token(&logits, &opts)?;
            let text = tokenizer.decode(&[token_id]);

            let is_stop = token_id == tokenizer.eos_token_id()
                || check_stop_sequences(&[token_id], &opts.stop_sequences);

            yield Token { id: token_id, text, logprob: None, is_stop };

            if is_stop { break; }
            prev_token = token_id;
        }
    }
}
```

Implement all samplers in `hologram-ai-stream::sampler`:

```rust
pub fn sample_token(logits: &Tensor, opts: &GenerateOptions) -> Result<u32> {
    let mut probs = softmax(apply_temperature(logits, opts.temperature));

    if opts.top_k > 0 {
        probs = top_k_filter(&probs, opts.top_k);
    }
    if opts.top_p < 1.0 {
        probs = top_p_filter(&probs, opts.top_p);
    }
    if opts.min_p > 0.0 {
        probs = min_p_filter(&probs, opts.min_p);
    }

    multinomial_sample(&probs, opts.seed)
}
```

---

## Task 5: CLI `generate` subcommand

Update `hologram-ai-cli` to support:

```
hologram-ai generate <model.gguf> "<prompt>" [--max-tokens N] [--temperature F] [--top-p F]
```

Example implementation using `clap`:

```rust
#[derive(Parser)]
struct GenerateArgs {
    model: PathBuf,
    prompt: String,
    #[arg(long, default_value = "256")]
    max_tokens: usize,
    #[arg(long, default_value = "0.8")]
    temperature: f32,
    #[arg(long, default_value = "0.95")]
    top_p: f32,
}
```

Use `futures::executor::block_on` to drive the stream synchronously in the CLI.
Print tokens as they arrive (unbuffered output for interactive feel).

---

## Task 6: Multi-turn test

```rust
#[test]
fn multi_turn_kvcache_consistent() {
    let model = compile_tinyllama_fixture();
    let mut session = model.session(SessionOptions {
        max_seq_len: Some(512),
        kv_cache_dtype: DType::F16,
        ..Default::default()
    }).unwrap();

    // Turn 1
    let t1 = session.generate(&[1, 2, 3], &GenerateOptions {
        max_new_tokens: 5, temperature: 0.0, ..Default::default()
    }).unwrap();
    assert_eq!(t1.len(), 5);

    // Turn 2 — context is extended, not reset
    let t2 = session.generate(&[t1[4]], &GenerateOptions {
        max_new_tokens: 5, temperature: 0.0, ..Default::default()
    }).unwrap();
    assert_eq!(t2.len(), 5);

    // reset
    session.reset_cache();
    assert_eq!(session.kv_cache.as_ref().unwrap().present_len, 0);
}
```

---

## Acceptance Criteria

- `InferenceSession::generate()` produces consistent output for greedy sampling
- `TokenStream` yields correct number of tokens before stop
- Multi-turn test passes
- `hologram-ai generate tinyllama.gguf "Hello"` prints a response to stdout
- Top-1 greedy token matches llama.cpp reference on a fixed prompt (golden test)

# Research Memo: Mixture-of-Experts and Conditional Memory (Engram)

- Date: 2026-03-22
- Status: Exploratory research
- Author: Architecture
- Source: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models" (Cheng et al., 2026, DeepSeek-AI / Peking University)

---

## 1. Executive Summary

This memo captures the architectural implications of two sparse model primitives — **Mixture-of-Experts (MoE)** and **Engram (conditional memory)** — for the hologram-ai compiler. Neither requires immediate implementation, but both are likely to appear in future open-weight models (particularly from DeepSeek).

**Key findings:**

1. MoE is already deployed in production models (DeepSeek-V3, Mixtral, Qwen-MoE). hologram-ai has no MoE support today. When needed, the right compilation strategy is **batched gather-scatter** (tensor ops), not runtime `If`/`Branch` control flow.
2. Engram is a new primitive from DeepSeek that complements MoE with O(1) hash-based embedding lookups. Models using it don't exist as open weights yet, but likely will.
3. Both primitives are compilable within hologram's flat DAG execution model — no runtime branching needed.
4. The main blocker for both is hologram base's `FloatOp` coverage (same blocker as current fusion work).

---

## 2. Mixture-of-Experts (MoE) — Primer

### What it is

MoE replaces the single dense feed-forward network (FFN) in each transformer block with **N smaller "expert" FFNs**, of which only **K are activated per token** based on a learned routing function. This scales total model parameters without proportionally increasing compute.

**Dense FFN (standard transformer):**
```
hidden → gate_proj(hidden) → activation → up_proj → down_proj → output
```
One set of weight matrices. Every token uses the same FFN.

**MoE FFN:**
```
hidden → router(hidden) → softmax → top-K selection
  → expert_i(hidden) for each selected expert i ∈ {1..K}
  → weighted_sum(expert_outputs, router_weights) → output
```
N sets of weight matrices (experts). Each token uses only K of them.

### Why it scales

| Metric | Dense (4B) | MoE (27B) | Ratio |
|--------|-----------|-----------|-------|
| Total parameters | 4.1B | 26.7B | 6.5x |
| Activated parameters per token | 3.8B | 3.8B | 1.0x |
| Training FLOPs per token | baseline | ~baseline | ~1.0x |

The "free" parameters (unselected experts) increase model capacity without increasing per-token compute. This is why MoE models can be much larger than dense models at equivalent inference cost.

### Key components

| Component | Role | Shape |
|-----------|------|-------|
| **Router** | Linear layer: hidden → expert logits | `[batch*seq, hidden] @ [hidden, num_experts] → [batch*seq, num_experts]` |
| **Gating** | Softmax + top-K selection | `[batch*seq, num_experts] → [batch*seq, K]` (indices + weights) |
| **Expert FFN** | Standard FFN (smaller) | Same structure as dense FFN, but narrower |
| **Combine** | Weighted sum of expert outputs | `sum(weight_i * expert_i_output)` per token |
| **Shared experts** | Always-active experts (DeepSeekMoE) | Same as expert FFN, but not routed |

### DeepSeekMoE variant (used in the Engram paper)

- **Fine-grained experts**: More experts, each smaller (72 routed + 2 shared, top-6)
- **Shared expert isolation**: 2 experts always active for every token (capture common patterns)
- **Auxiliary-loss-free load balancing**: Avoids auxiliary loss that can distort training dynamics
- **Production models**: DeepSeek-V3 (671B total / 37B active), DeepSeek-R1

### Key models using MoE

| Model | Total params | Active params | Experts | Top-K |
|-------|-------------|---------------|---------|-------|
| Mixtral-8x7B | 46.7B | 12.9B | 8 | 2 |
| DeepSeek-V3 | 671B | 37B | 256+1 | 8 |
| Qwen-MoE | 14.3B | 2.7B | 60+4 | 4 |
| Engram-27B (this paper) | 26.7B | 3.8B | 55+2 | 6 |

---

## 3. Engram — Conditional Memory

### What it is

Engram introduces **conditional memory** as a complement to MoE's conditional computation. Where MoE sparsely activates neural computation (experts), Engram sparsely retrieves static knowledge (embeddings) via O(1) hash lookups.

The insight: a substantial portion of language modeling — named entities, formulaic phrases, idiomatic expressions — is **local, static, and stereotyped**. Standard transformers waste early-layer compute reconstructing these patterns through attention and FFN layers. Engram offloads this to a direct lookup table.

### Architecture

The Engram module processes each token position in two phases:

**Phase 1 — Sparse Retrieval:**
```
1. Tokenizer compression: raw token IDs → canonical IDs via NFKC normalization
   (Apple, ␣apple → same canonical ID; achieves ~23% vocab reduction)
2. N-gram formation: suffix N-grams g_t,n = (x'_{t-n+1}, ..., x'_t) for n ∈ {2,3}
3. Multi-head hashing: K hash functions per N-gram order
   z_{t,n,k} = φ_{n,k}(g_{t,n})  — multiplicative-XOR hash
   e_{t,n,k} = E_{n,k}[z_{t,n,k}]  — embedding table lookup (prime-sized)
4. Concatenation: e_t = concat(all retrieved embeddings) ∈ R^{d_mem}
```

**Phase 2 — Context-Aware Gating:**
```
1. Key/Value projection: k_t = W_K @ e_t,  v_t = W_V @ e_t
2. Gating: α_t = sigmoid(RMSNorm(h_t)^T @ RMSNorm(k_t) / sqrt(d))
   - h_t is the current hidden state (has global context from preceding attention)
   - If retrieved memory contradicts context, gate → 0 (suppresses noise)
3. Gated value: v̄_t = α_t · v_t
4. Depthwise causal conv: Y = SiLU(Conv1D(RMSNorm(V̄))) + V̄
   - kernel_size=4, dilation=max_N_gram_order, causal padding
5. Residual add: H^(ℓ) ← H^(ℓ) + Y
```

### Sparsity allocation — the U-shaped scaling law

Given a fixed total parameter budget, the paper asks: how should sparse capacity be split between MoE experts (conditional computation) and Engram embeddings (conditional memory)?

Define **allocation ratio** ρ ∈ [0,1]:
- ρ = 1: Pure MoE (all inactive params are routed experts)
- ρ = 0: Pure Engram (all inactive params are embedding slots)

**Key finding:** Optimal allocation is ρ ≈ 75-80% across compute regimes. Pure MoE (ρ=100%) is suboptimal. Reallocating ~20-25% of sparse params to Engram consistently improves validation loss.

The optimum is stable across scales, suggesting this is a robust architectural principle rather than an artifact of a particular model size.

### Engram-27B configuration

| Parameter | Value |
|-----------|-------|
| Total params | 26.7B (iso with MoE-27B baseline) |
| Routed experts | 55 (reduced from 72) |
| Shared experts | 2 |
| Top-K | 6 |
| Engram params | 5.7B |
| Engram layers | [2, 15] |
| N-gram orders | {2, 3} |
| Hash heads | 8 |
| Embedding dim | 1280 |
| Vocab size (after compression) | 2,262,400 slots |

### Key results vs iso-parameter MoE-27B

| Benchmark | MoE-27B | Engram-27B | Delta |
|-----------|---------|------------|-------|
| MMLU | 57.4 | 60.4 | +3.0 |
| BBH | 50.9 | 55.9 | +5.0 |
| ARC-Challenge | 70.1 | 73.8 | +3.7 |
| HumanEval | 37.8 | 40.8 | +3.0 |
| MATH | 28.3 | 30.7 | +2.4 |
| Multi-Query NIAH | 84.2 | 97.0 | +12.8 |

Gains are **not limited to knowledge-intensive tasks** — general reasoning and code/math improve even more, because Engram frees up early-layer compute depth for complex reasoning.

### Inference offloading

Engram's key system advantage: **deterministic addressing**. Unlike MoE routing (depends on runtime hidden states), Engram indices depend only on the input token sequence and are known before the forward pass begins.

This enables:
- **Asynchronous prefetch** from host memory via PCIe, overlapping with preceding transformer block computation
- **Multi-level cache hierarchy**: frequent N-grams in HBM, long tail on host DRAM or NVMe (Zipfian distribution)
- **Result**: 100B-parameter Engram table offloaded to host memory incurs <3% throughput penalty

---

## 4. IR Ops Needed in hologram-ai

### For MoE

```
AiOp::MoERouter {
    num_experts: u32,
}
// Router is just MatMul + Softmax, but semantically distinct for fusion

AiOp::MoEGate {
    top_k: u32,
    num_experts: u32,
}
// TopK selection of expert indices + weights

AiOp::MoECombine {
    top_k: u32,
}
// Weighted sum of expert outputs

// Fused variant (post-optimization):
AiOp::FusedMoEFFN {
    num_experts: u32,
    top_k: u32,
    num_shared_experts: u32,
}
```

### For Engram

```
AiOp::NgramHashLookup {
    ngram_orders: Vec<u32>,    // e.g. [2, 3]
    num_heads: u32,            // hash heads per order
    table_sizes: Vec<u32>,     // prime-sized embedding tables
    embedding_dim: u32,
}
// Deterministic hash → gather from embedding tables

AiOp::ContextAwareGate {
    use_rmsnorm: bool,
}
// sigmoid(RMSNorm(h)^T @ RMSNorm(W_K @ e) / sqrt(d)) * (W_V @ e)
```

### GGUF importer changes

Extend `ArchParams` with:
- `num_experts`, `experts_per_token`, `num_shared_experts`
- `engram_layers`, `engram_ngram_orders`, `engram_num_heads`, `engram_dim`

Parse expert weight tensors: `blk.{layer}.ffn_experts.{expert_id}.{gate|up|down}.weight`
Parse router weights: `blk.{layer}.ffn_router.weight`
Parse Engram tables: `engram.{layer}.{ngram_order}.{head}.embedding`

---

## 5. Execution Model — Why hologram's Flat DAG Works

### The wrong approach: runtime `If` branching

hologram's executor is a flat DAG of parallel levels. When it encounters an `If` node today, it **flattens both branches** and merges with `Where`. For MoE with 72 experts, this would execute all 72 expert FFNs and discard 66 — a 12x compute waste that defeats the purpose of MoE.

Adding true runtime branching (instruction-pointer dispatch, conditional jumps) would require fundamental changes to hologram's execution model and is not necessary.

### The right approach: batched gather-scatter

MoE inference is implementable as pure tensor operations:

```
# 1. Route: compute expert assignments
router_logits = hidden @ router_weight           # [B*S, num_experts]
topk_weights, topk_idx = topk(router_logits, K)  # [B*S, K] each

# 2. Gather: group tokens by assigned expert
#    For each expert e, collect tokens where topk_idx contains e
#    This is a permutation + scatter, not a branch

# 3. Execute: batched expert matmul
#    Each expert processes its assigned tokens
#    This is a grouped/batched GEMM — well-understood GPU kernel

# 4. Scatter-combine: route outputs back and weight
output = scatter_add(expert_outputs * topk_weights, original_positions)
```

All ops are **statically shaped** (assuming pad-to-max for expert batch sizes) and fit hologram's flat DAG. The "conditional" part is index-based gathering, which is a `Gather` op hologram already has.

### For Engram

Even simpler — no routing needed. Hash function maps input token IDs to embedding indices deterministically. This is just a `Gather` with a precomputed index.

---

## 6. Fusion Opportunities

### MoE fusions

| Pattern | Fused Op | Benefit |
|---------|----------|---------|
| `MatMul(hidden, router_weight) + Softmax + TopK` | `FusedMoERouter` | Single kernel, avoid materializing full logit matrix |
| `Norm + Router + Expert FFN + Combine` | `FusedMoEBlock` | Eliminates all intermediates for the entire MoE layer |

### Engram fusions

| Pattern | Fused Op | Benefit |
|---------|----------|---------|
| `RMSNorm(h) @ RMSNorm(W_K @ e) / sqrt(d) → sigmoid` | `FusedEngramGate` | Single kernel for the gating computation |
| `SiLU(Conv1D(RMSNorm(v))) + v` | `FusedEngramConv` | Analogous to existing SwiGLU fusion |
| `W_V` shared + M distinct `W_K^(m)` for multi-branch | `FusedMultiBranchGate` | Collapse into single FP8 matmul |

---

## 7. hologram Base Dependencies

Both MoE and Engram require new `FloatOp` variants in the hologram base crate — the same class of blocker that currently affects MatMul+Activation and Concat+MatMul fusion.

### Required FloatOp variants

| Op | Purpose | Kernel complexity |
|----|---------|-------------------|
| `TopK` | Select top-K expert indices from router logits | Partial sort — well-understood |
| `BatchedGEMM` | Run K different weight matrices on K token subsets | Grouped matmul — standard GPU pattern |
| `ScatterAdd` | Route expert outputs back to original positions | Index-based reduction |
| `HashGather` | Hash N-gram → index → embedding lookup | Trivial: hash + table index |
| `DepthwiseConv1D` | Engram's causal convolution | Standard conv kernel |

None of these are exotic. The hologram base crate simply needs to expand its `FloatOp` enum and provide dispatch handlers.

---

## 8. Priority and Timeline

### Current assessment: **Not now**

1. **No test models**: No Engram model exists as open weights. MoE models exist (Mixtral, DeepSeek-V3) but are large (46B-671B params), making conformance testing impractical without a small MoE test model.
2. **Same blocker**: hologram base `FloatOp` coverage. Until the current fusion work (MatMul+Activation, Concat+MatMul) lands and the FloatOp surface area expands, MoE/Engram ops would have nowhere to lower to.
3. **Higher-leverage work exists**: Getting dense LLaMA fast, expanding to BERT/SD/Whisper, and landing current fusion passes benefits more users.

### When to revisit

- When DeepSeek or another lab releases an Engram model as GGUF/ONNX
- When hologram base has broader FloatOp coverage (post current sprint)
- When a small MoE test model is available for conformance testing

### Estimated effort (when ready)

| Component | Effort |
|-----------|--------|
| IR ops (AiOp enum + shape rules) | 2 days |
| GGUF importer (expert weights + metadata) | 2-3 days |
| Optimization passes (MoEFusion) | 2-3 days |
| Lowering + dispatch | 2 days |
| **Subtotal (hologram-ai side)** | **~1.5 weeks** |
| hologram base FloatOp + kernels | 2-4 weeks (separate workstream) |
| Integration + conformance testing | 1-2 weeks |

---

## References

- Cheng et al., "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models" (2026). Code: https://github.com/deepseek-ai/Engram
- Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" (2024)
- Liu et al., "DeepSeek-V3 Technical Report" (2024)
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)

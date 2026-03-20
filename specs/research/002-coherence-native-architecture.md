# Research Memo: Coherence-Native Model Architecture

- Date: 2026-03-18
- Status: Exploratory research
- Author: Architecture
- Prompt: `specs/prompts/005-coherence-exploration.md`

---

## 1. Executive Summary

### What is a coherence-native model?

A coherence-native model replaces the transformer's token-to-token attention mechanism with computation based on **field dynamics**: inputs are lifted into complex-valued (amplitude + phase) states, deposited into a shared latent field, and processed through phase evolution, interference, resonance gating, and spectral transforms. The output is obtained by probing the resulting field structure rather than by accumulating weighted value vectors.

The core idea: instead of asking "which tokens should attend to which?" (O(n²) pairwise), ask "what is the evolved state of the field after all tokens have deposited energy into it?" — which can be computed via spectral methods in O(n log n) or via recurrent state-space dynamics in O(n).

### What it replaces or augments in transformers

| Transformer primitive | Coherence replacement |
|---|---|
| Pairwise attention scores | Interference patterns in shared field |
| Softmax normalization | Energy conservation / unitarity constraints |
| KV cache (growing linearly with context) | Standing-wave modal memory (fixed-size) |
| Position encodings (RoPE, ALiBi) | Phase velocity / wave number parameterization |
| Feedforward layers | Spectral mixing + nonlinear gating |

### Why this might be promising

1. **Subquadratic context**: Field-based computation naturally avoids the O(n²) attention bottleneck. Spectral transforms are O(n log n); recurrent state propagation is O(n).
2. **Fixed-size memory**: Standing-wave memory doesn't grow with context length — a fundamental advantage over KV caches for long-context inference.
3. **Natural compositionality**: Interference and superposition allow multiple "threads of meaning" to coexist in the same field without explicit routing, potentially enabling richer compositional representations.
4. **Physics-inspired inductive bias**: Wave dynamics have well-understood mathematical properties (conservation laws, spectral decomposition, resonance) that provide strong structural priors.
5. **Convergent evidence**: SSMs (Mamba), Fourier models (FNet, GFNet), complex-valued networks (CvNNs), and energy-based models each capture fragments of this idea. A unified framework might capture synergies that piecemeal approaches miss.

### Why this might fail

1. **Training instability**: Complex-valued networks are notoriously difficult to train. Phase collapse, gradient explosions in the angular domain, and poor conditioning of complex Jacobians are unsolved at scale.
2. **Hardware mismatch**: Modern accelerators are optimized for real-valued matrix multiplications. Complex arithmetic doubles FLOPs and memory. Spectral transforms (FFT) have poor cache locality on GPUs compared to dense matmul.
3. **SSMs already exist**: Mamba-2 achieves competitive results with a simpler recurrent formulation. The coherence framing adds mathematical elegance but unclear practical benefit over what SSMs already provide.
4. **No existence proof**: Nobody has trained a coherence-native model at any meaningful scale. All claims about scaling behavior are extrapolation.
5. **Discrete reasoning gap**: Field dynamics are inherently continuous. Language is discrete. Bridging this gap requires hybrid mechanisms that may negate the elegance of the approach.

---

## 2. Transformer-to-Coherence Mapping

| Transformer component | Role | Coherence-native replacement | Mechanism | Complexity |
|---|---|---|---|---|
| Token embedding | Map discrete tokens to vectors | **Phase-amplitude embedding**: map to complex vector z = a·e^(iφ) | Learned amplitude + phase per token | Same |
| Positional encoding (RoPE) | Encode position | **Wave-number encoding**: position encoded as phase velocity ω·t | Multiply by e^(iωt) where ω is learned per dimension | Same |
| Self-attention (QKV) | Pairwise token mixing | **Field deposition + interference**: tokens deposit energy, read via probe | Spectral convolution or recurrent propagation | O(n log n) or O(n) vs O(n²) |
| Softmax | Normalize attention weights | **Energy normalization**: enforce ||ψ||² = const | Unitarity constraint or amplitude renormalization | Same |
| Feedforward (MLP) | Per-token nonlinear transform | **Spectral mixing + resonance gating** | FFT → learned filter → IFFT → nonlinear gate | O(n log n) |
| Residual connection | Gradient highway | **Superposition**: ψ_out = ψ_in + Δψ | Same principle, complex-valued | Same |
| Layer normalization | Stabilize activations | **Phase normalization**: normalize amplitude, preserve phase | ||ψ|| → 1 while keeping arg(ψ) | Same |
| KV cache | Store past context | **Modal memory**: fixed-rank standing-wave buffer | Top-k resonant modes retained; updated via deposit/decay | O(k) fixed |
| MoE routing | Sparse expert selection | **Resonance routing**: tokens activate experts via frequency match | Dot product in spectral domain → top-k | Same sparsity |
| Recurrence / memory | Long-range state | **Selective state-space memory**: gated complex diagonal recurrence | SSM with complex eigenvalues + selective gating (à la Mamba) | O(n) |

---

## 3. Core Mathematical Model

### 3.1 State Representation

Each token position t holds a complex-valued latent state:

```
ψ_t ∈ ℂ^d
```

Decomposed into amplitude and phase:

```
ψ_t = a_t ⊙ exp(i·φ_t)     where a_t ∈ ℝ₊^d, φ_t ∈ [0, 2π)^d
```

The full sequence forms a **field**:

```
Ψ = [ψ_1, ψ_2, ..., ψ_n] ∈ ℂ^{n × d}
```

### 3.2 Phase-Amplitude Embedding

Token embedding lifts discrete tokens into complex space:

```
ψ_t = E_a[token_t] ⊙ exp(i · E_φ[token_t])
```

where E_a ∈ ℝ^{V×d} and E_φ ∈ ℝ^{V×d} are learned amplitude and phase embedding tables.

Position is encoded as phase rotation:

```
ψ_t ← ψ_t ⊙ exp(i · ω · t)
```

where ω ∈ ℝ^d is a learned frequency vector (analogous to RoPE frequencies, but applied multiplicatively in the complex domain).

### 3.3 Field Deposition

All tokens deposit energy into a shared spectral field via DFT:

```
Φ(k) = Σ_t  ψ_t · exp(-2πi·k·t/n)     (DFT of the sequence field)
```

This transforms the position-domain field into a frequency-domain representation where global structure is local.

### 3.4 Spectral Evolution

In the frequency domain, apply a learned complex filter:

```
Φ'(k) = H(k) ⊙ Φ(k)
```

where H(k) ∈ ℂ^d is a learned spectral transfer function. This is the key operation: it replaces attention's O(n²) pairwise mixing with O(n log n) global mixing via pointwise multiplication in the frequency domain.

This is equivalent to a circular convolution in position space — every token "sees" every other token, weighted by a learned kernel.

### 3.5 Interference and Resonance

After spectral filtering, transform back:

```
ψ'_t = IDFT[Φ'](t)
```

The resulting field exhibits **interference patterns**: positions where multiple source signals reinforce (constructive) or cancel (destructive). This is the coherence-native analog of attention — information flows not by explicit pairwise weighting, but by wave superposition.

**Resonance score** for selective gating:

```
r_t = |⟨ψ_t, ψ'_t⟩|² / (||ψ_t||² · ||ψ'_t||²)
```

This measures how much the evolved field at position t "resonates" with the original signal. High resonance → the field reinforced this position; low resonance → the field is carrying information away from this position.

### 3.6 Energy Conservation

To prevent signal explosion/collapse, enforce an energy budget:

```
Σ_t ||ψ_t||² = E₀     (constant across layers)
```

In practice, this is implemented via amplitude normalization after each block:

```
ψ_t ← ψ_t · √(E₀ / Σ_t ||ψ_t||²)     (global)
```

or per-position:

```
ψ_t ← ψ_t / ||ψ_t||                     (local, analogous to RMSNorm)
```

### 3.7 Candidate Formulation Summary

A single coherence block computes:

```
1. Φ = DFT(Ψ)                          — deposit into spectral field
2. Φ' = H ⊙ Φ                          — spectral evolution (learned filter)
3. Ψ' = IDFT(Φ')                       — read evolved field
4. r = resonance(Ψ, Ψ')                — compute resonance scores
5. Ψ'' = gate(r) ⊙ Ψ' + (1-gate(r)) ⊙ Ψ   — selective update
6. Ψ_out = PhaseNorm(Ψ'' + Ψ)          — residual + normalize
```

**Alternative: SSM formulation** (for causal / autoregressive):

Replace steps 1-3 with a complex diagonal state-space recurrence:

```
h_t = A ⊙ h_{t-1} + B ⊙ ψ_t           where A ∈ ℂ^d, |A| < 1
ψ'_t = C ⊙ h_t
```

This is essentially Mamba with complex-valued states, providing O(n) causal processing. The spectral (DFT) path and SSM path are dual formulations — the DFT version is non-causal but parallelizable; the SSM version is causal but sequential.

---

## 4. Layer/Block Design

### 4.1 Coherence Block (Full — non-causal)

```
┌──────────────────────────────────────────┐
│            CoherenceBlock                 │
│                                          │
│  Input: Ψ ∈ ℂ^{n × d}                   │
│                                          │
│  ┌─────────────────────┐                 │
│  │ 1. PhaseNorm(Ψ)     │  normalize amp  │
│  └──────────┬──────────┘                 │
│             ▼                            │
│  ┌─────────────────────┐                 │
│  │ 2. DFT along seq    │  O(n log n)     │
│  └──────────┬──────────┘                 │
│             ▼                            │
│  ┌─────────────────────┐                 │
│  │ 3. H ⊙ Φ            │  pointwise      │
│  │    (learned filter)  │  O(n·d)         │
│  └──────────┬──────────┘                 │
│             ▼                            │
│  ┌─────────────────────┐                 │
│  │ 4. IDFT along seq   │  O(n log n)     │
│  └──────────┬──────────┘                 │
│             ▼                            │
│  ┌─────────────────────┐                 │
│  │ 5. Resonance gate   │  per-position   │
│  │    g = σ(W·[ψ;ψ'])  │  gating         │
│  └──────────┬──────────┘                 │
│             ▼                            │
│  ┌─────────────────────┐                 │
│  │ 6. SpectralMLP      │  per-token      │
│  │    FFT→filter→IFFT  │  nonlinear      │
│  │    + SiLU gate       │                 │
│  └──────────┬──────────┘                 │
│             ▼                            │
│  │ 7. Residual: + Ψ    │                 │
│  │ 8. PhaseNorm         │                 │
│                                          │
│  Output: Ψ_out ∈ ℂ^{n × d}              │
└──────────────────────────────────────────┘
```

**Complexity**: O(n·d·log(n)) per block (dominated by FFT). Compare transformer: O(n²·d).

**Parameters per block**: 2·d (spectral filter H, real+imag) + gating MLP + spectral MLP filter.
Roughly: ~6·d² parameters (vs. ~12·d² for a standard transformer block with 4x MLP expansion).

**Hardware characteristics**:
- FFT: poor GPU utilization at small n (batch dimension helps). Good at n > 1024.
- Pointwise complex multiply: memory-bound, same as elementwise ops.
- No large matmuls — this is a concern for GPU utilization. The SpectralMLP sub-block can be reformulated as real matmul to recover some utilization.

### 4.2 Causal Coherence Block (for autoregressive generation)

Replace the DFT/IDFT path with a complex diagonal SSM:

```
┌──────────────────────────────────────────┐
│         CausalCoherenceBlock             │
│                                          │
│  Input: ψ_t ∈ ℂ^d, state h ∈ ℂ^d       │
│                                          │
│  1. PhaseNorm(ψ_t)                       │
│  2. h ← A ⊙ h + B ⊙ ψ_t                │
│     where A = diag(exp(λ)), |λ| < 0     │
│     (stable complex recurrence)          │
│  3. ψ'_t = C ⊙ h                        │
│  4. Resonance gate: g = σ(W·[ψ_t;ψ'_t]) │
│  5. SpectralMLP (local FFT or real MLP)  │
│  6. Residual + PhaseNorm                 │
│                                          │
│  Output: ψ_out ∈ ℂ^d, state h' ∈ ℂ^d   │
└──────────────────────────────────────────┘
```

**Complexity**: O(d) per token per block. State size: d complex values per layer.

**This is essentially Mamba with complex states.** The novelty (if any) is the phase-aware normalization, resonance gating, and spectral MLP — not the recurrence itself.

### 4.3 Hybrid Block (recommended starting point)

Alternate between causal coherence blocks and narrow attention blocks:

```
Layer 1: CausalCoherence  (O(n·d))
Layer 2: CausalCoherence  (O(n·d))
Layer 3: NarrowAttention   (O(n·w·d), w = local window)
Layer 4: CausalCoherence  (O(n·d))
Layer 5: CausalCoherence  (O(n·d))
Layer 6: NarrowAttention   (O(n·w·d))
...
```

Ratio: ~4:1 coherence-to-attention. The attention blocks handle discrete, precise retrieval tasks that field dynamics struggle with. This is the most pragmatic architecture.

---

## 5. Memory Architecture

### 5.1 Multi-Tier Memory Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Architecture                       │
│                                                             │
│  Tier 0: Transient Coherence (per-layer state)              │
│  ├── SSM hidden state h ∈ ℂ^d                               │
│  ├── Decays naturally via |A| < 1                           │
│  ├── Lifetime: ~100–1000 tokens                             │
│  └── Analogous to: short-term working memory                │
│                                                             │
│  Tier 1: Modal Memory (standing-wave buffer)                │
│  ├── Fixed-rank bank: M ∈ ℂ^{k × d}, k = 64–256           │
│  ├── Each row = one resonant mode (frequency + amplitude)   │
│  ├── Updated via deposit: M ← decay·M + project(ψ_t)       │
│  ├── Read via probe: m_t = M^H · q_t (query in freq domain)│
│  ├── Lifetime: entire context window                        │
│  └── Analogous to: compressed KV cache with fixed budget    │
│                                                             │
│  Tier 2: Selective Persistent Memory                        │
│  ├── Gated write: only high-resonance events update         │
│  ├── Implemented as top-k spectral components               │
│  ├── Persists across segments / turns                       │
│  └── Analogous to: episodic memory / scratchpad             │
│                                                             │
│  Tier 3: External Retrieval (optional)                      │
│  ├── When modal memory resonance is low → retrieve          │
│  ├── Interface: query vector → external index → inject      │
│  └── Analogous to: RAG / tool use                           │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 How This Differs From KV Cache

| Property | KV cache | Modal memory |
|---|---|---|
| Size | Grows linearly with context (2·L·n·d) | Fixed: k·d per layer |
| Content | Exact past keys and values | Compressed spectral summary |
| Retrieval | Exact via attention dot product | Approximate via resonance probe |
| Long context | Memory-bound; requires paging, compression, or eviction | Naturally bounded; information competes for modes |
| Precision | Perfect recall of cached tokens | Lossy — early tokens "fade" unless they resonate |

**The fundamental tradeoff**: Modal memory is O(1) in context length but lossy. KV cache is exact but O(n). For tasks requiring precise retrieval of early context (e.g., "what was the 5th word?"), modal memory will fail without Tier 3 external retrieval or sparse attention backup.

### 5.3 Modal Memory Update Rule

```
# Deposit: project input onto modal basis
deposit = W_deposit · ψ_t                    ∈ ℂ^k

# Decay existing modes
M ← diag(γ) · M       where γ ∈ (0,1)^k     (learned decay rates)

# Selective write: only update modes with high resonance
gate = σ(|deposit|² - threshold)
M ← M + gate ⊙ (deposit · ψ_t^H)           (outer product update, gated)

# Read: probe modal memory with query
q = W_query · ψ_t
m_t = M^H · q                               ∈ ℂ^d
```

This is O(k·d) per token — constant regardless of context length.

---

## 6. Hybrid Control Strategy

### 6.1 The Discreteness Problem

Field dynamics are inherently continuous. Language requires:
- Discrete token selection (vocabulary lookup)
- Exact copying (proper nouns, code, quotes)
- Logical reasoning (if-then-else, negation)
- Counting and arithmetic
- Structured output (JSON, code syntax)

Pure coherence dynamics will fail at all of these. The architecture must explicitly introduce discreteness at controlled points.

### 6.2 Hybrid Strategy: Coherence + Sparse Control

```
┌─────────────────────────────────────────────────┐
│              Hybrid Architecture                 │
│                                                  │
│  ┌──────────────┐  ┌──────────────────────────┐ │
│  │  Coherence    │  │   Discrete Controller    │ │
│  │  Backbone     │  │                          │ │
│  │              │  │  • Sparse attention       │ │
│  │  CausalSSM   │──│    (every 4th layer)      │ │
│  │  + spectral   │  │  • Copy mechanism        │ │
│  │  + resonance  │  │  • Symbolic scratchpad   │ │
│  │              │  │  • Router (MoE-style)     │ │
│  └──────────────┘  └──────────────────────────┘ │
│         │                    │                   │
│         └────────┬───────────┘                   │
│                  ▼                                │
│         ┌──────────────┐                         │
│         │   Merge Gate  │                        │
│         │  g·coherence  │                        │
│         │  + (1-g)·ctrl │                        │
│         └──────────────┘                         │
└─────────────────────────────────────────────────┘
```

### 6.3 Where Discreteness Enters

1. **Sparse attention layers** (every 4th layer): Standard causal attention with local window + global tokens. Handles precise retrieval and copying. Uses real-valued QKV projected from complex states: Q = Re(W_q · ψ), K = Re(W_k · ψ), V = Re(W_v · ψ).

2. **Copy mechanism**: A learned gate that can bypass the coherence path entirely and copy input tokens to output. Essential for code generation and quoting.

3. **Resonance routing** (MoE-style): Instead of a learned router, use spectral similarity between token state and expert "signature frequencies":
   ```
   score_e = |⟨FFT(ψ_t), f_e⟩|²     where f_e is expert e's frequency signature
   top-k experts activated
   ```

4. **Output head**: The final layer must produce logits over a discrete vocabulary. Project complex state to real: logits = Re(W_out · ψ_t) + bias. The imaginary component is discarded — or used as a confidence signal.

### 6.4 Tool Use and Program-Like Control

For structured generation (JSON, code, tool calls):
- The coherence backbone provides semantic context and high-level planning.
- A **constrained decoding** layer post-output enforces syntax (grammar-guided sampling).
- This is identical to how transformers handle it — the coherence architecture doesn't change this interface.

---

## 7. Dataflow and Runtime Model

### 7.1 Multi-Plane Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Runtime Planes                         │
│                                                          │
│  ┌──────────────────┐                                    │
│  │ Representation   │  Token IDs → complex embeddings    │
│  │ Plane            │  Embedding tables, tokenizer       │
│  └────────┬─────────┘                                    │
│           ▼                                              │
│  ┌──────────────────┐                                    │
│  │ Field Plane      │  Core coherence computation        │
│  │                  │  FFT/IFFT, spectral filters,       │
│  │                  │  complex SSM recurrence,            │
│  │                  │  resonance gating                   │
│  └────────┬─────────┘                                    │
│           ▼                                              │
│  ┌──────────────────┐                                    │
│  │ Memory Plane     │  Modal memory bank (Tier 1)        │
│  │                  │  Selective persistent (Tier 2)      │
│  │                  │  SSM state (Tier 0)                 │
│  └────────┬─────────┘                                    │
│           ▼                                              │
│  ┌──────────────────┐                                    │
│  │ Control Plane    │  Sparse attention layers            │
│  │                  │  MoE routing, copy mechanism        │
│  │                  │  Constrained decoding               │
│  └────────┬─────────┘                                    │
│           ▼                                              │
│  ┌──────────────────┐                                    │
│  │ Execution Plane  │  Scheduling, memory allocation,    │
│  │                  │  kernel dispatch, backend selection │
│  └──────────────────┘                                    │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Data Flow Between Planes

```
Representation → Field:    Complex embedding vectors (ℂ^d per token)
Field → Memory:            Deposit vectors for modal update (ℂ^k per token)
Memory → Field:            Retrieved modal context (ℂ^d per token)
Field → Control:           Real projections for attention/routing (ℝ^d per token)
Control → Field:           Gated control signals merged back into complex state
Field → Representation:    Real projection for output logits (ℝ^V per token)
```

### 7.3 Backend-Agnostic Implementation

The core operations decompose into a small set of kernels:

| Kernel | Used in | CPU | CUDA | Metal | WebGPU |
|---|---|---|---|---|---|
| Complex FFT/IFFT | Field plane | FFTW / rustfft | cuFFT | vDSP / Accelerate | Custom compute shader |
| Complex pointwise multiply | Spectral filter | SIMD (AVX/NEON) | Elementwise kernel | SIMD | Compute shader |
| Complex diagonal recurrence | Causal SSM | Sequential loop | Parallel scan (Blelloch) | Sequential / threadgroup | Sequential |
| Real MatMul | Control plane (attention, MLP) | BLAS | cuBLAS | MPSMatMul | Custom |
| Phase normalization | Every block | Vectorized | Fused kernel | Vectorized | Compute shader |
| Modal memory update | Memory plane | BLAS (rank-1 update) | Custom kernel | BLAS | Compute shader |

**Key implementation note**: The complex diagonal SSM's parallel scan is the most hardware-sensitive kernel. On GPU, Blelloch-style parallel prefix scan achieves O(n/p) with p processors. On CPU, sequential is fine. On WebGPU, workgroup-level scan with shared memory.

### 7.4 Mapping to hologram's Execution Model

The coherence runtime maps naturally to hologram's graph IR:

- **Field plane ops** → `GraphOp::Float(FloatOp::...)` for existing ops, `GraphOp::Custom` for novel kernels
- **Memory plane** → Stateful execution nodes (hologram already supports stateful KV ops: `KvWrite`, `KvRead`, `KvUpdate`, `KvSlice`)
- **Scheduling** → hologram's layer-based execution plan with dependency tracking
- **Weights** → `.holo` archive sections for spectral filters, embedding tables, modal memory init

See Section 12 for the detailed op mapping.

---

## 8. Training Strategy

### 8.1 Staged Training Pipeline

**Stage 1: Substrate Pretraining (small scale, synthetic)**
- Task: Next-token prediction on synthetic sequences with known spectral structure (e.g., superpositions of sine waves, Markov chains with long-range dependencies).
- Purpose: Verify that the coherence block can learn spectral filters that capture long-range correlations.
- Scale: d=128, n=1024, ~10M parameters. Single GPU.
- Success criterion: Loss competitive with a same-size transformer on same tasks.
- Duration: Days, not weeks.

**Stage 2: Language Pretraining (medium scale)**
- Task: Causal language modeling on standard corpora (FineWeb, RedPajama).
- Architecture: Hybrid (4:1 coherence-to-attention).
- Scale: d=768, L=24, ~350M parameters. 8–64 GPUs.
- Key risk: Training instability from complex-valued gradients.
- Mitigation: See §8.2.
- Success criterion: Perplexity within 10% of same-size transformer on held-out set.
- Duration: Weeks.

**Stage 3: Scaling Validation (if Stage 2 succeeds)**
- Scale to 1B–7B parameters.
- Compare directly with Mamba-2, Llama-class transformers.
- Measure: perplexity, downstream task accuracy, throughput (tokens/sec/GPU).

**Stage 4: Hybrid Fine-Tuning**
- Instruction tuning, RLHF, multimodal alignment.
- Same techniques as transformers — the output interface is identical.

### 8.2 Training Stability Mitigations

| Problem | Mitigation |
|---|---|
| Phase collapse (all phases converge) | Auxiliary loss: maximize phase entropy across dimensions |
| Gradient explosion in angular domain | Parameterize A in log-space: A = exp(log_A), constrain Re(log_A) < 0 |
| Complex Adam instability | Use separate learning rates for amplitude and phase components |
| Spectral filter divergence | Initialize H near identity (passthrough); clip gradient norm |
| Energy drift across layers | Energy conservation loss: L_energy = (||Ψ_out||² - E₀)² |
| Mode collapse in modal memory | Diversity regularizer: maximize rank of modal memory M |

### 8.3 Auxiliary Objectives

1. **Phase consistency**: ΔL_phase = -H(arg(ψ)) — maximize entropy of phase distribution to prevent collapse.
2. **Modal sparsity**: ΔL_sparse = ||M||₁ — encourage sparse modal memory usage so modes specialize.
3. **Spectral smoothness**: ΔL_smooth = ||∇_k H(k)||² — prevent sharp spectral discontinuities that cause ringing artifacts.
4. **Energy conservation**: ΔL_energy = Σ_l (||Ψ_l||² - E₀)² — keep field energy bounded across layers.

### 8.4 Likely Optimization Problems

- **Slow convergence**: Complex-valued networks typically converge 2–5x slower than real-valued equivalents. Budget for longer training.
- **Sensitivity to initialization**: Phase initialization matters enormously. Random uniform phase is likely better than zero-init, but this needs empirical validation.
- **Mixed-precision issues**: f16 has limited phase precision (~3 decimal digits for angles near π). May need f32 for phase components, f16 for amplitudes — increasing memory pressure.
- **Gradient of FFT**: Well-defined (FFT is linear, gradient is IFFT of upstream gradient), but numerical precision of gradient accumulation through many FFT layers is untested at scale.

---

## 9. Evaluation Plan

### 9.1 Synthetic Benchmarks (Stage 1)

| Task | Tests | Expected coherence advantage |
|---|---|---|
| **Long-range retrieval**: "What was token X at position P?" | Exact memory over 1K–100K context | Modal memory vs. KV cache scaling |
| **Spectral pattern completion**: Predict next value in superposition of waves | Long-range periodic structure | Direct — spectral filters match task structure |
| **Selective copying**: Copy specific subsequences to output | Exact discrete copying | Likely weak — tests hybrid control necessity |
| **Compositional nesting**: Evaluate nested arithmetic expressions | Hierarchical structure | Unclear — tests resonance-as-composition hypothesis |
| **Multi-scale association**: Associate pairs across varying distances | Interference-based binding | Potential advantage if interference patterns encode associations |

### 9.2 Realistic Language Benchmarks (Stage 2+)

| Benchmark | Purpose |
|---|---|
| Perplexity (validation set) | Core language modeling quality |
| MMLU | Broad knowledge and reasoning |
| HellaSwag | Commonsense reasoning |
| RULER / NIAH (needle-in-a-haystack) | Long-context retrieval accuracy |
| GSM8K | Mathematical reasoning (tests discrete control) |
| HumanEval / MBPP | Code generation (tests exact copying) |
| Throughput (tokens/sec/GPU) | Practical efficiency |

### 9.3 Ablation Studies

| Ablation | Purpose |
|---|---|
| Remove spectral filter → identity | Is learned spectral mixing doing anything? |
| Remove resonance gating → always pass through | Is selective gating necessary? |
| Remove modal memory → SSM state only | Does multi-tier memory help? |
| Remove sparse attention layers → pure coherence | How much does discrete control contribute? |
| Real-valued states → complex-valued states | Does phase carry useful information? |
| DFT path → learned dense mixing matrix | Is FFT better than dense mixing? |

### 9.4 Interpretability Analysis

- **Spectral filter visualization**: Plot learned H(k) per layer — do they develop interpretable bandpass/lowpass structure?
- **Modal memory inspection**: Do modes specialize (e.g., one mode for syntax, another for entities)?
- **Phase coherence maps**: Visualize phase alignment between token positions — do semantically related tokens align in phase?
- **Resonance score distributions**: Do high-resonance tokens correspond to salient/important positions?

---

## 10. Failure Modes and Criticism

### 10.1 Training Instability (HIGH RISK)

Complex-valued networks at scale are essentially uncharted territory. Every claim about training stability in this document is based on small-scale experiments (d < 256) from academic papers. Scaling to d=4096+ may reveal entirely new failure modes. The auxiliary losses in §8.3 add hyperparameters that interact in unknown ways.

**Verdict**: This is the most likely point of failure. If a 350M coherence model cannot train stably within 2x the compute of a same-size transformer, the approach is likely not viable without fundamental advances in complex-valued optimization.

### 10.2 Lack of Discrete Reasoning (HIGH RISK)

The hybrid control strategy (§6) is an admission that pure coherence dynamics cannot handle discrete tasks. But the hybrid introduces the same attention mechanisms we're trying to avoid. If the model ends up relying heavily on the sparse attention layers for most useful work, the coherence backbone becomes expensive decoration.

**Test**: Ablation study — if removing the coherence backbone and keeping only the attention layers yields similar performance, the coherence contribution is nil.

### 10.3 Interpretability Limitations (MEDIUM RISK)

Phase and spectral structure are interpretable to signal processing experts, not to ML practitioners. The interpretability story may be *different* from transformers but not necessarily *better*. Attention maps are already poorly understood; interference patterns may be worse.

### 10.4 Hardware Inefficiency (HIGH RISK)

Modern GPUs achieve peak FLOPS on large matmuls (GEMM). The coherence architecture replaces these with FFTs and elementwise operations, which are memory-bandwidth-bound rather than compute-bound. Estimated utilization:

- Transformer on A100: ~50–60% of peak FLOPS (GEMM-dominated)
- Coherence model on A100: ~15–25% of peak FLOPS (FFT + elementwise dominated)

This means a coherence model may need 2–4x more hardware to match transformer throughput, even if the theoretical FLOP count is lower. **This alone could kill the approach for practical deployment.**

Mitigation: Reformulate spectral mixing as real matmul where possible (the SpectralMLP can use dense matrices instead of FFT→filter→IFFT). This trades O(n log n) complexity for better hardware utilization — but then we're just building a transformer with extra steps.

### 10.5 Inability to Match Transformer Scaling (MEDIUM RISK)

Transformers exhibit smooth, predictable scaling laws (Chinchilla). There is no evidence that coherence-native architectures follow similar scaling laws. They might scale worse (diminishing returns from more spectral filters) or better (more efficient information routing) — we simply don't know.

### 10.6 Unclear Benefit Over Existing SSMs (HIGH RISK)

Mamba-2 already provides:
- O(n) complexity (better than our O(n log n) spectral path)
- Selective state-space recurrence
- Competitive with transformers on most benchmarks

The coherence framing adds:
- Complex-valued states (Mamba uses real)
- Spectral mixing (Mamba uses input-dependent selection)
- Resonance gating (novel but unproven)
- Modal memory (novel but unproven)

If the complex-valued aspect and spectral mixing don't provide measurable improvements over Mamba's simpler real-valued selection, the entire coherence framing is unnecessary complexity.

**This is the most important question to answer in Stage 1 prototyping**: Does complex-valued state + spectral mixing outperform real-valued state + selective gating on long-range tasks?

---

## 11. Research Roadmap

### Prototype A: Complex-Valued SSM Baseline

**Scope**: Minimal viable experiment. Take Mamba's architecture, replace real-valued states with complex-valued states. Add phase normalization. No spectral mixing, no modal memory, no resonance gating.

**Purpose**: Isolate the question "does complex-valued state help?"

**Scale**: d=128–256, L=8–12, ~10M–50M params. Synthetic + small language tasks.

**Success metrics**:
- Matches or beats real-valued Mamba on long-range retrieval (RULER-style tasks)
- Training stability: converges without NaN within 1.5x the steps of real-valued baseline
- Phase entropy stays distributed (no collapse)

**Discard if**: Complex-valued variant trains unstably or shows no improvement after hyperparameter sweep.

**Estimated effort**: 2–4 weeks, 1–2 people, single node.

### Prototype B: Spectral Coherence Block

**Scope**: Add the spectral mixing path (DFT→filter→IDFT) and resonance gating on top of Prototype A. This tests the core coherence hypothesis.

**Purpose**: Does spectral mixing provide something that the SSM recurrence alone doesn't?

**Scale**: d=256–512, L=12–24, ~100M–350M params. Language modeling on FineWeb subset.

**Success metrics**:
- Perplexity within 10% of same-FLOP transformer
- Clear ablation showing spectral mixing contributes (removing it hurts perplexity)
- Resonance scores correlate with token salience (interpretability signal)

**Discard if**: Perplexity gap to transformer exceeds 15%, or spectral mixing ablation shows no contribution.

**Estimated effort**: 4–8 weeks, 2–3 people, 8–64 GPUs.

### Prototype C: Multi-Tier Memory

**Scope**: Add modal memory (Tier 1) and selective persistent memory (Tier 2) from §5. Test on long-context tasks.

**Purpose**: Does the multi-tier memory replace KV cache effectively for long contexts?

**Scale**: Same as B, but with context lengths 4K–32K.

**Success metrics**:
- Maintains performance as context grows (no degradation beyond 4K)
- Fixed memory footprint (no linear growth with context)
- Acceptable precision loss on needle-in-a-haystack vs. KV-cache baseline

**Discard if**: Precision on retrieval tasks degrades sharply beyond 2K tokens.

**Estimated effort**: 4–6 weeks, 2–3 people.

### Benchmark Phase

**Scope**: Full hybrid architecture (4:1 coherence-to-attention) at 350M–1B scale. Head-to-head comparison with Mamba-2 and Llama-class transformer.

**Success criteria for continuing**:
- Performance (perplexity, downstream accuracy) within 5% of best baseline
- Throughput (tokens/sec) competitive or better
- Memory usage demonstrably lower at long context lengths
- At least one benchmark where coherence model clearly wins

### Systems Phase

**Scope**: Optimize kernels for target hardware. Implement the coherence execution path in hologram.

**Deliverables**:
- Custom CUDA/Metal kernels for complex FFT, parallel scan, modal memory
- hologram FloatOp additions (ComplexMul, FFT, PhaseNorm, ModalUpdate)
- hologram-ai importer for coherence model format
- .holo archive support for complex-valued weights and modal memory state

### Production-Readiness Criteria

All of the following must be true:
- [ ] Performance matches transformer baseline within 5% on standard benchmarks
- [ ] Throughput (tokens/sec) competitive on target hardware
- [ ] Training recipe is reproducible and documented
- [ ] Long-context performance demonstrably superior to KV-cache baselines
- [ ] At least one unique capability not achievable by transformers (e.g., truly fixed-memory long context)
- [ ] Interpretability story is coherent (pun intended) and publishable
- [ ] hologram execution path validated end-to-end

---

## 12. Hologram Ecosystem Alignment

### 12.1 Operator Mapping: Coherence Ops → hologram FloatOps

| Coherence operation | Existing hologram FloatOp? | Notes |
|---|---|---|
| Complex embedding lookup | `FloatOp::Embed` (real part) | Need two lookups (amplitude + phase) or one 2x-wide |
| Phase rotation (position encoding) | `FloatOp::Mul` + `FloatOp::Sin`/`FloatOp::Cos` | Euler formula: e^(iωt) = cos(ωt) + i·sin(ωt). Apply as two real ops |
| FFT / IFFT | **MISSING** | hologram has no FFT op. Needs `FloatOp::FFT { size, inverse: bool }` |
| Complex pointwise multiply | `FloatOp::Mul`, `FloatOp::Sub`, `FloatOp::Add` | (a+bi)(c+di) = (ac-bd) + (ad+bc)i — 4 real muls, 1 add, 1 sub |
| Complex diagonal SSM recurrence | **MISSING** (as fused op) | Can decompose into complex multiply + add, but fused kernel needed for speed |
| Phase normalization | `FloatOp::Sqrt`, `FloatOp::Add`, `FloatOp::Mul`, `FloatOp::Reciprocal` | norm = sqrt(re² + im²); normalize = (re,im) / norm |
| Resonance gate | `FloatOp::Sigmoid`, `FloatOp::Mul` | Standard gating on real-valued resonance score |
| Spectral filter (H ⊙ Φ) | Complex pointwise multiply (see above) | After FFT, this is just elementwise |
| Resonance routing (MoE) | `FloatOp::MatMul` + top-k | Top-k selection exists as `FloatOp::TopK` |
| Modal memory deposit | `FloatOp::MatMul` (rank-1 outer product) | Decompose: gate ⊙ (deposit · ψ^H) |
| Modal memory read | `FloatOp::MatMul` (M^H · q) | Standard matmul on complex (use 2x-wide real) |
| Modal memory decay | `FloatOp::Mul` (elementwise by γ) | Exists |
| Sparse attention (hybrid layers) | `FloatOp::Attention` | Exists with GQA support |
| Energy normalization | `FloatOp::ReduceSum` + `FloatOp::Sqrt` + `FloatOp::Mul` | Compose from existing ops |
| Output projection (ℂ → ℝ) | `FloatOp::Slice` or `FloatOp::Gather` | Extract real component from interleaved complex |
| SiLU gate (SpectralMLP) | `FloatOp::Silu` | Exists |
| RMSNorm (hybrid layers) | `FloatOp::RmsNorm` | Exists |

### 12.2 Kernel Gaps — New FloatOps Needed

These are **general-purpose** additions that benefit the ecosystem beyond coherence models:

1. **`FloatOp::FFT { size: u32, inverse: bool }`**
   - 1D complex FFT along the last dimension
   - Input: interleaved complex (re, im, re, im, ...) as f32 buffer
   - Useful for: signal processing, spectral analysis, any Fourier-domain model
   - Backend: rustfft (CPU), cuFFT (CUDA), vDSP (Metal), custom (WebGPU)

2. **`FloatOp::ComplexMul`**
   - Fused complex multiply on interleaved buffers
   - (a+bi)(c+di) in one kernel instead of 6 real ops
   - Useful for: any complex-valued computation

3. **`FloatOp::ComplexDiagRecurrence { size: u32 }`**
   - Fused complex diagonal SSM scan: h_t = A ⊙ h_{t-1} + B ⊙ x_t
   - Critical for causal coherence blocks (hot inner loop)
   - Backend: parallel prefix scan (GPU), sequential (CPU)

4. **`FloatOp::PhaseNorm { size: u32, epsilon: u32 }`**
   - Normalize complex vector to unit amplitude, preserving phase
   - Fused: compute norm, divide both components
   - Analogous to `RmsNorm` but for complex data

### 12.3 Archive Format Extensions

New `.holo` archive sections for coherence models:

| Section | Content |
|---|---|
| `SECTION_COHERENCE_META` | Architecture config: d, k (modal rank), num_layers, hybrid ratio, decay rates |
| `SECTION_MODAL_MEMORY_INIT` | Initial modal memory state M₀ (from training) |
| `SECTION_SPECTRAL_FILTERS` | Learned H(k) per layer (compact: 2·d values per layer) |

These use hologram's existing `EmbeddableSection` trait — no core changes needed.

### 12.4 Compilation Path

```
hologram-coherence (training)
    │
    │  exports: CoherenceModelFormat (.cmf or ONNX-extended)
    │  contains: spectral filters, complex embeddings, SSM parameters, modal memory init
    │
    ▼
hologram-ai-coherence (importer crate)
    │
    │  parses export format
    │  maps coherence ops → AiOp variants
    │  lowers via hologram-ai-common strategy.rs
    │
    ▼
hologram-ai (compiler)
    │
    │  AiGraph → optimization passes → hologram::Graph
    │  complex ops → interleaved real representation
    │  FFT/ComplexMul/PhaseNorm → FloatOp or Custom
    │
    ▼
hologram (execution)
    │
    │  .holo archive with coherence sections
    │  FloatOp dispatch (existing + new complex ops)
    │  KvExecutor handles modal memory as stateful nodes
    │
    ▼
inference
```

### 12.5 Dependency Architecture

```
hologram-coherence ──depends on──► hologram          (graph IR, execution)
                   ──depends on──► hologram-ai-common (AiOp, DimExpr)
                   ──depends on──► PyTorch/tch-rs     (training, autograd)

hologram-ai ◄── hologram-ai-coherence (thin importer, new crate)
hologram    ◄── gains FloatOp::FFT, ComplexMul, ComplexDiagRecurrence, PhaseNorm
                (general-purpose additions, not coherence-specific)
```

**hologram and hologram-ai do NOT depend on hologram-coherence.** The new crate sits above the stack, depending downward. hologram gains general-purpose ops. hologram-ai gains a thin importer.

---

## Appendix: Key References

- Trabelsi et al., "Deep Complex Networks" (ICLR 2018) — complex-valued neural network foundations
- Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces (S4)" (ICLR 2022)
- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- Dao & Gu, "Transformers are SSMs" (Mamba-2, 2024)
- Lee-Thorp et al., "FNet: Mixing Tokens with Fourier Transforms" (2022)
- Rao et al., "Global Filter Networks for Image Classification with Fourier Transform" (GFNet, NeurIPS 2021)
- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022) — energy-based model framing
- Li et al., "Fourier Neural Operator for Parametric PDEs" (FNO, ICLR 2021)

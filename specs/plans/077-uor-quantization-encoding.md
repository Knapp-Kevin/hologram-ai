# Plan 077: UOR Encoding for Quantization

## Context

Weight quantization in hologram-ai is currently a compiler graph pass that rewrites `Float(MatMul)` nodes into `MatMulLut4/8/2` with pre-serialized quantized bytes. This creates 9+ scattered hooks across lowering code, separate streaming/in-memory paths, and fragile mmap offset tracking. The streaming path is broken on `feat/post-lowering-quantization` (TinyLlama falls to f32).

The fundamental insight: hologram's data model is built on UOR's ring structure (Z/2^nZ at quantum levels Q0-Q15). Quantization is just a precision projection — moving from Q3 (32-bit, f32 reinterpreted) to Q0 (8-bit) or sub-Q0 (4-bit) with scale fibers. Weight lookup should be content-addressed (BLAKE3 digest already exists on every tensor), and encoding metadata should travel with the data rather than being a graph-level concern.

**Goals:**
1. Unify streaming/mmap/in-memory behind content-addressed weight resolution
2. Make quantization a self-describing encoding on weight data, not a graph rewrite
3. Remove 9+ scattered quantization hooks from builder.rs
4. Fix the streaming quantization bug (TinyLlama)
5. Connect to UOR's geometric addressing model

**Non-goals:**
- Full UOR ring-native per-element encoding (Phase 4, future)
- Changing runtime kernels (MatMulLut4/8 dispatch stays the same)
- Breaking existing archives (backward compatible)

**Coordination with Plan 067 (ComputeBackend+ComputeMemory):**
The new backend rewrite changes how weights are loaded — all data lives on the target device, no CPU<->GPU transfers. UOR encoding must design the resolution API so that:
- `EncodingDescriptor` travels with the weight data into the backend's memory model
- The backend's `ComputeMemory` can decode encoded weights on-device (GPU decode of Q4 blocks)
- Content-addressed resolution returns a device-native handle, not necessarily `&[u8]`
- The `resolve()` API should be trait-based so backends can override (e.g., Metal shader for Q4 dequant directly from mmap'd encoded data)

This means Phase 1's `resolve_constant_bytes` returns `&[u8]` as a stopgap, but the trait design should anticipate `resolve_to_device(address, device) -> DeviceBuffer`.

---

## Phase 1: TensorEncoding Type + Content-Addressed Constants

**Branch:** `feat/uor-quantization` (from `feat/post-lowering-quantization`)

### 1.1 Add `TensorEncoding` to hologram-archive

**File:** `hologram/crates/hologram-archive/src/weight/encoding.rs` (new)

```rust
#[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum TensorEncoding {
    /// Raw IEEE 754 at native dtype. No transformation.
    Identity,
    /// Block-quantized with per-block scale fibers.
    /// UOR interpretation: Q3 -> Q0 projection with scale fiber per group.
    BlockQuantized {
        bits: u8,           // 2, 4, or 8
        block_size: u32,    // elements per block (32 or 256)
        variant: BlockVariant,
    },
    /// K-means clustered (LUT-GEMM): values mapped to centroid indices.
    /// UOR interpretation: Q3 -> sub-Q0 with centroid fiber.
    Clustered {
        bits: u8,           // 2, 4, or 8
        num_centroids: u16, // 4, 16, or 256
        rows: u32,
        cols: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum BlockVariant { Q4_0, Q8_0, Q2K, Q4K, Q6K }
```

### 1.2 Add `encoding` field to `TensorMetadata`

**File:** `hologram/crates/hologram-archive/src/weight/mod.rs`

```rust
pub struct TensorMetadata {
    // ... existing fields preserved ...
    pub encoding: Option<TensorEncoding>,  // NEW — None for legacy archives
}
```

### 1.3 Add `ConstantData::ContentAddressed` variant

**File:** `hologram/crates/hologram-graph/src/constant/mod.rs`

```rust
pub enum ConstantData {
    Bytes(Vec<u8>),
    Deferred { byte_size: u64, source_id: u64 },
    /// Content-addressed: resolved via BLAKE3 digest + ContentAddressIndex.
    ContentAddressed {
        byte_size: u64,
        digest: [u8; 32],
        encoding: TensorEncoding,
    },
}
```

### 1.4 Add `ContentAddressIndex` section

**File:** `hologram/crates/hologram-archive/src/weight/content_addr.rs` (new)

```rust
pub struct ContentAddressEntry {
    pub digest: [u8; 32],
    pub offset: u64,
    pub size: u64,
}

pub struct ContentAddressIndex {
    pub entries: Vec<ContentAddressEntry>,  // sorted by digest for binary search
}

impl ContentAddressIndex {
    pub fn resolve(&self, digest: &[u8; 32]) -> Option<&ContentAddressEntry> {
        self.entries.binary_search_by_key(digest, |e| e.digest)
            .ok().map(|i| &self.entries[i])
    }
}
```

### 1.5 Update resolution in hologram-exec

**File:** `hologram/crates/hologram-exec/src/kv/weight_cache.rs` (or equivalent)

Add third arm to constant resolution:
```rust
ConstantData::ContentAddressed { byte_size, digest, .. } => {
    let entry = content_index.resolve(digest)?;
    &weights[entry.offset as usize..(entry.offset + byte_size) as usize]
}
```

Design the resolution as a trait to anticipate Plan 067's `ComputeBackend`:
```rust
/// Trait for resolving weight data — backends can override for device-native loading.
pub trait WeightResolver {
    /// Resolve encoded weight bytes. Default: slice from mmap'd blob.
    fn resolve(&self, digest: &[u8; 32], byte_size: u64) -> Result<&[u8]>;
}
```
Phase 1 impl: `MmapWeightResolver` wrapping `ContentAddressIndex` + `&[u8]` blob.
Plan 067 adds: `DeviceWeightResolver` that loads directly to GPU memory.

### 1.6 Archive format: version bump + flag

**File:** `hologram/crates/hologram-archive/src/format/header.rs`
- Add `FLAG_CONTENT_ADDRESSED: u32 = 1 << 5`
- Version 2 -> 3 (accept both)
- `LoadedPlan` gains `content_index: Option<ContentAddressIndex>`

---

## Phase 2: Encoding-Aware Compiler (hologram-ai)

### 2.1 `QuantAnnotation` side table

**File:** `hologram-ai-common/src/lower/quant_annotation.rs` (new)

```rust
pub struct QuantAnnotation {
    pub encoding: TensorEncoding,
    pub pre_encoded: bool,  // true if data already quantized (GGUF pre-quant)
}

pub type QuantTable = HashMap<ConstantId, QuantAnnotation>;
```

### 2.2 `annotate_quant` pass

**File:** `hologram-ai-common/src/lower/annotate_quant.rs` (new)

Walks the lowered Graph, finds `Float(MatMul)`/`Float(Gemm)`/`Float(Conv2d)` nodes, decides encoding based on `QuantStrategy` + model size + error thresholds. Populates `QuantTable`. Logic extracted from existing `quantize_graph.rs` decision code.

### 2.3 `resolve_encodings` pass (replaces `quantize_graph.rs`)

**File:** `hologram-ai-common/src/lower/resolve_encodings.rs` (new)

For each annotated weight:
1. Read raw bytes (from `Bytes` OR `Deferred` via mmap — unified path)
2. Encode using `encode_weight(f32_data, &annotation.encoding) -> Vec<u8>`
3. Compute BLAKE3 digest of encoded bytes
4. Register as new constant with `ConstantData::ContentAddressed`
5. Replace `Float(MatMul)` -> `MatMulLut4`/`MatMulLut8`/`MatMulLut2` based on encoding bits
6. For `Float(Conv2d)` -> same flow: reshape 4D->2D, encode, replace with `Conv2dLut4`/etc.

**Covers:** MatMul, Gemm, AND Conv2d (needed for SD UNet). Single code path for all weight-bearing ops.

**Key fix for streaming bug:** Resolution uses `ContentAddressed` with digest-based lookup instead of raw byte offsets, eliminating the offset math bug.

### 2.4 Remove old hooks from builder.rs

Delete/disable in `hologram-ai-common/src/lower/builder.rs`:
- `do_early_quant` block (lines ~148-239)
- `q4_eligible` set tracking
- `early_quant_bytes` cache
- Deferred Q4 interception (lines ~410-442)
- `quantize_weight_on_demand` (line ~2189)
- `quantize_weight_q8_on_demand` (line ~2138)
- `try_convert_f32_to_lut4/8/2` functions

### 2.5 Archive writer integration

In `compiler.rs`, after `resolve_encodings`:
- Build `ContentAddressIndex` in memory during write (accumulate `(digest, offset, size)` as each weight is written — ~64 bytes per tensor, negligible memory)
- Compute BLAKE3 incrementally as each encoded weight streams to the blob
- Write `ContentAddressIndex` as the **final section** (after all weights are flushed)
- Set `FLAG_CONTENT_ADDRESSED` in header
- Populate `TensorMetadata.encoding` for each weight tensor
- No second pass, no seeking — append-only streaming write pattern

---

## Phase 3: UOR Ring Connection (future)

### 3.1 `TensorEncoding::Ring` variant

```rust
TensorEncoding::Ring {
    quantum_index: u8,    // Q0=0, Q1=1, Q3=3, etc.
    encoding_name: &str,  // "signed", "unsigned", "angle"
}
```

Maps to hologram-ring's `Encoding<W>` trait for per-element encode/decode.

### 3.2 Address<Q> per-block content addressing

Each quantized block (32 elements) gets a UOR `Address<Q>` (Braille content-address). Weight lookup becomes: "resolve Address<Q0> for block at coordinate (layer, head, row_group)".

### 3.3 Fiber model for precision

- f32 weight = Q3 datum with all 32 fibers free
- Q8 encoding = pin 24 fibers (keep 8 free) + scale fiber per block
- Q4 encoding = pin 28 fibers (keep 4 free) + centroid fiber
- Q2 encoding = pin 30 fibers (keep 2 free) + centroid fiber

Residual entropy: S = free_fibers * ln(2). Q4 has S = 4*ln(2) ~ 2.77 bits per element.

---

## Critical Files

| Component | Path |
|-----------|------|
| TensorEncoding type | `hologram/crates/hologram-archive/src/weight/encoding.rs` (new) |
| TensorMetadata | `hologram/crates/hologram-archive/src/weight/mod.rs` |
| ContentAddressIndex | `hologram/crates/hologram-archive/src/weight/content_addr.rs` (new) |
| ConstantData enum | `hologram/crates/hologram-graph/src/constant/mod.rs` |
| Weight resolution | `hologram/crates/hologram-exec/src/kv/weight_cache.rs` |
| HoloHeader | `hologram/crates/hologram-archive/src/format/header.rs` |
| Current quant pass | `hologram-ai/crates/hologram-ai-common/src/lower/quantize_graph.rs` |
| Builder hooks | `hologram-ai/crates/hologram-ai-common/src/lower/builder.rs` |
| Compiler entry | `hologram-ai/src/compiler.rs` |
| hologram-ring | `hologram/crates/hologram-ring/src/` (Datum, Address, QuantumLevel) |
| UOR Foundation | `hologram/UOR-Framework/foundation/src/kernel/` |

---

## Verification

1. **TinyLlama streaming Q4**: Must achieve 30+ tok/s (currently falls to f32 = 2.5 tok/s)
2. **Qwen2 in-memory Q8**: Must maintain 10.5 tok/s (no regression)
3. **SD UNet Conv2d Q4**: Conv2d weights encoded and load correctly (verify via inspector)
4. **Archive backward compat**: v2 archives load on new runtime without errors
5. **Content dedup**: Two identical weight tensors share one blob entry (verify via ContentAddressIndex)
6. **Round-trip**: `encode(f32_data, Q4) |> decode == dequant_q4(quantize_q4(f32_data))`
7. **WeightResolver trait**: Confirm trait can be implemented by both MmapWeightResolver and a mock DeviceWeightResolver (unit test)

```bash
# Test commands
cargo test -p hologram-ai -- --ignored tinyllama_streaming_q4
cargo test -p hologram-ai -- --ignored qwen2_q8_decode
cargo test -p hologram-ai -- --ignored sd_unet_conv2d_q4
cargo test -p hologram-archive -- content_address
cargo test -p hologram-exec -- weight_resolver
```

---

## Execution Order

1. Create branch `feat/uor-quantization` from `feat/post-lowering-quantization`
2. Phase 1.1-1.2: Add types to hologram base (hologram repo)
3. Phase 1.3-1.6: Add ContentAddressed variant + resolution
4. Phase 2.1-2.3: Build annotation + resolution passes in hologram-ai
5. Phase 2.4: Remove old hooks (only after 2.3 passes tests)
6. Phase 2.5: Wire archive writer
7. Verify TinyLlama + Qwen2 + SD UNet Conv2d
8. Phase 3: Future work (separate plan)

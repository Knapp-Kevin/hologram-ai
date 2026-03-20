# Plan 016: Paged Attention for KV Cache

## Context

KV cache currently pre-allocates contiguous flat buffers sized to `max_seq_len` upfront
in `KvCacheState::new(n_layers, n_kv_heads, head_dim, max_seq)`. This wastes memory
when actual generation is much shorter than the maximum, and requires knowing `max_seq_len`
at compile time.

For TinyLlama (22 layers, 4 KV heads, 64 head_dim, 2048 max_seq):
`2 * 22 * 4 * 2048 * 64 * 4 bytes ≈ 90 MB` — even for a 5-token generation.

For production-scale models (70B params, 128K context), the cache alone can exceed
available memory. Additionally, there is no dynamic growth — unused capacity is dead memory.

**Paged attention** (borrowed from OS virtual memory) replaces monolithic per-layer buffers
with small fixed-size pages allocated on demand. A block table maps logical positions to
physical pages, enabling:

- **Dynamic growth**: allocate pages only as the sequence extends
- **No upfront max_seq**: construction needs only page_size, not max_seq_len
- **Foundation for batching**: per-request block tables sharing a global page pool (future)

**Design principle**: Paging is purely a runtime concern. `AiOp` and `FloatOp` are unchanged —
the attention kernel still sees flat contiguous tensors. Paging is internal to `KvCacheState`.

---

## Step 1: PagedKvCache data structure

**Scope:** hologram base crate
**File:** `hologram-exec/src/kv_cache.rs`

```rust
/// A single physical page holding `page_size` token slots for one head-group.
/// Layout: [page_size, n_kv_heads, head_dim] stored as flat f32 vec.
struct Page {
    data: Vec<f32>,  // len = page_size * n_kv_heads * head_dim
}

/// Monotonic physical page identifier.
type PageId = u32;

/// Arena of physical pages with a free list for recycling.
struct PagePool {
    pages: Vec<Page>,           // indexed by PageId
    free_list: Vec<PageId>,     // recycled pages available for reuse
    page_size: usize,
    slot_stride: usize,         // n_kv_heads * head_dim (floats per token-slot)
}

/// Per-layer mapping: logical block index → physical PageId.
/// block_table[i] holds the PageId for tokens [i*page_size .. (i+1)*page_size).
struct BlockTable {
    k_blocks: Vec<PageId>,
    v_blocks: Vec<PageId>,
}

/// Paged KV cache replacing flat-buffer KvCacheState.
pub struct PagedKvCache {
    pool: PagePool,
    layers: Vec<BlockTable>,    // one per transformer layer
    n_kv_heads: u32,
    head_dim: u32,
    page_size: u32,
    write_pos: usize,           // global token position (same semantic as KvCacheState)
}
```

---

## Step 2: Page allocator

**Scope:** hologram base crate
**File:** `hologram-exec/src/kv_cache.rs`

```rust
impl PagePool {
    /// Allocate a page — recycle from free list or grow the arena.
    fn alloc(&mut self) -> PageId {
        if let Some(id) = self.free_list.pop() {
            // Zero the recycled page
            self.pages[id as usize].data.fill(0.0);
            id
        } else {
            let id = self.pages.len() as PageId;
            self.pages.push(Page {
                data: vec![0.0; self.page_size * self.slot_stride],
            });
            id
        }
    }

    /// Return pages to free list (for reset or truncation).
    fn free(&mut self, page_id: PageId) {
        self.free_list.push(page_id);
    }
}

impl PagedKvCache {
    pub fn new(n_layers: u32, n_kv_heads: u32, head_dim: u32, page_size: u32) -> Self;

    /// Called on KvWrite — ensures a page exists for the current write position.
    /// Allocates a new page when crossing a page boundary.
    fn ensure_page(&mut self, layer: u32, is_key: bool) {
        let block_idx = self.write_pos / self.page_size as usize;
        let table = &mut self.layers[layer as usize];
        let blocks = if is_key { &mut table.k_blocks } else { &mut table.v_blocks };
        while blocks.len() <= block_idx {
            blocks.push(self.pool.alloc());
        }
    }

    /// Reset cache for a new sequence — recycle all pages, reset write_pos.
    pub fn reset(&mut self) {
        for table in &mut self.layers {
            for &id in table.k_blocks.iter().chain(table.v_blocks.iter()) {
                self.pool.free(id);
            }
            table.k_blocks.clear();
            table.v_blocks.clear();
        }
        self.write_pos = 0;
    }
}
```

**Allocation trigger:** When `write_pos % page_size == 0` and we're about to write,
`ensure_page()` allocates for all layers' K and V. During prefill (writing N tokens),
this allocates `ceil(N / page_size)` pages per layer per K/V = `2 * n_layers * ceil(N / page_size)` pages total.

---

## Step 3: KvWrite / KvRead dispatch changes

**Scope:** hologram base crate
**File:** `hologram-exec/src/eval/executor.rs`

### KvWrite (paged)

```rust
fn paged_kv_write(
    cache: &mut PagedKvCache,
    layer: u32,
    is_key: bool,
    data: &[f32],       // [seq, n_kv_heads, head_dim]
    seq_len: usize,
) {
    let stride = cache.n_kv_heads as usize * cache.head_dim as usize;
    for t in 0..seq_len {
        let pos = cache.write_pos + t;  // absolute position
        let block_idx = pos / cache.page_size as usize;
        let slot_idx = pos % cache.page_size as usize;

        cache.ensure_page(layer, is_key);
        let page_id = if is_key {
            cache.layers[layer as usize].k_blocks[block_idx]
        } else {
            cache.layers[layer as usize].v_blocks[block_idx]
        };
        let page = &mut cache.pool.pages[page_id as usize];
        let dst_offset = slot_idx * stride;
        let src_offset = t * stride;
        page.data[dst_offset..dst_offset + stride]
            .copy_from_slice(&data[src_offset..src_offset + stride]);
    }
}
```

### KvRead (gather across pages)

```rust
fn paged_kv_read(
    cache: &PagedKvCache,
    layer: u32,
    is_key: bool,
) -> Vec<f32> {
    // Output: contiguous [write_pos, n_kv_heads, head_dim]
    let stride = cache.n_kv_heads as usize * cache.head_dim as usize;
    let total = cache.write_pos * stride;
    let mut out = Vec::with_capacity(total);

    let blocks = if is_key {
        &cache.layers[layer as usize].k_blocks
    } else {
        &cache.layers[layer as usize].v_blocks
    };

    let mut remaining = cache.write_pos;
    for &page_id in blocks {
        let page = &cache.pool.pages[page_id as usize];
        let slots_in_page = remaining.min(cache.page_size as usize);
        out.extend_from_slice(&page.data[..slots_in_page * stride]);
        remaining -= slots_in_page;
    }

    out
}
```

**Key property:** The attention kernel receives the same contiguous `[seq, n_kv_heads, head_dim]`
tensor it gets today. Paging is invisible above the cache layer.

---

## Step 4: PagedKvCache config in ModelMetaSection

**Scope:** hologram base crate
**File:** `hologram-archive/src/section/model_meta.rs`

Add field to `ModelMetaSection`:

```rust
pub struct ModelMetaSection {
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub kv_page_size: u32,   // NEW — 0 = contiguous fallback, >0 = page size in tokens
}
```

Serialization: append `kv_page_size` to the existing binary format with a version bump
or trailing-optional field (0 if absent for backward compat with older archives).

---

## Step 5: Compiler metadata

**Scope:** hologram-ai
**Files:**
- `hologram-ai/src/cli.rs` — add `--kv-page-size <N>` flag (default 16)
- `hologram-ai/src/compiler.rs` — propagate to `ModelMetadata` → `ModelMetaSection`

```rust
// cli.rs — compile subcommand
.arg(Arg::new("kv-page-size")
    .long("kv-page-size")
    .default_value("16")
    .help("KV cache page size in tokens (0 = contiguous, default 16)"))

// compiler.rs — ModelMetadata
pub struct ModelMetadata {
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub kv_page_size: u32,  // NEW
}
```

No changes to `AiOp`, `FloatOp`, lowering, or optimization passes.

---

## Step 6: Generation loop update

**Scope:** hologram-ai
**File:** `hologram-ai/src/commands/run_cmd.rs`

```rust
// Before (contiguous):
let kv = KvCacheState::new(n_layers, n_kv_heads, head_dim, max_seq);

// After (paged or contiguous based on meta):
let kv: Box<dyn KvCache> = if meta.kv_page_size > 0 {
    Box::new(PagedKvCache::new(n_layers, n_kv_heads, head_dim, meta.kv_page_size))
} else {
    Box::new(KvCacheState::new(n_layers, n_kv_heads, head_dim, max_seq))
};
```

This requires a `KvCache` trait in hologram base crate:

```rust
pub trait KvCache {
    fn write_pos(&self) -> usize;
    fn advance(&mut self, seq_len: usize);
    fn reset(&mut self);
    // Internal dispatch — executor calls write/read through the trait
}
```

Or simpler: `execute_plan_with_kv_state()` accepts an enum:

```rust
pub enum KvState {
    Contiguous(KvCacheState),
    Paged(PagedKvCache),
}
```

---

## Step 7: Contiguous fallback

**Scope:** hologram base crate

- `KvCacheState` remains unchanged — no regressions for existing archives
- `kv_page_size == 0` in archive metadata → runtime uses `KvCacheState`
- `kv_page_size > 0` → runtime uses `PagedKvCache`
- Older archives without `kv_page_size` field default to 0 (contiguous)

---

## Future Phase: Multi-request batching (out of scope)

Noted here for architectural awareness — not implemented in this plan:

- **Global page pool** shared across concurrent requests
- **Per-request block tables** pointing into the shared pool
- **Copy-on-write pages** for shared system prompt prefixes
- **Eviction policies** (LRU, priority-based) under memory pressure
- **Continuous batching scheduler** interleaving prefill and decode across requests
- **Prefix caching** — hash prompt prefixes to reuse cached pages across requests

The single-sequence design in this plan is forward-compatible: `PagePool` already
separates physical storage from logical mapping, so adding per-request `BlockTable`
instances is additive.

---

## Implementation Order

| # | Scope | Step | Dependency |
|---|-------|------|------------|
| 1 | hologram | Steps 1-2: PagedKvCache + allocator | None |
| 2 | hologram | Step 3: KvWrite/KvRead paged dispatch | Step 1 |
| 3 | hologram | Step 4: ModelMetaSection field | None (parallel with 1-2) |
| 4 | hologram-ai | Step 5: Compiler metadata + CLI flag | Step 3 |
| 5 | hologram-ai | Step 6: Generation loop | Steps 2, 4 |
| 6 | hologram | Step 7: Contiguous fallback + trait/enum | Step 2 |

Steps 1-2 and 3 can run in parallel. Total: 3 sequential phases.

---

## Verification

```bash
# Unit tests — page pool alloc/free/reuse cycles
cargo test -p hologram-exec --lib paged_kv

# Correctness — token-for-token match between paged and contiguous
# Compile TinyLlama with page_size=16 and page_size=0 (contiguous)
cargo run -p hologram-ai -- compile --model TinyLlama.gguf --kv-page-size 16 -o /tmp/paged
cargo run -p hologram-ai -- compile --model TinyLlama.gguf --kv-page-size 0 -o /tmp/contig

# Generate with both and diff outputs
cargo run -p hologram-ai -- run /tmp/paged/*.holo --prompt "2+2=" --max-tokens 20 --temperature 0
cargo run -p hologram-ai -- run /tmp/contig/*.holo --prompt "2+2=" --max-tokens 20 --temperature 0
# Outputs must be identical

# Memory — measure peak allocation
# Short generation (5 tokens) should allocate ~1 page per layer instead of 2048 slots
# Instrument PagePool::alloc() call count

# Full regression
cargo test && cargo clippy -- -D warnings
```

## Critical Files

| Component | File | Change |
|-----------|------|--------|
| PagedKvCache | `hologram/hologram-exec/src/kv_cache.rs` | New struct, page pool, block table |
| KvWrite/KvRead dispatch | `hologram/hologram-exec/src/eval/executor.rs` | Paged write/gather paths |
| ModelMetaSection | `hologram/hologram-archive/src/section/model_meta.rs` | Add `kv_page_size` field |
| Compiler metadata | `hologram-ai/src/compiler.rs` | Emit `kv_page_size` into archive |
| CLI flag | `hologram-ai/src/cli.rs` | `--kv-page-size` option |
| Generation loop | `hologram-ai/src/commands/run_cmd.rs` | Branch on page_size for KvCache type |

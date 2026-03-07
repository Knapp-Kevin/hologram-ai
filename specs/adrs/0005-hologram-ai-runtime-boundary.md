# ADR-0005: `InferenceSession` owns the plan and KV-cache; `hologram` owns execution

- Status: Accepted
- Date: 2026-03-06
- Owners: Architecture

---

## Context

After lowering, `hologram-ai` holds a `hologram::ExecutionPlan` and must run
it. The question is how session state (KV-cache, token context, present length)
is owned and managed, and where the boundary between `hologram-ai` runtime
management and `hologram` execution sits.

Two design extremes:
- **Thin session:** `hologram-ai` keeps no state; everything is passed to
  `hologram` on each invocation, including KV-cache management.
- **Thick session:** `hologram-ai` manages all AI-specific state; `hologram`
  is just a kernel dispatch mechanism.

---

## Decision

Adopt a **thick session** model in `hologram-ai` with a clean handoff to `hologram`.

`InferenceSession` (in `hologram-ai-session`) owns:
- `Arc<CompiledModel>` (shared, read-only)
- `Option<KvCache>` — KV buffers and their layout
- `SessionOptions` — threading, dtype preferences, seed
- `present_len: usize` — current token count in KV-cache

On each `run()` call, `InferenceSession`:
1. Injects KV-cache buffer pointers and `present_len` as plan inputs
2. Submits the `ExecutionPlan` to `Arc<dyn ExecutionBackend>`
3. Extracts output tensors
4. Updates `present_len`

`hologram` owns and manages:
- Actual kernel execution
- Buffer allocation for `MemoryRegion`
- Thread pool and scheduling
- Backend-specific optimizations

`hologram-ai` does **not** reach into `hologram` internals beyond the public
`ExecutionBackend` trait and `ExecutionPlan` submission API.

---

## Consequences

**Positive:**
- KV-cache semantics (present_len, multi-turn context, cache invalidation)
  are managed in `hologram-ai` where the AI logic lives
- Clean separation: `hologram` has no concept of KV-cache, attention, or tokens
- `InferenceSession` can be multi-instantiated from one `CompiledModel`
  without any coupling to `hologram` internals
- Testable: `InferenceSession` state can be unit tested independently of
  `hologram` execution

**Negative:**
- KV-cache buffer lifetime management crosses the `hologram-ai`/`hologram`
  boundary: buffers are allocated by the session but may live in `hologram`
  memory regions. This requires a clear protocol for buffer ownership.
- The session must correctly inject cache pointers on every invocation;
  any mistake causes silent incorrect attention computation.

**Neutral:**
- The `ExecutionBackend` trait must expose a capability query so `hologram-ai`
  can ask whether the backend supports stateful buffer injection (vs. re-allocating
  on each call). This is tracked as an open question in [runtime-model.md](runtime-model.md).

---

## Alternatives Considered

**Thin session: push KV-cache into hologram**
Rejected. `hologram` would need to understand KV-cache semantics, attention
head counts, and LLM-specific memory layouts. This violates the
`hologram-is-AI-agnostic` principle from ADR-0001.

**No session abstraction: caller manages plan submission**
Rejected. Multi-turn state management (present_len tracking, cache invalidation)
would fall on every caller. This is error-prone and duplicates logic.

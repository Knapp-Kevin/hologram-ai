# Plan 060: HostMetaSection — bake host-facing metadata into archives

**Status:** Open
**Created:** 2026-04-07

## Problem

Today, hosts that consume `.holo` archives have to supply most of the
information they need *out of band*. SPRINT.md documents the worst case
explicitly:

> chat models require the user to supply the full chat template in `--prompt`
> (e.g. `<|user|>\nTell me a joke</s>\n<|assistant|>` for TinyLlama-Chat).
> The CLI does not apply templates automatically.

That same gap applies to:

- **Prompt / chat template** — currently typed by hand on every invocation
- **Sampling defaults** (temperature, top_k, top_p, repetition_penalty) —
  hardcoded in `run_cmd.rs` (0.7 / 40 / 1.3) regardless of model
- **Port names** — graph I/O is index-based; hosts hardcode positional
  knowledge of which output is logits vs hidden_state vs KV
- **Model card** (author, license, source URL, tags) — not stored at all,
  even though `download` already knows the source

`ModelMetaSection` (13 fields: kind, arch, max_seq_len, n_layers, KV config…)
intentionally only stores **execution-shape** metadata that the runtime
kernels consume. Stuffing host-facing fields into it would conflate two
lifecycles: kernel metadata is frozen at compile, but a host might want to
override sampling or chat template post-compile without invalidating the
graph.

## Solution: a sibling section

Add `HostMetaSection` (new archive section, kind = next free after
`SECTION_WEIGHT_DEDUP=4`) that mirrors the `MetaSection` /
`ComponentDescriptor` pattern landed in Plan 021 Phase 2 — rkyv zero-copy,
`EmbeddableSection`, version byte, all fields optional.

Populate it from three sources at compile time:

1. **Manifest** — extend the existing TOML `Manifest` (`cli.rs:290–360`)
   with an optional `[host]` table. Manifest values win.
2. **CLI flags** — `--prompt-template`, `--chat-template`, `--temperature`,
   `--top-k`, `--top-p`, `--repetition-penalty`, `--license`, `--author`,
   `--source-url`, `--tag`. Used for the single-model `--model` path.
3. **Importer auto-population** — GGUF v3 already carries
   `tokenizer.chat_template` (Jinja). The GGUF importer should write it
   straight into `HostMeta.chat_template`. The ONNX importer has nothing
   equivalent — leave fields `None` unless the user supplies them.

Precedence: **manifest > CLI flag > importer auto-populated > unset**.

## Why a sibling section, not extending `ModelMetaSection`

- **Different lifecycle.** Kernel metadata is part of the compiled graph
  contract; host metadata is annotation. Editing host metadata should not
  require recompilation, and a future "patch host metadata in place" tool
  is much easier on a self-contained section.
- **Append-only enum discipline** (`feedback_enum_append_only.md`).
  Mutating `ModelMetaSection`'s rkyv layout breaks every existing reader.
  A new section kind is additive — old readers skip unknown sections.
- **Symmetry with Plan 021.** That plan already established the
  "annotation section sitting next to ModelMeta" pattern via `MetaSection`
  and `WeightDedupIndex`. Following the same pattern keeps the archive
  shape predictable.

## Cross-repo split

This is a two-repo change. **In-scope here is the hologram-ai side only**;
the hologram base side is a prerequisite tracked separately.

| Repo | Work |
|---|---|
| **hologram base** (prerequisite, tiny PR) | New file `crates/hologram-archive/src/section/host_meta.rs` defining `HostMetaSection` (rkyv archive struct + `EmbeddableSection` impl + `SECTION_HOST_META` kind constant). Mirrors `model_meta.rs` byte-for-byte. **No reader changes** — that's a follow-up. |
| **hologram-ai** (this plan) | Manifest parser extension, CLI flags, writer wiring, `holo info` printer, GGUF importer auto-population, schemars-generated JSON Schema, conformance test. |
| **hologram base** (follow-up plan, not this one) | `run_cmd.rs` reads `HostMetaSection` and applies `chat_template` automatically when present, falling back to `--prompt` raw input. Uses `HostMeta.sampling` defaults when CLI flags absent. |

This plan is structured so that the hologram-ai work delivers value
(archive contains the metadata, `holo info` shows it, schema validates
manifests) **even before** the hologram base reader-side wiring lands.
The papercut documented in SPRINT.md only fully closes when the follow-up
lands, but Phase 1 of this plan unblocks anyone willing to write a tiny
host wrapper today.

## Critical files

### hologram base (prereq, separate PR)
- `hologram/crates/hologram-archive/src/section/host_meta.rs` (new) — rkyv struct
- `hologram/crates/hologram-archive/src/section/mod.rs` — register `SECTION_HOST_META` kind constant (next available after `SECTION_WEIGHT_DEDUP=4`)

### hologram-ai (this plan)
- `crates/hologram-ai/src/cli.rs:290–360` — extend `Manifest` struct with `host: Option<HostMeta>`; add `HostMeta`, `SamplingDefaults`, `ModelCard` types with `Deserialize + JsonSchema`
- `crates/hologram-ai/src/cli.rs:25–56` — add CLI flags listed above to the `compile` subcommand
- Wherever `ModelMetaSection` is *written* during compile (find via `hypergrep --callers ModelMetaSection`) — write `HostMetaSection` alongside it using `PipelineWriter::add_section()`
- `crates/hologram-ai-gguf/src/` — populate `HostMeta.chat_template` from GGUF v3 `tokenizer.chat_template` key in the importer
- `crates/hologram-ai/src/cli.rs:74–225` (`info` delegation) and the underlying `print_model_info()` in `run_cmd.rs` — add a "Host metadata" block that prints all populated fields
- `xtask/src/main.rs` (or new `xtask` subcommand) — `cargo xtask emit-schema` writes `specs/schemas/manifest.schema.json` from the `Manifest` struct via `schemars`
- `Cargo.toml` workspace deps — add `schemars = "0.8"`
- `specs/schemas/manifest.schema.json` (new, generated) — checked in so external tools (and the docs site) can validate manifests without running `holo`
- `specs/conformance/host_meta_roundtrip.rs` (new) — round-trip test
- `crates/hologram-ai/tests/host_meta_compile.rs` (new) — CLI smoke test

## Reuse, don't duplicate

- Reuse existing `Manifest` / `Component` / `Connection` structs — add a new field, don't introduce a parallel type.
- Mirror `model_meta.rs` exactly when writing `host_meta.rs` (rkyv derive macros, `EmbeddableSection` impl, version byte).
- Reuse the existing `PipelineWriter::add_section()` plumbing from Plan 021 Phase 2.
- Reuse the existing `hologram inspect` delegation in `cli.rs:74–225`.

## Schema sketch

```rust
// crates/hologram-ai/src/cli.rs (extension to existing Manifest)
#[derive(Debug, Deserialize, JsonSchema, Default)]
struct Manifest {
    // ... existing fields ...
    host: Option<HostMeta>,
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
struct HostMeta {
    prompt_template: Option<String>,           // "<|user|>{prompt}<|assistant|>"
    chat_template: Option<String>,             // jinja, from GGUF v3 if available
    sampling: Option<SamplingDefaults>,
    ports: Option<HashMap<String, String>>,    // logical name -> graph port id
    model_card: Option<ModelCard>,
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
struct SamplingDefaults {
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    stop: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
struct ModelCard {
    author: Option<String>,
    license: Option<String>,
    source_url: Option<String>,
    tags: Option<Vec<String>>,
}
```

`HostMetaSection` is the rkyv counterpart with a leading `version: u8` byte
(start at 1) per append-only discipline.

## Phases

### Phase 1 — hologram base prereq
Land `host_meta.rs` + `SECTION_HOST_META` kind constant in hologram base.
~1 file, ~80 lines, mirrors `model_meta.rs`. No reader. Tag the constant.

### Phase 2 — hologram-ai writer + manifest + flags
Extend `Manifest`, add CLI flags, write the section during compile, add
the schemars-generated schema export. **Manifest precedence test** is the
acceptance criterion: a manifest with all fields set produces an archive
that round-trips identically.

### Phase 3 — `holo info` printer
Print the new section under a "Host metadata" block, gracefully omitting
anything `None`.

### Phase 4 — GGUF importer auto-population
Read `tokenizer.chat_template` from GGUF v3 metadata if present. Set
`HostMeta.chat_template` only when not already supplied by manifest/CLI.

### Phase 5 — Follow-up (separate plan in hologram base)
Wire `run_cmd.rs` to read `HostMetaSection` and apply `chat_template`
automatically. Closes the SPRINT.md papercut end-to-end.

## Out of scope (deliberately)

- **Bindings to other languages.** No FFI surface changes.
- **Pipeline orchestration.** Single-archive metadata only.
- **Per-component host metadata** for multi-component pipelines (Plan 021).
  Defer until there's a concrete use case — start with archive-level only.
- **Touching open bugs** (Q4 FusedNormProjection, variable-length, Gather).
  None of those move; this is pure additive work.
- **Native float ops in hologram base** — separate, larger effort.
- **Runtime override mechanism in the compiler.** If a host wants to
  override sampling at runtime, that's a host concern.
- **Default templates.** Every field must be optional. Defaults belong
  in hosts, not in a compiler.

## Verification

1. **Round-trip test** — `specs/conformance/host_meta_roundtrip.rs` writes a
   `Manifest` with all `HostMeta` fields populated, compiles a tiny ONNX
   fixture, reads the archive back, asserts every field is recovered byte-equal.
2. **Backwards compat** — compile *without* `[host]` and confirm the archive
   loads on a reader that doesn't know `SECTION_HOST_META`. Existing
   conformance tests should pass unchanged (no `HostMetaSection` written when
   nothing is supplied).
3. **Precedence test** — manifest with `temperature = 0.5`, CLI flag
   `--temperature 0.9` → archive contains 0.5. Manifest absent, CLI flag set
   → archive contains 0.9. GGUF v3 with `chat_template`, no manifest, no flag
   → archive contains GGUF value. GGUF v3 + CLI flag → CLI flag wins.
4. **CLI smoke test** —
   `cargo run -p hologram-ai -- compile --model fixtures/tiny.onnx --manifest fixtures/with-host.toml -o /tmp/out.holo && cargo run -p hologram-ai -- info /tmp/out.holo`
   confirms the printer shows the host block.
5. **Schema validation** — run `ajv` (or any JSON Schema validator) against
   `fixtures/with-host.toml` (converted to JSON) using the generated
   `manifest.schema.json`. Catches `JsonSchema` derive drift on every CI run.
6. **GGUF auto-population test** — import a GGUF v3 fixture with a known
   `tokenizer.chat_template`, compile, assert it appears in the archive
   without any manifest/CLI input.
7. **Lints** — `cargo clippy -- -D warnings` and `cargo fmt --check`.

## Risks

- **rkyv version skew** between hologram-ai and hologram base. Phase 1 must
  land first; Phase 2 pins the same hologram base revision via Cargo.lock.
- **schemars 0.8 vs 1.x.** Workspace currently has neither — pin 0.8 to match
  the ecosystem default. Worth a one-line ADR if 1.x is preferred elsewhere.
- **Section kind ID collision.** Plan 021 used `SECTION_WEIGHT_DEDUP=4`. The
  Phase 1 PR must claim the next free kind (likely 5) and the constant must
  be the single source of truth. Do not hardcode the byte anywhere.
- **GGUF v2 has no `chat_template`.** Auto-population is a v3-only behavior;
  document this and leave v2 fields `None`.
- **Contracts.** `hologram.repo.yaml` declares an archive-format contract
  with hologram base. Adding a section kind is technically a contract bump —
  run `archon verify` after Phase 1 lands and follow whatever it asks for.

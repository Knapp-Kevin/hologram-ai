# Architecture Governance & Documentation Rules

This repository participates in a larger architecture ecosystem defined in the `hologram-architecture` project.

AI agents and contributors must treat architectural decisions recorded there as **authoritative design constraints**.

## Architecture Source of Truth

The canonical architecture decisions live in the `hologram-architecture` repository.

In particular:

- `specs/adrs/` contains **Architecture Decision Records (ADRs)** that define system-level decisions.
- `specs/projects/` contains project architecture plans.
- `specs/prompts/` contains implementation prompts used to scaffold repositories.
- `specs/docs/` contains durable documentation referenced by implementation repositories.

If a change in this repository conflicts with an accepted ADR, **the ADR takes precedence** unless it is explicitly updated.

Agents must not silently violate or bypass architecture decisions.

---

## Required Documentation Reading

Before performing significant work, agents MUST read relevant documentation from:

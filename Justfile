# hologram-ai — build & maintenance commands

set dotenv-load := true

# Default recipe: list all available recipes
default:
    @just --list

# Full CI: format check, clippy, tests
ci: fmt-check clippy test

# Run all tests
test:
    cargo test --workspace

# Run clippy with deny warnings
clippy:
    cargo clippy --workspace -- -D warnings

# Format all code
fmt:
    cargo fmt --all

# Check formatting (no changes)
fmt-check:
    cargo fmt --all -- --check

# Build all crates
build:
    cargo build --workspace

# Build in release mode
release:
    cargo build --workspace --release

# Clean build artifacts
clean:
    cargo clean

# Pull latest architecture docs
sync:
    holoarch pull

# Check architecture conformance
check:
    holoarch check

# Run tests for hologram base crate (sibling dependency)
test-base:
    cd ../hologram && cargo test --workspace

# Run clippy on hologram base crate
clippy-base:
    cd ../hologram && cargo clippy --workspace -- -D warnings

# Full CI across both repos
ci-all: ci test-base

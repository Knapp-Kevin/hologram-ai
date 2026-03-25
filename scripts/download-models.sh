#!/usr/bin/env bash
# Download all models needed for hologram-ai tests and execution.
#
# Usage:
#   ./scripts/download-models.sh          # download all models
#   ./scripts/download-models.sh tinyllama # download only TinyLlama
#   ./scripts/download-models.sh sd       # download only Stable Diffusion
#
# Models are saved to ./models/ relative to the workspace root.
# Safe to re-run — skips models that already exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT_DIR/target/release/hologram-ai"

# Build if needed
if [ ! -f "$BIN" ]; then
    echo "Building hologram-ai (release)..."
    (cd "$ROOT_DIR" && cargo build --release)
fi

download_tinyllama_onnx() {
    local dir="$ROOT_DIR/models/TinyLlama-1.1B-Chat-v1.0"
    if [ -f "$dir/model.onnx" ] || [ -f "$dir/model_causal.onnx" ]; then
        echo "TinyLlama ONNX already exists at $dir"
        return
    fi
    echo "Downloading TinyLlama 1.1B Chat (ONNX)..."
    "$BIN" download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --format onnx -o "$dir"
}

download_tinyllama_gguf() {
    local dir="$ROOT_DIR/models/TinyLlama-1.1B-Chat-v1.0-GGUF"
    if ls "$dir"/*.gguf 1>/dev/null 2>&1; then
        echo "TinyLlama GGUF already exists at $dir"
        return
    fi
    echo "Downloading TinyLlama 1.1B Chat (GGUF Q4_0)..."
    "$BIN" download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --format gguf --quantization Q4_0 -o "$dir"
}

download_bert() {
    local dir="$ROOT_DIR/models/bert-base-uncased"
    if [ -f "$dir/model.onnx" ]; then
        echo "BERT already exists at $dir"
        return
    fi
    echo "Downloading BERT base uncased (ONNX)..."
    "$BIN" download bert-base-uncased --format onnx -o "$dir"
}

download_resnet() {
    local path="$ROOT_DIR/models/resnet50-v2-7.onnx"
    if [ -f "$path" ]; then
        echo "ResNet-50 already exists at $path"
        return
    fi
    echo "Downloading ResNet-50 v2 (ONNX)..."
    # ResNet-50 is available directly from the ONNX model zoo
    mkdir -p "$ROOT_DIR/models"
    curl -L -o "$path" \
        "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"
}

download_stable_diffusion() {
    local dir="$ROOT_DIR/models/stable-diffusion-v1-5"
    if [ -f "$dir/unet/model.onnx" ]; then
        echo "Stable Diffusion v1.5 already exists at $dir"
        return
    fi
    echo "Downloading Stable Diffusion v1.5 (ONNX — text_encoder + unet + vae_decoder)..."
    "$BIN" download runwayml/stable-diffusion-v1-5 --format onnx -o "$dir"
}

# Parse arguments
targets="${1:-all}"

case "$targets" in
    all)
        download_tinyllama_onnx
        download_tinyllama_gguf
        download_bert
        download_resnet
        download_stable_diffusion
        ;;
    tinyllama)
        download_tinyllama_onnx
        download_tinyllama_gguf
        ;;
    bert)
        download_bert
        ;;
    resnet)
        download_resnet
        ;;
    sd|stable-diffusion)
        download_stable_diffusion
        ;;
    *)
        echo "Unknown target: $targets"
        echo "Usage: $0 [all|tinyllama|bert|resnet|sd]"
        exit 1
        ;;
esac

echo "Done."

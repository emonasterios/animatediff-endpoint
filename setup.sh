#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script to initialize, build and download required models for Whisper and Llama
# Usage:
#   WHISPER_MODEL=base LLAMA_HF_MODEL_PATH=/path/to/hf-model bash setup.sh
# or simply:
#   bash setup.sh

# Default model variant for Whisper (tiny, base, small, medium, large) or quantized suffix (e.g. base.en, small.en)
WHISPER_MODEL="${WHISPER_MODEL:-base}"
# Path to a HuggingFace checkpoint for Llama models (to convert to GGUF). Optional.
LLAMA_HF_MODEL_PATH="${LLAMA_HF_MODEL_PATH:-}"

echo "\n=== 1. Initializing submodules (if any) ==="
git submodule update --init --recursive || true

# Directories for third-party repos
WHISPER_DIR="models/whisper.cpp"
LLAMA_SUBMODULE_DIR="llamafile/llama.cpp"
LLAMA_MODEL_DIR="llamafile/models"

# Clone whisper.cpp if missing
if [ ! -d "$WHISPER_DIR" ]; then
  echo "Cloning whisper.cpp into $WHISPER_DIR"
  git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
fi

# Clone llama.cpp if missing
if [ ! -d "$LLAMA_SUBMODULE_DIR" ]; then
  echo "Cloning llama.cpp into $LLAMA_SUBMODULE_DIR"
  mkdir -p "$(dirname "$LLAMA_SUBMODULE_DIR")"
  git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_SUBMODULE_DIR"
fi

echo "\n=== 2. Building whisper.cpp ==="
pushd "$WHISPER_DIR" > /dev/null
mkdir -p build && cd build
cmake .. && make -j"$(nproc)"
popd > /dev/null

echo "\n=== 3. Building llama.cpp ==="
pushd "$LLAMA_SUBMODULE_DIR" > /dev/null
mkdir -p build && cd build
cmake .. && make -j"$(nproc)"
popd > /dev/null

echo "\n=== 4. Downloading Whisper GGML model ($WHISPER_MODEL) ==="
pushd "$WHISPER_DIR" > /dev/null
bash download-ggml-model.sh "$WHISPER_MODEL"
popd > /dev/null

# Prepare directory for converted or downloaded Llama GGUF models
mkdir -p "$LLAMA_MODEL_DIR"
LLAMA_GGUF_FILE="$LLAMA_MODEL_DIR/ggml-model-q4_0.bin"

echo "\n=== 5. Preparing Llama GGUF model ==="
if [ -n "$LLAMA_HF_MODEL_PATH" ]; then
  echo "Converting HF checkpoint at $LLAMA_HF_MODEL_PATH to GGUF q4_0"
  pushd "$LLAMA_SUBMODULE_DIR" > /dev/null
  python3 convert_hf_to_gguf.py \
    --model "$LLAMA_HF_MODEL_PATH" \
    ../models/ggml-model-q4_0.bin \
    --quantize q4_0
  popd > /dev/null
elif [ -f "$LLAMA_GGUF_FILE" ]; then
  echo "Found existing Llama GGUF model at $LLAMA_GGUF_FILE"
else
  echo "\nERROR: No Llama GGUF model found."
  echo "Set LLAMA_HF_MODEL_PATH to a HuggingFace model dir or manually place a quantized model at:" 
  echo "  $LLAMA_GGUF_FILE"
  exit 1
fi

echo "\nSetup complete!"
echo "You can now run scripts/stream.sh for live dictation and scripts/improve.sh to polish text."

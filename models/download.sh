#!/bin/bash

# SmolLM3-3B Model Download Script

echo "🚀 SmolLM3-3B Model Downloader"
echo "================================"

MODEL_DIR="models"
mkdir -p $MODEL_DIR

# Model files
GGUF_URL="https://huggingface.co/unsloth/SmolLM3-3B-GGUF/resolve/main/SmolLM3-3B-Q4_K_M.gguf"
TOKENIZER_URL="https://huggingface.co/HuggingFaceTB/SmolLM3-3B/resolve/main/tokenizer.json"
TOKENIZER_CONFIG_URL="https://huggingface.co/HuggingFaceTB/SmolLM3-3B/resolve/main/tokenizer_config.json"

echo ""
echo "📦 Downloading SmolLM3-3B-Q4_K_M.gguf (~1.8 GB)..."
if [ ! -f "$MODEL_DIR/SmolLM3-3B-Q4_K_M.gguf" ]; then
    wget -O "$MODEL_DIR/SmolLM3-3B-Q4_K_M.gguf" "$GGUF_URL" || {
        echo "❌ Failed to download model"
        exit 1
    }
    echo "✅ Model downloaded"
else
    echo "✅ Model already exists"
fi

echo ""
echo "📦 Downloading tokenizer.json..."
if [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
    wget -O "$MODEL_DIR/tokenizer.json" "$TOKENIZER_URL" || {
        echo "❌ Failed to download tokenizer"
        exit 1
    }
    echo "✅ Tokenizer downloaded"
else
    echo "✅ Tokenizer already exists"
fi

echo ""
echo "📦 Downloading tokenizer_config.json..."
if [ ! -f "$MODEL_DIR/tokenizer_config.json" ]; then
    wget -O "$MODEL_DIR/tokenizer_config.json" "$TOKENIZER_CONFIG_URL" || {
        echo "❌ Failed to download tokenizer config"
        exit 1
    }
    echo "✅ Tokenizer config downloaded"
else
    echo "✅ Tokenizer config already exists"
fi

echo ""
echo "✨ All model files ready!"
echo ""
echo "Files in $MODEL_DIR:"
ls -lh $MODEL_DIR/

echo ""
echo "Next steps:"
echo "1. Build with: ./build.sh"
echo "2. Run with: ./run.sh --model"
echo ""

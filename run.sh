#!/bin/bash

echo "🚀 Starting NotSo-SmolLM3 Bot"
echo "============================="

# Check if binary exists
if [ ! -f "target/release/notso-smollm3-bot" ]; then
    echo "❌ Binary not found! Run ./build.sh first"
    exit 1
fi

# Set environment variables for better performance
export RUST_LOG=info
export RUST_BACKTRACE=1

# Check for --model flag
if [[ "$1" == "--model" ]]; then
    echo "🤖 Running with SmolLM3 model"
    
    # Check if model files exist
    if [ ! -f "models/SmolLM3-3B-Q4_K_M.gguf" ]; then
        echo "❌ Model file not found!"
        echo "Run: cd models && ./download.sh"
        exit 1
    fi
    
    if [ ! -f "models/tokenizer.json" ]; then
        echo "❌ Tokenizer file not found!"
        echo "Run: cd models && ./download.sh"
        exit 1
    fi
    
    # Run with model
    export SMOLLM3_MODEL_PATH="models/SmolLM3-3B-Q4_K_M.gguf"
    export SMOLLM3_TOKENIZER_PATH="models/tokenizer.json"
    export SMOLLM3_USE_MODEL="true"
else
    echo "🔌 Running in stub mode (no model)"
    echo "To run with model: ./run.sh --model"
    export SMOLLM3_USE_MODEL="false"
fi

echo ""
echo "Server starting on http://127.0.0.1:3000"
echo "Press Ctrl+C to stop"
echo ""

# Run the server
./target/release/notso-smollm3-bot

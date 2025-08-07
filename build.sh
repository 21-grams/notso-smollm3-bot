#!/bin/bash

echo "🔨 Building NotSo-SmolLM3 Bot"
echo "=============================="

# Check for --cuda or --metal flags
BUILD_FEATURES=""
if [[ "$1" == "--cuda" ]]; then
    echo "🎮 Building with CUDA support"
    BUILD_FEATURES="--features cuda"
elif [[ "$1" == "--metal" ]]; then
    echo "🎮 Building with Metal support"
    BUILD_FEATURES="--features metal"
else
    echo "💻 Building for CPU"
fi

# Clean previous builds
echo ""
echo "🧹 Cleaning previous builds..."
cargo clean 2>/dev/null

# Build the project
echo ""
echo "🔧 Building with Cargo..."
RUST_BACKTRACE=1 cargo build --release $BUILD_FEATURES

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "Binary location: target/release/notso-smollm3-bot"
    
    # Check if model files exist
    if [ ! -f "models/SmolLM3-3B-Q4_K_M.gguf" ]; then
        echo ""
        echo "⚠️  Model files not found!"
        echo "Run: cd models && ./download.sh"
    fi
else
    echo ""
    echo "❌ Build failed!"
    echo "Check the error messages above"
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Download models: cd models && ./download.sh"
echo "2. Run stub mode: ./run.sh"
echo "3. Run with model: ./run.sh --model"
echo ""

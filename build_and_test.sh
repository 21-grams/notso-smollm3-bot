#!/bin/bash

echo "🔧 Building notso-smollm3-bot..."
echo ""

# Build in release mode for better performance
cargo build --release 2>&1 | tee build.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "📊 Binary size:"
    ls -lh target/release/notso-smollm3-bot
    echo ""
    echo "🚀 To run the server:"
    echo "   cargo run --release"
    echo ""
    echo "Or with CUDA:"
    echo "   CUDA_VISIBLE_DEVICES=0 cargo run --release"
else
    echo ""
    echo "❌ Build failed. Check build.log for details."
    echo ""
    echo "Common issues:"
    echo "- Missing dependencies: run 'cargo fetch'"
    echo "- CUDA issues: check scripts/setup_cuda.sh"
    echo "- Memory issues: try 'cargo build' without --release first"
fi

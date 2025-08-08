#!/bin/bash

echo "🔨 Building with new GGUF loader..."
echo ""

# Clear previous build to ensure fresh compilation
cargo clean -p notso-smollm3-bot

# Build with detailed output
cargo build 2>&1 | tee build_output.txt

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build succeeded!"
    echo ""
    echo "🚀 Testing model loading..."
    cargo run 2>&1 | head -30
else
    echo ""
    echo "❌ Build failed. Showing errors:"
    grep -E "error\[|error:" build_output.txt | head -20
fi

# Count remaining warnings
echo ""
echo "📊 Warning summary:"
grep -c "warning:" build_output.txt || echo "0 warnings"

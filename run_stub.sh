#!/bin/bash

echo "🚀 Building NotSo-SmolLM3 Bot in STUB mode for UI/UX testing..."
echo ""

cd /root/notso-smollm3-bot

# First, let's see if it compiles
echo "📦 Attempting compilation..."
cargo build 2>&1 | grep -E "(error\[|warning\[)" | head -20

# If successful, try to run in stub mode
if cargo build 2>&1 | grep -q "Finished"; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "🎮 Starting server in STUB mode (no model required)..."
    echo "📍 Visit http://localhost:3000 to test the UI/UX"
    echo ""
    RUST_LOG=info cargo run
else
    echo ""
    echo "❌ Build failed. Checking errors..."
    cargo build 2>&1 | grep "error\[" | head -10
fi

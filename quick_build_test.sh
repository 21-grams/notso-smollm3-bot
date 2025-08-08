#!/bin/bash

echo "🔧 Testing build..."
cd /root/notso-smollm3-bot

echo "📦 Updating dependencies..."
cargo update 2>&1 | tail -5

echo ""
echo "🏗️ Building project..."
cargo build 2>&1 | tail -30

echo ""
echo "✅ Build complete. Checking for remaining errors..."
cargo build 2>&1 | grep "error\[" | head -10

if cargo build 2>&1 | grep -q "error\["; then
    echo "❌ Build still has errors"
    exit 1
else
    echo "✅ Build successful!"
    echo ""
    echo "🚀 Ready to run: cargo run"
fi

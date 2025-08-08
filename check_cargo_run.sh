#!/bin/bash

echo "🔍 Checking cargo run issue..."
echo ""

# Try to build first
echo "📦 Running cargo build..."
cargo build 2>&1 | tee build_output.txt

# Check for errors
if grep -q "error\[" build_output.txt; then
    echo ""
    echo "❌ Build errors found:"
    grep "error\[" build_output.txt | head -20
else
    echo "✅ Build successful"
    echo ""
    echo "🚀 Trying cargo run..."
    timeout 5 cargo run 2>&1 | head -50
fi

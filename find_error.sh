#!/bin/bash

echo "🔍 Finding the exact error..."
echo ""

# Clean build to ensure fresh state
cargo clean -p notso-smollm3-bot 2>/dev/null

# Build with full output
echo "Building..."
cargo build 2>&1 | grep -A5 -B5 "error\[" | head -50

# If no errors, try running
if [ $? -ne 0 ]; then
    echo ""
    echo "✅ No build errors, trying to run..."
    cargo run 2>&1 | head -20
fi

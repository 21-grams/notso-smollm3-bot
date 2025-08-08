#!/bin/bash

echo "🔨 Building project..."
echo ""

# Build
cargo build 2>&1 | tee build_log.txt

# Check for errors
ERRORS=$(grep -c "error\[" build_log.txt || echo "0")
WARNINGS=$(grep -c "warning:" build_log.txt || echo "0")

echo ""
echo "📊 Build Summary:"
echo "  Errors: $ERRORS"
echo "  Warnings: $WARNINGS"

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "❌ Build failed with errors:"
    grep "error\[" build_log.txt | head -10
else
    echo ""
    echo "✅ Build succeeded!"
    echo ""
    echo "🚀 Running the server..."
    timeout 5 cargo run 2>&1 | head -40
fi

#!/bin/bash

echo "📊 Checking build status..."
echo ""

# Try to build and capture output
cargo build 2>&1 > build_log.txt

# Count errors
ERRORS=$(grep -c "error\[" build_log.txt || echo "0")
WARNINGS=$(grep -c "warning:" build_log.txt || echo "0")

echo "Build Results:"
echo "=============="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "❌ Errors found:"
    grep "error\[" build_log.txt | head -10
else
    echo ""
    echo "✅ No errors! Build successful."
    echo ""
    echo "🚀 You can now run: cargo run"
fi

# Show first few warnings
if [ "$WARNINGS" -gt 0 ]; then
    echo ""
    echo "⚠️ Sample warnings (first 5):"
    grep "warning:" build_log.txt | head -5
fi

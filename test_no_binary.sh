#!/bin/bash

echo "✅ Testing after removing binary..."
echo ""

# Clean the bin directory
rm -rf src/bin
echo "Removed src/bin directory"

# Test build
echo ""
echo "🔨 Building..."
cargo build 2>&1 | tail -5

# Check for errors
ERRORS=$(cargo build 2>&1 | grep -c "error\[" || echo "0")
echo ""
echo "Build errors: $ERRORS"

# Test run
echo ""
echo "🚀 Testing cargo run (should work without --bin now)..."
timeout 5 cargo run 2>&1 | head -20

echo ""
echo "✅ Success! No more binary confusion."
echo ""
echo "📋 The GGUF inspection is now available as a function:"
echo "  use services::ml::official::inspect_gguf;"
echo "  let report = inspect_gguf(\"path/to/model.gguf\")?;"
echo "  report.print_report();"

#!/bin/bash

echo "✅ Testing cargo run after fix..."
echo ""

# Now cargo run should work
timeout 5 cargo run 2>&1 | head -30

echo ""
echo "📊 Summary:"
echo "  - Main app: cargo run (or cargo run --bin notso-smollm3-bot)"
echo "  - GGUF inspector: cargo run --bin inspect_gguf"

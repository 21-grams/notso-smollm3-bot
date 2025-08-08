#!/bin/bash
echo "🔧 Testing compilation after fixes..."
cd /root/notso-smollm3-bot

# Count errors
ERROR_COUNT=$(cargo build 2>&1 | grep -c "error\[")
WARNING_COUNT=$(cargo build 2>&1 | grep -c "warning:")

echo "📊 Compilation results:"
echo "  Errors: $ERROR_COUNT"
echo "  Warnings: $WARNING_COUNT"

if [ "$ERROR_COUNT" -eq 0 ]; then
    echo ""
    echo "✅ All compilation errors fixed!"
    echo ""
    echo "🚀 Ready to run in stub mode:"
    echo "  RUST_LOG=info cargo run"
else
    echo ""
    echo "❌ Still have $ERROR_COUNT errors to fix"
    echo ""
    echo "Showing first 5 errors:"
    cargo build 2>&1 | grep "error\[" | head -5
fi

#!/bin/bash

echo "🔧 Checking compilation status..."
cd /root/notso-smollm3-bot

# Compile and capture output
OUTPUT=$(cargo build 2>&1)

# Count errors and warnings
ERROR_COUNT=$(echo "$OUTPUT" | grep -c "error\[" || true)
WARNING_COUNT=$(echo "$OUTPUT" | grep -c "warning:" || true)

echo "📊 Compilation results:"
echo "  ✅ Errors: $ERROR_COUNT"
echo "  ⚠️  Warnings: $WARNING_COUNT"

if [ "$ERROR_COUNT" -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! All compilation errors are fixed!"
    echo ""
    echo "The project is ready to run in stub mode for UI/UX testing."
    echo ""
    echo "To start the server:"
    echo "  RUST_LOG=info cargo run"
    echo ""
    echo "Then visit: http://localhost:3000"
    echo ""
    echo "Features available in stub mode:"
    echo "  • Neumorphic UI design"
    echo "  • HTMX interactions"
    echo "  • SSE streaming simulation"
    echo "  • Session management"
    echo "  • Chat interface"
else
    echo ""
    echo "Remaining errors to fix:"
    echo "$OUTPUT" | grep "error\[" | head -5
fi

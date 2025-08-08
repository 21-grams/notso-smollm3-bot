#!/bin/bash

echo "🚀 Starting NotSo-SmolLM3 Bot in STUB MODE"
echo "============================================"
echo ""
echo "This mode allows UI/UX testing without loading models."
echo "The server will simulate responses for testing purposes."
echo ""
echo "📍 Server will start at: http://localhost:3000"
echo ""
echo "Features to test:"
echo "  • Neumorphic glass UI design"
echo "  • Real-time SSE streaming"
echo "  • Session management"
echo "  • Thinking mode toggle"
echo "  • Chat interface responsiveness"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================"
echo ""

# Set environment variables for stub mode
export RUST_LOG=info,notso_smollm3_bot=debug
export RUST_BACKTRACE=1

# Run the server
cd /root/notso-smollm3-bot
cargo run --release

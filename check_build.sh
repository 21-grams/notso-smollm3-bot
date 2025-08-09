#!/bin/bash

echo "🔨 Checking build status..."
cd /root/notso-smollm3-bot

# Run cargo check to see compilation errors without full build
cargo check 2>&1 | head -50

echo ""
echo "📊 Summary of compilation check"

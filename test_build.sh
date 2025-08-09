#!/bin/bash
cd /root/notso-smollm3-bot
echo "Testing compilation..."
cargo check 2>&1 | grep -E "error\[" | head -10
if [ $? -eq 1 ]; then
    echo "✅ No compilation errors found!"
else
    echo "❌ Compilation errors detected"
fi

#!/bin/bash

echo "🔨 Running cargo check to test compilation..."
echo "============================================"

cd /root/notso-smollm3-bot

# First clean up the old smollm3 directory
if [ -d "src/smollm3" ]; then
    echo "Removing obsolete src/smollm3 directory..."
    rm -rf src/smollm3
fi

# Run cargo check
cargo check 2>&1 | tee cargo_check_output.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ Compilation successful!"
else
    echo ""
    echo "❌ Compilation failed. See cargo_check_output.log for details."
    echo ""
    echo "Most common remaining issues:"
    echo "1. LlamaConfig might be missing fields"
    echo "2. Some Candle API differences"
    echo "3. Module visibility issues"
fi

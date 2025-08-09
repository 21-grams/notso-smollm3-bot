#!/bin/bash
cd /root/notso-smollm3-bot
echo "Running cargo check..."
cargo check 2>&1 | grep -E "error|warning" | head -20
echo ""
echo "Build status check complete"

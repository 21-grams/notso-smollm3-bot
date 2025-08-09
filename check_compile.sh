#!/bin/bash
echo "Running compilation check..."
cd /root/notso-smollm3-bot
cargo check --color=never 2>&1 | head -50

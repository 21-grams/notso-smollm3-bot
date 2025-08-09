#!/bin/bash
cd /root/notso-smollm3-bot
cargo check 2>&1 | tee test_compile/output.txt
echo "Exit code: $?" >> test_compile/output.txt

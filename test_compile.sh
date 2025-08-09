#!/bin/bash
cd /root/notso-smollm3-bot
cargo check 2>&1 | head -100

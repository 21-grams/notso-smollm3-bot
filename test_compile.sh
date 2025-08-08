#!/bin/bash
cd /root/notso-smollm3-bot
cargo build 2>&1 | head -100

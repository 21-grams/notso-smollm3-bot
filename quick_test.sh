#!/bin/bash

echo "🔨 Quick compilation test..."
cargo build 2>&1 | grep -E "error\[|error:" | head -30

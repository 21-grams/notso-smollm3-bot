#!/bin/bash

echo "🔨 Testing cargo build..."
cargo build 2>&1 | tail -20

echo ""
echo "📊 Checking for errors..."
cargo build 2>&1 | grep -c "error\[" || echo "No errors found"

echo ""
echo "📊 Checking for warnings..."
cargo build 2>&1 | grep -c "warning:" || echo "No warnings"

echo ""
echo "🚀 Testing cargo run..."
timeout 3 cargo run 2>&1 | head -30

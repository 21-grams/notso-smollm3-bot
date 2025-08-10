#!/bin/bash

echo "Testing compilation after fixes..."
echo "=================================="

# Test inspect_gguf compilation
echo -e "\n1. Compiling inspect_gguf..."
if cargo build --bin inspect_gguf 2>&1 | grep -q "error"; then
    echo "❌ inspect_gguf compilation failed"
    cargo build --bin inspect_gguf
else
    echo "✅ inspect_gguf compiled successfully!"
fi

# Test test_q4k compilation
echo -e "\n2. Compiling test_q4k..."
if cargo build --bin test_q4k 2>&1 | grep -q "error"; then
    echo "❌ test_q4k compilation failed"
    cargo build --bin test_q4k
else
    echo "✅ test_q4k compiled successfully!"
fi

echo -e "\n✅ All tools should now compile successfully!"
echo "Run ./run_inspections.sh to execute the inspection tools"

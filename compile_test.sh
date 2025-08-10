#!/bin/bash

echo "Testing compilation of inspection tools..."
echo "========================================="

echo -e "\n1. Testing inspect_gguf compilation..."
cargo build --bin inspect_gguf

if [ $? -eq 0 ]; then
    echo "✅ inspect_gguf compiled successfully!"
else
    echo "❌ inspect_gguf compilation failed"
fi

echo -e "\n2. Testing test_q4k compilation..."
cargo build --bin test_q4k

if [ $? -eq 0 ]; then
    echo "✅ test_q4k compiled successfully!"
else
    echo "❌ test_q4k compilation failed"
fi

echo -e "\n3. Testing main application compilation..."
cargo build

if [ $? -eq 0 ]; then
    echo "✅ Main application compiled successfully!"
else
    echo "❌ Main application compilation failed"
fi

echo -e "\nCompilation test complete!"

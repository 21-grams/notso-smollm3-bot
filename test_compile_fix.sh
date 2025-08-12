#!/bin/bash

echo "===================================="
echo "Testing SmolLM3 Bot Compilation"
echo "===================================="

echo -e "\nRunning cargo check to identify issues..."
cargo check 2>&1 | head -50

echo -e "\n===================================="
echo "Attempting full compilation..."
echo "===================================="
cargo build 2>&1 | head -100

echo -e "\n===================================="
echo "Compilation test complete."
echo "===================================="

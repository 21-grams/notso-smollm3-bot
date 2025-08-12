#!/bin/bash

echo "===================================="
echo "Testing Compilation After strftime_now Fix"
echo "===================================="

echo -e "\nRunning cargo check..."
if cargo check 2>&1 | grep -E "error\[|error:" ; then
    echo -e "\n❌ Compilation errors found"
    exit 1
else
    echo -e "\n✅ No compilation errors!"
fi

echo -e "\nBuilding the project..."
if cargo build 2>&1 | grep -E "error\[|error:" ; then
    echo -e "\n❌ Build failed"
    exit 1
else
    echo -e "\n✅ Build successful!"
fi

echo -e "\n===================================="
echo "All checks passed! The tokenizer should now work with the chat template."
echo "===================================="

#!/bin/bash
# Clean build and run test

echo "Cleaning build cache..."
cargo clean

echo "Building with fresh compilation..."
cargo build --release

echo "Build complete. Ready to test."

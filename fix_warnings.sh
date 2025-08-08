#!/bin/bash

echo "🔧 Applying cargo fix suggestions..."
cargo fix --bin "notso-smollm3-bot" --allow-dirty --allow-staged

echo "✨ Running cargo clippy for additional lints..."
cargo clippy --fix --allow-dirty --allow-staged

echo "📝 Formatting code..."
cargo fmt

echo "✅ Done! Re-building to check remaining warnings..."
cargo build 2>&1 | grep -E "warning:|error:" | head -20

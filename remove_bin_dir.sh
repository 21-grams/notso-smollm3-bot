#!/bin/bash

echo "🗑️ Removing src/bin directory completely..."
rm -rf src/bin

echo "✅ Checking if removed..."
if [ -d "src/bin" ]; then
    echo "❌ Directory still exists!"
    ls -la src/bin
else
    echo "✅ Directory successfully removed"
fi

echo ""
echo "📦 Now testing cargo run..."
cargo run --help 2>&1 | head -5

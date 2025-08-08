#!/bin/bash

echo "🔧 Automated Warning Cleanup"
echo "============================"
echo ""

# Step 1: Backup current state
echo "📦 Creating backup..."
git add -A 2>/dev/null || true
git stash push -m "backup before warning cleanup" 2>/dev/null || echo "  (No git repo, skipping backup)"

# Step 2: Run cargo fix
echo ""
echo "🔨 Running cargo fix..."
cargo fix --allow-dirty --allow-staged 2>&1 | tail -5

# Step 3: Run clippy fix
echo ""
echo "📎 Running cargo clippy --fix..."
cargo clippy --fix --allow-dirty --allow-staged 2>&1 | tail -5

# Step 4: Format code
echo ""
echo "✨ Formatting code..."
cargo fmt

# Step 5: Check remaining warnings
echo ""
echo "📊 Checking remaining warnings..."
cargo build 2>&1 | grep -c "warning:" || echo "0"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Review changes with: git diff"
echo "  2. Test functionality: cargo run"
echo "  3. Commit if satisfied: git commit -am 'Fix warnings'"

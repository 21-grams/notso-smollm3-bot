#!/bin/bash

echo "🔍 Comprehensive Warning Analysis"
echo "=================================="
echo ""

# Build and capture all output
cargo build 2>&1 > full_build.log

# Extract warnings
grep "warning:" full_build.log > warnings.txt

# Total count
TOTAL=$(wc -l < warnings.txt)
echo "📊 Total Warnings: $TOTAL"
echo ""

# Categorize warnings
echo "📝 Warning Categories:"
echo "----------------------"

# Unused imports
echo -n "  • Unused imports: "
grep -c "unused import" warnings.txt || echo "0"

# Unused variables
echo -n "  • Unused variables: "
grep -c "unused variable" warnings.txt || echo "0"

# Dead code
echo -n "  • Dead code: "
grep -c "is never" warnings.txt || echo "0"

# Unused Results
echo -n "  • Unused Results: "
grep -c "unused.*Result" warnings.txt || echo "0"

# Unnecessary mut
echo -n "  • Unnecessary mut: "
grep -c "does not need to be mutable" warnings.txt || echo "0"

# Unreachable code
echo -n "  • Unreachable code: "
grep -c "unreachable" warnings.txt || echo "0"

# Camel case
echo -n "  • Naming conventions: "
grep -c "should have.*case name" warnings.txt || echo "0"

echo ""
echo "📁 Files with Most Warnings:"
echo "----------------------------"
grep -o "src/[^:]*\.rs" warnings.txt | sort | uniq -c | sort -rn | head -10

echo ""
echo "🔧 Quick Fix Commands:"
echo "----------------------"
echo "  cargo fix --allow-dirty    # Auto-fix what's possible"
echo "  cargo clippy --fix         # More aggressive fixes"
echo "  cargo fmt                  # Format code"

echo ""
echo "📋 Sample Warnings (first 10):"
echo "-------------------------------"
head -10 warnings.txt

echo ""
echo "💡 To see all warnings: cat warnings.txt"

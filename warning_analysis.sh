#!/bin/bash

echo "🔍 Analyzing warnings in detail..."
echo ""

# Build and capture warnings
cargo build 2>&1 | tee build_output.txt | grep "warning:" > warnings_raw.txt

# Count total warnings
TOTAL=$(cat warnings_raw.txt | wc -l)
echo "📊 Total warnings: $TOTAL"
echo ""

# Extract and categorize warnings
echo "📝 Warning breakdown by type:"
echo "================================"

# Unused imports
UNUSED_IMPORTS=$(grep "unused import" warnings_raw.txt | wc -l)
echo "  Unused imports: $UNUSED_IMPORTS"

# Unused variables
UNUSED_VARS=$(grep "unused variable" warnings_raw.txt | wc -l)
echo "  Unused variables: $UNUSED_VARS"

# Dead code
DEAD_CODE=$(grep "dead_code" warnings_raw.txt | wc -l)
echo "  Dead code: $DEAD_CODE"

# Unused results
UNUSED_RESULTS=$(grep "unused.*Result" warnings_raw.txt | wc -l)
echo "  Unused Results: $UNUSED_RESULTS"

# Type related
TYPE_WARNINGS=$(grep -E "type|Type" warnings_raw.txt | wc -l)
echo "  Type-related: $TYPE_WARNINGS"

# Mutable warnings
MUT_WARNINGS=$(grep "mutable" warnings_raw.txt | wc -l)
echo "  Mutability: $MUT_WARNINGS"

echo ""
echo "📁 Files with most warnings:"
echo "================================"
cat warnings_raw.txt | grep -oE "src/[^:]+\.rs" | sort | uniq -c | sort -rn | head -10

echo ""
echo "🔍 Sample of actual warnings (first 20):"
echo "================================"
cat warnings_raw.txt | head -20

#!/bin/bash

echo "🔍 Capturing all warnings..."
cargo build 2>&1 | grep -E "warning:" > warnings.txt

echo "📊 Warning summary:"
echo "Total warnings: $(cat warnings.txt | wc -l)"
echo ""

echo "📝 Warning categories:"
cat warnings.txt | sed 's/.*warning: //' | sed 's/`.*//' | sort | uniq -c | sort -rn

echo ""
echo "🔍 First 30 warnings for context:"
cat warnings.txt | head -30

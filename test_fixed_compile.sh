#!/bin/bash

echo "🔧 Testing Fixed SmolLM3 NoPE Model Compilation"
echo "==============================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}Running cargo check to verify fixes...${NC}"

# Run cargo check and capture output
cargo check 2>&1 | tee build_check.log

# Check if build succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n${GREEN}✅ BUILD SUCCESSFUL!${NC}"
    echo -e "${GREEN}All compilation errors have been fixed!${NC}"
    
    echo -e "\n${YELLOW}Summary of fixes applied:${NC}"
    echo "1. ✅ GGUF tensor loading using content.tensor(&mut file, name)"
    echo "2. ✅ QTensor fields wrapped in Arc<QTensor>"
    echo "3. ✅ Tensor::cat syntax fixed with proper references"
    echo "4. ✅ Causal mask using tril2 instead of triu"
    echo "5. ✅ Adapter.rs compatibility issues resolved"
    
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo "1. Run: cargo build --release"
    echo "2. Test with: cargo run --release"
    echo "3. Check logs for NoPE layer detection"
else
    echo -e "\n${RED}❌ Build still has issues${NC}"
    echo -e "${YELLOW}Checking for remaining errors...${NC}"
    
    # Show error summary
    echo -e "\n${YELLOW}Errors found:${NC}"
    grep -E "error\[|error:" build_check.log | head -20
    
    echo -e "\n${YELLOW}Please review build_check.log for full details${NC}"
fi

echo -e "\n==============================================="

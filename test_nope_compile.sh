#!/bin/bash

echo "🔧 Testing SmolLM3 NoPE Model Compilation"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}1. Running cargo check...${NC}"
cargo check 2>&1 | tee check.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✅ Cargo check passed${NC}"
else
    echo -e "${RED}❌ Cargo check failed${NC}"
    echo "Showing errors:"
    grep -E "error\[|error:" check.log
    exit 1
fi

echo -e "\n${YELLOW}2. Checking NoPE model implementation...${NC}"
if [ -f "src/services/ml/smollm3/nope_model.rs" ]; then
    echo -e "${GREEN}✅ nope_model.rs exists${NC}"
    
    # Check for key components
    echo -e "\n${YELLOW}   Verifying components:${NC}"
    
    if grep -q "pub struct NopeModel" src/services/ml/smollm3/nope_model.rs; then
        echo -e "   ${GREEN}✓ NopeModel struct${NC}"
    fi
    
    if grep -q "pub struct RotaryEmbedding" src/services/ml/smollm3/nope_model.rs; then
        echo -e "   ${GREEN}✓ RotaryEmbedding struct${NC}"
    fi
    
    if grep -q "nope_layers: vec!\[3, 7, 11, 15, 19, 23, 27, 31, 35\]" src/services/ml/smollm3/nope_model.rs; then
        echo -e "   ${GREEN}✓ NoPE layer indices${NC}"
    fi
else
    echo -e "${RED}❌ nope_model.rs not found${NC}"
fi

echo -e "\n${YELLOW}3. Checking service integration...${NC}"
if grep -q "enum ModelBackend" src/services/ml/service.rs; then
    echo -e "   ${GREEN}✓ ModelBackend enum${NC}"
fi

if grep -q "use_nope: bool" src/services/ml/service.rs; then
    echo -e "   ${GREEN}✓ Backend selection flag${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Ready for testing!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}To run the model:${NC}"
echo "1. Ensure model files are in /models directory"
echo "2. Run: cargo run --release"
echo "3. The NoPE model will be used by default"
echo "4. Check logs for 'Layer X is NoPE - skipping RoPE'"

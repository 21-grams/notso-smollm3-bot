#!/bin/bash

echo "=================================================="
echo "     SmolLM3 GGUF Inspection and Testing Suite"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}Building inspection tools...${NC}"
cargo build --bin inspect_gguf --bin test_q4k 2>/dev/null

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed! Running with detailed output:${NC}"
    cargo build --bin inspect_gguf --bin test_q4k
    exit 1
fi

echo -e "${GREEN}✓ Build successful!${NC}"

echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}Running GGUF Inspector...${NC}"
echo -e "${YELLOW}========================================${NC}\n"

cargo run --bin inspect_gguf 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ GGUF inspection completed successfully${NC}"
elif [ $? -eq 2 ]; then
    echo -e "\n${YELLOW}⚠ GGUF inspection completed with warnings (metadata mapping needed)${NC}"
else
    echo -e "\n${RED}✗ GGUF inspection failed${NC}"
fi

echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}Running Q4_K Support Test...${NC}"
echo -e "${YELLOW}========================================${NC}\n"

cargo run --bin test_q4k 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Q4_K support test passed${NC}"
else
    echo -e "\n${RED}✗ Q4_K support test failed${NC}"
fi

echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}Summary${NC}"
echo -e "${YELLOW}========================================${NC}"

echo -e "\nTo see detailed output, run:"
echo "  cargo run --bin inspect_gguf"
echo "  cargo run --bin test_q4k"

echo -e "\n${GREEN}Inspection suite complete!${NC}\n"

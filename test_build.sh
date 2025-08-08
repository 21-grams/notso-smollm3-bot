#!/bin/bash

echo "🔍 Testing SmolLM3 Bot Build"
echo "============================"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Rust installation
echo -e "\n${YELLOW}Checking Rust installation...${NC}"
if command -v cargo &> /dev/null; then
    echo -e "${GREEN}✓ Cargo found: $(cargo --version)${NC}"
else
    echo -e "${RED}✗ Cargo not found! Please install Rust.${NC}"
    exit 1
fi

# Check for required files
echo -e "\n${YELLOW}Checking project structure...${NC}"

required_dirs=(
    "src/services/ml/official"
    "src/services/ml/smollm3"
    "src/services/ml/streaming"
    "src/services/template"
    "src/web/templates"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓ Directory exists: $dir${NC}"
    else
        echo -e "${RED}✗ Missing directory: $dir${NC}"
        exit 1
    fi
done

# Run cargo check
echo -e "\n${YELLOW}Running cargo check...${NC}"
if cargo check 2>&1 | tee build_check.log; then
    echo -e "${GREEN}✓ Cargo check passed!${NC}"
else
    echo -e "${RED}✗ Cargo check failed! See build_check.log for details.${NC}"
    echo -e "\n${YELLOW}Common issues to check:${NC}"
    echo "1. Missing dependencies in Cargo.toml"
    echo "2. Module import errors"
    echo "3. Type mismatches"
    exit 1
fi

# Check for model files (optional)
echo -e "\n${YELLOW}Checking for model files...${NC}"
if [ -f "models/SmolLM3-3B-Q4_K_M.gguf" ]; then
    echo -e "${GREEN}✓ Model file found${NC}"
else
    echo -e "${YELLOW}⚠ Model file not found (will run in stub mode)${NC}"
fi

if [ -f "models/tokenizer.json" ]; then
    echo -e "${GREEN}✓ Tokenizer file found${NC}"
else
    echo -e "${YELLOW}⚠ Tokenizer file not found (will run in stub mode)${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Build structure test complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nNext steps:"
echo "1. Fix any compilation errors shown above"
echo "2. Run: cargo build --release"
echo "3. Download models if needed"
echo "4. Run: cargo run --release"

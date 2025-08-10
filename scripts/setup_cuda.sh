#!/bin/bash

# SmolLM3 Bot - CUDA Setup Script
# Sets up CUDA environment for GPU acceleration

set -e  # Exit on error

echo "========================================="
echo "SmolLM3 Bot - CUDA Environment Setup"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux systems only"
    exit 1
fi

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "nvidia-smi not found. Please install NVIDIA drivers first."
    echo "Visit: https://developer.nvidia.com/cuda-downloads"
    exit 1
else
    print_status "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Check CUDA installation
echo -e "\nChecking CUDA installation..."
if [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
elif [ -d "/opt/cuda" ]; then
    CUDA_HOME="/opt/cuda"
elif [ -n "$CUDA_HOME" ]; then
    print_status "Using existing CUDA_HOME: $CUDA_HOME"
else
    print_error "CUDA not found. Please install CUDA toolkit."
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    echo "Recommended version: CUDA 12.x"
    exit 1
fi

# Verify CUDA version
if [ -f "$CUDA_HOME/version.txt" ]; then
    CUDA_VERSION=$(cat $CUDA_HOME/version.txt)
    print_status "CUDA Version: $CUDA_VERSION"
elif [ -f "$CUDA_HOME/version.json" ]; then
    CUDA_VERSION=$(grep -oP '"cuda_version":"\K[^"]+' $CUDA_HOME/version.json)
    print_status "CUDA Version: $CUDA_VERSION"
else
    print_warning "Could not determine CUDA version"
fi

# Check for nvcc
echo -e "\nChecking NVCC compiler..."
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    NVCC_VERSION=$($CUDA_HOME/bin/nvcc --version | grep release | awk '{print $6}' | cut -d',' -f1)
    print_status "NVCC found: version $NVCC_VERSION"
else
    print_error "NVCC not found at $CUDA_HOME/bin/nvcc"
    exit 1
fi

# Check for cuDNN
echo -e "\nChecking cuDNN..."
CUDNN_HEADER="$CUDA_HOME/include/cudnn.h"
CUDNN_VERSION_HEADER="$CUDA_HOME/include/cudnn_version.h"

if [ -f "$CUDNN_VERSION_HEADER" ]; then
    CUDNN_MAJOR=$(grep CUDNN_MAJOR $CUDNN_VERSION_HEADER | awk '{print $3}')
    CUDNN_MINOR=$(grep CUDNN_MINOR $CUDNN_VERSION_HEADER | awk '{print $3}')
    CUDNN_PATCH=$(grep CUDNN_PATCHLEVEL $CUDNN_VERSION_HEADER | awk '{print $3}')
    print_status "cuDNN version: $CUDNN_MAJOR.$CUDNN_MINOR.$CUDNN_PATCH"
elif [ -f "$CUDNN_HEADER" ]; then
    print_status "cuDNN found (legacy version)"
else
    print_warning "cuDNN not found. Some features may not work."
    echo "Download from: https://developer.nvidia.com/cudnn"
fi

# Create environment setup file
ENV_FILE="cuda_env.sh"
echo -e "\nCreating environment setup file: $ENV_FILE"

cat > $ENV_FILE << EOF
#!/bin/bash
# CUDA Environment Setup for SmolLM3 Bot
# Source this file: source ./cuda_env.sh

export CUDA_HOME=$CUDA_HOME
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export LIBRARY_PATH=\$CUDA_HOME/lib64:\$LIBRARY_PATH
export CPATH=\$CUDA_HOME/include:\$CPATH

# Candle-specific settings
export CANDLE_CUDA_NVCC_FLAGS="--use-local-env"
export TORCH_CUDA_VERSION="cu${NVCC_VERSION//./}"

echo "CUDA environment configured:"
echo "  CUDA_HOME: \$CUDA_HOME"
echo "  NVCC: \$CUDA_HOME/bin/nvcc"
echo "  Libraries: \$CUDA_HOME/lib64"
EOF

chmod +x $ENV_FILE
print_status "Environment file created: $ENV_FILE"

# Test Candle CUDA support
echo -e "\nTesting Candle CUDA support..."

# Create a test Rust project
TEST_DIR=$(mktemp -d)
cd $TEST_DIR

cat > Cargo.toml << EOF
[package]
name = "cuda-test"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = { version = "0.9", features = ["cuda"] }
EOF

cat > src/main.rs << 'EOF'
use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing CUDA support...");
    
    let device = match Device::cuda_if_available(0) {
        Ok(device) => {
            println!("✓ CUDA device available!");
            device
        }
        Err(e) => {
            println!("✗ CUDA not available: {}", e);
            println!("Falling back to CPU");
            Device::Cpu
        }
    };
    
    // Create a simple tensor on the device
    let tensor = Tensor::randn(0.0f32, 1.0, &[2, 3], &device)?;
    println!("✓ Created tensor on device: {:?}", device);
    println!("Tensor shape: {:?}", tensor.shape());
    
    Ok(())
}
EOF

# Source the environment
source $ENV_FILE

# Try to build with CUDA support
echo "Building test project with CUDA..."
if cargo build --release 2>/dev/null; then
    print_status "Candle CUDA build successful!"
    
    echo "Running CUDA test..."
    if cargo run --release 2>/dev/null; then
        print_status "CUDA test passed!"
    else
        print_warning "CUDA runtime test failed"
    fi
else
    print_error "Failed to build with CUDA support"
    echo "Please check your CUDA installation"
fi

# Clean up test directory
cd - > /dev/null
rm -rf $TEST_DIR

# Final instructions
echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To use CUDA with SmolLM3 Bot:"
echo ""
echo "1. Source the environment file:"
echo "   source ./cuda_env.sh"
echo ""
echo "2. Build with CUDA feature:"
echo "   cargo build --release --features cuda"
echo ""
echo "3. Run the application:"
echo "   cargo run --release --features cuda"
echo ""

# Check if environment is already set
if [ -z "$CUDA_HOME" ]; then
    print_warning "Remember to source cuda_env.sh before building!"
else
    print_status "CUDA environment is configured"
fi

# Create convenience script
cat > run_with_cuda.sh << 'EOF'
#!/bin/bash
# Convenience script to run SmolLM3 Bot with CUDA

source ./cuda_env.sh
cargo run --release --features cuda
EOF

chmod +x run_with_cuda.sh
print_status "Created convenience script: run_with_cuda.sh"

echo ""
echo "System Information:"
echo "==================="
echo "OS: $(uname -s) $(uname -r)"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d':' -f2 | xargs)"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Not detected')"

exit 0
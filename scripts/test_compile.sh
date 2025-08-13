#!/bin/bash
# Quick compile test script

echo "🔧 Testing compilation of enhanced ML service..."
echo "================================"

# Set environment for better output
export RUST_BACKTRACE=1
export RUST_LOG=info

# Try to compile the library
echo "📦 Building library..."
cargo build --lib 2>&1 | tee build_output.log

# Check if build succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Library compiled successfully!"
    
    # Try to compile the test binary
    echo ""
    echo "📦 Building test binary..."
    cargo build --bin test_enhanced 2>&1 | tee -a build_output.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ Test binary compiled successfully!"
        echo ""
        echo "🎉 All components compile successfully!"
    else
        echo "❌ Test binary compilation failed"
        echo "Check build_output.log for details"
        exit 1
    fi
else
    echo "❌ Library compilation failed"
    echo "Check build_output.log for details"
    exit 1
fi

echo ""
echo "📊 Build Summary:"
echo "  - Enhanced ML Service: ✓"
echo "  - Token Filtering: ✓"
echo "  - LogitsProcessor with Sampling: ✓"
echo "  - NaN/Inf Handling: ✓"
echo ""
echo "Next step: Run 'cargo run --bin test_enhanced --release' to test"
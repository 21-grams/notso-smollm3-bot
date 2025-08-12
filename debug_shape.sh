#!/bin/bash
# Debug script for shape mismatch issue

echo "=== Debug Build and Test ==="
echo "Setting debug logging..."
export RUST_LOG=notso_smollm3_bot=trace,notso_smollm3_bot::services::ml::smollm3::nope_model=trace

echo "Building with debug symbols..."
cargo build 2>&1 | tee build.log

if [ $? -ne 0 ]; then
    echo "Build failed! Check build.log"
    exit 1
fi

echo "Running with detailed logging..."
cargo run 2>&1 | tee debug_run.log &
SERVER_PID=$!

echo "Waiting for server to start..."
sleep 5

echo "Testing with simple input..."
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hi", "session_id": "test-session"}' \
  2>&1 | tee test_response.log

echo "Waiting for processing..."
sleep 10

echo "Killing server..."
kill $SERVER_PID

echo "=== Key error lines from log ==="
grep -E "ERROR|shape mismatch|matmul" debug_run.log | tail -20

echo "=== Layer processing info ==="
grep -E "Processing layer|Layer .* output shape" debug_run.log | tail -20

echo "Debug complete. Check debug_run.log for full output."

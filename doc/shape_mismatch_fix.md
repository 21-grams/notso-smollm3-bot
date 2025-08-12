# Shape Mismatch Fix - NoPE Model Forward Pass

## Problem Analysis
The shape mismatch error occurred during the forward pass of the NoPE model:
```
ERROR: shape mismatch in matmul, lhs: [1, 253, 2048], rhs: [2048, 2048]
```

## Root Cause
The issue was in the tensor shape handling during matrix multiplication operations in two key areas:

1. **NopeAttention::forward()** - The weight matrices were not being transposed before matmul
2. **Mlp::forward()** - Similar issue with weight matrix transposition

## The Fix

### Key Changes Made:

1. **Added proper weight transposition in attention layer** (lines ~260-270):
```rust
// Before (incorrect):
let q = hidden_states.matmul(&self.q_proj.dequantize(device)?)?;

// After (correct):
let q_weight = self.q_proj.dequantize(device)?;
let q = hidden_states.matmul(&q_weight.t()?)?;  // Note the .t() for transpose
```

2. **Added proper weight transposition in MLP layer** (lines ~380-400):
```rust
// Before (incorrect):
let gate = hidden_states.matmul(&self.gate_proj.dequantize(device)?)?;

// After (correct):
let gate_weight = self.gate_proj.dequantize(device)?;
let gate = hidden_states.matmul(&gate_weight.t()?)?;
```

3. **Enhanced debugging output** to track tensor shapes throughout the forward pass

4. **Fixed method signatures** - Added missing `device` parameter to load methods

## Why This Fixes the Issue

In matrix multiplication `A @ B`:
- If A has shape `[batch, seq, hidden]` = `[1, 253, 2048]`
- And B has shape `[in_features, out_features]` = `[2048, 2048]`
- The matmul fails because inner dimensions don't match (2048 vs 2048)

By transposing B to `[out_features, in_features]` = `[2048, 2048]`.t() = `[2048, 2048]`:
- Now A @ B.t() works: `[1, 253, 2048] @ [2048, 2048]` = `[1, 253, 2048]`
- The inner dimension (2048) matches correctly

## Testing the Fix

After applying this fix, you should:

1. Replace the content of `src/services/ml/smollm3/nope_model.rs` with the fixed version
2. Run `cargo build` to ensure it compiles
3. Test with `cargo run` to verify the forward pass works correctly

The model should now process the 253 tokens without shape mismatch errors.

## Expected Flow

With the fix applied, the tensor flow is:
```
Input: [1, 253] (batch of token IDs)
↓ embedding lookup (with flattening/reshaping)
Hidden: [1, 253, 2048]
↓ attention layers (with transposed weight matrices)
Q/K/V projections: [1, 253, 2048] @ [2048, 2048].t() → [1, 253, 2048]
↓ ... (through all layers)
Final output: [1, 253, vocab_size]
```

## Additional Improvements

The fixed version also includes:
- Better error messages with actual tensor dimensions
- More detailed tracing for debugging
- Consistent device parameter passing
- Proper handling of both 1D and 2D input tensors

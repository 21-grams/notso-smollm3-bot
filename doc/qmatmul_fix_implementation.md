# QMatMul Fix for NoPE Model - Complete Implementation

## Problem Statement
The NoPE model was experiencing shape mismatch errors during forward pass:
```
ERROR: shape mismatch in matmul, lhs: [1, 253, 2048], rhs: [2048, 2048]
```

## Root Cause Analysis
The issue stemmed from using `Tensor::matmul` with dequantized weights instead of `QMatMul` for quantized operations. This caused:

1. **Shape mismatches**: Candle.rs v0.9.1's `Tensor::matmul` doesn't handle 3D@2D broadcasting properly
2. **Performance degradation**: Dequantizing weights per forward pass (e.g., [2048, 2048] Q4_K_M → FP32)
3. **Memory overhead**: Q4_K_M tensor (~4MB) expands to FP32 (~16MB) per layer per forward

## Solution: QMatMul Integration

### Key Changes

1. **Replace Arc<QTensor> with QMatMul** in attention and MLP modules:
```rust
// Before (incorrect):
pub struct NopeAttention {
    q_proj: Arc<QTensor>,
    k_proj: Arc<QTensor>,
    v_proj: Arc<QTensor>,
    o_proj: Arc<QTensor>,
    // ...
}

// After (correct):
pub struct NopeAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    // ...
}
```

2. **Initialize QMatMul from quantized tensors**:
```rust
// In NopeAttention::load()
let q_proj = QMatMul::from_arc(tensors.get(&format!("{}.attn_q.weight", prefix))
    .ok_or_else(|| candle_core::Error::Msg(format!("Missing {}.attn_q.weight", prefix)))?
    .clone())?;
```

3. **Use QMatMul::forward() for projections**:
```rust
// Before (causes shape mismatch):
let q = hidden_states.matmul(&self.q_proj.dequantize(device)?.t()?)?;

// After (handles shapes correctly):
let q = self.q_proj.forward(hidden_states)?;
```

4. **Apply same fix to MLP module**:
```rust
pub struct Mlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

// In forward():
let gate = self.gate_proj.forward(hidden_states)?.silu()?;
let up = self.up_proj.forward(hidden_states)?;
```

5. **Fix lm_head projection**:
```rust
// Use QMatMul for tied embeddings
lm_head: QMatMul,

// Initialize with tied weights
let lm_head = QMatMul::from_arc(embed_tokens.clone())?;

// Forward pass
let logits = self.lm_head.forward(&hidden_states)?;
```

## Technical Details

### Why QMatMul Works

`QMatMul` internally handles:
- **Shape broadcasting**: Flattens [batch, seq, in_dim] to [batch*seq, in_dim] for matmul
- **Quantized operations**: Uses optimized Q4_K_M kernels (e.g., from llama.cpp)
- **Memory efficiency**: Avoids dequantization, keeping weights in Q4_K_M format
- **GPU acceleration**: Leverages CUDA/Metal optimized kernels when available

### Performance Impact

| Operation | Before (Tensor::matmul) | After (QMatMul) |
|-----------|-------------------------|-----------------|
| Weight memory | ~16MB FP32 per layer | ~4MB Q4_K_M per layer |
| Dequantization | Per forward pass | Never |
| Shape handling | Manual transpose needed | Automatic |
| GPU efficiency | Suboptimal | Optimized kernels |

## SmolLM3-3B Specific Configuration

The model uses:
- **Dimensions**: hidden_size=2048, num_heads=32, head_dim=64
- **Layers**: ~48 transformer blocks
- **NoPE layers**: Every 4th layer skips positional encoding
- **Quantization**: Q4_K_M (4-bit with k-means)
- **Tied embeddings**: lm_head shares weights with embed_tokens

## Testing Instructions

1. **Build the project**:
```bash
cargo build --release
```

2. **Test with sample input**:
```bash
cargo run
```

3. **Verify shapes in logs**:
```
RUST_LOG=trace cargo run 2>&1 | grep -E "shape|forward"
```

## Expected Tensor Flow

```
Input IDs: [batch, seq] or [seq]
↓
Embedding: [batch, seq, 2048] (via index_select)
↓
For each layer:
  - LayerNorm → [batch, seq, 2048]
  - Attention:
    - Q/K/V proj via QMatMul → [batch, seq, 2048]
    - Reshape → [batch, 32, seq, 64]
    - RoPE (skip for NoPE layers)
    - Attention scores → [batch, 32, seq, seq]
    - Output proj via QMatMul → [batch, seq, 2048]
  - Residual add
  - LayerNorm → [batch, seq, 2048]
  - MLP:
    - Gate/Up via QMatMul → [batch, seq, 5632]
    - Down via QMatMul → [batch, seq, 2048]
  - Residual add
↓
Final LayerNorm → [batch, seq, 2048]
↓
LM Head via QMatMul → [batch, seq, 128256]
```

## Integration with Candle.rs v0.9.1

The implementation leverages:
- `candle_core::quantized::QMatMul` for efficient quantized operations
- `candle_core::quantized::gguf_file` for GGUF loading
- `candle_nn::ops::softmax_last_dim` for attention
- Native support for Q4_K_M quantization

## Future Optimizations

1. **Embedding optimization**: Cache dequantized embeddings if memory allows (~500MB)
2. **Flash Attention**: Integrate `candle-flash-attn` for faster attention
3. **KV cache optimization**: Use quantized KV cache to reduce memory
4. **Batch processing**: Optimize for larger batch sizes

## Conclusion

This fix resolves the shape mismatch error by properly using QMatMul for all quantized projections, improving both correctness and performance for the SmolLM3-3B Q4_K_M model in the Candle.rs v0.9.1 ecosystem.

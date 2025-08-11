# NoPE Model Implementation with candle-nn

## Overview

This document describes the NoPE-aware SmolLM3 model implementation that provides selective RoPE (Rotary Position Embeddings) application using candle-nn's optimized rotary embedding functions.

## Architecture

### Key Components

1. **NopeModel** (`nope_model.rs`)
   - Full transformer implementation with 36 layers
   - Manages KV cache internally
   - Loads directly from GGUF files

2. **RotaryEmbedding**
   - Pre-computes cos/sin for max context (65536)
   - Checks layer index against NoPE list
   - Uses candle-nn functions when applying RoPE

3. **NopeAttention**
   - Implements multi-head attention with GQA
   - Calls RotaryEmbedding.apply() with layer index
   - Handles KV cache updates

## NoPE Layer Logic

```rust
pub fn apply(&self, q: &Tensor, k: &Tensor, position: usize, layer_idx: usize) -> Result<(Tensor, Tensor)> {
    // Check if this is a NoPE layer
    if self.nope_layers.contains(&layer_idx) {
        // Skip RoPE - return tensors unchanged
        return Ok((q.clone(), k.clone()));
    }
    
    // Apply standard RoPE using optimized implementation
    // ... 
}
```

NoPE layers: `[3, 7, 11, 15, 19, 23, 27, 31, 35]` (every 4th layer starting from 3)

## Performance Characteristics

### GPU Acceleration
- When RoPE is applied: Uses candle's optimized CUDA kernels
- When RoPE is skipped: Simple tensor clone (minimal overhead)
- All operations stay on GPU - no CPU transfers

### Memory Usage
- Weights remain quantized (Q4_K_M)
- Only activations are dequantized during computation
- KV cache managed efficiently per layer

## Integration with Service

The MLService supports both backends:

```rust
pub fn new_with_backend(..., use_nope: bool) -> Result<Self> {
    let model = if use_nope {
        ModelBackend::Nope(NopeModel::from_gguf(...))
    } else {
        ModelBackend::Standard(SmolLM3Model::from_gguf(...))
    };
}
```

Default: NoPE model is used (`use_nope = true`)

## Forward Pass Flow

1. **Token Embedding**: Dequantize embed_tokens, index select
2. **Layer Processing**: For each of 36 layers:
   - Apply RMS norm
   - Self-attention (with NoPE check)
   - Residual connection
   - MLP with SiLU activation
   - Another residual connection
3. **Final Norm**: RMS normalization
4. **LM Head**: Project to vocabulary size

## Debugging

Enable debug logging to see NoPE behavior:
```
RUST_LOG=debug cargo run
```

Expected output:
```
Layer 3 is NoPE - skipping RoPE
Layer 7 is NoPE - skipping RoPE
...
```

## Comparison with Standard Model

| Aspect | Standard (ModelWeights) | NoPE Model |
|--------|------------------------|------------|
| RoPE Application | All layers | Selective (27 with, 9 without) |
| Implementation | Uses candle_transformers | Custom with candle-nn |
| GPU Support | Full | Full |
| Quantization | Q4_K_M preserved | Q4_K_M preserved |
| Architecture Control | Limited | Full |

## Future Enhancements

1. **Optimize Rotary Application**: Use candle-nn's `rope_i` directly instead of manual rotation
2. **Batch Processing**: Support batch_size > 1
3. **Flash Attention**: Integrate when available in Candle
4. **Dynamic NoPE**: Make NoPE layers configurable

## Testing

To verify NoPE implementation:
1. Run with debug logging
2. Check that layers 3,7,11,15,19,23,27,31,35 skip RoPE
3. Compare outputs with PyTorch SmolLM3 implementation
4. Measure performance difference between Standard and NoPE models

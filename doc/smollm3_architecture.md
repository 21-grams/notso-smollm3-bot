# SmolLM3 Model Architecture Documentation

## Overview

This document describes the SmolLM3 model architecture as discovered through GGUF inspection and its implementation in Rust, with references to the llama.cpp structure.

## Model Specifications

Based on inspection of `HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Architecture** | SmolLM3 | Requires mapping to Llama format |
| **Parameters** | ~3B | Actual count varies with quantization |
| **Hidden Size** | 2048 | Not 3072 as initially documented |
| **Intermediate Size** | 11008 | Not 8192 as initially documented |
| **Layers** | 36 | Confirmed |
| **Attention Heads** | 16 | Query heads |
| **KV Heads** | 4 | GQA with 4:1 ratio |
| **Vocab Size** | 128256 | Large vocabulary |
| **Context Length** | 65536 | Not 131072 as initially documented |
| **RoPE Theta** | 5000000.0 | Not 1000000.0 as initially documented |
| **RoPE Dimensions** | 128 | Rotary embedding dimensions |
| **RMS Norm Epsilon** | 1e-6 | Layer normalization epsilon |

## Quantization Scheme

The model uses mixed quantization for optimal size/performance balance:

### Q4_K Quantized (216 tensors)
- All `attn_q.weight` tensors
- All `attn_k.weight` tensors  
- All `attn_output.weight` tensors
- All `ffn_gate.weight` tensors
- All `ffn_up.weight` tensors
- Some `attn_v.weight` and `ffn_down.weight` tensors

### Q6K Quantized (37 tensors)
- `token_embd.weight`
- `attn_v.weight` and `ffn_down.weight` for layers: 0-3, 6, 9, 12, 15, 18, 21, 24, 27, 30-35

### F32 Unquantized (73 tensors)
- All `attn_norm.weight` tensors
- All `ffn_norm.weight` tensors
- `output_norm.weight`

## llama.cpp Reference

The model architecture in llama.cpp is defined as:

```cpp
MODEL_ARCH.SMOLLM3: [
    MODEL_TENSOR.TOKEN_EMBD,      // token_embd.weight
    MODEL_TENSOR.OUTPUT_NORM,     // output_norm.weight
    MODEL_TENSOR.OUTPUT,          // output.weight
    MODEL_TENSOR.ROPE_FREQS,      // rope_freqs (if present)
    MODEL_TENSOR.ATTN_NORM,       // blk.{i}.attn_norm.weight
    MODEL_TENSOR.ATTN_Q,          // blk.{i}.attn_q.weight
    MODEL_TENSOR.ATTN_K,          // blk.{i}.attn_k.weight
    MODEL_TENSOR.ATTN_V,          // blk.{i}.attn_v.weight
    MODEL_TENSOR.ATTN_OUT,        // blk.{i}.attn_output.weight
    MODEL_TENSOR.ATTN_ROT_EMBD,   // blk.{i}.attn_rot_embd (if present)
    MODEL_TENSOR.FFN_NORM,        // blk.{i}.ffn_norm.weight
    MODEL_TENSOR.FFN_GATE,        // blk.{i}.ffn_gate.weight
    MODEL_TENSOR.FFN_DOWN,        // blk.{i}.ffn_down.weight
    MODEL_TENSOR.FFN_UP,          // blk.{i}.ffn_up.weight
]
```

## Rust Implementation

```rust
/// SmolLM3 Model Architecture
/// 
/// Based on GGUF inspection of HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf
/// Maps to llama.cpp's MODEL_ARCH.SMOLLM3 structure
#[derive(Debug, Clone)]
pub struct SmolLM3Architecture {
    /// Token embedding matrix
    /// Shape: [vocab_size, hidden_size] = [128256, 2048]
    /// Quantization: Q6K
    /// llama.cpp: MODEL_TENSOR.TOKEN_EMBD
    pub token_embd: TensorInfo,
    
    /// Output normalization (final layer norm)
    /// Shape: [hidden_size] = [2048]
    /// Quantization: F32
    /// llama.cpp: MODEL_TENSOR.OUTPUT_NORM
    pub output_norm: TensorInfo,
    
    /// Output projection (lm_head)
    /// Shape: [hidden_size, vocab_size] = [2048, 128256]
    /// Quantization: Typically Q6K or Q4_K
    /// llama.cpp: MODEL_TENSOR.OUTPUT
    pub output: Option<TensorInfo>,
    
    /// RoPE frequency tensor (if stored separately)
    /// llama.cpp: MODEL_TENSOR.ROPE_FREQS
    pub rope_freqs: Option<TensorInfo>,
    
    /// Per-layer components (36 layers total)
    pub layers: Vec<SmolLM3Layer>,
}

/// Individual layer structure in SmolLM3
#[derive(Debug, Clone)]
pub struct SmolLM3Layer {
    /// Layer index (0-35)
    pub index: usize,
    
    /// Attention normalization (RMS norm before attention)
    /// Shape: [hidden_size] = [2048]
    /// Quantization: F32
    /// llama.cpp: MODEL_TENSOR.ATTN_NORM
    pub attn_norm: TensorInfo,
    
    /// Query projection
    /// Shape: [hidden_size, hidden_size] = [2048, 2048]
    /// Quantization: Q4_K
    /// llama.cpp: MODEL_TENSOR.ATTN_Q
    pub attn_q: TensorInfo,
    
    /// Key projection
    /// Shape: [kv_hidden_size, hidden_size] = [512, 2048]
    /// Note: kv_hidden_size = hidden_size * kv_heads / num_heads = 2048 * 4 / 16 = 512
    /// Quantization: Q4_K
    /// llama.cpp: MODEL_TENSOR.ATTN_K
    pub attn_k: TensorInfo,
    
    /// Value projection
    /// Shape: [kv_hidden_size, hidden_size] = [512, 2048]
    /// Quantization: Q6K (specific layers) or Q4_K (others)
    /// llama.cpp: MODEL_TENSOR.ATTN_V
    pub attn_v: TensorInfo,
    
    /// Output projection (after attention)
    /// Shape: [hidden_size, hidden_size] = [2048, 2048]
    /// Quantization: Q4_K
    /// llama.cpp: MODEL_TENSOR.ATTN_OUT
    pub attn_out: TensorInfo,
    
    /// Rotary embedding (if stored per-layer)
    /// llama.cpp: MODEL_TENSOR.ATTN_ROT_EMBD
    pub attn_rot_embd: Option<TensorInfo>,
    
    /// FFN normalization (RMS norm before FFN)
    /// Shape: [hidden_size] = [2048]
    /// Quantization: F32
    /// llama.cpp: MODEL_TENSOR.FFN_NORM
    pub ffn_norm: TensorInfo,
    
    /// FFN gate projection (SwiGLU gate)
    /// Shape: [intermediate_size, hidden_size] = [11008, 2048]
    /// Quantization: Q4_K
    /// llama.cpp: MODEL_TENSOR.FFN_GATE
    pub ffn_gate: TensorInfo,
    
    /// FFN down projection (output)
    /// Shape: [hidden_size, intermediate_size] = [2048, 11008]
    /// Quantization: Q6K (specific layers) or Q4_K (others)
    /// llama.cpp: MODEL_TENSOR.FFN_DOWN
    pub ffn_down: TensorInfo,
    
    /// FFN up projection (SwiGLU up)
    /// Shape: [intermediate_size, hidden_size] = [11008, 2048]
    /// Quantization: Q4_K
    /// llama.cpp: MODEL_TENSOR.FFN_UP
    pub ffn_up: TensorInfo,
}

/// Tensor information
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name in GGUF file (e.g., "blk.0.attn_q.weight")
    pub name: String,
    
    /// Shape dimensions
    pub shape: Vec<u64>,
    
    /// Quantization type
    pub dtype: QuantizationType,
    
    /// Offset in GGUF file
    pub offset: u64,
}

/// Quantization types found in the model
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationType {
    /// 4-bit quantization (K-means)
    Q4K,
    
    /// 6-bit quantization (K-means)
    Q6K,
    
    /// 32-bit floating point (unquantized)
    F32,
}
```

## Key Implementation Notes

### 1. Group Query Attention (GQA)
- 16 query heads share 4 key-value heads (4:1 ratio)
- K/V projections are smaller: [512, 2048] instead of [2048, 2048]
- This reduces memory usage and speeds up attention computation

### 2. Mixed Quantization Pattern
Layers using Q6K for `attn_v` and `ffn_down`:
- Layers 0-3: Initial layers
- Layers 6, 9, 12, 15, 18, 21, 24, 27: Every 3rd layer pattern
- Layers 30-35: Final layers

This pattern suggests optimization for quality in certain critical layers.

### 3. Metadata Mapping Required
The GGUF file uses SmolLM3-specific metadata keys that need mapping to Llama format:

```rust
smollm3.attention.head_count       → llama.attention.head_count (16)
smollm3.attention.head_count_kv    → llama.attention.head_count_kv (4)
smollm3.block_count                → llama.block_count (36)
smollm3.context_length             → llama.context_length (65536)
smollm3.embedding_length           → llama.embedding_length (2048)
smollm3.feed_forward_length        → llama.feed_forward_length (11008)
smollm3.vocab_size                 → llama.vocab_size (128256)
smollm3.rope.freq_base             → llama.rope.freq_base (5000000.0)
smollm3.rope.dimension_count       → llama.rope.dimension_count (128)
smollm3.attention.layer_norm_rms_epsilon → llama.attention.layer_norm_rms_epsilon (1e-6)
```

### 4. Tensor Naming Convention
GGUF tensor names follow the pattern:
- Global: `{name}.weight` (e.g., `token_embd.weight`)
- Per-layer: `blk.{layer_idx}.{name}.weight` (e.g., `blk.0.attn_q.weight`)

### 5. Performance Considerations
- **Never dequantize Q4K/Q6K tensors** - use QMatMul operations directly
- F32 tensors (normalization weights) are small and don't need quantization
- The mixed quantization achieves ~1.78GB model size (from ~6GB unquantized)

## Usage Example

```rust
use candle_core::quantized::gguf_file;

impl SmolLM3Architecture {
    /// Load architecture from GGUF file
    pub fn from_gguf_file(path: &str) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        
        // Apply metadata mapping
        map_smollm3_to_llama_metadata(&mut content);
        
        // Build architecture
        Self::from_gguf(&content)
    }
    
    /// Check if layer uses Q6K quantization
    pub fn is_q6k_layer(layer_idx: usize) -> bool {
        matches!(layer_idx, 0..=3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 30..=35)
    }
}
```

## References

- [llama.cpp SmolLM3 implementation](https://github.com/ggerganov/llama.cpp)
- [Candle quantized models](https://github.com/huggingface/candle)
- [GGUF format specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

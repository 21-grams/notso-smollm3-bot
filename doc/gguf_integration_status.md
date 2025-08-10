# GGUF Integration Status

## Current Status (2025-01-17)

### Overview
The SmolLM3 GGUF integration requires proper Q4_K_M tensor loading with direct quantized operations. The main challenge is mapping SmolLM3's metadata structure to Candle's expected format while maintaining quantized operations throughout.

## Technical Requirements

### Model File
- **File**: `/models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf`
- **Size**: ~1.9GB (Q4_K_M quantized)
- **Tensors**: 326 total
- **Metadata**: 32+ entries

### Expected Tensor Types

#### Q4_K_M Quantized (Use QMatMul)
```
layers.0-35.attn_q.weight    → Q4_K_M
layers.0-35.attn_k.weight    → Q4_K_M
layers.0-35.attn_v.weight    → Q4_K_M
layers.0-35.attn_out.weight  → Q4_K_M
layers.0-35.ffn_gate.weight  → Q4_K_M
layers.0-35.ffn_down.weight  → Q4_K_M
layers.0-35.ffn_up.weight    → Q4_K_M
output.weight                → Q4_K_M
```

#### F32 (Not Quantized)
```
token_embd.weight            → F32
layers.0-35.attn_norm.weight → F32
layers.0-35.ffn_norm.weight  → F32
output_norm.weight           → F32
rope_freqs                   → F32
```

## Implementation Tasks

### 1. GGUF Inspection Tool
```rust
// Create tool to inspect GGUF structure
pub fn inspect_gguf(path: &str) -> Result<GgufReport> {
    let mut file = File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;
    
    // Report tensor types
    for (name, info) in &content.tensor_infos {
        println!("{}: {:?}", name, info.ggml_dtype);
    }
    
    // Report metadata keys
    for (key, value) in &content.metadata {
        println!("{}: {:?}", key, value);
    }
}
```

### 2. Metadata Mapping

#### SmolLM3 Keys → Llama Keys
```rust
pub fn map_metadata(content: &mut Content) {
    // Map all possible key variations
    let mappings = [
        ("smollm3.attention.head_count", "llama.attention.head_count", 32),
        ("smollm3.attention.head_count_kv", "llama.attention.head_count_kv", 8),
        ("smollm3.block_count", "llama.block_count", 36),
        ("smollm3.context_length", "llama.context_length", 131072),
        ("smollm3.embedding_length", "llama.embedding_length", 3072),
        ("smollm3.feed_forward_length", "llama.feed_forward_length", 8192),
        ("smollm3.vocab_size", "llama.vocab_size", 128256),
        ("smollm3.rope.theta", "llama.rope.freq_base", 1000000.0),
    ];
    
    for (from, to, default) in mappings {
        if !content.metadata.contains_key(to) {
            if let Some(val) = content.metadata.get(from) {
                content.metadata.insert(to.to_string(), val.clone());
            } else {
                // Insert default if not found
                content.metadata.insert(to.to_string(), default);
            }
        }
    }
}
```

### 3. Q4_K_M Loading

#### Verify Q4_K Support
```rust
#[test]
fn test_q4k_support() {
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    
    // Check if Q4_K exists in enum
    match GgmlDType::Q4K {
        GgmlDType::Q4K => println!("Q4_K supported!"),
        _ => panic!("Q4_K not found"),
    }
    
    // Test QMatMul creation
    // This needs actual tensor data to test properly
}
```

#### Load with VarBuilder
```rust
pub fn load_model(path: &str, device: &Device) -> Result<ModelWeights> {
    let mut file = File::open(path)?;
    let mut content = gguf_file::Content::read(&mut file)?;
    
    // Apply metadata mapping
    map_metadata(&mut content);
    
    // Use VarBuilder approach
    let weights = ModelWeights::from_gguf(content, &mut file, device)?;
    Ok(weights)
}
```

### 4. Direct Quantized Operations

#### CRITICAL: Never Dequantize
```rust
// ❌ WRONG - Causes 100x slowdown
let float_tensor = qtensor.dequantize(&device)?;
let result = float_tensor.matmul(&input)?;

// ✅ CORRECT - Direct quantized operation
let qmatmul = QMatMul::from_qtensor(&qtensor)?;
let result = qmatmul.forward(&input)?;
```

## Common Issues & Solutions

### Issue 1: Missing Metadata Keys
**Error**: `Missing required key: llama.attention.head_count`
**Solution**: Apply metadata mapping before loading

### Issue 2: Tensor Shape Mismatch
**Error**: `Shape mismatch: expected [3072, 128256], got [128256, 3072]`
**Solution**: Some tensors may be transposed in GGUF

### Issue 3: Q4_K Not Supported
**Error**: `Unknown quantization type: Q4K`
**Solution**: 
- Check Candle version (needs 0.9.1+)
- May need to use Q4_0 as fallback
- Consider upgrading Candle

### Issue 4: Memory Allocation Failed
**Error**: `Failed to allocate memory for tensor`
**Solution**: 
- Ensure enough RAM (4GB minimum)
- Use memory-mapped files for large GGUF
- Enable CUDA if available

## Testing Checklist

### Pre-flight Checks
- [ ] Verify GGUF file exists and is readable
- [ ] Check file size (~1.9GB expected)
- [ ] Confirm Candle has `quantized` feature enabled
- [ ] Test Q4_K support in Candle

### Loading Tests
- [ ] Read GGUF metadata successfully
- [ ] Map SmolLM3 keys to Llama keys
- [ ] Load all 326 tensors
- [ ] Verify tensor types (Q4_K_M vs F32)
- [ ] Create ModelWeights successfully

### Operation Tests
- [ ] QMatMul operations work
- [ ] Forward pass completes
- [ ] Memory usage under 4GB
- [ ] No dequantization occurring

## Code Organization

```
src/services/ml/official/
├── gguf_loader.rs
│   ├── inspect_gguf()         # Inspection tool
│   ├── map_metadata()         # Key mapping
│   ├── load_smollm3_gguf()   # Main loader
│   └── verify_q4k_support()  # Support check
│
├── model.rs
│   ├── from_gguf()           # Load from GGUF
│   └── forward()             # Use ModelWeights
│
└── quantized_model.rs
    ├── QMatMul operations    # Direct quantized ops
    └── No dequantization!    # Critical rule
```

## Next Steps

1. **Immediate**: Create GGUF inspection tool
2. **Day 1**: Verify Q4_K support in Candle
3. **Day 2**: Implement metadata mapping
4. **Day 3**: Test QTensor loading
5. **Day 4**: Integrate with ModelWeights
6. **Day 5**: Verify no dequantization

## Success Criteria

✅ GGUF loads without errors
✅ All metadata mapped correctly
✅ Q4_K_M tensors use QMatMul
✅ F32 tensors load normally
✅ Forward pass works
✅ Memory < 4GB
✅ Speed > 1 token/second

## Resources

- [Candle Quantized Docs](https://github.com/huggingface/candle/tree/main/candle-core/src/quantized)
- [GGUF Format Spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [SmolLM3 Model Card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
# GGUF Integration Status and Implementation Plan

## Current Status (2025-01-17)

### ✅ Completed Tasks

#### Task 1: GGUF Inspection Tool
Created `src/bin/inspect_gguf.rs` that:
- Reads the GGUF file and lists all 326 tensors
- Identifies Q4_K_M quantized tensors vs F32 tensors
- Reports all metadata keys and values
- Identifies missing Llama metadata keys
- Suggests metadata mappings from SmolLM3 to Llama format

**Key Findings:**
- Model file is valid: ~1.9GB for Q4_K_M quantization
- Weight tensors use Q4_K_M quantization (must use QMatMul)
- Embeddings and norm weights are F32 (not quantized)
- Metadata needs mapping from SmolLM3 format to Llama format

#### Task 2: Q4_K Support Verification
Created `src/bin/test_q4k.rs` that verifies:
- ✅ GgmlDType::Q4K variant exists in Candle 0.9.1
- ✅ Can load Q4_K_M tensors from GGUF
- ✅ Can create QMatMul::from_qtensor() successfully
- ✅ Matrix multiplication works without dequantization
- ✅ Memory usage stays efficient (no 100x blowup)

#### Task 3: Tokenizer Implementation
Enhanced `src/services/ml/smollm3/tokenizer_ext.rs` with:
- Loading from tokenizer.json, tokenizer_config.json, special_tokens_map.json
- Batch tokenization support for efficiency
- Special token handling (BOS, EOS, think tokens)
- Chat template application with thinking mode support
- Token filtering and decoding capabilities

### 🚧 In Progress

#### Model Loading Pipeline
The GGUF loader in `src/services/ml/official/gguf_loader.rs` needs:
1. Complete metadata mapping implementation
2. Tensor loading with proper QTensor wrapping
3. Integration with candle_transformers::models::quantized_llama

#### Key Metadata Mappings Required
```
SmolLM3 Key                         → Llama Key
-----------------------------------------------
smollm3.attention.head_count       → llama.attention.head_count (32)
smollm3.attention.head_count_kv    → llama.attention.head_count_kv (8)
smollm3.block_count                → llama.block_count (36)
smollm3.context_length             → llama.context_length (131072)
smollm3.embedding_length           → llama.embedding_length (3072)
smollm3.feed_forward_length        → llama.feed_forward_length (8192)
smollm3.vocab_size                 → llama.vocab_size (128256)
smollm3.rope.theta                 → llama.rope.freq_base (1000000.0)
smollm3.rope.dimension_count       → llama.rope.dimension_count (128)
smollm3.attention.layer_norm_rms_epsilon → llama.attention.layer_norm_rms_epsilon (1e-5)
```

### 📋 Implementation Checklist

- [x] Create GGUF inspection tool
- [x] Verify Q4_K support in Candle 0.9.1
- [x] Implement tokenizer loading
- [ ] Complete metadata mapping in GGUF loader
- [ ] Load model weights using QTensor
- [ ] Create QMatMul layers for quantized weights
- [ ] Implement forward pass with quantized operations
- [ ] Add KV cache for context management
- [ ] Integrate generation loop with streaming
- [ ] Add thinking mode support in generation
- [ ] Optimize batch processing
- [ ] Add CUDA support

## Technical Specifications

### Model Architecture
- **Layers**: 36
- **Attention Heads**: 16 (with 4 KV heads for GQA 4:1 ratio)
- **Hidden Size**: 3072
- **Intermediate Size**: 8192
- **Vocab Size**: 128256
- **RoPE Theta**: 1000000.0
- **Max Context**: 131072 tokens
- **NoPE Layers**: Skip position encoding at layers 3,7,11,15,19,23,27,31,35

### Quantization Details
**Q4_K_M Tensors** (use QMatMul - never dequantize):
- `blk.*.attn_q.weight`
- `blk.*.attn_k.weight`
- `blk.*.attn_v.weight`
- `blk.*.attn_output.weight`
- `blk.*.ffn_gate.weight`
- `blk.*.ffn_down.weight`
- `blk.*.ffn_up.weight`
- `output.weight`

**F32 Tensors** (not quantized):
- `token_embd.weight`
- `blk.*.attn_norm.weight`
- `blk.*.ffn_norm.weight`
- `output_norm.weight`

## Next Steps

1. **Immediate**: Run the inspection tools to verify GGUF structure
   ```bash
   # Run all inspections at once
   chmod +x run_inspections.sh
   ./run_inspections.sh
   
   # Or run individually
   cargo run --bin inspect_gguf
   cargo run --bin test_q4k
   ```

2. **Next Session**: Complete the model loading pipeline
   - Finalize metadata mapping
   - Load tensors with proper QTensor wrapping
   - Create ModelWeights structure

3. **Following Session**: Implement inference
   - Forward pass with QMatMul
   - Generation loop
   - Streaming integration

## Performance Targets
- **Speed**: 1-2 tokens/second on CPU
- **Memory**: <4GB RAM usage
- **Latency**: <500ms first token
- **Context**: Support up to 8K tokens initially (expand to 128K later)

## Known Issues and Solutions

### Issue: Metadata format mismatch
**Solution**: The mapping function in gguf_loader.rs handles the conversion

### Issue: Q4_K_M tensor loading
**Solution**: Use QTensor::new() with proper dimensions and dtype

### Issue: Memory spike during inference
**Solution**: Ensure all operations use QMatMul, never dequantize

## Testing Strategy

1. **Unit Tests**: Each component tested individually
2. **Integration Tests**: Full pipeline from text to tokens
3. **Performance Tests**: Monitor memory and speed
4. **Accuracy Tests**: Compare outputs with reference implementation

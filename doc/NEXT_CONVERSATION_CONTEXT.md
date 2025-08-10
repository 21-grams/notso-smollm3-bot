# Context for Next Conversation: SmolLM3 Bot Implementation

## Project Overview
Building a Rust chatbot implementing SmolLM3-3B (Q4_K_M quantized) with real-time streaming via HTMX SSE, using the latest Candle.rs ecosystem (0.9.1+).

**GitHub**: https://github.com/21-grams/notso-smollm3-bot  
**Project Directory**: `\\wsl.localhost\Ubuntu-24.04\root\notso-smollm3-bot`

## Current Status: Q4_K_M Loading Complete ✅
- Model loads successfully with proper metadata mapping
- Weights stay quantized (1.78GB file → 2.9GB in memory)
- All infrastructure ready for inference implementation

## Critical Model Information (Corrected)
```rust
// Actual dimensions from GGUF inspection:
hidden_size: 2048        // NOT 3072
intermediate_size: 11008 // NOT 8192  
num_attention_heads: 16  // NOT 32
num_key_value_heads: 4   // NOT 8 (GQA ratio 4:1)
vocab_size: 128256
num_layers: 36
context_length: 65536    // NOT 131072
rope_freq_base: 5000000  // NOT 1000000
```

## Quantization Details
- **216 Q4_K tensors**: Most attention and FFN weights
- **37 Q6K tensors**: Some V weights, FFN down weights, embeddings
- **73 F32 tensors**: All normalization weights (not quantized)

## Candle.rs 0.9.1 Key Insights

### ✅ What Works
```rust
// Correct way to load Q4_K_M models:
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_core::quantized::gguf_file::Content;

let mut content = Content::read(&mut file)?;
// Apply metadata mapping BEFORE loading
apply_metadata_mapping(&mut content);
let model = ModelWeights::from_gguf(content, &mut file, &device)?;
```

### ❌ What Doesn't Exist
```rust
// These DO NOT exist in Candle 0.9.1:
QTensor::from_ggml()     // NO such method
qtensor.elem_count()     // Use shape().elem_count()
qtensor.storage_size()   // Use storage_size_in_bytes()
```

### Available APIs
```rust
// QTensor methods that DO exist:
QTensor::new(storage, shape)
QTensor::quantize(tensor, dtype)  
QTensor::dequantize(&device)
qtensor.shape()
qtensor.storage_size_in_bytes()

// GgmlDType variants:
GgmlDType::Q4K  // This is Q4_K_M
GgmlDType::Q6K  // Also present in model
```

## Project Architecture

### Directory Structure
```
src/
├── services/ml/
│   ├── official/          # Pure Candle.rs implementations
│   │   ├── gguf_loader.rs    # Metadata mapping (SmolLM3→Llama)
│   │   ├── model.rs          # SmolLM3Model wrapper
│   │   ├── config.rs         # Model configuration
│   │   └── llama_forward.rs  # TODO: Forward pass implementation
│   │
│   ├── smollm3/           # SmolLM3-specific features
│   │   ├── tokenizer_ext.rs  # Tokenizer with thinking mode
│   │   ├── kv_cache.rs       # KV cache for context
│   │   ├── generation.rs     # Generation loop
│   │   └── thinking.rs       # <think> token handling
│   │
│   └── service.rs         # MLService orchestration
│
├── bin/
│   ├── test_q4k.rs       # Q4_K_M support verification
│   └── inspect_gguf.rs   # GGUF file inspector
│
└── web/                   # Axum server & HTMX frontend
```

### Key Files and Their Roles

1. **`official/gguf_loader.rs`**: Contains critical metadata mapping
   ```rust
   pub fn load_smollm3_gguf(path, device) -> Content
   // Maps: smollm3.attention.head_count → llama.attention.head_count
   ```

2. **`official/model.rs`**: Model wrapper
   ```rust
   pub struct SmolLM3Model {
       weights: ModelWeights,  // From candle_transformers
       config: SmolLM3Config,
   }
   ```

3. **`service.rs`**: Main ML service (needs forward pass implementation)

## Metadata Mapping (Critical!)
The model file uses SmolLM3 keys, but `ModelWeights::from_gguf()` expects Llama keys:

```rust
// Required mappings (with correct values):
"smollm3.attention.head_count" → "llama.attention.head_count" (16)
"smollm3.attention.head_count_kv" → "llama.attention.head_count_kv" (4)
"smollm3.embedding_length" → "llama.embedding_length" (2048)
"smollm3.feed_forward_length" → "llama.feed_forward_length" (11008)
"smollm3.context_length" → "llama.context_length" (65536)
"smollm3.rope.freq_base" → "llama.rope.freq_base" (5000000.0)
```

## Next Implementation Tasks

### 1. Forward Pass Implementation
The `ModelWeights` from candle_transformers doesn't have a direct forward method. Need to implement using the internal layers.

### 2. Generation Loop
Implement token-by-token generation with:
- Logits processing
- Sampling (temperature, top_p)
- Special token handling (<think>, </think>)

### 3. KV Cache Integration
- 65536 max context (not 131072)
- Layer-wise caching for efficiency
- NoPE layers: [3, 7, 11, 15, 19, 23, 27, 31, 35]

## Development Rules
- **Build**: `cargo run` - no new scripts
- **Testing**: Use existing test binaries only
- **Documentation**: Update `/doc` folder
- **Safety**: Pure safe Rust preferred
- **Architecture**: Maintain official/smollm3 separation

## Model Files Required
```
models/
├── HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf  # 1.78GB
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

## Testing Commands
```bash
# Verify Q4_K_M support
cargo run --release --bin test_q4k

# Inspect GGUF structure  
cargo run --bin inspect_gguf

# Run main application
cargo run --release
```

## Known Issues
- Forward pass returns placeholder (needs implementation)
- 127 compiler warnings to clean up
- Generation loop not connected

## Important Notes
1. **DO NOT** create new binaries or test scripts
2. **DO NOT** guess at Candle APIs - they often don't exist
3. **ALWAYS** apply metadata mapping before `ModelWeights::from_gguf()`
4. **USE** the corrected model dimensions (2048 hidden, not 3072)
5. **REMEMBER** mixed quantization: Q4K + Q6K tensors

## Success Criteria
When implementation is complete:
1. Model performs forward pass through all 36 layers
2. Generates coherent text token by token
3. Handles thinking mode with special tokens
4. Maintains conversation context with KV cache
5. Streams responses via SSE to web interface

## Reference Documentation
- Candle repo: https://github.com/huggingface/candle
- Specifically: `candle-transformers/src/models/quantized_llama.rs`
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

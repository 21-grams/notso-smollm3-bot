# System Architecture

## Overview

SmolLM3 Bot implements a three-tier architecture with clean separation between official Candle.rs operations, SmolLM3-specific features, and web infrastructure.

## Architecture Layers

### Layer 1: Official Candle (`/official`)
Pure Candle.rs implementations using only documented APIs.

```
src/services/ml/official/
├── gguf_loader.rs         # GGUF file parsing and metadata mapping
├── model.rs               # Wraps candle_transformers::models::quantized_llama
├── quantized_model.rs     # Direct QMatMul operations for Q4_K_M
├── config.rs              # LlamaConfig with SmolLM3 parameters
└── device.rs              # CUDA/CPU device management
```

**Key Responsibilities:**
- Load GGUF files with metadata mapping
- Use `ModelWeights::from_gguf()` directly
- Implement QMatMul operations without dequantization
- Manage device allocation

### Layer 2: SmolLM3 Extensions (`/smollm3`)
Model-specific features built on top of official layer.

```
src/services/ml/smollm3/
├── tokenizer_ext.rs       # Batch tokenization support
├── chat_template.rs        # External template application
├── generation.rs          # Token generation with buffering
├── thinking.rs            # <think> token detection
├── kv_cache.rs           # 128K context cache management
├── nope_layers.rs        # Skip position encoding (every 4th layer)
└── adapter.rs            # Bridge between layers
```

**Key Responsibilities:**
- Apply chat templates before inference
- Handle thinking mode at generation level
- Manage KV cache for long contexts
- Implement NoPE layer logic

### Layer 3: Web Infrastructure (`/web`)
User interface and streaming infrastructure.

```
src/web/
├── handlers/             # HTTP request handlers
├── templates/            # HTML templates
├── static/              # CSS, JavaScript
└── routes.rs            # Route configuration

src/services/
├── streaming/           # SSE streaming service
├── session.rs          # Session management
└── metrics.rs          # Performance monitoring
```

## Data Flow

### 1. Input Processing
```
User Input → Chat Template → Tokenizer → Token IDs
```

### 2. Model Inference
```
Token IDs → Official Model → QMatMul Ops → Logits
```

### 3. Generation Pipeline
```
Logits → Sampling → Token Buffer → Decode → Output
```

### 4. Response Streaming
```
Generated Tokens → SSE Stream → HTMX → Browser
```

## Key Design Patterns

### Direct Quantized Operations
```rust
// All Q4_K_M tensors use QMatMul directly
pub struct QuantizedLayer {
    q_attn: QMatMul,  // Attention weights
    q_ffn: QMatMul,   // Feed-forward weights
}

impl QuantizedLayer {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Direct operation without dequantization
        self.q_attn.forward(input)
    }
}
```

### Token Buffering
```rust
pub struct TokenBuffer {
    tokens: Vec<u32>,
    threshold: usize,  // Flush after N tokens
}

impl TokenBuffer {
    pub fn push(&mut self, token: u32) {
        self.tokens.push(token);
        if self.tokens.len() >= self.threshold {
            self.flush();
        }
    }
    
    pub fn flush(&mut self) -> String {
        let output = tokenizer.decode(&self.tokens);
        self.tokens.clear();
        output
    }
}
```

### KV Cache with Sliding Window
```rust
pub struct ExtendedKVCache {
    cache: Vec<(Tensor, Tensor)>,
    max_context: usize,    // 131072
    window_size: usize,    // 1024
}

impl ExtendedKVCache {
    pub fn update(&mut self, k: Tensor, v: Tensor, position: usize) {
        // Sliding window for long contexts
        if position > self.max_context - self.window_size {
            self.slide_window();
        }
        self.cache[position] = (k, v);
    }
}
```

## Model Specifications

### Tensor Mapping

| Tensor Name | Type | Quantization | Operation |
|-------------|------|--------------|-----------|
| `token_embd.weight` | Embedding | F32 | Direct |
| `layers.*.attn_q.weight` | Attention Q | Q4_K_M | QMatMul |
| `layers.*.attn_k.weight` | Attention K | Q4_K_M | QMatMul |
| `layers.*.attn_v.weight` | Attention V | Q4_K_M | QMatMul |
| `layers.*.attn_out.weight` | Attention Out | Q4_K_M | QMatMul |
| `layers.*.attn_norm.weight` | Layer Norm | F32 | Direct |
| `layers.*.ffn_gate.weight` | FFN Gate | Q4_K_M | QMatMul |
| `layers.*.ffn_down.weight` | FFN Down | Q4_K_M | QMatMul |
| `layers.*.ffn_up.weight` | FFN Up | Q4_K_M | QMatMul |
| `layers.*.ffn_norm.weight` | FFN Norm | F32 | Direct |
| `output.weight` | Output | Q4_K_M | QMatMul |
| `output_norm.weight` | Output Norm | F32 | Direct |

### Special Features

#### Grouped Query Attention (GQA)
- 16 attention heads
- 4 KV heads (4:1 ratio)
- 75% memory savings

#### NoPE Layers
- Layers 3, 7, 11, 15, 19, 23, 27, 31, 35
- Skip position encoding
- Content-based attention only

#### Thinking Mode
- Start token: `<think>` (128002)
- End token: `</think>` (128003)
- Handled at generation level

## Performance Considerations

### Memory Management
- Model: ~1.9GB (Q4_K_M)
- KV Cache: ~2GB (for 128K context)
- Total: < 4GB target

### Optimization Strategies
1. **Never dequantize** Q4_K_M tensors
2. **Buffer tokens** before decoding
3. **Use sliding window** for long contexts
4. **Batch operations** when possible
5. **Reuse tensors** to minimize allocation

### CUDA Acceleration
```bash
# Environment setup
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Build with CUDA
cargo build --release --features cuda
```

## Error Handling

### Graceful Fallbacks
1. **Model Loading Failure** → Stub mode
2. **CUDA Unavailable** → CPU inference
3. **OOM Error** → Reduce context window
4. **Tokenizer Error** → Default tokens

### Logging Strategy
```rust
tracing::info!("🚀 Starting inference");
tracing::debug!("Token: {} -> {}", id, text);
tracing::warn!("⚠️ Falling back to CPU");
tracing::error!("❌ Model loading failed: {}", e);
```

## Testing Strategy

### Unit Tests
- GGUF metadata parsing
- Tokenizer encode/decode
- QMatMul operations
- KV cache management

### Integration Tests
- Full inference pipeline
- Multi-turn conversations
- Thinking mode toggle
- Context overflow handling

### Performance Tests
- Token generation speed
- Memory usage monitoring
- Context length scaling
- Quantization accuracy

## Security Considerations

- Input sanitization for chat templates
- Token limit enforcement
- Memory bounds checking
- Session isolation

## Future Extensions

### Planned Features
- Batch inference support
- Streaming tokenization
- Tool calling integration
- Multi-model support

### Architecture Extensions
- Plugin system for custom features
- Distributed inference
- Model hot-swapping
- Fine-tuning support
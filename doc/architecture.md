# System Architecture

## Overview

NotSo-SmolLM3 Bot implements a production-ready SmolLM3 language model with a clean 3-tier architecture designed specifically for the model's unique features.

## SmolLM3 Model Architecture

SmolLM3 is a 3B parameter transformer model with several distinctive features:

- **Grouped Query Attention (GQA)**: 4 key-value groups vs 16 query heads (4:1 ratio) for 75% memory reduction
- **NoPE Layers**: No Position Encoding on layers [3,7,11,15,19,23,27,31,35] for better long-context performance
- **Thinking Mode**: Native `<think>` token support for chain-of-thought reasoning
- **Tool Calling**: Built-in function calling capabilities
- **128k Vocabulary**: Extended tokenizer for better multilingual support

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      Web Layer (HTMX)                       │
│            Neumorphic UI • SSE Streaming • API              │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│                    Service Layer                            │
│     Session Management • Orchestration • Templates          │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│                  ML Foundation Layer                        │
├──────────────┬──────────────────┬──────────────────────────┤
│   Official   │     SmolLM3      │      Streaming          │
│    Candle    │   Extensions     │       Pipeline          │
└──────────────┴──────────────────┴──────────────────────────┘
```

## Layer Details

### 1. Web Layer (`/src/web/`)
**Purpose**: User interface and HTTP handling

**Components**:
- **HTMX Interface**: Progressive enhancement without heavy JavaScript
- **SSE Streaming**: Real-time token streaming to browser
- **Neumorphic Design**: Minimalist UI with soft shadows and depth
- **API Endpoints**: RESTful + SSE for chat, thinking toggle, context tracking

**Key Files**:
- `handlers/chat.rs` - Message submission and response orchestration
- `handlers/sse.rs` - Server-sent event streaming
- `templates/chat.html` - Main chat interface
- `static/css/chat.css` - Neumorphic styling

### 2. Service Layer (`/src/services/`)
**Purpose**: Business logic and orchestration

**Components**:
- **ML Service**: High-level API for model interaction
- **Session Manager**: Conversation state and context tracking
- **Template Engine**: MiniJinja with markdown support
- **Response Buffer**: Token batching for efficient streaming

**Key Files**:
- `ml/service.rs` - MLService orchestrator
- `session/manager.rs` - Session lifecycle management
- `template/engine.rs` - Template rendering with filters

### 3. ML Foundation Layer (`/src/services/ml/`)

#### 3a. Official Candle (`/official/`)
**Purpose**: Pure Candle.rs foundation

**Components**:
- **Model Wrapper**: `OfficialSmolLM3Model` wrapping `quantized_llama`
- **GGUF Loader**: Direct model loading from quantized files
- **Device Manager**: Automatic CUDA/Metal/CPU selection
- **Configuration**: Model parameters and hyperparameters

**Key Integration Points**:
```rust
// Direct use of Candle's quantized operations
ModelWeights::from_gguf(path, device)?
QMatMul::forward(&weights, &input)?
```

#### 3b. SmolLM3 Extensions (`/smollm3/`)
**Purpose**: Model-specific features

**Components**:
- **KV Cache**: Optimized for 4-group GQA architecture
  - 75% memory reduction vs standard MHA
  - Layer-wise caching for 50-100x speedup after first token
- **Thinking Detector**: Handles `<think>` and `</think>` tokens
- **NoPE Handler**: Manages position encoding for specific layers
- **Generation Pipeline**: Token-by-token generation with streaming
- **Chat Templates**: Proper message formatting with role support

**SmolLM3-Specific Optimizations**:
```rust
// GQA-aware KV cache
pub struct KVCache {
    cache: HashMap<usize, (Tensor, Tensor)>, // 4 KV pairs vs 16
    // Optimized for SmolLM3's architecture
}

// Thinking mode detection
if token == THINK_TOKEN_ID {
    self.in_thinking_mode = true;
    // Route to thinking display
}
```

#### 3c. Streaming Pipeline (`/streaming/`)
**Purpose**: Real-time generation

**Components**:
- **Response Buffer**: Accumulates 5-10 tokens before sending
- **SSE Events**: Structured event types (token, status, thinking, tool_use)
- **Pipeline Orchestrator**: Coordinates generation → buffer → SSE flow

**Optimization Strategy**:
```rust
// Buffer tokens to reduce DOM updates
if buffer.len() >= 5 || elapsed > 100ms {
    flush_to_sse(buffer.drain().collect())
}
```

## Data Flow for SmolLM3 Inference

```
1. User Input
   ↓
2. Apply Chat Template
   "User: {input}\nAssistant:"
   ↓
3. Tokenization (128k vocab)
   [token_ids]
   ↓
4. Thinking Mode Check
   Insert <think> tokens if enabled
   ↓
5. Model Forward Pass
   - GQA attention (4 groups)
   - NoPE on specific layers
   - KV cache utilization
   ↓
6. Token Generation Loop
   - Sample from logits
   - Update KV cache
   - Stream via SSE
   ↓
7. Post-processing
   - Extract follow-ups
   - Format markdown
   - Update context count
```

## SmolLM3-Specific Design Decisions

### 1. **GQA-Optimized Caching**
The KV cache is specifically designed for SmolLM3's 4-group architecture, storing only 4 key-value pairs per layer instead of 16, directly mapping to the model's GQA design.

### 2. **Thinking Mode Integration**
Native support for SmolLM3's thinking tokens, with special UI treatment to show/hide thinking process based on user preference.

### 3. **NoPE Layer Handling**
Layers [3,7,11,15,19,23,27,31,35] bypass position encoding as per SmolLM3's architecture for improved long-context performance.

### 4. **Streaming Optimization**
Token buffering tuned for SmolLM3's generation speed (1-2 tok/s target) to balance responsiveness with DOM efficiency.

### 5. **Quantization Strategy**
Q4_K_M quantization chosen specifically for SmolLM3's size (3B params) to fit in consumer GPU memory while maintaining quality.

## Memory Layout

```
Model Weights (Q4_K_M):  ~1.5GB
KV Cache (GQA):          ~200MB (for 2K context)
Tensors/Buffers:         ~300MB
-----------------------------------
Total GPU Memory:        ~2GB

Traditional MHA would require ~800MB for KV cache
SmolLM3 GQA saves 75% on cache memory
```

## Performance Targets

- **First Token Latency**: <500ms
- **Generation Speed**: 1-2 tokens/second
- **KV Cache Speedup**: 50-100x after first token
- **Memory Usage**: <2GB GPU RAM
- **Context Length**: 2048 tokens (expandable to 32K)

## Error Handling Strategy

1. **Model Loading Failures**: Fallback to stub mode
2. **OOM Errors**: Clear KV cache and retry with smaller context
3. **Generation Errors**: Return partial response with error indicator
4. **Timeout**: 30s generation limit with graceful termination

## Security Considerations

- **Input Sanitization**: All user inputs HTML-escaped
- **Token Limits**: Hard cap at 2048 tokens per request
- **Rate Limiting**: Per-session generation limits
- **No Code Execution**: Tool calls simulated, not executed

## Future Architecture Enhancements

1. **Multi-Model Support**: Abstract model interface for different variants
2. **Distributed Inference**: Model parallelism across GPUs
3. **Persistent KV Cache**: Redis-backed cache for session continuity
4. **Dynamic Batching**: Process multiple requests simultaneously
5. **Quantization Levels**: Runtime selection of Q4/Q8/F16 based on hardware
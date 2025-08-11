# Context for Next Conversation: Implementing Generation Loop

## Project Overview
We're building **notso-smollm3-bot**, a high-performance Rust chatbot using SmolLM3-3B (Q4_K_M/Q6_K quantized) with NoPE layers. The model successfully loads, and we need to implement the generation loop.

## Current Status
✅ **Model Loading Works**: NoPE-aware model loads all 326 tensors successfully
✅ **Architecture Established**: Three-layer separation (Official/SmolLM3/Web)
✅ **Web Interface Ready**: HTMX + SSE streaming functional
❌ **Generation Not Implemented**: `generate_streaming` returns "not yet implemented"

## Architecture Strategy (CRITICAL)

### Three-Layer Modular Design
```
┌─────────────────────────────────────────┐
│          Web Layer (Axum + HTMX)       │
│  - Handles HTTP, SSE streaming         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      SmolLM3 Layer (Custom Features)    │
│  - NoPE layers, Thinking mode, Cache   │
│  - Generation loop, Sampling            │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│    Official Layer (Pure Candle.rs)      │
│  - ONLY documented Candle APIs         │
│  - No custom modifications             │
└─────────────────────────────────────────┘
```

### Key Design Decisions
1. **Official Layer**: Uses ONLY documented Candle.rs APIs - no hacks or private APIs
2. **SmolLM3 Layer**: All model-specific features go here (NoPE, thinking, generation)
3. **Clean Interfaces**: Each layer has clear boundaries and typed interfaces
4. **No Mixing**: Never put SmolLM3-specific code in the official layer

## Model Architecture Details

### NoPE Implementation
- **Model Type**: Custom `NopeModel` in `src/services/ml/smollm3/nope_model.rs`
- **NoPE Layers**: [3, 7, 11, 15, 19, 23, 27, 31, 35] skip position encoding
- **Working**: Model loads and forward pass structure exists

### Model Specifications
- **Layers**: 36 transformer blocks
- **Hidden Size**: 2048
- **Vocab Size**: 128,256
- **Context**: 65,536 tokens max
- **Attention**: 16 heads, 4 KV heads (GQA 4:1)
- **Quantization**: Mixed (216 Q4_K_M, 37 Q6_K, 73 F32 tensors)
- **Tied Embeddings**: Output shares weights with input

### Critical Files
```
src/services/ml/
├── service.rs              # MLService with generate_streaming() stub
├── smollm3/
│   ├── nope_model.rs      # NopeModel::forward() works
│   ├── generation.rs      # Empty generation loop
│   ├── tokenizer_ext.rs  # SmolLM3Tokenizer wrapper
│   └── kv_cache.rs        # KV cache structure
└── official/
    └── [Candle-only code]
```

## What Needs Implementation

### 1. Generation Loop (`service.rs`)
```rust
pub async fn generate_streaming(
    &mut self,
    prompt: &str,
    buffer: &mut StreamingBuffer,
) -> Result<()> {
    // TODO: Implement this
    // 1. Tokenize prompt
    // 2. Run forward pass
    // 3. Sample from logits
    // 4. Stream tokens
    // 5. Update KV cache
}
```

### 2. Key Challenges
- **Position Tracking**: Start at 0, jump to prompt_len, then increment
- **KV Cache**: Currently inside NopeModel, needs proper update logic
- **Sampling**: Need temperature, top-p, repetition penalty
- **Streaming**: Connect to SSE buffer for real-time output
- **Stop Tokens**: Handle EOS (128001) and thinking tokens (128070/128071)

### 3. Tokenizer Integration
- Tokenizer exists at `models/tokenizer.json`
- `SmolLM3Tokenizer` wrapper needs completion
- Special tokens: `<think>` (128070), `</think>` (128071)

## Technical Constraints

### Must Follow
1. **No unsafe code** unless absolutely necessary
2. **Keep official layer pure** - only documented Candle APIs
3. **Maintain type safety** - leverage Rust's type system
4. **Handle errors gracefully** - no panics in production paths
5. **Stream incrementally** - don't buffer entire response

### Performance Targets
- Token generation: 1-2 tokens/second minimum
- Memory usage: Stay under 4GB total
- First token latency: < 2 seconds
- Streaming: Real-time token delivery

## Generation Algorithm Outline

```rust
// Pseudo-code for generation
let tokens = tokenizer.encode(prompt)?;
let mut position = 0;

// Process prompt
let prompt_tensor = Tensor::new(&tokens, device)?;
let logits = model.forward(&prompt_tensor, position)?;
position = tokens.len();

// Generation loop
for _ in 0..max_tokens {
    // Sample next token
    let next_token = sample_from_logits(&logits, temperature, top_p)?;
    
    // Check stop conditions
    if next_token == EOS_TOKEN { break; }
    
    // Decode and stream
    let text = tokenizer.decode(&[next_token])?;
    buffer.push(&text).await?;
    
    // Forward pass for next token
    let input = Tensor::new(&[next_token], device)?;
    logits = model.forward(&input, position)?;
    position += 1;
}
```

## Questions to Address
1. Should KV cache be managed inside NopeModel or outside?
2. How to handle batch processing (currently batch_size=1)?
3. Where should sampling logic live (in service.rs or generation.rs)?
4. How to implement repetition penalty efficiently?
5. Should we pre-allocate tensors for performance?

## Success Criteria
- Generate coherent text from a prompt
- Stream tokens in real-time to web interface
- Properly handle NoPE layers during generation
- Stop at EOS token
- Memory usage stays stable during long generations

## Reference Code Locations
- Forward pass: `src/services/ml/smollm3/nope_model.rs:forward()`
- Tokenizer: `src/services/ml/smollm3/tokenizer_ext.rs`
- Streaming buffer: `src/services/streaming/buffer.rs`
- Current stub: `src/services/ml/service.rs:generate_streaming()`

## Start Point
Begin by implementing the basic generation loop in `generate_streaming()`, focusing on getting a single token to generate correctly before adding streaming and optimization.

---

**Your task**: Implement the generation loop while maintaining the clean architecture separation and following the established patterns. The NoPE model loads successfully - now make it generate text!

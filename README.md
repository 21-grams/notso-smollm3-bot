# SmolLM3 Bot - notso-smollm3-bot

A high-performance Rust chatbot implementing SmolLM3-3B (Q4_K_M quantized) with real-time streaming via HTMX SSE, using the latest Candle.rs ecosystem (0.9.1+).

## 🎯 Project Goal

Build a fully-featured inference engine for SmolLM3-3B with:
- **Direct quantized operations** (Q4_K_M) for 50-100x speedup
- **Thinking mode** with `<think>` tokens for chain-of-thought reasoning
- **128K context support** with efficient KV cache
- **Real-time streaming** via Server-Sent Events
- **Clean architecture** separating official Candle from SmolLM3 features

## 📊 Current Status

**Version**: 0.7.0  
**Date**: 2025-01-17  
**Phase**: Q4_K_M Loading Complete ✅

### ✅ Complete
- **Web Infrastructure**: Axum 0.8 server with HTMX SSE streaming
- **UI/UX**: Beautiful chat interface with markdown rendering
- **Session Management**: Multi-session support with UUID v7
- **GGUF Inspector**: Tool to analyze model quantization and metadata
- **Q4_K_M Support**: ✅ **FULLY VERIFIED** in Candle 0.9.1
- **Metadata Mapping**: ✅ SmolLM3 → Llama format working
- **Model Loading**: ✅ `ModelWeights::from_gguf()` successful
- **Memory Efficiency**: ✅ Weights stay quantized (1.78GB file → 2.9GB in memory)

### 🚧 In Progress
- **Forward Pass**: Implementing actual inference through the model
- **Generation Loop**: Token-by-token generation with logits processing
- **KV Cache**: Implementation for 65536 context length

### ⏳ Next Steps
1. Implement forward pass using ModelWeights
2. Complete generation loop with sampling
3. Add KV cache for conversation context
4. Integrate thinking mode (`<think>` tokens)

## 🛠️ Development Tools

### Inspection Tools
Two binary tools are provided for development:

```bash
# Inspect GGUF file structure and metadata
cargo run --bin inspect_gguf

# Test Q4_K support in Candle
cargo run --bin test_q4k
```

### Key Findings
- ✅ GGUF file contains 326 tensors (216 Q4K, 37 Q6K, 73 F32)
- ✅ **Model dimensions corrected**:
  - Hidden size: **2048** (not 3072)
  - Intermediate: **11008** (not 8192)
  - Heads: **16** (not 32)
  - KV heads: **4** (not 8)
- ✅ **Metadata mapping working**: SmolLM3 → Llama format
- ✅ **Q4_K_M fully supported** via `GgmlDType::Q4K`
- ✅ **Efficient loading**: 1.78GB file uses ~2.9GB memory (reasonable)

## 🏗️ Architecture

### Clean Layer Separation
```
src/services/ml/
├── official/           # Pure Candle.rs implementations
│   ├── gguf_loader.rs     # GGUF → QTensor loading
│   ├── model.rs           # Wraps quantized_llama
│   ├── quantized_model.rs # Direct QMatMul operations
│   └── config.rs          # Model configuration
│
├── smollm3/           # SmolLM3-specific features
│   ├── tokenizer_ext.rs   # Batch tokenization
│   ├── chat_template.rs   # Template application
│   ├── generation.rs      # Token generation
│   ├── thinking.rs        # Thinking mode
│   └── kv_cache.rs        # 128K context cache
│
└── service.rs         # Orchestration layer
```

### Key Design Principles
- **Official layer**: Uses ONLY documented Candle APIs
- **SmolLM3 layer**: Adds model-specific features
- **No dequantization**: Direct Q4_K_M operations throughout
- **Token buffering**: Efficient batch processing

## 🔧 Setup

### Prerequisites
- Rust 1.75+ 
- CUDA toolkit 12.x (optional, for GPU)
- 8GB+ RAM
- Model files in `/models`:
  - `HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/21-grams/notso-smollm3-bot
cd notso-smollm3-bot
```

2. Verify model files:
```bash
ls -la models/
# Should show GGUF file (~1.9GB) and tokenizer files
```

3. Build and run:
```bash
cargo build --release
cargo run --release
```

4. Open browser to `http://localhost:3000`

### CUDA Setup (Optional)
```bash
# Run setup script
./scripts/setup_cuda.sh

# Build with CUDA support
cargo build --release --features cuda
```

## 🎯 Technical Requirements

### Model Specifications
- **Architecture**: SmolLM3-3B
- **Quantization**: Q4_K_M (~1.9GB)
- **Layers**: 36
- **Attention**: 16 heads (4 KV heads for GQA)
- **Hidden Size**: 3072
- **Vocab Size**: 128256
- **Context**: Up to 131072 tokens

### Performance Targets
- **Speed**: 1-2 tokens/second minimum
- **Memory**: < 4GB total usage
- **Latency**: < 50ms per token
- **Context**: 128K with sliding window

## 📖 Documentation

### Technical Documents
- [Technical Requirements](doc/technical_requirements.md) - Detailed specifications
- [Implementation Status](doc/implementation_status.md) - Current progress
- [GGUF Integration](doc/gguf_integration_status.md) - Model loading details
- [Architecture](doc/architecture.md) - System design

### API References
- [Candle Reference](doc/candle_reference.md) - Candle.rs patterns
- [Model Loading](doc/model_loading_reference.md) - GGUF tensor mapping

## 🚀 Roadmap

### Phase 1: Core Implementation (Current)
- [x] Web infrastructure
- [x] Architecture design
- [ ] GGUF inspection tool
- [ ] Q4_K support verification
- [ ] Basic tokenizer loading
- [ ] Model loading with QMatMul

### Phase 2: Features
- [ ] Chat template application
- [ ] Generation loop
- [ ] Thinking mode (`<think>` tokens)
- [ ] KV cache for conversations
- [ ] Batch tokenization

### Phase 3: Optimization
- [ ] CUDA acceleration
- [ ] 128K context support
- [ ] Performance tuning
- [ ] Production deployment

### Future Features
- [ ] Batch inference
- [ ] Streaming tokenization
- [ ] Pause/resume inference
- [ ] Tool calling support

## 🤝 Collaboration Guidelines

### Development Rules
- **Build**: `cargo run` - Create setup scripts for dependencies
- **Testing**: Unit tests for core features only
- **Documentation**: Use `///` comments, maintain `/doc` folder
- **Safety**: Pure safe Rust preferred, justify `unsafe` blocks
- **Architecture**: Maintain official/smollm3 separation strictly

### Contribution Process
1. Check existing issues
2. Follow Rust best practices
3. Update documentation
4. Test thoroughly
5. Submit clear PR

## 🔍 Technical Highlights

### Q4_K_M Loading with Proper Typing
```rust
use candle_core::quantized::{GgmlDType, QTensor, QMatMul};

// Load Q4_K_M tensor from GGUF
let qtensor = QTensor::from_ggml(
    GgmlDType::Q4K,  // Q4_K_M format
    &raw_data,        // Quantized bytes
    &dims             // Tensor shape
)?;

// Create QMatMul for efficient operations
let qmatmul = QMatMul::from_qtensor(qtensor)?;

// Forward pass - weights stay quantized!
let output = qmatmul.forward(&input_f32)?;
```

### Memory Efficiency Verified
```rust
// Q4_K_M: ~3.8GB for 3B model (vs 12GB unquantized)
// Block structure: 32 weights → 144 bytes
// Effective: ~4.5 bits per weight
```

### Efficient Token Buffering
```rust
// Collect tokens before decoding
let mut token_buffer = Vec::new();
for _ in 0..max_tokens {
    token_buffer.push(generate_token()?);
}
// Decode once
let output = tokenizer.decode(&token_buffer)?;
```

## 🐛 Known Issues

- Forward pass returns placeholder (generation loop incomplete)
- 127 compiler warnings to clean up
- KV cache not yet integrated
- CUDA features not tested

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Candle.rs team for the ML ecosystem
- HuggingFace for SmolLM3 model
- HTMX team for the framework

---

**Project Status**: 🟢 Active Development (55% Complete)

**Critical Next Steps**:
1. ✅ ~~Verify Candle Q4_K support~~ **COMPLETE**
2. ✅ ~~Create GGUF inspection tool~~ **COMPLETE**
3. ✅ ~~Implement proper model loading~~ **COMPLETE**
4. ✅ ~~Fix metadata mapping~~ **COMPLETE**
5. 🚧 Implement forward pass and generation loop

For questions or contributions, please open an issue on [GitHub](https://github.com/21-grams/notso-smollm3-bot).
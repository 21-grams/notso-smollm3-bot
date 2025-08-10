# SmolLM3 Bot - notso-smollm3-bot

A high-performance Rust chatbot implementing SmolLM3-3B (Q4_K_M quantized) with real-time streaming via HTMX SSE, using the latest Candle.rs ecosystem (0.9.1+).

## 🎯 Project Goal

Build a fully-featured inference engine for SmolLM3-3B with:
- **Direct quantized operations** (Q4_K_M) for 50-100x speedup
- **Thinking mode** with `<think>` tokens
- **128K context support** with KV cache
- **Real-time streaming** via Server-Sent Events
- **Clean architecture** separating official Candle from SmolLM3 features

## 📊 Current Status

**Version**: 0.4.0  
**Date**: 2025-01-17  
**Phase**: Model Integration (Architecture Complete)

### Working ✅
- Web server with Axum 0.8
- HTMX SSE streaming interface
- Beautiful chat UI with markdown
- Session management
- Stub mode for testing

### In Progress 🚧
- GGUF model loading with Q4_K_M support
- Tokenizer integration
- Generation pipeline

### Pending ❌
- Direct QMatMul operations
- KV cache implementation
- CUDA acceleration

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

### Phase 2: Features (Week 1-2)
- [ ] Chat template application
- [ ] Generation loop
- [ ] Thinking mode (`<think>` tokens)
- [ ] KV cache for conversations
- [ ] Batch tokenization

### Phase 3: Optimization (Week 3)
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

### Direct Quantized Operations
```rust
// ✅ CORRECT - 50-100x faster
let qmatmul = QMatMul::from_qtensor(&qtensor)?;
let result = qmatmul.forward(&input)?;

// ❌ WRONG - Never dequantize
let float = qtensor.dequantize(&device)?;
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

## 📈 Performance Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Token Speed | 1-2 tok/s | N/A | ❌ |
| Memory Usage | <4GB | ~500MB | ⚠️ |
| Context Length | 128K | N/A | ❌ |
| Quantization | Q4_K_M | N/A | ❌ |

## 🐛 Known Issues

- Model loading incomplete (metadata mapping needed)
- Q4_K support not verified in Candle
- Forward pass returns placeholder
- 127 compiler warnings to clean up

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Candle.rs team for the ML ecosystem
- HuggingFace for SmolLM3 model
- HTMX team for the framework

---

**Project Status**: 🟡 Active Development (30% Complete)

**Critical Next Steps**:
1. Verify Candle Q4_K support
2. Create GGUF inspection tool
3. Implement proper model loading
4. Connect tokenizer to inference

For questions or contributions, please open an issue on [GitHub](https://github.com/21-grams/notso-smollm3-bot).
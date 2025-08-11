# SmolLM3 Bot - notso-smollm3-bot

A high-performance Rust chatbot implementing SmolLM3-3B (Q4_K_M quantized) with NoPE (No Position Encoding) layers for content-based attention, using the latest Candle.rs ecosystem (>0.9.1).

## 📋 Table of Contents
- [Project Objectives](#-project-objectives)
- [SmolLM3 Model Features](#-smollm3-model-features)
- [Architecture Strategy](#-architecture-strategy)
- [Implementation Features](#-implementation-features)
- [System Requirements](#-system-requirements)
- [Installation & Setup](#-installation--setup)
- [Development Tools](#-development-tools)
- [Project Structure](#-project-structure)
- [Technical Reference](#-technical-reference)
- [Contributing](#-contributing)

## 🎯 Project Objectives

### Primary Goals
1. **Implement SmolLM3-3B with NoPE layers** - Enable content-based attention on specific layers
2. **Leverage Candle.rs ecosystem** - Use official Candle tools for tensor operations
3. **Maintain clean architecture** - Separate official Candle usage from SmolLM3-specific features
4. **Achieve production performance** - Direct quantized operations for 50-100x speedup
5. **Support extended context** - Handle up to 128K tokens with efficient KV cache

### Design Principles
- **Modularity**: Clear separation between layers (official/SmolLM3/web)
- **Performance**: Quantized operations without dequantization where possible
- **Maintainability**: Type-safe Rust with comprehensive error handling
- **Extensibility**: Easy to add new features without breaking existing code

## 🤖 SmolLM3 Model Features

### Model Architecture
SmolLM3-3B is a transformer-based language model with unique architectural innovations:

- **Parameters**: 3 billion
- **Layers**: 36 transformer blocks
- **Hidden Size**: 2048
- **Intermediate Size**: 11008
- **Attention Heads**: 16 (with 4 KV heads for GQA)
- **Vocabulary**: 128,256 tokens
- **Context Length**: 65,536 tokens (extended from base)
- **RoPE Theta**: 5,000,000 (for extended context)

### NoPE (No Position Encoding) Innovation
SmolLM3 introduces selective position encoding:
- **NoPE Layers**: [3, 7, 11, 15, 19, 23, 27, 31, 35]
- **Pattern**: Every 4th layer starting from layer 3
- **Purpose**: Enable content-based attention without positional bias
- **Benefit**: Better understanding of semantic relationships

### Quantization Strategy
- **Q4_K_M**: Primary quantization for weights (4-bit)
- **Q6_K**: Select layers for higher precision
- **F32**: Normalization layers remain unquantized
- **Mixed Precision**: Optimal balance of speed and quality

### Special Features
- **Thinking Mode**: `<think>` and `</think>` tokens for chain-of-thought
- **Tied Embeddings**: Shared weights between input and output layers
- **GQA (Grouped Query Attention)**: 4:1 KV head ratio for efficiency

## 🏗️ Architecture Strategy

### Three-Layer Architecture

```
┌─────────────────────────────────────────┐
│          Web Layer (Axum + HTMX)       │
│  - HTTP server, SSE streaming, UI      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      SmolLM3 Layer (Custom Features)    │
│  - NoPE layers, Thinking mode, Cache   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│    Official Layer (Pure Candle.rs)      │
│  - Tensor ops, GGUF loading, Quant     │
└─────────────────────────────────────────┘
```

### Official Layer (Candle.rs Tools)
Uses only documented Candle.rs APIs:
- `candle_core`: Core tensor operations
- `candle_nn`: Neural network layers
- `candle_transformers`: Transformer components
- `tokenizers`: HuggingFace tokenizers
- GGUF support for quantized models

### SmolLM3 Layer (Model-Specific)
Implements SmolLM3-specific features:
- NoPE layer selection logic
- Thinking mode token handling
- Custom KV cache for 128K context
- Chat template application
- Generation strategies

### Web Layer (User Interface)
Modern web stack for interaction:
- Axum 0.8 for HTTP server
- HTMX for dynamic updates
- SSE (Server-Sent Events) for streaming
- Markdown rendering for formatted output

## 🚀 Implementation Features

### Core Features
- ✅ **NoPE-aware model**: Selective RoPE application
- ✅ **GGUF loading**: Direct quantized model loading
- ✅ **Mixed quantization**: Q4_K_M, Q6_K, F32 support
- ✅ **Tied embeddings**: Weight sharing for efficiency
- ✅ **Web interface**: Full chat UI with streaming
- ✅ **Session management**: Multi-user support
- 🚧 **Generation loop**: Token-by-token generation
- 🚧 **Thinking mode**: CoT reasoning support
- 🚧 **KV cache**: 128K context handling

### Technical Capabilities
- **Direct quantized operations**: No unnecessary dequantization
- **Streaming responses**: Real-time token generation
- **Markdown support**: Rich text formatting
- **Slash commands**: `/quote`, `/status`, etc.
- **Debug mode**: Comprehensive logging

## 💻 System Requirements

### Minimum Requirements
- **CPU**: x86_64 processor with AVX2 support
- **RAM**: 8GB (model uses ~3GB)
- **Storage**: 5GB free space
- **OS**: Linux, macOS, or Windows with WSL2

### Recommended Requirements
- **CPU**: Modern multi-core processor
- **RAM**: 16GB for comfortable operation
- **GPU**: NVIDIA with 8GB+ VRAM (optional)
- **Storage**: SSD for faster model loading

### Development Requirements
- **Rust**: 1.75 or newer
- **CUDA**: 12.x toolkit (optional, for GPU)
- **Git**: For version control
- **Build tools**: C++ compiler, CMake

## 🔧 Installation & Setup

### 1. Prerequisites

#### Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

#### Install CUDA (Optional, for GPU)
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

### 2. Clone Repository
```bash
git clone https://github.com/21-grams/notso-smollm3-bot
cd notso-smollm3-bot
```

### 3. Download Model Files

Create `models` directory and download required files:
```bash
mkdir -p models
cd models

# Download quantized model (1.78GB)
wget https://huggingface.co/HuggingFaceTB/SmolLM3-3B-GGUF/resolve/main/SmolLM3-3B-Q4_K_M.gguf

# Download tokenizer files
wget https://huggingface.co/HuggingFaceTB/SmolLM3-3B/resolve/main/tokenizer.json
wget https://huggingface.co/HuggingFaceTB/SmolLM3-3B/resolve/main/tokenizer_config.json
wget https://huggingface.co/HuggingFaceTB/SmolLM3-3B/resolve/main/special_tokens_map.json

cd ..
```

### 4. Build and Run

#### CPU Version
```bash
cargo build --release
cargo run --release
```

#### GPU Version (CUDA)
```bash
cargo build --release --features cuda
cargo run --release --features cuda
```

### 5. Access the Interface
Open your browser and navigate to:
```
http://localhost:3000
```

### Configuration

Environment variables (optional):
```bash
# Server configuration
export HOST=0.0.0.0
export PORT=3000

# Model paths (defaults shown)
export MODEL_PATH=models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf
export TOKENIZER_PATH=models/tokenizer.json

# Device selection
export DEVICE=cuda  # or cpu, metal

# Logging
export RUST_LOG=info  # or debug for verbose output
```

## 🛠️ Development Tools

### Binary Tools

#### GGUF Inspector
Analyze model file structure and metadata:
```bash
cargo run --bin inspect_gguf [path_to_gguf]
```

Output includes:
- Tensor count and types
- Quantization distribution
- Metadata mapping requirements
- Memory usage estimates

#### Q4K Test Tool
Verify Candle's quantization support:
```bash
cargo run --bin test_q4k
```

### Debug Mode
Enable detailed logging:
```bash
RUST_LOG=debug cargo run
```

Provides:
- Tensor loading progress
- NoPE layer detection
- Memory allocation tracking
- Generation pipeline steps

### Performance Profiling
```bash
cargo build --release --features profiling
CARGO_PROFILE_RELEASE_DEBUG=true cargo run --release
```

## 📁 Project Structure

```
notso-smollm3-bot/
├── Cargo.toml                 # Dependencies and features
├── README.md                  # This file
├── PROJECT_STATUS.md          # Implementation progress
│
├── src/
│   ├── main.rs               # Application entry point
│   ├── lib.rs                # Library exports
│   ├── config.rs             # Configuration management
│   ├── state.rs              # Application state
│   │
│   ├── services/
│   │   ├── ml/               # Machine learning service
│   │   │   ├── official/     # Candle.rs integrations
│   │   │   │   ├── model.rs         # Model wrapper
│   │   │   │   ├── gguf_loader.rs   # GGUF file handling
│   │   │   │   └── config.rs        # Model configuration
│   │   │   │
│   │   │   ├── smollm3/      # SmolLM3-specific
│   │   │   │   ├── nope_model.rs    # NoPE implementation
│   │   │   │   ├── tokenizer_ext.rs # Extended tokenizer
│   │   │   │   ├── thinking.rs      # Thinking mode
│   │   │   │   ├── generation.rs    # Generation loop
│   │   │   │   └── kv_cache.rs      # Context cache
│   │   │   │
│   │   │   └── service.rs    # ML service orchestration
│   │   │
│   │   ├── streaming/        # SSE streaming service
│   │   ├── session.rs        # Session management
│   │   └── template/         # Template rendering
│   │
│   ├── web/
│   │   ├── server.rs         # Axum server setup
│   │   ├── routes.rs         # Route definitions
│   │   ├── handlers/         # Request handlers
│   │   ├── middleware.rs     # Custom middleware
│   │   └── static/           # CSS, JS assets
│   │
│   └── types/                # Shared types
│       ├── message.rs        # Message structures
│       ├── events.rs         # Event definitions
│       └── errors.rs         # Error types
│
├── models/                   # Model files (not in git)
│   ├── *.gguf               # Quantized model
│   └── tokenizer.json       # Tokenizer config
│
├── doc/                      # Documentation
│   ├── architecture.md      # System design
│   ├── nope_implementation.md # NoPE details
│   └── api.md               # API reference
│
└── scripts/                  # Utility scripts
    ├── setup_cuda.sh        # CUDA setup
    └── download_models.sh   # Model download
```

## 📚 Technical Reference

### Model Configuration
```rust
pub struct SmolLM3Config {
    pub num_hidden_layers: usize,        // 36
    pub num_attention_heads: usize,      // 16
    pub num_key_value_heads: usize,      // 4
    pub hidden_size: usize,              // 2048
    pub intermediate_size: usize,        // 11008
    pub vocab_size: usize,               // 128256
    pub max_position_embeddings: usize,  // 65536
    pub rope_theta: f32,                 // 5000000.0
    pub head_dim: usize,                 // 128
    pub nope_layer_indices: Vec<usize>,  // [3,7,11,15,19,23,27,31,35]
}
```

### GGUF Tensor Naming
- **Embeddings**: `token_embd.weight`
- **Layers**: `blk.{i}.*`
- **Attention**: `blk.{i}.attn_{q,k,v,output}.weight`
- **MLP**: `blk.{i}.ffn_{gate,up,down}.weight`
- **Norms**: `blk.{i}.{attn,ffn}_norm.weight`
- **Output**: `output_norm.weight` (lm_head tied to embeddings)

### API Endpoints
- `GET /` - Web interface
- `GET /health` - Health check
- `POST /api/chat` - Send message
- `GET /api/stream/{session_id}` - SSE stream
- `POST /api/command` - Slash commands

### Performance Metrics
- **Model Load Time**: ~5-6 seconds
- **Memory Usage**: ~2.9GB active
- **Token Generation**: 1-2 tokens/second (target)
- **Context Window**: 65,536 tokens
- **Batch Size**: 1 (current), expandable

🔍 SmolLM3 GGUF Inspector v1.0
================================

Reading GGUF file: models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf


## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install pre-commit hooks: `cargo install cargo-husky`
4. Make changes with tests
5. Submit pull request

### Key Areas for Contribution
- Generation loop optimization
- Batch processing support
- Additional quantization formats
- Web UI enhancements
- Documentation improvements
- Test coverage expansion

### Code Style
- Follow Rust standard formatting: `cargo fmt`
- Ensure clippy passes: `cargo clippy`
- Add documentation comments for public APIs
- Include unit tests for new features

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **SmolLM3 Team** - For the innovative NoPE architecture
- **Candle.rs Team** - For the excellent tensor framework
- **Hugging Face** - For model hosting and tokenizers
- **Community Contributors** - For feedback and improvements

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/21-grams/notso-smollm3-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/21-grams/notso-smollm3-bot/discussions)
- **Documentation**: [Project Wiki](https://github.com/21-grams/notso-smollm3-bot/wiki)

---

For implementation details and progress, see [PROJECT_STATUS.md](PROJECT_STATUS.md)


================================================================================
                    GGUF INSPECTION REPORT
================================================================================

📁 FILE INFORMATION
----------------------------------------
Path: models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf
Size: 1.78 GB
Total tensors: 326
Total metadata: 32

🏗️  ARCHITECTURE
----------------------------------------
Architecture: smollm3
Vocab size: Some(128256)
Hidden size: Some(2048)
Number of layers: Some(36)

📊 TENSOR QUANTIZATION REPORT
----------------------------------------
Q4_K_M tensors (quantized): 216
F32 tensors (not quantized): 73
Other tensor types: 37

  Q4_K_M Tensors (MUST use QMatMul):
    - blk.0.attn_k.weight: shape [512, 2048]
    - blk.0.attn_output.weight: shape [2048, 2048]
    - blk.0.attn_q.weight: shape [2048, 2048]
    - blk.0.ffn_gate.weight: shape [11008, 2048]
    - blk.0.ffn_up.weight: shape [11008, 2048]
    - blk.1.attn_k.weight: shape [512, 2048]
    - blk.1.attn_output.weight: shape [2048, 2048]
    - blk.1.attn_q.weight: shape [2048, 2048]
    - blk.1.ffn_gate.weight: shape [11008, 2048]
    - blk.1.ffn_up.weight: shape [11008, 2048]
    - blk.10.attn_k.weight: shape [512, 2048]
    - blk.10.attn_output.weight: shape [2048, 2048]
    - blk.10.attn_q.weight: shape [2048, 2048]
    - blk.10.attn_v.weight: shape [512, 2048]
    - blk.10.ffn_down.weight: shape [2048, 11008]
    - blk.10.ffn_gate.weight: shape [11008, 2048]
    - blk.10.ffn_up.weight: shape [11008, 2048]
    - blk.11.attn_k.weight: shape [512, 2048]
    - blk.11.attn_output.weight: shape [2048, 2048]
    - blk.11.attn_q.weight: shape [2048, 2048]
    - blk.11.attn_v.weight: shape [512, 2048]
    - blk.11.ffn_down.weight: shape [2048, 11008]
    - blk.11.ffn_gate.weight: shape [11008, 2048]
    - blk.11.ffn_up.weight: shape [11008, 2048]
    - blk.12.attn_k.weight: shape [512, 2048]
    - blk.12.attn_output.weight: shape [2048, 2048]
    - blk.12.attn_q.weight: shape [2048, 2048]
    - blk.12.ffn_gate.weight: shape [11008, 2048]
    - blk.12.ffn_up.weight: shape [11008, 2048]
    - blk.13.attn_k.weight: shape [512, 2048]
    - blk.13.attn_output.weight: shape [2048, 2048]
    - blk.13.attn_q.weight: shape [2048, 2048]
    - blk.13.attn_v.weight: shape [512, 2048]
    - blk.13.ffn_down.weight: shape [2048, 11008]
    - blk.13.ffn_gate.weight: shape [11008, 2048]
    - blk.13.ffn_up.weight: shape [11008, 2048]
    - blk.14.attn_k.weight: shape [512, 2048]
    - blk.14.attn_output.weight: shape [2048, 2048]
    - blk.14.attn_q.weight: shape [2048, 2048]
    - blk.14.attn_v.weight: shape [512, 2048]
    - blk.14.ffn_down.weight: shape [2048, 11008]
    - blk.14.ffn_gate.weight: shape [11008, 2048]
    - blk.14.ffn_up.weight: shape [11008, 2048]
    - blk.15.attn_k.weight: shape [512, 2048]
    - blk.15.attn_output.weight: shape [2048, 2048]
    - blk.15.attn_q.weight: shape [2048, 2048]
    - blk.15.ffn_gate.weight: shape [11008, 2048]
    - blk.15.ffn_up.weight: shape [11008, 2048]
    - blk.16.attn_k.weight: shape [512, 2048]
    - blk.16.attn_output.weight: shape [2048, 2048]
    - blk.16.attn_q.weight: shape [2048, 2048]
    - blk.16.attn_v.weight: shape [512, 2048]
    - blk.16.ffn_down.weight: shape [2048, 11008]
    - blk.16.ffn_gate.weight: shape [11008, 2048]
    - blk.16.ffn_up.weight: shape [11008, 2048]
    - blk.17.attn_k.weight: shape [512, 2048]
    - blk.17.attn_output.weight: shape [2048, 2048]
    - blk.17.attn_q.weight: shape [2048, 2048]
    - blk.17.attn_v.weight: shape [512, 2048]
    - blk.17.ffn_down.weight: shape [2048, 11008]
    - blk.17.ffn_gate.weight: shape [11008, 2048]
    - blk.17.ffn_up.weight: shape [11008, 2048]
    - blk.18.attn_k.weight: shape [512, 2048]
    - blk.18.attn_output.weight: shape [2048, 2048]
    - blk.18.attn_q.weight: shape [2048, 2048]
    - blk.18.ffn_gate.weight: shape [11008, 2048]
    - blk.18.ffn_up.weight: shape [11008, 2048]
    - blk.19.attn_k.weight: shape [512, 2048]
    - blk.19.attn_output.weight: shape [2048, 2048]
    - blk.19.attn_q.weight: shape [2048, 2048]
    - blk.19.attn_v.weight: shape [512, 2048]
    - blk.19.ffn_down.weight: shape [2048, 11008]
    - blk.19.ffn_gate.weight: shape [11008, 2048]
    - blk.19.ffn_up.weight: shape [11008, 2048]
    - blk.2.attn_k.weight: shape [512, 2048]
    - blk.2.attn_output.weight: shape [2048, 2048]
    - blk.2.attn_q.weight: shape [2048, 2048]
    - blk.2.ffn_gate.weight: shape [11008, 2048]
    - blk.2.ffn_up.weight: shape [11008, 2048]
    - blk.20.attn_k.weight: shape [512, 2048]
    - blk.20.attn_output.weight: shape [2048, 2048]
    - blk.20.attn_q.weight: shape [2048, 2048]
    - blk.20.attn_v.weight: shape [512, 2048]
    - blk.20.ffn_down.weight: shape [2048, 11008]
    - blk.20.ffn_gate.weight: shape [11008, 2048]
    - blk.20.ffn_up.weight: shape [11008, 2048]
    - blk.21.attn_k.weight: shape [512, 2048]
    - blk.21.attn_output.weight: shape [2048, 2048]
    - blk.21.attn_q.weight: shape [2048, 2048]
    - blk.21.ffn_gate.weight: shape [11008, 2048]
    - blk.21.ffn_up.weight: shape [11008, 2048]
    - blk.22.attn_k.weight: shape [512, 2048]
    - blk.22.attn_output.weight: shape [2048, 2048]
    - blk.22.attn_q.weight: shape [2048, 2048]
    - blk.22.attn_v.weight: shape [512, 2048]
    - blk.22.ffn_down.weight: shape [2048, 11008]
    - blk.22.ffn_gate.weight: shape [11008, 2048]
    - blk.22.ffn_up.weight: shape [11008, 2048]
    - blk.23.attn_k.weight: shape [512, 2048]
    - blk.23.attn_output.weight: shape [2048, 2048]
    - blk.23.attn_q.weight: shape [2048, 2048]
    - blk.23.attn_v.weight: shape [512, 2048]
    - blk.23.ffn_down.weight: shape [2048, 11008]
    - blk.23.ffn_gate.weight: shape [11008, 2048]
    - blk.23.ffn_up.weight: shape [11008, 2048]
    - blk.24.attn_k.weight: shape [512, 2048]
    - blk.24.attn_output.weight: shape [2048, 2048]
    - blk.24.attn_q.weight: shape [2048, 2048]
    - blk.24.ffn_gate.weight: shape [11008, 2048]
    - blk.24.ffn_up.weight: shape [11008, 2048]
    - blk.25.attn_k.weight: shape [512, 2048]
    - blk.25.attn_output.weight: shape [2048, 2048]
    - blk.25.attn_q.weight: shape [2048, 2048]
    - blk.25.attn_v.weight: shape [512, 2048]
    - blk.25.ffn_down.weight: shape [2048, 11008]
    - blk.25.ffn_gate.weight: shape [11008, 2048]
    - blk.25.ffn_up.weight: shape [11008, 2048]
    - blk.26.attn_k.weight: shape [512, 2048]
    - blk.26.attn_output.weight: shape [2048, 2048]
    - blk.26.attn_q.weight: shape [2048, 2048]
    - blk.26.attn_v.weight: shape [512, 2048]
    - blk.26.ffn_down.weight: shape [2048, 11008]
    - blk.26.ffn_gate.weight: shape [11008, 2048]
    - blk.26.ffn_up.weight: shape [11008, 2048]
    - blk.27.attn_k.weight: shape [512, 2048]
    - blk.27.attn_output.weight: shape [2048, 2048]
    - blk.27.attn_q.weight: shape [2048, 2048]
    - blk.27.ffn_gate.weight: shape [11008, 2048]
    - blk.27.ffn_up.weight: shape [11008, 2048]
    - blk.28.attn_k.weight: shape [512, 2048]
    - blk.28.attn_output.weight: shape [2048, 2048]
    - blk.28.attn_q.weight: shape [2048, 2048]
    - blk.28.attn_v.weight: shape [512, 2048]
    - blk.28.ffn_down.weight: shape [2048, 11008]
    - blk.28.ffn_gate.weight: shape [11008, 2048]
    - blk.28.ffn_up.weight: shape [11008, 2048]
    - blk.29.attn_k.weight: shape [512, 2048]
    - blk.29.attn_output.weight: shape [2048, 2048]
    - blk.29.attn_q.weight: shape [2048, 2048]
    - blk.29.attn_v.weight: shape [512, 2048]
    - blk.29.ffn_down.weight: shape [2048, 11008]
    - blk.29.ffn_gate.weight: shape [11008, 2048]
    - blk.29.ffn_up.weight: shape [11008, 2048]
    - blk.3.attn_k.weight: shape [512, 2048]
    - blk.3.attn_output.weight: shape [2048, 2048]
    - blk.3.attn_q.weight: shape [2048, 2048]
    - blk.3.ffn_gate.weight: shape [11008, 2048]
    - blk.3.ffn_up.weight: shape [11008, 2048]
    - blk.30.attn_k.weight: shape [512, 2048]
    - blk.30.attn_output.weight: shape [2048, 2048]
    - blk.30.attn_q.weight: shape [2048, 2048]
    - blk.30.ffn_gate.weight: shape [11008, 2048]
    - blk.30.ffn_up.weight: shape [11008, 2048]
    - blk.31.attn_k.weight: shape [512, 2048]
    - blk.31.attn_output.weight: shape [2048, 2048]
    - blk.31.attn_q.weight: shape [2048, 2048]
    - blk.31.ffn_gate.weight: shape [11008, 2048]
    - blk.31.ffn_up.weight: shape [11008, 2048]
    - blk.32.attn_k.weight: shape [512, 2048]
    - blk.32.attn_output.weight: shape [2048, 2048]
    - blk.32.attn_q.weight: shape [2048, 2048]
    - blk.32.ffn_gate.weight: shape [11008, 2048]
    - blk.32.ffn_up.weight: shape [11008, 2048]
    - blk.33.attn_k.weight: shape [512, 2048]
    - blk.33.attn_output.weight: shape [2048, 2048]
    - blk.33.attn_q.weight: shape [2048, 2048]
    - blk.33.ffn_gate.weight: shape [11008, 2048]
    - blk.33.ffn_up.weight: shape [11008, 2048]
    - blk.34.attn_k.weight: shape [512, 2048]
    - blk.34.attn_output.weight: shape [2048, 2048]
    - blk.34.attn_q.weight: shape [2048, 2048]
    - blk.34.ffn_gate.weight: shape [11008, 2048]
    - blk.34.ffn_up.weight: shape [11008, 2048]
    - blk.35.attn_k.weight: shape [512, 2048]
    - blk.35.attn_output.weight: shape [2048, 2048]
    - blk.35.attn_q.weight: shape [2048, 2048]
    - blk.35.ffn_gate.weight: shape [11008, 2048]
    - blk.35.ffn_up.weight: shape [11008, 2048]
    - blk.4.attn_k.weight: shape [512, 2048]
    - blk.4.attn_output.weight: shape [2048, 2048]
    - blk.4.attn_q.weight: shape [2048, 2048]
    - blk.4.attn_v.weight: shape [512, 2048]
    - blk.4.ffn_down.weight: shape [2048, 11008]
    - blk.4.ffn_gate.weight: shape [11008, 2048]
    - blk.4.ffn_up.weight: shape [11008, 2048]
    - blk.5.attn_k.weight: shape [512, 2048]
    - blk.5.attn_output.weight: shape [2048, 2048]
    - blk.5.attn_q.weight: shape [2048, 2048]
    - blk.5.attn_v.weight: shape [512, 2048]
    - blk.5.ffn_down.weight: shape [2048, 11008]
    - blk.5.ffn_gate.weight: shape [11008, 2048]
    - blk.5.ffn_up.weight: shape [11008, 2048]
    - blk.6.attn_k.weight: shape [512, 2048]
    - blk.6.attn_output.weight: shape [2048, 2048]
    - blk.6.attn_q.weight: shape [2048, 2048]
    - blk.6.ffn_gate.weight: shape [11008, 2048]
    - blk.6.ffn_up.weight: shape [11008, 2048]
    - blk.7.attn_k.weight: shape [512, 2048]
    - blk.7.attn_output.weight: shape [2048, 2048]
    - blk.7.attn_q.weight: shape [2048, 2048]
    - blk.7.attn_v.weight: shape [512, 2048]
    - blk.7.ffn_down.weight: shape [2048, 11008]
    - blk.7.ffn_gate.weight: shape [11008, 2048]
    - blk.7.ffn_up.weight: shape [11008, 2048]
    - blk.8.attn_k.weight: shape [512, 2048]
    - blk.8.attn_output.weight: shape [2048, 2048]
    - blk.8.attn_q.weight: shape [2048, 2048]
    - blk.8.attn_v.weight: shape [512, 2048]
    - blk.8.ffn_down.weight: shape [2048, 11008]
    - blk.8.ffn_gate.weight: shape [11008, 2048]
    - blk.8.ffn_up.weight: shape [11008, 2048]
    - blk.9.attn_k.weight: shape [512, 2048]
    - blk.9.attn_output.weight: shape [2048, 2048]
    - blk.9.attn_q.weight: shape [2048, 2048]
    - blk.9.ffn_gate.weight: shape [11008, 2048]
    - blk.9.ffn_up.weight: shape [11008, 2048]

  F32 Tensors (not quantized):
    - blk.0.attn_norm.weight: shape [2048]
    - blk.0.ffn_norm.weight: shape [2048]
    - blk.1.attn_norm.weight: shape [2048]
    - blk.1.ffn_norm.weight: shape [2048]
    - blk.10.attn_norm.weight: shape [2048]
    - blk.10.ffn_norm.weight: shape [2048]
    - blk.11.attn_norm.weight: shape [2048]
    - blk.11.ffn_norm.weight: shape [2048]
    - blk.12.attn_norm.weight: shape [2048]
    - blk.12.ffn_norm.weight: shape [2048]
    - blk.13.attn_norm.weight: shape [2048]
    - blk.13.ffn_norm.weight: shape [2048]
    - blk.14.attn_norm.weight: shape [2048]
    - blk.14.ffn_norm.weight: shape [2048]
    - blk.15.attn_norm.weight: shape [2048]
    - blk.15.ffn_norm.weight: shape [2048]
    - blk.16.attn_norm.weight: shape [2048]
    - blk.16.ffn_norm.weight: shape [2048]
    - blk.17.attn_norm.weight: shape [2048]
    - blk.17.ffn_norm.weight: shape [2048]
    - blk.18.attn_norm.weight: shape [2048]
    - blk.18.ffn_norm.weight: shape [2048]
    - blk.19.attn_norm.weight: shape [2048]
    - blk.19.ffn_norm.weight: shape [2048]
    - blk.2.attn_norm.weight: shape [2048]
    - blk.2.ffn_norm.weight: shape [2048]
    - blk.20.attn_norm.weight: shape [2048]
    - blk.20.ffn_norm.weight: shape [2048]
    - blk.21.attn_norm.weight: shape [2048]
    - blk.21.ffn_norm.weight: shape [2048]
    - blk.22.attn_norm.weight: shape [2048]
    - blk.22.ffn_norm.weight: shape [2048]
    - blk.23.attn_norm.weight: shape [2048]
    - blk.23.ffn_norm.weight: shape [2048]
    - blk.24.attn_norm.weight: shape [2048]
    - blk.24.ffn_norm.weight: shape [2048]
    - blk.25.attn_norm.weight: shape [2048]
    - blk.25.ffn_norm.weight: shape [2048]
    - blk.26.attn_norm.weight: shape [2048]
    - blk.26.ffn_norm.weight: shape [2048]
    - blk.27.attn_norm.weight: shape [2048]
    - blk.27.ffn_norm.weight: shape [2048]
    - blk.28.attn_norm.weight: shape [2048]
    - blk.28.ffn_norm.weight: shape [2048]
    - blk.29.attn_norm.weight: shape [2048]
    - blk.29.ffn_norm.weight: shape [2048]
    - blk.3.attn_norm.weight: shape [2048]
    - blk.3.ffn_norm.weight: shape [2048]
    - blk.30.attn_norm.weight: shape [2048]
    - blk.30.ffn_norm.weight: shape [2048]
    - blk.31.attn_norm.weight: shape [2048]
    - blk.31.ffn_norm.weight: shape [2048]
    - blk.32.attn_norm.weight: shape [2048]
    - blk.32.ffn_norm.weight: shape [2048]
    - blk.33.attn_norm.weight: shape [2048]
    - blk.33.ffn_norm.weight: shape [2048]
    - blk.34.attn_norm.weight: shape [2048]
    - blk.34.ffn_norm.weight: shape [2048]
    - blk.35.attn_norm.weight: shape [2048]
    - blk.35.ffn_norm.weight: shape [2048]
    - blk.4.attn_norm.weight: shape [2048]
    - blk.4.ffn_norm.weight: shape [2048]
    - blk.5.attn_norm.weight: shape [2048]
    - blk.5.ffn_norm.weight: shape [2048]
    - blk.6.attn_norm.weight: shape [2048]
    - blk.6.ffn_norm.weight: shape [2048]
    - blk.7.attn_norm.weight: shape [2048]
    - blk.7.ffn_norm.weight: shape [2048]
    - blk.8.attn_norm.weight: shape [2048]
    - blk.8.ffn_norm.weight: shape [2048]
    - blk.9.attn_norm.weight: shape [2048]
    - blk.9.ffn_norm.weight: shape [2048]
    - output_norm.weight: shape [2048]

  Other Tensor Types:
    - blk.0.attn_v.weight: Q6K shape [512, 2048]
    - blk.0.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.1.attn_v.weight: Q6K shape [512, 2048]
    - blk.1.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.12.attn_v.weight: Q6K shape [512, 2048]
    - blk.12.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.15.attn_v.weight: Q6K shape [512, 2048]
    - blk.15.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.18.attn_v.weight: Q6K shape [512, 2048]
    - blk.18.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.2.attn_v.weight: Q6K shape [512, 2048]
    - blk.2.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.21.attn_v.weight: Q6K shape [512, 2048]
    - blk.21.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.24.attn_v.weight: Q6K shape [512, 2048]
    - blk.24.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.27.attn_v.weight: Q6K shape [512, 2048]
    - blk.27.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.3.attn_v.weight: Q6K shape [512, 2048]
    - blk.3.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.30.attn_v.weight: Q6K shape [512, 2048]
    - blk.30.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.31.attn_v.weight: Q6K shape [512, 2048]
    - blk.31.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.32.attn_v.weight: Q6K shape [512, 2048]
    - blk.32.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.33.attn_v.weight: Q6K shape [512, 2048]
    - blk.33.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.34.attn_v.weight: Q6K shape [512, 2048]
    - blk.34.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.35.attn_v.weight: Q6K shape [512, 2048]
    - blk.35.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.6.attn_v.weight: Q6K shape [512, 2048]
    - blk.6.ffn_down.weight: Q6K shape [2048, 11008]
    - blk.9.attn_v.weight: Q6K shape [512, 2048]
    - blk.9.ffn_down.weight: Q6K shape [2048, 11008]
    - token_embd.weight: Q6K shape [128256, 2048]

🦙 SMOLLM3 METADATA
----------------------------------------
  smollm3.embedding_length: 2048
  smollm3.block_count: 36
  smollm3.feed_forward_length: 11008
  smollm3.rope.dimension_count: 128
  smollm3.attention.head_count_kv: 4
  smollm3.context_length: 65536
  smollm3.attention.head_count: 16
  smollm3.attention.layer_norm_rms_epsilon: 0.000001
  smollm3.rope.freq_base: 5000000
  smollm3.vocab_size: 128256

🦙 LLAMA METADATA STATUS
----------------------------------------

❌ Missing Llama keys:
  - llama.attention.head_count
  - llama.attention.head_count_kv
  - llama.block_count
  - llama.context_length
  - llama.embedding_length
  - llama.feed_forward_length
  - llama.vocab_size
  - llama.rope.freq_base
  - llama.rope.dimension_count
  - llama.attention.layer_norm_rms_epsilon

🔄 SUGGESTED MAPPINGS
----------------------------------------
  smollm3.attention.head_count → llama.attention.head_count
  smollm3.attention.head_count_kv → llama.attention.head_count_kv
  smollm3.block_count → llama.block_count
  smollm3.context_length → llama.context_length
  smollm3.embedding_length → llama.embedding_length
  smollm3.feed_forward_length → llama.feed_forward_length
  smollm3.vocab_size → llama.vocab_size
  smollm3.rope.dimension_count → llama.rope.dimension_count
  smollm3.attention.layer_norm_rms_epsilon → llama.attention.layer_norm_rms_epsilon

📝 SUMMARY
----------------------------------------
✅ Valid GGUF file
✅ 326 tensors found (216 quantized, 73 F32)
================================================================================


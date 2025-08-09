# NotSo-SmolLM3 Bot

A high-performance, production-ready chatbot implementation of SmolLM3-3B using pure Rust and the Candle machine learning framework. This project demonstrates efficient quantized inference with streaming responses, all running natively without Python dependencies.

## 🎯 Project Goal

To create a fully-featured SmolLM3 inference system in Rust that:
- Runs Q4_K_M quantized models efficiently on consumer hardware
- Provides real-time token streaming with intelligent buffering
- Supports SmolLM3's unique features (thinking mode, GQA, NoPE layers)
- Delivers a smooth chat experience through a clean web interface
- Serves as a reference implementation for Rust-based LLM inference

## 🎯 Target Model

**SmolLM3-3B Q4_K_M GGUF** - Specifically optimized for:
- **4-bit quantization** with K-means clustering for quality preservation
- **~1.5GB model size** - fits in consumer GPU memory
- **Direct quantized operations** - no dequantization overhead
- **Efficient inference** - maintains original model's optimization intent

The Q4_K_M format provides the best balance of model quality, inference speed, and memory efficiency for the 3B parameter SmolLM3 model.

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  Web Layer (Axum 0.8 + HTMX 2.0.6)         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • HTTP Routes      • Persistent SSE per Session    │   │
│  │  • Chat UI          • Slash Commands                │   │
│  │  • MiniJinja 2      • Progressive Markdown          │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────┬───────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────┐
│                     Service Layer                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • Session Manager (Single Receiver Pattern)        │   │
│  │  • Unified StreamingBuffer (500ms/10 tokens)        │   │
│  │  • Template Engine with Custom Filters              │   │
│  │  • Event Pipeline (mpsc channels)                   │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────┬───────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────┐
│                    ML Foundation Layer                      │
│  ┌──────────────────┬──────────────────┬──────────────┐   │
│  │  Official Candle │  SmolLM3 Adapter │  Quantized   │   │
│  │  Integration     │  & Extensions    │  Operations  │   │
│  │                  │                  │              │   │
│  │  • GGUF Loader   │  • Thinking Mode │  • Q4_K_M    │   │
│  │  • Model Weights │  • KV Cache (GQA)│  • Direct    │   │
│  │  • Forward Pass  │  • NoPE Layers   │    MatMul    │   │
│  │  • Device Mgmt   │  • Chat Template │  • No Deq.   │   │
│  └──────────────────┴──────────────────┴──────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
src/
├── main.rs                 # Application entry point
├── config.rs              # Global configuration
├── state.rs               # Application state management
├── lib.rs                 # Library exports
│
├── web/                   # Web Layer
│   ├── handlers/
│   │   ├── api.rs        # Main API endpoints & command routing
│   │   ├── chat.rs       # Chat page rendering
│   │   ├── commands.rs   # Slash command handlers
│   │   └── health.rs     # Health check endpoint
│   ├── routes.rs         # Route definitions
│   ├── templates/
│   │   └── chat.html     # Single unified chat template
│   └── static/
│       ├── css/          # Neumorphic UI styles
│       └── js/           # Slash commands & chat logic
│
├── services/              # Service Layer
│   ├── ml/
│   │   ├── service.rs    # ML service orchestration
│   │   ├── official/     # Candle integration
│   │   │   ├── model.rs  # SmolLM3 model wrapper
│   │   │   ├── gguf_loader.rs  # GGUF metadata mapping
│   │   │   └── config.rs # Model configuration
│   │   └── smollm3/      # SmolLM3-specific features
│   │       ├── adapter.rs       # Model adapter
│   │       ├── chat_template.rs # Chat formatting
│   │       ├── thinking.rs      # Thinking mode
│   │       ├── kv_cache.rs      # GQA-optimized cache
│   │       ├── nope_layers.rs   # Position encoding
│   │       └── generation.rs    # Token generation
│   ├── streaming/
│   │   ├── buffer.rs     # Unified streaming buffer
│   │   └── sse_handler.rs # SSE event handling
│   ├── session.rs        # Session management
│   └── template/         # Template rendering
│
└── types/                 # Shared types
    ├── events.rs         # Stream event types
    ├── message.rs        # Message structures
    └── session.rs        # Session types
```

## 🔄 Data Flow

### 1. **User Input Flow**
```
User Input → Web Handler → Command Detection → Service Layer
                                ↓
                        Regular Chat / Slash Command
                                ↓
                         Unified Processing
```

### 2. **Streaming Response Flow**
```
Model/Command Output → Streaming Buffer → SSE Events → Client
         ↓                    ↓              ↓           ↓
   Token Generation    Accumulate (10)   "message"   Progressive
                          or 500ms        event      Markdown
```

### 3. **Session & Streaming Architecture**

#### Session Management
- **Single Receiver Pattern**: Each session creates one receiver, taken once by SSE
- **Persistent SSE**: One connection per session, auto-reconnects with `Last-Event-ID`
- **Clean Separation**: Senders remain with sessions, receivers consumed by SSE endpoints

#### Buffer Strategy
- Accumulates tokens until threshold (10 tokens or 500ms)
- Reduces DOM updates for smooth user experience
- Single buffer implementation for all output types
- Natural boundary detection (sentences, paragraphs)
- Automatic completion signaling with metadata

## 🚀 Key Features

### Latest Technology Stack (August 2025)
- **Rust**: Stable 1.80+ with async/await, Arc<RwLock> patterns
- **Axum 0.8**: Latest with `/{param}` syntax, improved SSE, graceful shutdown
- **HTMX 2.0.6**: Core + SSE extension 2.2.2, morphing support
- **MiniJinja 2.11+**: Template inheritance, custom filters, safe rendering
- **UUID v7**: Sortable, timestamp-based message IDs

### SmolLM3-Specific Optimizations
- **Grouped Query Attention (GQA)**: 4:1 KV head ratio for 75% memory savings
- **NoPE Layers**: Layers [3,7,11,15,19,23,27,31,35] skip position encoding
- **Thinking Mode**: Native `<think>` token support for reasoning
- **128k Vocabulary**: Extended tokenizer for better coverage

### Technical Highlights
- **Pure Rust**: No Python dependencies, fully native
- **Q4_K_M Quantization**: 4-bit weights with minimal quality loss
- **Direct Quantized Ops**: No dequantization overhead
- **Unified Streaming**: Single pipeline for all content types
- **Persistent SSE**: One connection per session with auto-reconnect
- **Progressive Markdown**: Real-time rendering during streaming
- **Template Components**: Modular MiniJinja templates for messages

## 🏭 Modular Design Philosophy

### Leveraging Official Candle Ecosystem
The project maximizes use of official Candle (v0.9.1+) and Tokenizers (v0.21+) crates:

```rust
// Official Candle components we build upon:
candle_core::quantized::gguf_file     // GGUF file handling
candle_core::quantized::QMatMul       // Quantized matrix operations
candle_transformers::models::quantized_llama  // Base model architecture
tokenizers::Tokenizer                 // HuggingFace tokenizers
```

### SmolLM3 Adaptation Layer
Model-specific features are cleanly isolated in `src/services/ml/smollm3/`:
- **Extends** official Candle without modifying it
- **Adapts** generic Llama to SmolLM3's architecture
- **Preserves** quantization throughout inference
- **Contained** in a single directory for maintainability

### Quantization Preservation
```rust
// We maintain quantized weights throughout:
QTensor → QMatMul → Quantized Output
        ↑
   No dequantization!
```

This design ensures:
1. **Maximum compatibility** with Candle updates
2. **Optimal performance** from quantized operations
3. **Clean separation** of concerns
4. **Easy maintenance** as Candle evolves

## 🛠️ Setup & Installation

### Prerequisites
- Rust 1.75+ (for async trait support)
- 4GB+ RAM
- (Optional) CUDA 12+ or Metal for GPU acceleration

### Required Model Files
The system specifically requires:
- **Model**: `SmolLM3-3B-Q4_K_M.gguf` (~1.5GB)
- **Tokenizer**: `tokenizer.json` from SmolLM3-3B
- **Format**: GGUF with Q4_K_M quantization (4-bit)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/21-grams/notso-smollm3-bot.git
cd notso-smollm3-bot

# Download model files
cd models
wget https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Q4_K_M.gguf
wget https://huggingface.co/HuggingFaceTB/SmolLM3-3B/tokenizer.json
cd ..

# Build and run
cargo run --release

# For GPU support
cargo run --release --features cuda  # NVIDIA
cargo run --release --features metal # Apple Silicon
```

Access the chat interface at `http://localhost:3000`

## 📊 Performance Targets

- **First Token Latency**: < 500ms
- **Generation Speed**: 1-2 tokens/second
- **Memory Usage**: ~2GB with Q4_K_M (1.5GB model + overhead)
- **Context Length**: 2048 tokens (expandable to 32K)
- **Streaming Buffer**: 10 tokens or 500ms flush
- **SSE Keep-Alive**: 30-second intervals
- **Quantization**: Maintained throughout inference (no dequantization)
- **KV Cache**: 75% memory savings with GQA optimization

## 🔧 Configuration

Key settings in `config.rs`:
- Model paths and device selection
- Temperature and sampling parameters  
- Buffer thresholds (500ms, 10 tokens)
- Thinking mode defaults
- Session timeout and cleanup
- SSE keep-alive intervals

## 🧪 Testing

The `/quote` command serves as an integration test for the streaming pipeline:
- Tests buffer accumulation and flushing
- Validates SSE event flow
- Demonstrates markdown rendering
- Confirms proper stream completion

## 📚 Documentation

Detailed technical documentation is available in the `/doc` directory:
- `architecture.md` - System design and component details
- `latest-tech-stack-2025.md` - Current versions and features reference
- `gguf_integration_status.md` - GGUF metadata mapping
- `streaming-architecture.md` - SSE and buffer design

## 📝 License

MIT

## 🙏 Acknowledgments

- [Candle](https://github.com/huggingface/candle) - Rust ML framework
- [HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) - SmolLM3 models
- [Axum](https://github.com/tokio-rs/axum) - Web framework
- [HTMX](https://htmx.org) - Hypermedia-driven UI

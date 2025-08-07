# NotSo-SmolLM3 Bot

Production-ready SmolLM3 chatbot using Candle.rs with a clean 3-tier architecture and neumorphic UI.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Web Layer (HTMX)                      │
│                 Neumorphic UI, SSE Streaming             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  Service Layer                           │
│         ML Orchestration, Session Management             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│               ML Foundation Layer                        │
│    ┌──────────────┐  ┌─────────────┐  ┌──────────────┐ │
│    │   Official   │  │   SmolLM3   │  │  Streaming   │ │
│    │    Candle    │  │ Extensions  │  │   Pipeline   │ │
│    └──────────────┘  └─────────────┘  └──────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## 📁 Project Structure & File Descriptions

```
notso-smollm3-bot/
├── doc/                           # Documentation
│   ├── architecture.md           # System design and architecture decisions
│   ├── implementation_status.md  # Current progress and roadmap
│   ├── candle_reference.md       # Candle.rs patterns and examples
│   └── ui_ux_interaction.md      # UI/UX flow documentation
│
├── models/                        # Model storage (gitignored)
│   ├── download.sh               # Script to download GGUF models
│   └── [*.gguf, *.json]         # Model and tokenizer files
│
├── src/
│   ├── web/                      # Web Layer - Self-contained UI
│   │   ├── static/
│   │   │   ├── css/
│   │   │   │   ├── main.css    # Global neumorphic design system
│   │   │   │   └── chat.css    # Chat-specific neumorphic styles
│   │   │   └── js/
│   │   │       └── chat.js     # HTMX SSE handlers, markdown rendering
│   │   │
│   │   ├── templates/
│   │   │   ├── base.html       # Base template with layout
│   │   │   ├── chat.html       # Main chat interface
│   │   │   └── partials/       # Reusable components
│   │   │
│   │   ├── handlers/            # HTTP Request Handlers
│   │   │   ├── mod.rs          # Module exports
│   │   │   ├── chat.rs         # Chat endpoints (POST /api/chat)
│   │   │   ├── api.rs          # API endpoints (thinking toggle, context)
│   │   │   ├── sse.rs          # SSE streaming endpoint
│   │   │   └── health.rs       # Health check endpoint
│   │   │
│   │   ├── server.rs            # Axum server configuration
│   │   ├── routes.rs            # Route definitions and middleware
│   │   └── mod.rs               # Web module exports
│   │
│   ├── services/                 # Service Layer - Business Logic
│   │   ├── ml/                  # Machine Learning Services
│   │   │   │
│   │   │   ├── official/        # Official Candle Foundation
│   │   │   │   ├── model.rs    # OfficialSmolLM3Model - quantized_llama wrapper
│   │   │   │   ├── config.rs   # SmolLM3Config - model parameters
│   │   │   │   ├── loader.rs   # OfficialLoader - GGUF file loading
│   │   │   │   ├── device.rs   # DeviceManager - CUDA/CPU detection
│   │   │   │   └── mod.rs      # Module exports
│   │   │   │
│   │   │   ├── smollm3/         # SmolLM3-Specific Extensions
│   │   │   │   ├── adapter.rs      # SmolLM3Adapter - bridges to official
│   │   │   │   ├── generation.rs   # SmolLM3Generator - token generation
│   │   │   │   ├── thinking.rs     # ThinkingDetector - <think> token handling
│   │   │   │   ├── kv_cache.rs     # KVCache - 4-group GQA optimization
│   │   │   │   ├── nope_layers.rs  # NopeHandler - NoPE layer management
│   │   │   │   ├── tokenizer_ext.rs # SmolLM3TokenizerExt - chat templates
│   │   │   │   ├── stub_mode.rs    # StubModeService - testing without models
│   │   │   │   └── mod.rs          # Module exports
│   │   │   │
│   │   │   ├── streaming/       # Real-time Streaming
│   │   │   │   ├── buffer.rs   # ResponseBuffer - token batching
│   │   │   │   ├── events.rs   # StreamEvent - SSE event types
│   │   │   │   ├── pipeline.rs # StreamingPipeline - orchestration
│   │   │   │   └── mod.rs      # Module exports
│   │   │   │
│   │   │   ├── service.rs      # MLService - high-level orchestration
│   │   │   └── mod.rs          # ML module exports
│   │   │
│   │   ├── template/            # Template Rendering
│   │   │   ├── engine.rs       # TemplateEngine - MiniJinja setup
│   │   │   ├── filters.rs      # Custom filters (markdown, datetime)
│   │   │   └── mod.rs          # Module exports
│   │   │
│   │   ├── session/             # Session Management
│   │   │   ├── manager.rs      # SessionManager - in-memory sessions
│   │   │   ├── store.rs        # SessionStore - persistence layer
│   │   │   └── mod.rs          # Module exports
│   │   │
│   │   └── mod.rs               # Services module exports
│   │
│   ├── types/                   # Shared Type Definitions
│   │   ├── events.rs           # StreamEvent, GenerationEvent types
│   │   ├── message.rs          # Message, ChatMessage structures
│   │   ├── session.rs          # Session, ChatSession types
│   │   ├── config.rs           # AppConfig, ModelConfig types
│   │   └── mod.rs              # Type exports
│   │
│   ├── config.rs                # Application configuration
│   ├── state.rs                 # AppState - shared application state
│   ├── lib.rs                   # Library exports
│   └── main.rs                  # Entry point - server initialization
│
├── tests/                        # Test Suite
│   ├── integration/             # Integration tests
│   └── unit/                    # Unit tests
│
├── Cargo.toml                   # Project dependencies and metadata
├── Cargo.lock                   # Locked dependency versions
├── README.md                    # This file
├── build.sh                     # Production build script
├── run.sh                       # Run script with environment setup
└── test_build.sh                # Quick test build script
```

## 🎯 Logical Organization

### **Separation of Concerns**

1. **Web Layer** (`/src/web`)
   - Self-contained UI with all assets
   - Handles HTTP requests and responses
   - Manages SSE streaming connections
   - No business logic, only presentation

2. **Service Layer** (`/src/services`)
   - Business logic and orchestration
   - ML model management
   - Session and state management
   - Template rendering

3. **ML Foundation** (`/src/services/ml`)
   - **Official**: Pure Candle.rs wrappers
   - **SmolLM3**: Model-specific features
   - **Streaming**: Real-time generation

### **Key Design Principles**

- **Modularity**: Each component has a single responsibility
- **Layered Architecture**: Clear boundaries between layers
- **Dependency Injection**: Services receive dependencies via constructors
- **Type Safety**: Strong typing throughout with shared type definitions
- **Progressive Enhancement**: HTMX for interactivity without heavy JS

## 🚀 Quick Start

```bash
# Setup
chmod +x *.sh
./test_build.sh  # Verify structure

# Download models (optional)
cd models && ./download.sh && cd ..

# Build & Run
cargo build --release
./run.sh

# Visit http://localhost:3000
```

## 🔧 Configuration

### Model Specifications
- **Architecture**: SmolLM3-3B with 4-group GQA
- **Quantization**: Q4_K_M (4-bit)
- **Context**: 2048 tokens (expandable to 32K)
- **Layers**: 36 with NoPE on layers [3,7,11,15,19,23,27,31,35]
- **Performance**: Target 1-2 tok/s

### Environment Variables
```bash
RUST_LOG=info                  # Logging level
MODEL_PATH=models/model.gguf   # Path to GGUF model
TOKENIZER_PATH=models/tokenizer.json  # Path to tokenizer
PORT=3000                       # Server port
```

## 📊 Implementation Status

### ✅ Complete
- Web UI with neumorphic design
- HTMX + SSE streaming
- Service layer architecture
- ML foundation structure
- Session management
- Template engine

### 🚧 In Progress
- Model integration
- Generation pipeline
- Tool use system
- Follow-up suggestions

### 📋 Planned
- Context management
- Performance optimization
- Persistent sessions
- Multi-model support

## 📚 Documentation

- `/doc/architecture.md` - System design decisions
- `/doc/implementation_status.md` - Detailed progress tracking
- `/doc/candle_reference.md` - Candle.rs usage patterns
- `/doc/ui_ux_interaction.md` - User interaction flow

## 📄 License

MIT
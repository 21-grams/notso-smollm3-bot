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
│             Inference Foundation Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Low-level   │  │  Generation  │  │   Services   │  │
│  │   Candle     │  │     Loop     │  │  ML/SmolLM3  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## 🚀 Latest Update (Phase: Web Server Route Fix)

### **Fixed Route Conflict Issue** ✅
Resolved the Axum router conflict where `/static` route was being registered twice:

**Problem**: 
```
thread 'main' panicked at src/web/server.rs:22:10:
Invalid route "/static/{*__private__axum_nest_tail_param}":
Insertion failed due to conflict with previously registered route
```

**Root Cause**:
- Static file route was defined in both `routes.rs` and `server.rs`
- Axum doesn't allow duplicate route patterns

**Solution**:
1. Removed duplicate `/static` registration from `src/web/routes.rs`
2. Kept single registration in `src/web/server.rs` for clarity
3. Added documentation comments for middleware stack order
4. Created `/doc/routing-architecture.md` for routing best practices

### **Architecture Improvements**
- **Clear Separation**: Static files served at app level, application routes in dedicated module
- **Single Responsibility**: Each module handles specific route types
- **Documentation**: Added routing architecture guide for future reference

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
│   ├── inference/                # Inference Foundation Layer [NEW]
│   │   ├── candle/              # Low-level Candle operations
│   │   │   ├── device.rs        # Device management & memory tracking
│   │   │   ├── kv_cache.rs      # KV cache tensor operations
│   │   │   ├── tensor_ops.rs    # Common tensor utilities
│   │   │   ├── model_loader.rs  # GGUF model loading
│   │   │   ├── quantized_ops.rs # Quantized operations
│   │   │   └── mod.rs           # Module exports
│   │   │
│   │   ├── generation.rs        # Core generation loop
│   │   ├── engine.rs            # Inference engine orchestration
│   │   └── mod.rs               # Inference exports
│   │
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
│   │   │   └── components/     # Reusable components
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
│   │   │   │   ├── device.rs   # DeviceManager - high-level device selection
│   │   │   │   └── mod.rs      # Module exports
│   │   │   │
│   │   │   ├── smollm3/         # SmolLM3-Specific Extensions
│   │   │   │   ├── adapter.rs      # SmolLM3Adapter - bridges to official
│   │   │   │   ├── generation.rs   # SmolLM3Generator - uses inference layer
│   │   │   │   ├── thinking.rs     # ThinkingDetector - <think> token handling
│   │   │   │   ├── kv_cache.rs     # KVCache - SmolLM3-specific cache logic
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
│   │   │   ├── chat.rs         # Chat-specific templates
│   │   │   └── mod.rs          # Module exports
│   │   │
│   │   ├── session.rs          # Session management
│   │   ├── streaming.rs        # Streaming service
│   │   ├── metrics.rs          # Performance metrics
│   │   └── mod.rs              # Services module exports
│   │
│   ├── types/                   # Shared Type Definitions
│   │   ├── events.rs           # StreamEvent, GenerationEvent types
│   │   ├── message.rs          # Message, ChatMessage structures
│   │   ├── session.rs          # Session, ChatSession types
│   │   ├── errors.rs           # Error types
│   │   └── mod.rs              # Type exports
│   │
│   ├── smollm3/                # SmolLM3 model core
│   │   ├── model.rs            # Model implementation
│   │   ├── config.rs           # Model configuration
│   │   ├── tokenizer.rs        # Tokenizer handling
│   │   ├── chat_template.rs    # Chat formatting
│   │   └── mod.rs              # Module exports
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

### **Three-Tier Architecture**

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

3. **Inference Foundation** (`/src/inference`)
   - **Candle**: Low-level tensor operations
   - **Generation**: Core generation loop
   - **Engine**: Inference orchestration

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
- **Memory**: ~2GB GPU RAM with KV cache

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
- **Inference layer (NEW)**
  - Device management
  - KV cache operations
  - Tensor utilities
  - Generation loop

### 🚧 In Progress
- Model integration with inference layer
- Generation pipeline optimization
- Tool use system
- Follow-up suggestions

### 📋 Planned
- Context management improvements
- Performance optimization
- Persistent sessions
- Multi-model support
- Distributed inference

## 📚 Documentation

- `/doc/architecture.md` - System design decisions
- `/doc/implementation_status.md` - Detailed progress tracking
- `/doc/candle_reference.md` - Candle.rs usage patterns
- `/doc/ui_ux_interaction.md` - User interaction flow

## 🛠️ Development Commands

```bash
# Build with all features
cargo build --release

# Run with environment setup
./run.sh

# Test compilation only
./test_compilation.sh

# Fix common issues
./fix_compilation.sh

# Make all scripts executable
./make_all_executable.sh
```

## 📄 License

MIT

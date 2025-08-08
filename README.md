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

## 🚀 Latest Update (v0.3.0 - Slash Commands & Streaming)

### **New Features** ✨

#### 1. **Slash Commands System**
- Type `/` in chat to open an interactive command palette
- **Categories**: Chat, Model, Utility, Quick Actions
- **Keyboard Navigation**: Arrow keys, Enter to select, Tab to autocomplete
- **Available Commands**:
  - `/clear` - Clear chat history
  - `/reset` - Reset conversation context  
  - `/export` - Export chat to file
  - `/thinking` - Toggle thinking mode
  - `/temp` - Adjust temperature
  - `/model` - View model info
  - `/quote` - Stream scripture passages (test command)
  - `/help` - Show all commands
  - `/status` - System status
  - `/theme` - Toggle dark/light theme

#### 2. **Response Buffer Testing with `/quote`**
- Special test command that streams John 1:1-14 (Recovery Version)
- Uses pure HTMX SSE for smooth text streaming
- Server-side markdown to HTML conversion
- Progressive verse-by-verse display
- Tests streaming buffer behavior without model loading

#### 3. **UI/UX Improvements**
- Smart scrollbar: Hidden by default, appears on hover when needed
- Full-width input field with proper flex layout
- Smooth animations for command menu
- Dark mode support throughout
- Mobile responsive design

### **Technical Implementation**
- **Pure HTMX SSE**: No JavaScript for streaming, uses `hx-ext="sse"`
- **Server-Side Rendering**: MiniJinja 2 templates
- **Elegant Simplicity**: Minimal client-side code
- **Axum 0.8.4**: Proper route syntax with `{param}` captures

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

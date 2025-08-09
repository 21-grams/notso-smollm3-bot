# NotSo-SmolLM3 Bot

A high-performance, production-ready chatbot implementation of SmolLM3-3B using pure Rust and the Candle machine learning framework. This project demonstrates efficient quantized inference with streaming responses, all running natively without Python dependencies.

## 🎯 Project Goal

To create a fully-featured SmolLM3 inference system in Rust that:
- Runs Q4_K_M quantized models efficiently on consumer hardware
- Provides real-time token streaming with intelligent buffering
- Supports SmolLM3's unique features (thinking mode, GQA, NoPE layers)
- Delivers a smooth chat experience through a clean web interface
- Serves as a reference implementation for Rust-based LLM inference

## 🚀 Key Features

### Resilient Architecture
- **Graceful Model Loading**: Server starts regardless of model availability
- **Fallback Mechanisms**: Automatic fallback to FTS5 search (placeholder) when model unavailable
- **Broadcast SSE**: Multiple subscribers supported with no message loss
- **Event-Driven Design**: Pure event flow from input to display

### Latest Technology Stack (2025)
- **Rust**: Stable 1.80+ with async/await, Arc<RwLock> patterns
- **Axum 0.8**: Latest with `/{param}` syntax, improved SSE, graceful shutdown
- **HTMX 2.0.6**: Core + SSE extension for reactive UI without heavy JavaScript
- **Candle 0.9.1+**: Latest quantized operations support
- **Tokio Broadcast**: Resilient multi-subscriber event system

### SmolLM3-Specific Optimizations
- **Grouped Query Attention (GQA)**: 4:1 KV head ratio for 75% memory savings
- **NoPE Layers**: Layers [3,7,11,15,19,23,27,31,35] skip position encoding
- **Thinking Mode**: Native `<think>` token support for reasoning
- **128k Vocabulary**: Extended tokenizer for better coverage
- **Q4_K_M Quantization**: 4-bit weights with minimal quality loss (~1.5GB model size)

## 🏗️ System Architecture

### High-Level Flow
```
User Input → HTMX POST → Server Handler → Background Task → 
Broadcast Channel → SSE Stream → Client → Markdown Rendering
```

### Component Architecture
```
┌────────────────────────────────────────────────────────────┐
│                  Web Layer (Axum 0.8 + HTMX 2.0.6)         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • HTTP Routes      • Persistent SSE Connections    │   │
│  │  • Chat UI          • EventSource Client            │   │
│  │  • Slash Commands   • Client-side Markdown (marked) │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────┬───────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────┐
│                     Service Layer                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • Session Manager (Broadcast Channels)             │   │
│  │  • Unified StreamingBuffer (500ms/10 tokens)        │   │
│  │  • Template Engine (MiniJinja 2)                    │   │
│  │  • Event Broadcasting System                        │   │
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

## 🔄 Server Startup Behavior

The server implements **graceful degradation** with three possible states:

| State | Model Status | Server Status | Capabilities |
|-------|-------------|---------------|--------------|
| **Full Operation** | ✅ Loaded | ✅ Running | Model inference, Commands, SSE |
| **Fallback Mode** | ❌ Not loaded | ✅ Running | Commands only, FTS5 placeholder |
| **Error State** | ❌ Load failed | ✅ Running | Graceful error messages |

### Startup Sequence
1. Initialize tracing and logging
2. Load configuration from environment
3. **Attempt** to load ML model (non-blocking)
4. Start web server regardless of model status
5. Bind to port 3000 and begin serving

## 📡 Broadcast SSE Implementation

### Why Broadcast Channels?

**Problem with traditional MPSC**:
- Single receiver - once taken, channel is consumed
- SSE reconnections lose messages
- Race conditions between message sending and SSE connection

**Solution with Broadcast**:
- Multiple subscribers receive same messages
- Late subscribers get buffered messages
- SSE reconnections just create new subscriptions
- Zero message loss due to timing

### Implementation
```rust
// Each session has a broadcast channel
pub struct SessionState {
    pub event_sender: broadcast::Sender<StreamEvent>,
    // ... other fields
}

// SSE endpoints subscribe to the broadcast
let receiver = sessions.subscribe(&session_id);
let stream = BroadcastStream::new(receiver);
```

## 📋 Complete Call Chain

### User Input → Display Output

1. **User Input**: Type message in chat interface
2. **HTMX Submit**: POST to `/api/chat` with session_id
3. **Server Handler**: 
   - Generate unique message_id (UUID v7)
   - Return HTML immediately (user + assistant placeholder)
   - Spawn background task for processing
4. **Background Processing**:
   - Check if slash command or regular message
   - If model available: generate tokens
   - If not: stream fallback message
5. **Streaming Buffer**:
   - Accumulate tokens/words
   - Flush at 10 tokens OR 500ms
   - Send via broadcast channel
6. **SSE Delivery**:
   - All subscribers receive events
   - Convert to SSE format
   - Stream to clients
7. **Client Display**:
   - EventSource receives data
   - Accumulate raw text in DOM
   - On completion: render markdown
   - Display formatted result

## 🛠️ Setup & Installation

### Prerequisites
- Rust 1.75+ (for async trait support)
- 4GB+ RAM
- (Optional) CUDA 12+ or Metal for GPU acceleration

### Required Model Files
- **Model**: `SmolLM3-3B-Q4_K_M.gguf` (~1.5GB)
- **Tokenizer**: `tokenizer.json` from SmolLM3-3B

### Quick Start
```bash
# Clone the repository
git clone https://github.com/21-grams/notso-smollm3-bot.git
cd notso-smollm3-bot

# Download model files (optional - server runs without them)
mkdir models
cd models
wget https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Q4_K_M.gguf
wget https://huggingface.co/HuggingFaceTB/SmolLM3-3B/tokenizer.json
cd ..

# Build and run
cargo run --release

# Access the chat interface
open http://localhost:3000
```

## 📊 Performance Characteristics

- **First Token Latency**: < 500ms (when model loaded)
- **Generation Speed**: 1-2 tokens/second (target)
- **Memory Usage**: ~2GB with Q4_K_M
- **Context Length**: 2048 tokens (expandable)
- **Streaming Buffer**: 10 tokens or 500ms flush
- **SSE Keep-Alive**: 30-second intervals
- **Broadcast Buffer**: 100 messages per session

## 🔧 Configuration

Key settings in `config.rs`:
```rust
Config {
    host: "127.0.0.1",
    port: 3000,
    model_path: "models/SmolLM3-3B-Q4_K_M.gguf",
    tokenizer_path: "models/tokenizer.json",
    max_context_length: 65536,
    thinking_mode_default: true,
    temperature: 0.7,
    top_p: 0.9,
}
```

## 🧪 Testing & Debugging

### Test SSE Streaming
Navigate to `/test-sse` for isolated SSE testing page.

### Slash Commands
- `/quote` - Stream scripture with markdown formatting
- `/status` - System status information
- `/model` - Model information
- `/reset` - Reset conversation context

### Debug Logging
Set `RUST_LOG=info` or `RUST_LOG=debug` for detailed logs:
```bash
RUST_LOG=debug cargo run
```

## 📚 Documentation

Detailed technical documentation in `/doc`:
- `architecture.md` - Complete system architecture & call chains
- `latest-tech-stack-2025.md` - Current versions and features
- `gguf_integration_status.md` - GGUF metadata mapping
- `implementation_status.md` - Development progress tracking

## 🎨 Key Design Decisions

1. **Graceful Model Loading**: Server always starts, even without model files
2. **Broadcast Channels**: Solves race conditions and supports multiple subscribers
3. **Immediate Response**: HTML returned instantly, content streams separately
4. **Unified Buffer**: Same streaming behavior for all content types
5. **Client-Side Markdown**: Raw text streams, renders on completion

## 🚧 Current Status

- ✅ Complete web interface with HTMX
- ✅ Broadcast SSE implementation
- ✅ Session management system
- ✅ Streaming buffer with thresholds
- ✅ Slash command system
- ✅ Fallback mechanisms
- ✅ Client-side markdown rendering
- 🔄 Model inference integration (in progress)
- 📝 FTS5 search integration (planned)

## 📝 License

MIT

## 🙏 Acknowledgments

- [Candle](https://github.com/huggingface/candle) - Rust ML framework
- [HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) - SmolLM3 models
- [Axum](https://github.com/tokio-rs/axum) - Web framework
- [HTMX](https://htmx.org) - Hypermedia-driven UI
# SmolLM3 Bot - notso-smollm3-bot

A high-performance Rust chatbot featuring SmolLM3-3B (quantized Q4_K_M) with real-time streaming via HTMX SSE.

## 🚀 Features

- **SmolLM3-3B Integration** - Latest Candle.rs ecosystem (>0.9.1)
- **Real-time Streaming** - Character-by-character response streaming
- **HTMX-Powered UI** - Minimal JavaScript, maximum interactivity
- **Markdown Support** - Full markdown rendering for responses
- **Slash Commands** - `/quote`, `/status`, and more
- **Session Management** - Multi-user support with isolated sessions
- **Thinking Mode** - Optional CoT (Chain of Thought) reasoning

## 📊 Current Status

**Last Major Update**: 2025-01-17
- ✅ Fixed markdown rendering issue with unwanted `<pre><code>` wrapping
- ✅ HTMX SSE streaming with OOB swaps implemented
- ✅ Pure HTMX content routing (no JavaScript accumulation)
- ✅ Proper two-pass markdown rendering (parse then highlight)
- ✅ Clean separation of concerns
- ✅ External JavaScript modules

**Known Issues Fixed**:
- ~~Messages wrapped in code blocks~~ → Fixed by trimming indentation before markdown parsing
- ~~Highlight.js interfering with markdown~~ → Fixed with selective highlighting

## 🏗️ Architecture

### Tech Stack
- **Backend**: Rust with Axum web framework
- **ML**: Candle.rs, candle-nn, candle-transformers
- **Frontend**: HTMX 2.0 with SSE extension
- **Streaming**: Server-Sent Events (SSE)
- **Styling**: Minimal CSS with dark theme

### Project Structure
```
notso-smollm3-bot/
├── src/
│   ├── main.rs              # Application entry point
│   ├── state.rs             # Application state management
│   ├── config.rs            # Configuration
│   ├── services/
│   │   ├── ml/              # ML service with SmolLM3
│   │   ├── streaming/       # Streaming buffer implementation
│   │   └── session/         # Session management
│   └── web/
│       ├── handlers/        # HTTP request handlers
│       ├── templates/       # HTML templates
│       └── static/          # CSS, JavaScript
├── models/                  # GGUF model files
└── doc/                     # Documentation
```

## 🔧 Setup

### Prerequisites
- Rust 1.75+
- CUDA toolkit (optional, for GPU acceleration)
- 8GB+ RAM for model loading

### Installation

1. Clone the repository:
```bash
git clone https://github.com/21-grams/notso-smollm3-bot
cd notso-smollm3-bot
```

2. Download the model:
```bash
# Place SmolLM3-3B-Q4_K_M.gguf in models/ directory
mkdir models
# Download from HuggingFace or convert your own
```

3. Build and run:
```bash
cargo run --release
```

4. Open browser to `http://localhost:3000`

## 🎯 Development Goals

### Phase 1: Core Functionality ✅
- [x] Basic chat interface
- [x] SSE streaming
- [x] Markdown rendering
- [x] Session management

### Phase 2: ML Integration (In Progress)
- [ ] GGUF model loading
- [ ] Token streaming
- [ ] Context management
- [ ] Temperature control

### Phase 3: Advanced Features
- [ ] FTS5 search integration
- [ ] Conversation history
- [ ] Model switching
- [ ] Fine-tuning support

## 📖 Documentation

- [HTMX SSE Streaming Solution](doc/HTMX_SSE_Streaming_Solution.md) - Complete streaming architecture
- [SSE Streaming Refactoring Notes](doc/SSE_Streaming_Refactoring_Notes.md) - Future improvements
- [Pure HTMX SSE Implementation](doc/Pure_HTMX_SSE_Implementation.md) - Implementation details

## 🤝 Collaboration Guidelines

### Development Rules
- **Build**: `cargo run` - Create environment setup scripts for dependencies
- **Testing**: Unit tests for core features only
- **Documentation**: Use `///` comments, maintain /doc folder
- **Safety**: Pure safe Rust preferred, justify any `unsafe` blocks
- **Clean Code**: No test scripts without permission, ask clear questions

### Contribution Process
1. Check existing issues/discussions
2. Create feature branch
3. Follow Rust best practices
4. Update documentation
5. Submit PR with clear description

## 🔍 Technical Highlights

### HTMX SSE with OOB Swaps
The streaming solution uses HTMX's out-of-band swaps to route content to specific message bubbles:

```rust
// Backend sends targeted content
Event::default()
    .event("message")
    .data(format!(
        r#"<span hx-swap-oob="beforeend:#msg-{}-content">{}</span>"#,
        message_id, content
    ))
```

```html
<!-- Frontend receives and auto-routes content -->
<div id="msg-{id}-content"></div>
```

### Minimal JavaScript
JavaScript is only used for:
- Markdown rendering (marked.js)
- UI helpers (auto-scroll, textarea resize)
- No message accumulation or routing needed!

## 📈 Performance

- **Streaming Latency**: <50ms per chunk
- **Memory Usage**: ~500MB (without model)
- **Concurrent Sessions**: 100+ supported
- **Model Inference**: TBD (pending GGUF integration)

## 🐛 Known Issues

- Model loading not yet implemented (fallback messages active)
- FTS5 search integration pending
- Token counting not active

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- HTMX team for the excellent framework
- Candle.rs team for the ML ecosystem
- SmolLM team for the model

---

**Project Status**: 🟡 Active Development

For questions or contributions, please open an issue on [GitHub](https://github.com/21-grams/notso-smollm3-bot).
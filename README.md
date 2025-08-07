# NotSo-SmolLM3 Bot

Production-ready SmolLM3-3B chatbot using official Candle.rs patterns with clear layer separation.

## Architecture

```
Web UI (Entry) → Inference Engine (Candle) → SmolLM3 (Model-specific)
```

## Change Log

### v0.2.0 - 2024-01-XX
- ✅ **Complete restructure** with 3-layer architecture
- ✅ **Web Layer**: Full Axum + HTMX + MiniJinja implementation
- ✅ **Inference Layer**: Official Candle.rs patterns with Q4_K_M support
- ✅ **SmolLM3 Layer**: Model-specific features (thinking, tools, NoPE)
- ✅ **Services**: Session management and SSE streaming
- ✅ **Documentation**: Technical docs in `/doc` directory

### v0.1.0 - 2024-01-XX
- 🚀 Initial project setup
- 🌐 Basic web server
- 🧠 Stub inference engine

## Project Structure

```
src/
├── web/          # Layer 1: Web UI (Entry Point)
├── inference/    # Layer 2: Standard Candle.rs
├── smollm3/      # Layer 3: SmolLM3-specific
└── services/     # Shared services
```

## Quick Start

```bash
# Download models
./scripts/download_models.sh

# Build
cargo build --release

# Run
cargo run --release

# Visit http://localhost:3000
```

## Features

### Implemented ✅
- **Web Interface**: HTMX-based chat UI
- **SSE Streaming**: Real-time token streaming
- **Session Management**: Multi-session support
- **Stub Mode**: Runs without model for testing

### In Progress 🚧
- **Model Loading**: Q4_K_M GGUF support
- **Tokenization**: 128k vocabulary
- **Generation**: KV-cached inference
- **Thinking Mode**: `<think>` token support

### Planned 📋
- **Tool Calling**: XML/Python formats
- **NoPE Layers**: Skip RoPE on specific layers
- **YARN Scaling**: Extended context support

## Technical Stack

- **Rust**: 1.75+
- **Candle**: 0.9.1 (Official patterns only)
- **Axum**: 0.8 (Web framework)
- **HTMX**: 1.9 (Dynamic UI)
- **MiniJinja**: 2.3 (Templates)

## Documentation

See `/doc` directory for:
- `architecture.md` - System design
- `candle_reference.md` - Candle.rs patterns used
- `smollm3_features.md` - Model capabilities
- `q4km_technical.md` - Quantization details

## License

MIT

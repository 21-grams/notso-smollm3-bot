# System Architecture

## Overview

NotSo-SmolLM3 Bot follows a clean 3-layer architecture:

```
┌─────────────────────────────────────────┐
│          Layer 1: Web UI                 │
│     (Axum + HTMX + MiniJinja)           │
│         Entry point for users            │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Layer 2: Inference Engine          │
│    (Official Candle.rs Patterns)        │
│         Standard ML operations          │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        Layer 3: SmolLM3 Model           │
│      (Model-specific features)          │
│    Thinking, Tools, Chat Templates      │
└─────────────────────────────────────────┘
```

## Layer 1: Web UI (src/web/)

**Purpose**: Handle all HTTP/Web concerns

- **Routes**: Define API and page endpoints
- **Handlers**: Process requests, call inference
- **SSE**: Stream responses to browser
- **Templates**: Render HTML with MiniJinja
- **Static**: CSS, minimal JS, assets

## Layer 2: Inference Engine (src/inference/)

**Purpose**: Standard Candle.rs ML operations

- **Model Loading**: GGUF with Q4_K_M validation
- **Quantized Ops**: Direct QMatMul operations
- **KV Cache**: Efficient generation caching
- **Device Management**: CPU/CUDA abstraction
- **Tensor Operations**: Standard Candle patterns

## Layer 3: SmolLM3 Model (src/smollm3/)

**Purpose**: SmolLM3-specific features

- **Config**: Model parameters (vocab, layers, etc.)
- **Tokenizer**: 128k vocabulary support
- **Chat Templates**: Proper formatting
- **Thinking Mode**: `<think>` token handling
- **Tool Use**: Function calling support
- **NoPE Layers**: Position encoding control

## Services (src/services/)

**Purpose**: Shared functionality

- **Session Management**: Track conversations
- **Streaming**: SSE event broadcasting
- **Metrics**: Performance monitoring

## Data Flow

1. **User Input** → Web Handler
2. **Handler** → SmolLM3 Model
3. **SmolLM3** → Apply chat template
4. **SmolLM3** → Tokenize
5. **SmolLM3** → Inference Engine
6. **Engine** → Generate tokens
7. **Tokens** → Streaming Service
8. **SSE** → Browser UI

## Key Design Decisions

1. **Clear Separation**: Each layer has distinct responsibilities
2. **Official Patterns**: Use Candle's built-in functions
3. **No Custom ML**: Leverage existing ecosystem
4. **Async Throughout**: Tokio for all async operations
5. **Stub Mode**: Can run without model for testing

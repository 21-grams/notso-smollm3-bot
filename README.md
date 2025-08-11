# SmolLM3 Bot - notso-smollm3-bot

A high-performance Rust chatbot implementing SmolLM3-3B (Q4_K_M quantized) with real-time streaming via HTMX SSE, using the latest Candle.rs ecosystem (0.9.1+).

## 🎯 Project Goal

Build a fully-featured inference engine for SmolLM3-3B with:
- **Direct quantized operations** (Q4_K_M) for 50-100x speedup
- **Thinking mode** with `<think>` tokens for chain-of-thought reasoning
- **128K context support** with efficient KV cache
- **Real-time streaming** via Server-Sent Events
- **Clean architecture** separating official Candle from SmolLM3 features
- **NoPE layers** via CustomOp for content-based attention

## 📊 Current Status

**Version**: 0.9.0  
**Date**: 2025-01-17  
**Phase**: NoPE Model Implementation with candle-nn 🚀

### ✅ Complete
- **Web Infrastructure**: Axum 0.8 server with HTMX SSE streaming
- **UI/UX**: Beautiful chat interface with markdown rendering
- **Session Management**: Multi-session support with UUID v7
- **GGUF Inspector**: Tool to analyze model quantization and metadata
- **Q4_K_M Support**: ✅ **FULLY VERIFIED** in Candle 0.9.1
- **Metadata Mapping**: ✅ SmolLM3 → Llama format working
- **Model Loading**: ✅ `ModelWeights::from_gguf()` successful
- **Memory Efficiency**: ✅ Weights stay quantized (1.78GB file → 2.9GB in memory)
- **NoPE Model**: ✅ Full implementation with selective RoPE using candle-nn
- **Position Tracking**: ✅ Proper position management (0 → prompt_len → +1)
- **Dual Backend**: ✅ Support for both Standard and NoPE models

### 🚧 In Progress
- **Forward Pass**: Using ModelWeights.forward() with CustomOp interception
- **Generation Loop**: Token-by-token generation with proper position tracking
- **KV Cache**: Integration with ModelWeights' internal cache
- **NoPE Layer Verification**: Debug logging for layers [3,7,11,15,19,23,27,31,35]

### ⏳ Next Steps
1. Debug and verify NoPE layer detection in forward pass
2. Ensure ModelWeights.forward() works with CustomOp
3. Test generation quality with NoPE layers
4. Optimize performance and memory usage

## 🛠️ Development Tools

### Inspection Tools
Two binary tools are provided for development:

```bash
# Inspect GGUF file structure and metadata
cargo run --bin inspect_gguf

# Test Q4_K support in Candle
cargo run --bin test_q4k
```

### Key Findings
- ✅ GGUF file contains 326 tensors (216 Q4K, 37 Q6K, 73 F32)
- ✅ **Model dimensions corrected**:
  - Hidden size: **2048** (not 3072)
  - Intermediate: **11008** (not 8192)
  - Heads: **16** (not 32)
  - KV heads: **4** (not 8)
- ✅ **Metadata mapping working**: SmolLM3 → Llama format
- ✅ **Q4_K_M fully supported** via `GgmlDType::Q4K`
- ✅ **Efficient loading**: 1.78GB file uses ~2.9GB memory (reasonable)
- ✅ **NoPE layers identified**: Every 4th layer starting from 3

## 🏗️ Architecture

### Clean Layer Separation
```
src/services/ml/
├── official/           # Pure Candle.rs implementations
│   ├── gguf_loader.rs     # GGUF → QTensor loading
│   ├── model.rs           # Wraps quantized_llama with CustomOp hooks
│   ├── quantized_model.rs # Direct QMatMul operations
│   └── config.rs          # Model configuration
│
├── smollm3/           # SmolLM3-specific features
│   ├── custom_ops.rs      # NoPE-aware RoPE CustomOp
│   ├── tokenizer_ext.rs   # Batch tokenization
│   ├── chat_template.rs   # Template application
│   ├── generation.rs      # Token generation
│   ├── thinking.rs        # Thinking mode
│   └── kv_cache.rs        # 128K context cache
│
└── service.rs         # Orchestration layer
```

### Key Design Principles
- **Official layer**: Uses ONLY documented Candle APIs + CustomOp
- **SmolLM3 layer**: Adds model-specific features and NoPE support
- **No dequantization**: Direct Q4_K_M operations throughout
- **Token buffering**: Efficient batch processing
- **CustomOp interception**: Automatic NoPE layer handling

### NoPE Implementation Details
The NoPE (No Position Encoding) layers are implemented using Candle's CustomOp2 trait:
- **NoPE Layers**: [3, 7, 11, 15, 19, 23, 27, 31, 35]
- **Mechanism**: CustomOp intercepts RoPE calls and skips them for NoPE layers
- **Global State**: Atomic counters track current layer and position
- **Debug Logging**: Verbose logging shows when NoPE layers skip RoPE

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
- **Architecture**: SmolLM3-3B with NoPE layers
- **Quantization**: Q4_K_M (~1.9GB)
- **Layers**: 36 (9 NoPE, 27 with RoPE)
- **Attention**: 16 heads (4 KV heads for GQA)
- **Hidden Size**: 2048
- **Intermediate**: 11008
- **Vocab Size**: 128256
- **Context**: Up to 65536 tokens
- **RoPE Theta**: 5,000,000

### Performance Targets
- **Speed**: 1-2 tokens/second minimum
- **Memory**: < 4GB total usage
- **Latency**: < 50ms per token
- **Context**: 65K with efficient KV cache

## 📝 Implementation Notes

### CustomOp Integration
The project uses Candle's CustomOp feature to add NoPE layer support without modifying the core ModelWeights implementation:

1. **Global State Management**: Atomic counters track layer and position
2. **CustomOp Registration**: Happens once at service initialization
3. **Automatic Interception**: RoPE operations are intercepted transparently
4. **Debug Visibility**: Extensive logging shows NoPE layer behavior

### Position Tracking
Position management follows SmolLM3's expected behavior:
- Start at position 0 for new sequences
- Jump to prompt_len after processing prompt
- Increment by 1 for each generated token
- Used for both RoPE angles and KV cache indexing

## 🚀 Latest Update (v0.8.0)

**Major Achievement**: Successfully integrated CustomOp for NoPE layer support!

- ✅ Created `NopeAwareRoPE` CustomOp that implements CustomOp2 trait
- ✅ Added global state tracking for layer and position
- ✅ Modified model forward pass to use CustomOp hooks
- ✅ Updated service layer to properly track position through generation
- ✅ Added extensive debug logging for NoPE layer verification

The implementation allows SmolLM3's NoPE layers to function correctly by:
1. Intercepting RoPE operations at the CustomOp level
2. Checking if current layer is NoPE (3, 7, 11, 15, 19, 23, 27, 31, 35)
3. Skipping position encoding for NoPE layers
4. Applying standard RoPE for other layers

This approach maintains clean separation between official Candle APIs and SmolLM3-specific features while enabling the unique NoPE architecture that makes SmolLM3 special.

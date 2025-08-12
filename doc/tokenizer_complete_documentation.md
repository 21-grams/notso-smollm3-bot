# SmolLM3 Tokenizer: Complete Implementation Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Call Chain](#call-chain)
4. [Design Considerations](#design-considerations)
5. [Implementation Details](#implementation-details)
6. [Pipeline Processing](#pipeline-processing)
7. [Template System](#template-system)
8. [Batch Encoding](#batch-encoding)
9. [Error Handling](#error-handling)
10. [Integration Points](#integration-points)

## Overview

The SmolLM3 tokenizer is a comprehensive text processing pipeline that transforms user input into batch-encoded token IDs ready for model inference. It implements a type-safe builder pattern to enforce processing order while maintaining consistency with the Candle.rs ecosystem.

### Key Features
- **Unified batch encoding** for all inputs (including batch_size=1)
- **Type-safe pipeline** with compile-time ordering enforcement
- **Jinja2 template integration** with custom functions
- **Multi-stage processing** with stubs for future safety features
- **Clean error boundaries** between service and ML layers

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   SmolLM3Tokenizer                   │
├─────────────────────────────────────────────────────┤
│ Components:                                          │
│ - tokenizers::Tokenizer (HuggingFace tokenizers)    │
│ - minijinja::Environment (Template engine)          │
│ - SpecialTokens (Token ID mappings)                 │
│ - Pipeline<S> (Type-safe builder)                   │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  Pipeline States                     │
├─────────────────────────────────────────────────────┤
│ Initial → Configured → Process                      │
│ (Enforces configuration before processing)          │
└─────────────────────────────────────────────────────┘
```

## Call Chain

### 1. Web Layer Entry Point
```rust
// src/web/handlers/api.rs
async fn generate_response_buffered() {
    service.generate_streaming(&message, &mut buffer, true).await
}
```

### 2. ML Service Layer
```rust
// src/services/ml/service.rs
pub async fn generate_streaming(
    &mut self,
    user_input: &str,
    buffer: &mut StreamingBuffer,
    thinking_enabled: bool,
) -> Result<()> {
    // Calls tokenizer pipeline
    let token_batch = self.tokenizer
        .process_input(user_input.to_string(), thinking_enabled)?;
}
```

### 3. Tokenizer Pipeline Entry
```rust
// src/services/ml/smollm3/tokenizer.rs
pub fn process_input(
    &self,
    user_input: String,
    thinking_enabled: bool,
) -> Result<Vec<Vec<u32>>> {
    Pipeline::new(self, user_input)
        .with_thinking(thinking_enabled)
        .process()
}
```

### 4. Pipeline Processing Chain
```rust
impl Pipeline<'a, Configured> {
    pub fn process(self) -> Result<Vec<Vec<u32>>> {
        // Step-by-step processing
        let sanitized = self.tokenizer.sanitize_input(self.data);
        let filtered = self.tokenizer.filter_prompt(sanitized)?;
        let templated = self.tokenizer.apply_chat_template(
            filtered,
            self.conversation_history.as_deref(),
            self.thinking_enabled,
        )?;
        let tokens = self.tokenizer.encode_batch(vec![templated])?;
        Ok(tokens)
    }
}
```

### 5. Template Application
```rust
fn apply_chat_template(
    &self,
    input: String,
    conversation_history: Option<&[ChatMessage]>,
    thinking_enabled: bool,
) -> Result<String> {
    // Build message structure
    let mut messages = /* ... */;
    
    // Create context with all required variables
    let ctx = minijinja::context! {
        messages => messages,
        add_generation_prompt => true,
        enable_thinking => thinking_enabled,
        xml_tools => false,
        python_tools => false,
        tools => false,
    };
    
    // Render Jinja2 template
    let template = self.chat_template.get_template("chat")?;
    template.render(ctx)
}
```

### 6. Batch Encoding
```rust
fn encode_batch(&self, inputs: Vec<String>) -> Result<Vec<Vec<u32>>> {
    // Always returns batch format [batch_size, seq_len]
    let encodings = self.tokenizer.encode_batch(inputs, true)?;
    
    // Extract token IDs
    let mut result = Vec::with_capacity(encodings.len());
    for encoding in encodings {
        result.push(encoding.get_ids().to_vec());
    }
    Ok(result)
}
```

### 7. Tensor Creation (ML Service)
```rust
// Back in service.rs
let tokens = token_batch.into_iter().next().unwrap(); // Extract from batch
let input_tensor = Tensor::new(&tokens[..], &self.device)?
    .unsqueeze(0)?;  // Add batch dimension [1, seq_len]
```

### 8. Model Forward Pass
```rust
let prompt_logits = match &mut self.model {
    ModelBackend::Nope(m) => m.forward(&input_tensor, position)?,
    // ...
};
```

## Design Considerations

### 1. Batch Encoding Consistency

**Decision**: Always use batch encoding, even for single inputs (batch_size=1)

**Rationale**:
- Maintains consistency with Candle.rs ecosystem patterns
- Simplifies tensor shape handling throughout the pipeline
- Prepares for future multi-batch processing
- Avoids special-casing single vs. batch inputs

**Implementation**:
```rust
// Always returns Vec<Vec<u32>> even for single input
pub fn encode_batch(&self, inputs: Vec<String>) -> Result<Vec<Vec<u32>>>
```

### 2. Type-Safe Pipeline Pattern

**Decision**: Use builder pattern with type states to enforce processing order

**Rationale**:
- Compile-time guarantee of correct processing order
- Prevents calling `process()` before configuration
- Self-documenting API through type system
- Minimal runtime overhead

**Implementation**:
```rust
// Type states enforce order
struct Initial;
struct Configured;

// Can't call process() on Initial state
Pipeline<Initial> → with_thinking() → Pipeline<Configured> → process()
```

### 3. Single Entry Point

**Decision**: Provide `process_input()` as the main API

**Rationale**:
- Simplifies usage for callers
- Encapsulates complexity
- Maintains backward compatibility option
- Clear separation of concerns

### 4. Error Boundary Management

**Decision**: Use `anyhow::Result` in tokenizer, `candle_core::Result` in model layers

**Rationale**:
- Tokenizer is part of service layer (uses anyhow)
- Model operations need candle-specific errors
- Clear conversion points at boundaries
- Better error messages with anyhow's context

### 5. Template Function Registration

**Decision**: Register custom functions like `strftime_now` in minijinja

**Rationale**:
- Supports official SmolLM3 template requirements
- Extensible for future template functions
- Clean integration with template engine
- Maintains template compatibility

## Implementation Details

### File Loading Strategy

The tokenizer loads three configuration files at startup:

1. **tokenizer.json**: Core tokenizer configuration
   - Vocabulary mappings
   - Merge rules
   - Special token definitions

2. **tokenizer_config.json**: Additional settings
   ```json
   {
     "pad_token_id": 128004,
     "bos_token_id": 128000,
     "eos_token_id": 128001,
     "padding_side": "left"
   }
   ```

3. **special_tokens_map.json**: Optional special token overrides

### Special Token Management

```rust
pub struct SpecialTokens {
    pub bos: u32,           // 128000 - <|begin_of_text|>
    pub eos: u32,           // 128001 - <|end_of_text|>
    pub thinking_start: u32, // 128002 - <think>
    pub thinking_end: u32,   // 128003 - </think>
    pub pad: u32,           // 128004 - <|finetune_right_pad_id|>
}
```

These tokens are used for:
- **BOS/EOS**: Mark sequence boundaries
- **Thinking tokens**: Filter internal reasoning during generation
- **Padding**: Align sequences in batches

### Chat Template Integration

The tokenizer embeds the official SmolLM3 Jinja2 template:

```rust
let template = include_str!("../../../../templates/smollm3_official.j2");
```

Template features:
- Metadata section with date and reasoning mode
- System message handling
- Tool support (stubs for future)
- Thinking mode toggle
- Conversation history formatting

## Pipeline Processing

### Stage 1: Input Sanitization (Stub)
```rust
fn sanitize_input(&self, input: String) -> String {
    // TODO: Implement
    // - Remove control characters
    // - Normalize whitespace
    // - Handle encoding issues
    input.trim().to_string()
}
```

**Future Implementation**:
- Unicode normalization
- HTML/script tag removal
- Length validation
- Character encoding fixes

### Stage 2: Prompt Filtering (Stub)
```rust
fn filter_prompt(&self, input: String) -> Result<String> {
    // TODO: Implement
    // - Check for prompt injection
    // - Filter harmful content
    // - Rate limiting checks
    Ok(input)
}
```

**Future Implementation**:
- Prompt injection detection
- Harmful content filtering
- PII detection and masking
- Rate limiting per session

### Stage 3: Template Application
Applies the SmolLM3 chat template with:
- Current conversation context
- Thinking mode configuration
- System message defaults
- Tool availability flags

### Stage 4: Batch Encoding
Converts templated text to token IDs:
- Uses HuggingFace tokenizers library
- Always returns batch format
- Handles special tokens automatically
- Supports padding configuration

## Template System

### Custom Functions

**strftime_now**: Provides current date formatting
```rust
chat_template.add_function("strftime_now", |format: String| {
    let now = Local::now();
    now.format(&format).to_string()
});
```

Usage in template:
```jinja
{%- set today = strftime_now("%d %B %Y") -%}
{{- "Today Date: " ~ today ~ "\n" -}}
```

### Context Variables

The template expects these variables:
```rust
{
    messages: Vec<ChatMessage>,      // Conversation history
    add_generation_prompt: bool,     // Add assistant prefix
    enable_thinking: bool,           // Enable thinking mode
    xml_tools: bool,                // XML tool support
    python_tools: bool,             // Python tool support
    tools: bool,                    // General tool support
}
```

### Reasoning Mode Logic

The template determines reasoning mode based on:
1. `enable_thinking` flag from context
2. `/think` or `/no_think` commands in system message
3. Default to thinking mode if enabled

## Batch Encoding

### Design Philosophy

**Always batch, even for single inputs**

This design choice ensures:
1. **Consistency**: Same code path for all inputs
2. **Compatibility**: Matches Candle.rs expectations
3. **Scalability**: Ready for multi-batch processing
4. **Simplicity**: No special cases to handle

### Tensor Shape Flow

```
User Input (String)
    ↓
Tokenizer: Vec<Vec<u32>> where outer.len() = 1
    ↓
Service: Tensor [1, seq_len] via unsqueeze(0)
    ↓
Model: Expects [batch_size, seq_len]
    ↓
Output: Logits [batch_size, seq_len, vocab_size]
```

## Error Handling

### Error Types by Layer

1. **Tokenizer Layer** (`anyhow::Result`):
   - File loading errors
   - Template rendering errors
   - Encoding failures
   - Configuration parsing errors

2. **Model Layer** (`candle_core::Result`):
   - Tensor operations
   - Device allocation
   - Forward pass errors
   - Shape mismatches

### Error Conversion Points

```rust
// At service boundary
let tokenizer = SmolLM3Tokenizer::from_files(tokenizer_dir)
    .map_err(|e| candle_core::Error::Msg(format!("Tokenizer: {}", e)))?;
```

### Error Propagation Chain

```
Web Handler → anyhow::Result
    ↓
ML Service → anyhow::Result (converts candle errors)
    ↓
Tokenizer → anyhow::Result
    ↓
Model → candle_core::Result
```

## Integration Points

### 1. Web Layer Integration
- Receives raw user input as String
- Passes thinking mode preference
- Handles streaming responses
- Manages session state

### 2. ML Service Integration
- Orchestrates tokenizer and model
- Manages KV cache
- Handles generation loop
- Filters thinking tokens

### 3. Model Integration
- Receives batch tensors [batch_size, seq_len]
- Processes through transformer layers
- Returns logits for sampling
- Maintains KV cache state

### 4. Streaming Integration
- Decodes tokens incrementally
- Filters thinking content
- Sends to SSE stream
- Handles completion signals

## Performance Considerations

### Memory Efficiency
- Templates compiled once at startup
- Special tokens cached in struct
- Batch allocation minimized
- String operations optimized

### Processing Efficiency
- Pipeline stages are lazy where possible
- Minimal string copying
- Direct tensor creation from token vectors
- Efficient batch encoding

### Scalability
- Ready for multi-batch processing
- Stateless tokenizer operations
- Thread-safe template rendering
- Concurrent request handling

## Future Enhancements

### Planned Features
1. **Real sanitization**: Implement comprehensive input cleaning
2. **Content filtering**: Add safety checks and prompt guards
3. **Conversation history**: Maintain context across messages
4. **Tool support**: Enable XML and Python tool integration
5. **Custom prompts**: Support user-defined system messages

### Extension Points
- Additional template functions
- Custom filtering rules
- Alternative template formats
- Multi-model support
- Caching strategies

## Testing Strategy

### Unit Tests
- Each pipeline stage independently
- Template rendering with various contexts
- Batch encoding edge cases
- Error handling paths

### Integration Tests
- Full pipeline flow
- Template + encoding combination
- Model integration
- Streaming output

### Performance Tests
- Tokenization speed
- Template rendering performance
- Batch processing throughput
- Memory usage patterns

## Conclusion

The SmolLM3 tokenizer implementation provides a robust, type-safe, and extensible pipeline for processing user input into model-ready tokens. Its design prioritizes:

1. **Consistency** with Candle.rs patterns
2. **Type safety** through builder pattern
3. **Extensibility** via template system
4. **Performance** through batch processing
5. **Maintainability** with clear separation of concerns

The implementation successfully bridges the gap between web input and model inference while maintaining clean architectural boundaries and preparing for future enhancements.

# Complete System Architecture & Call Chains

## Table of Contents
1. [Server Startup Call Chain](#server-startup-call-chain)
2. [Inference Engine Call Chain](#inference-engine-call-chain)
3. [UI Input/Output Call Chain](#ui-inputoutput-call-chain)
4. [Broadcast SSE Architecture](#broadcast-sse-architecture)
5. [Error Handling & Fallback Chains](#error-handling--fallback-chains)

---

## Server Startup Call Chain

### Complete Startup Sequence

```
main.rs::main()
│
├── tracing_subscriber::registry()
│   └── .with(EnvFilter::try_from_default_env())
│   └── .with(fmt::layer())
│   └── .init()
│
├── tracing::info!("🚀 Starting SmolLM3 Bot Server")
│
├── AppState::new().await?
│   │
│   ├── Config::from_env()?
│   │   ├── Read HOST (default: "127.0.0.1")
│   │   ├── Read PORT (default: "3000")
│   │   ├── Read MODEL_PATH (default: "models/SmolLM3-3B-Q4_K_M.gguf")
│   │   ├── Read TOKENIZER_PATH (default: "models/tokenizer.json")
│   │   └── Set device: DeviceConfig::Cpu
│   │
│   ├── MLService::new() [OPTIONAL - Non-blocking]
│   │   │
│   │   ├── SmolLM3Model::from_gguf(model_path, &device)?
│   │   │   ├── gguf_file::Content::read(&mut file)?
│   │   │   ├── Load metadata and tensors
│   │   │   ├── Initialize quantized weights
│   │   │   └── Return SmolLM3Model
│   │   │
│   │   ├── SmolLM3Tokenizer::from_file(tokenizer_path)?
│   │   │   ├── Read tokenizer.json
│   │   │   ├── Parse vocabulary
│   │   │   └── Setup special tokens
│   │   │
│   │   ├── SmolLM3KVCache::new(num_layers, max_seq_len, device)
│   │   │   └── Initialize empty cache layers
│   │   │
│   │   └── LogitsProcessor::new(seed, temperature, top_p)
│   │
│   ├── [On MLService Error]
│   │   ├── tracing::warn!("⚠️ Model not available: {}", e)
│   │   ├── tracing::info!("🌐 Server will start without model")
│   │   └── Set model = None
│   │
│   ├── TemplateEngine::new()?
│   │   └── Initialize MiniJinja templates
│   │
│   └── Return AppState {
│       config: Arc::new(config),
│       model: Arc::new(RwLock::new(ml_service)), // Option<MLService>
│       sessions: Arc::new(RwLock::new(SessionManager::new())),
│       templates: Arc::new(templates),
│   }
│
└── web::start_server(state).await?
    │
    ├── let app = create_app(state.clone())
    │   │
    │   ├── Router::new()
    │   │   ├── .route("/", get(handlers::chat::index))
    │   │   ├── .route("/api/chat", post(handlers::api::send_message))
    │   │   ├── .route("/api/stream/{session_id}", get(handlers::api::stream_session))
    │   │   ├── .route("/api/toggle-thinking", post(handlers::api::toggle_thinking))
    │   │   ├── .route("/test-sse", get(handlers::api::test_sse))
    │   │   └── ... (other routes)
    │   │
    │   ├── .nest_service("/static", ServeDir::new("src/web/static"))
    │   │
    │   └── .layer(ServiceBuilder::new()
    │       ├── .layer(CorsLayer::permissive())
    │       └── .layer(TraceLayer::new_for_http()))
    │
    ├── let addr = SocketAddr::from(([0, 0, 0, 0], state.config.port))
    │
    ├── tracing::info!("🌐 Server listening on http://{}", addr)
    │
    └── axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?
```

---

## Inference Engine Call Chain

### Model Generation Flow (When Model Available)

```
generate_response_buffered(state, session_id, message, message_id)
│
├── Get broadcast sender
│   └── sessions.get_or_create_sender(&session_id)
│       ├── Check if session exists
│       ├── If not: broadcast::channel(100)
│       └── Return sender clone
│
├── StreamingBuffer::new(sender, message_id)
│   └── Initialize with empty buffer
│
├── Check model availability
│   └── state.model.read().await
│
└── If Some(service):
    │
    └── service.generate_streaming(&message, &mut buffer).await
        │
        ├── tokenizer.encode(prompt)?
        │   ├── Convert prompt to tokens
        │   └── Return Vec<u32>
        │
        ├── Prepare input tensor
        │   └── Tensor::new(tokens, &device)?.unsqueeze(0)?
        │
        └── Generation loop (max_tokens iterations)
            │
            ├── model.forward_with_cache(&input_ids)?
            │   ├── Embedding lookup
            │   ├── Through transformer layers:
            │   │   ├── RMSNorm
            │   │   ├── Self-Attention (with GQA)
            │   │   ├── KV Cache update
            │   │   ├── Position encoding (skip on NoPE layers)
            │   │   ├── Feed-forward network
            │   │   └── Residual connections
            │   └── Output logits
            │
            ├── logits_processor.sample(&last_logits)?
            │   ├── Apply temperature
            │   ├── Apply top_p
            │   └── Sample token
            │
            ├── Check special tokens
            │   ├── If think_token_id: enter thinking mode
            │   ├── If think_end_token_id: exit thinking mode
            │   └── If EOS: break loop
            │
            ├── tokenizer.decode(&[next_token])?
            │
            └── buffer.push(token_text).await?
                ├── Accumulate in buffer
                ├── If 10 tokens OR 500ms elapsed:
                │   └── flush().await?
                │       ├── broadcast::send(StreamEvent::MessageContent)
                │       └── Reset buffer
                └── Continue
```

### Fallback Flow (When Model Unavailable)

```
If None (no model):
│
├── Create fallback message
│   └── "🔴 **Model not loaded**\n\n..."
│
├── tracing::warn!("No model available for message {}", message_id)
│
└── Stream fallback word by word
    │
    └── For each word in fallback.split_whitespace()
        ├── buffer.push(&format!("{} ", word)).await?
        ├── tokio::time::sleep(30ms).await
        └── Continue until done
```

---

## UI Input/Output Call Chain

### Complete User Interaction Flow

```
1. USER INPUT PHASE
   Browser
   │
   ├── User types in <textarea id="message-input">
   ├── User presses Enter (without Shift)
   └── JavaScript: document.getElementById('chat-form').requestSubmit()

2. HTMX PROCESSING PHASE
   HTMX
   │
   ├── Intercept form submission
   ├── Prepare POST request
   │   ├── URL: /api/chat
   │   ├── Data: {session_id, message}
   │   └── Headers: Content-Type: application/x-www-form-urlencoded
   └── Send async request

3. SERVER RECEPTION PHASE
   api::send_message()
   │
   ├── Extract Form data
   ├── Generate message_id (UUID v7)
   ├── Log: "Received message: '{}' for session: {}"
   │
   ├── Create immediate HTML response
   │   └── format!(r#"
   │       <div class="message user">
   │           <div class="message-bubble">{escaped_user_message}</div>
   │       </div>
   │       <div class="message assistant" id="msg-{message_id}">
   │           <div class="message-bubble">
   │               <span class="loading">Thinking...</span>
   │           </div>
   │       </div>"#)
   │
   ├── Return Html(html) [IMMEDIATE - User sees response]
   │
   └── tokio::spawn(async move { ... }) [BACKGROUND TASK]

4. BACKGROUND PROCESSING PHASE
   Background Task
   │
   ├── Check message type
   │   ├── If starts_with("/quote"): stream_quote_buffered()
   │   └── Else: generate_response_buffered()
   │
   └── Processing continues...
       [See Inference Engine Call Chain above]

5. EVENT BROADCASTING PHASE
   StreamingBuffer::flush()
   │
   ├── Create StreamEvent::MessageContent
   ├── broadcast::Sender::send(event)
   │   └── All subscribers receive copy
   └── Clear buffer

6. SSE DELIVERY PHASE
   stream_session() handler
   │
   ├── BroadcastStream receives event
   ├── Map to SSE format
   │   └── Event::default()
   │       .event("message")
   │       .data(format!("{}|{}", message_id, content))
   └── Send over SSE connection

7. CLIENT RECEPTION PHASE
   EventSource (Browser)
   │
   ├── Receive SSE event
   ├── JavaScript: eventSource.addEventListener('message', ...)
   ├── Parse: const [messageId, ...contentParts] = e.data.split('|')
   └── Update DOM
       ├── Find: document.querySelector(`#msg-${messageId} .message-bubble`)
       ├── Store: messageEl.dataset.rawContent += content
       └── Display: messageEl.textContent = messageEl.dataset.rawContent

8. COMPLETION & RENDERING PHASE
   On 'complete' event
   │
   ├── Get accumulated content: messageEl.dataset.rawContent
   ├── Render markdown: marked.parse(rawContent)
   ├── Update HTML: messageEl.innerHTML = formattedHtml
   └── Auto-scroll to bottom
```

---

## Broadcast SSE Architecture

### Why Broadcast Instead of MPSC

```
MPSC Problem:
┌──────────┐      ┌──────────┐      ┌──────────┐
│ Producer │─────>│ Channel  │─────>│ Consumer │
└──────────┘      └──────────┘      └──────────┘
                        ↓
                  Once taken, gone!
                  SSE reconnect = lost messages

Broadcast Solution:
┌──────────┐      ┌──────────┐      ┌──────────┐
│ Producer │─────>│ Broadcast│──┬──>│Consumer 1│
└──────────┘      │  Channel │  ├──>│Consumer 2│
                  └──────────┘  └──>│Consumer N│
                        ↓
                  Multiple subscribers
                  Late join = get buffered messages
```

### Implementation Details

```rust
// Session creation with broadcast
pub fn create_session(&mut self, session_id: &str) {
    if !self.sessions.contains_key(session_id) {
        let (tx, _rx) = broadcast::channel(100); // 100 message buffer
        
        let session = SessionState {
            id: session_id.to_string(),
            event_sender: tx,
            // ... other fields
        };
        
        self.sessions.insert(session_id.to_string(), session);
    }
}

// SSE subscription
pub fn subscribe(&mut self, session_id: &str) -> Option<broadcast::Receiver<StreamEvent>> {
    self.sessions.get(session_id)
        .map(|s| s.event_sender.subscribe()) // New receiver each time
}

// Broadcasting events
let _ = sender.send(StreamEvent::MessageContent {
    message_id,
    content,
}); // Non-async, returns Result<usize, SendError>
```

### Race Condition Solution

```
Timeline without broadcast (MPSC):
T0: User sends message
T1: Background task starts
T2: Task sends to channel [MESSAGE LOST - no receiver yet]
T3: SSE connects
T4: SSE takes receiver [Too late!]

Timeline with broadcast:
T0: User sends message
T1: Background task starts
T2: Task broadcasts message [BUFFERED in channel]
T3: SSE connects
T4: SSE subscribes [RECEIVES buffered message]
```

---

## Error Handling & Fallback Chains

### Model Loading Failure

```
AppState::new()
│
└── MLService::new() returns Err
    │
    ├── Log: tracing::warn!("⚠️ Model not available: {}", e)
    ├── Log: tracing::info!("🌐 Server will start without model")
    ├── Set: model = None
    └── Continue startup [SERVER STILL STARTS]
```

### Runtime Model Failure

```
generate_response_buffered()
│
└── service.generate_streaming() returns Err
    │
    ├── Log: tracing::error!("Model generation failed: {}", e)
    ├── Create error message with markdown
    ├── Stream fallback through buffer
    └── User sees: "⚠️ **Model generation failed**..."
```

### SSE Connection Failure

```
EventSource connection drops
│
├── Browser: Automatic reconnection attempt
├── Server: New subscribe() call
├── Broadcast: New receiver created
└── Messages: Continue flowing (no loss)
```

### Buffer Overflow Handling

```
broadcast::channel(100) fills up
│
├── Oldest messages dropped (lagged)
├── BroadcastStream receives Err(RecvError::Lagged)
├── Map to StreamEvent::KeepAlive
└── Continue operation (graceful degradation)
```

---

## Summary

The architecture demonstrates:

1. **Resilient Startup**: Server always starts, model is optional
2. **Event-Driven Flow**: From input to display via broadcast channels
3. **No Message Loss**: Broadcast buffers handle timing issues
4. **Graceful Degradation**: Fallbacks at every level
5. **Clean Separation**: Each layer has clear responsibilities

The system elegantly handles complex async flows while maintaining simplicity and robustness.
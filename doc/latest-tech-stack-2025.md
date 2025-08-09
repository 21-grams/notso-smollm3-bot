# Latest Technology Stack Reference (August 2025)

This document outlines the latest features, syntax, and best practices for our technology stack as of August 2025.

## Rust (Edition 2021, targeting stable 1.80+)

### Key Language Features Used

#### Type System Enhancements
```rust
// UUID v7 for sortable, timestamp-based IDs
use uuid::Uuid;
let message_id = Uuid::now_v7(); // Sortable by creation time

// Arc<Mutex<T>> pattern for async shared state
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};

pub struct AppState {
    pub sessions: Arc<RwLock<SessionManager>>, // Read-heavy
    pub model: Arc<Mutex<MLService>>,          // Write-heavy
}
```

#### Async/Await Patterns
```rust
// Modern async stream handling
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use async_stream::try_stream;

let stream = try_stream! {
    while let Some(msg) = receiver.recv().await {
        yield process_message(msg)?;
    }
};
```

#### Error Handling
```rust
// anyhow for application errors
use anyhow::{Result, Context};

async fn process() -> Result<String> {
    let data = fetch_data().await
        .context("Failed to fetch data")?;
    Ok(data)
}
```

## Axum 0.8 (January 2025 Release)

### Breaking Changes from 0.7
- Path syntax: `/{param}` instead of `/:param`
- `/{*rest}` for catch-all instead of `/*rest`
- Improved `Option<T>` extractors with `OptionalFromRequestParts`

### SSE Implementation
```rust
use axum::{
    response::sse::{Event, Sse, KeepAlive},
    extract::{State, Path},
};
use std::time::Duration;
use futures::stream::Stream;

pub async fn sse_handler(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = ReceiverStream::new(receiver)
        .map(|msg| {
            Ok(Event::default()
                .id(Uuid::now_v7().to_string()) // For reconnection
                .event("message")                // Named event
                .json_data(json!({               // JSON payload
                    "html": msg.content,
                    "target": msg.target,
                }))
                .unwrap())
        });
    
    Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(30))
                .text("keep-alive")
        )
}
```

### State Management
```rust
// Axum 0.8 automatically wraps in Arc, but explicit is clearer
let app = Router::new()
    .route("/api/send", post(send_message))
    .route("/api/stream/{session_id}", get(sse_handler))
    .with_state(Arc::new(app_state)); // Explicit Arc
```

### Graceful Shutdown
```rust
// Proper SSE connection handling during shutdown
axum::serve(listener, app)
    .with_graceful_shutdown(shutdown_signal())
    .await?;

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for ctrl-c");
    tracing::info!("Shutdown signal received");
}
```

### New Extractors
```rust
// Optional extractors that properly handle errors
use axum::extract::Query;

async fn handler(
    Query(params): Query<HashMap<String, String>>, // Required
    auth: Option<AuthUser>,  // Optional, won't fail request
) -> Result<Json<Response>, AppError> {
    // ...
}
```

## HTMX 2.0.6 (Latest as of August 2025)

### Installation
```html
<!-- Core HTMX -->
<script src="https://cdn.jsdelivr.net/npm/htmx.org@2.0.6/dist/htmx.min.js" 
        integrity="sha384-Akqfrbj/HpNVo8k11SXBb6TlBWmXXlYQrCSqEWmyKJe+hDm3Z/B2WVG4smwBkRVm" 
        crossorigin="anonymous"></script>

<!-- SSE Extension (separately versioned) -->
<script src="https://cdn.jsdelivr.net/npm/htmx-ext-sse@2.2.2/dist/sse.min.js"></script>

<!-- Optional: Morphing for smooth DOM updates -->
<script src="https://cdn.jsdelivr.net/npm/htmx-ext-idiomorph@2.0.0/dist/idiomorph-ext.min.js"></script>
```

### SSE Pattern (Persistent Connection)
```html
<body hx-ext="sse">
    <!-- Single persistent SSE connection per session -->
    <div id="sse-connection" 
         sse-connect="/api/stream/{{session_id}}"
         sse-close="session-end">
    </div>
    
    <!-- Messages are targeted by SSE events -->
    <div id="messages">
        <div id="msg-123" sse-swap="message-123">
            <!-- Content updated via SSE -->
        </div>
    </div>
</body>
```

### Key HTMX 2.0 Features
- **Extensions separated**: SSE, WebSocket now external extensions
- **Shadow DOM support**: Better Web Component integration
- **View Transitions API**: Smooth page transitions
- **Morphing swaps**: Preserve focus and form state during updates
- **DELETE compliance**: Uses query params per HTTP spec

### Event Handling
```javascript
// New event handling patterns
document.body.addEventListener('htmx:sseMessage', function(evt) {
    const data = JSON.parse(evt.detail.data);
    // Handle SSE message
});

// Reconnection handling
document.body.addEventListener('htmx:sseError', function(evt) {
    setTimeout(() => {
        htmx.trigger('#sse-connection', 'htmx:sseReconnect');
    }, 5000);
});
```

## MiniJinja 2.11+ Features

### Template Setup
```rust
use minijinja::{Environment, context, Value};

let mut env = Environment::new();

// Add templates
env.add_template("message", include_str!("templates/message.html"))?;

// Custom filters
env.add_filter("markdown", markdown_filter);
env.add_filter("timeago", timeago_filter);

// Custom functions
env.add_function("generate_id", generate_id);
```

### Advanced Template Features
```html
{# Template inheritance #}
{% extends "base.html" %}

{# Macros for reusable components #}
{% macro message_bubble(content, role) %}
<div class="message {{ role }}">
    {{ content|markdown|safe }}
</div>
{% endmacro %}

{# Conditional rendering with modern syntax #}
{% if thinking_mode %}
    <details class="thinking">
        <summary>Thinking...</summary>
        {{ thinking_content }}
    </details>
{% endif %}

{# Loop with index #}
{% for msg in messages %}
    <div id="msg-{{ loop.index }}">
        {{ msg.content|escape }}
    </div>
{% endfor %}
```

### Context Passing
```rust
// Structured context with serde
#[derive(Serialize)]
struct MessageContext {
    id: String,
    role: String,
    content: String,
    timestamp: DateTime<Utc>,
}

let html = tmpl.render(context! {
    message => MessageContext { ... },
    session_id => session_id,
    thinking_mode => state.thinking_mode,
})?;
```

## Session Architecture

### Single Receiver Pattern
```rust
pub struct SessionManager {
    sessions: HashMap<String, Session>,
    // Receivers stored separately until claimed by SSE
    pending_receivers: HashMap<String, mpsc::Receiver<StreamEvent>>,
}

pub struct Session {
    pub id: String,
    pub sender: mpsc::Sender<StreamEvent>, // Only sender stored
    // Receiver taken once by SSE connection
}

impl SessionManager {
    /// Creates session with receiver ready for SSE
    pub fn ensure_session_exists(&mut self, id: &str) {
        if !self.sessions.contains_key(id) {
            let (tx, rx) = mpsc::channel(100);
            
            self.sessions.insert(id.to_string(), Session {
                id: id.to_string(),
                sender: tx,
            });
            
            self.pending_receivers.insert(id.to_string(), rx);
        }
    }
    
    /// SSE endpoint takes receiver once
    pub fn take_receiver(&mut self, id: &str) -> Option<mpsc::Receiver<StreamEvent>> {
        self.ensure_session_exists(id);
        self.pending_receivers.remove(id) // Taken once!
    }
}
```

## Streaming Architecture

### Unified StreamingBuffer
```rust
pub struct StreamingBuffer {
    buffer: Vec<String>,
    token_threshold: usize,        // 10 tokens
    time_threshold: Duration,       // 500ms
    last_flush: Instant,
    sender: Sender<StreamEvent>,
}

impl StreamingBuffer {
    pub async fn push(&mut self, content: &str) -> Result<()> {
        self.buffer.push(content.to_string());
        
        if self.should_flush() {
            self.flush().await?;
        }
        Ok(())
    }
    
    fn should_flush(&self) -> bool {
        self.buffer.len() >= self.token_threshold ||
        self.last_flush.elapsed() >= self.time_threshold
    }
}
```

## Performance Optimizations

### Memory Management
- **KV Cache**: Optimized for SmolLM3's 4-group GQA (75% memory savings)
- **Streaming Buffers**: Token batching reduces DOM updates
- **Arc Usage**: Explicit Arc wrapping for shared state clarity

### Network Optimizations
- **Keep-Alive**: 30-second SSE keep-alive prevents proxy timeouts
- **Single SSE**: One persistent connection per session
- **Event IDs**: Enable reconnection with `Last-Event-ID` header

### Rendering Optimizations
- **Debounced Markdown**: Client-side rendering every 100ms
- **Morphing**: Preserves DOM state during updates
- **Progressive Rendering**: Markdown rendered during streaming

## Security Considerations

### Content Security
```rust
// HTML escaping
use html_escape;
let safe_content = html_escape::encode_text(&user_input);

// CSP nonce support
let nonce = generate_nonce();
response.headers_mut().insert(
    "Content-Security-Policy",
    format!("script-src 'nonce-{}'", nonce).parse()?
);
```

### Session Security
- UUIDs v7 for unpredictable session IDs
- Single-use receivers prevent hijacking
- Automatic session cleanup on disconnect

## Error Handling Patterns

### Graceful Degradation
```rust
// Fallback to stub mode when model unavailable
match load_model().await {
    Ok(model) => MLService::new(model),
    Err(e) => {
        tracing::warn!("Model load failed: {}, using stub", e);
        MLService::stub_mode()
    }
}
```

### SSE Error Recovery
```javascript
// Automatic reconnection with exponential backoff
let retryCount = 0;
function reconnectSSE() {
    const delay = Math.min(1000 * Math.pow(2, retryCount), 30000);
    setTimeout(() => {
        htmx.trigger('#sse-connection', 'htmx:sseReconnect');
        retryCount++;
    }, delay);
}
```

## Testing Patterns

### Async Testing
```rust
#[tokio::test]
async fn test_sse_stream() {
    let (tx, rx) = mpsc::channel(10);
    let stream = create_sse_stream(rx);
    
    tx.send(StreamEvent::Content("test".into())).await.unwrap();
    
    let events: Vec<_> = stream.take(1).collect().await;
    assert_eq!(events.len(), 1);
}
```

## Deployment Considerations

- **RUST_LOG**: Set to `info` for production
- **Tokio Runtime**: Multi-threaded with work stealing
- **Connection Limits**: Configure based on expected SSE connections
- **Proxy Headers**: Handle `X-Forwarded-For` for SSE connections

# HTMX SSE Streaming Solution Documentation

## Overview
A clean implementation of real-time text streaming using HTMX Server-Sent Events (SSE) with minimal JavaScript for markdown rendering.

## Architecture

### Key Components
1. **HTMX SSE Extension** - Manages SSE connection and handles out-of-band (OOB) swaps
2. **Rust Backend** - Streams content via SSE with message routing
3. **Minimal JavaScript** - Only for markdown rendering and UI helpers

### Data Flow
```
User Input → Backend Processing → SSE Stream → HTMX OOB Swaps → DOM Updates → Markdown Rendering
```

## Implementation Details

### 1. Session Management
- **Session ID**: Unique identifier per browser tab/connection
- **Message ID**: Unique identifier per message bubble
- **SSE Connection**: One persistent connection per session

### 2. Backend (Rust/Axum)

#### Message Structure
Each message creates a bubble with a unique ID:
```html
<div class="message assistant" id="msg-{message_id}">
    <div class="message-bubble">
        <span class="loading">Thinking...</span>
        <div id="msg-{message_id}-content" style="display:inline;"></div>
    </div>
</div>
```

#### SSE Events

**Content Streaming:**
```rust
StreamEvent::MessageContent { message_id, content } => {
    Event::default()
        .event("message")
        .data(format!(
            r#"<span hx-swap-oob="beforeend:#msg-{}-content">{}</span>"#,
            message_id,
            html_escape::encode_text(&content)
        ))
}
```
- Uses `hx-swap-oob` to target specific message content div
- Appends content chunks to the correct message bubble
- HTML-escaped for safety

**Completion Event:**
```rust
StreamEvent::MessageComplete { message_id } => {
    Event::default()
        .event("complete")
        .data(message_id)
}
```
- Sends just the message ID
- Triggers client-side markdown rendering

### 3. Frontend (HTMX + JavaScript)

#### HTMX Configuration
```html
<div class="chat-messages" id="chat-messages"
     hx-ext="sse"
     sse-connect="/api/stream/{{ session_id }}">
    <!-- Hidden SSE swap targets -->
    <div sse-swap="message" style="display:none;"></div>
    <div sse-swap="complete" style="display:none;"></div>
    <div sse-swap="message-error" style="display:none;"></div>
</div>
```

#### JavaScript (chat.js)
Minimal JavaScript for:
1. **Markdown Rendering** - Triggered by complete events
2. **UI Helpers** - Textarea resize, auto-scroll, form cleanup

```javascript
// Listen for SSE complete events to render markdown
document.addEventListener('sse:complete', function(e) {
    const messageId = e.detail?.data || e.data;
    const contentDiv = document.querySelector(`#msg-${messageId}-content`);
    
    if (contentDiv && contentDiv.textContent) {
        // Remove loading indicator
        const loading = contentDiv.parentElement.querySelector('.loading');
        if (loading) loading.remove();
        
        // Render markdown
        contentDiv.innerHTML = marked.parse(contentDiv.textContent);
        
        // Auto-scroll
        const messages = document.getElementById('chat-messages');
        if (messages) messages.scrollTop = messages.scrollHeight;
    }
});
```

## How It Works

### Streaming Flow
1. **User sends message** → Form submission via HTMX
2. **Backend returns HTML** → User message + Assistant message bubble with unique ID
3. **Backend starts streaming** → Sends SSE events to session channel
4. **HTMX processes OOB swaps** → Content appends to specific message div
5. **Completion triggers markdown** → JavaScript renders accumulated text

### Key Insights

#### Why OOB Swaps?
- **Problem**: Multiple messages can be streaming simultaneously
- **Solution**: `hx-swap-oob` targets specific elements by ID
- **Benefit**: No JavaScript needed for content routing

#### Why Minimal JavaScript?
- **HTMX Handles**: SSE connection, DOM updates, content appending
- **JavaScript Only For**: Markdown rendering (not possible in HTMX), UI polish

#### Session vs Message IDs
- **Session ID**: Routes SSE events to correct browser tab
- **Message ID**: Routes content to correct message bubble
- Both are essential for proper content delivery

## Benefits

1. **Clean Separation of Concerns**
   - HTMX: Transport and DOM manipulation
   - JavaScript: Presentation logic (markdown) only
   - Backend: Stateless streaming

2. **Efficient Streaming**
   - No JavaScript accumulation needed
   - Direct DOM updates via OOB swaps
   - Minimal overhead

3. **Maintainable Code**
   - External JavaScript file (chat.js)
   - Clean HTML templates
   - Simple backend events

4. **Real-time Experience**
   - Instant "Thinking..." feedback
   - Character-by-character streaming
   - Smooth markdown rendering on completion

## Trade-offs

### What We Achieved
- ✅ Pure HTMX for content streaming
- ✅ No JavaScript message accumulation
- ✅ Clean event-driven architecture
- ✅ Proper content routing via OOB swaps

### What Still Requires JavaScript
- 📝 Markdown rendering (unavoidable - HTMX can't parse markdown)
- 🎨 UI helpers (textarea resize, auto-scroll)
- 🔄 Loading indicator removal

## Future Improvements

1. **Server-side Markdown** - Could eliminate JavaScript entirely but loses raw text preview
2. **HTMX Extension** - Custom extension for markdown rendering
3. **WebSocket Upgrade** - For bidirectional communication if needed

## Code Metrics
- **JavaScript**: ~80 lines (only for markdown + UI)
- **HTMX Attributes**: 5 (minimal configuration)
- **Backend Complexity**: Simple, stateless streaming
- **Dependencies**: HTMX, SSE Extension, Marked.js

## Conclusion

This solution achieves near-pure HTMX streaming with the absolute minimum JavaScript required for markdown rendering. The use of OOB swaps for content routing is elegant and efficient, eliminating the need for JavaScript message handling entirely.

The architecture is:
- **Simple** - Easy to understand and maintain
- **Efficient** - Minimal overhead, direct DOM updates
- **Scalable** - Stateless backend, client-side rendering
- **Clean** - Clear separation of concerns

This represents the optimal balance between HTMX purity and practical requirements.
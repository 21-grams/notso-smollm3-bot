# Pure HTMX SSE Implementation

## Summary
Successfully implemented a pure HTMX SSE solution where the backend sends HTML fragments with out-of-band (OOB) swaps, eliminating the need for JavaScript message handling.

## Changes Made

### Frontend (`chat.html`)
- **Pure HTMX SSE**: Uses `sse-connect` and `sse-swap` for message streaming
- **Minimal JavaScript**: Only for markdown rendering and UI helpers
- **Clean event handling**: Listens for `render-markdown` trigger from complete events

### Backend (`api.rs`)
Modified SSE event formatting to send HTML fragments:

1. **Message Content Events**:
   ```html
   event: message
   data: <div id="msg-{id}-content" hx-swap-oob="append">{content}</div>
   ```
   - Uses `hx-swap-oob="append"` to append chunks to the correct message bubble
   - Content is HTML-escaped for safety

2. **Complete Events**:
   ```html
   event: complete
   data: <div id="msg-{id}" hx-trigger="render-markdown"></div>
   ```
   - Triggers markdown rendering via custom event
   - JavaScript listener converts accumulated text to formatted HTML

3. **Error Events**:
   ```html
   event: message-error
   data: <div id="msg-{id}-content" hx-swap-oob="innerHTML"><div class="error-message">Error: {error}</div></div>
   ```
   - Replaces content with error message

4. **Initial HTML Structure**:
   - Added `msg-{id}-content` div inside message bubble
   - Provides target for OOB swaps

## How It Works

1. **User sends message** → HTMX posts to `/api/chat`
2. **Server returns HTML** with message structure including `msg-{id}-content` div
3. **SSE streams chunks** as HTML with `hx-swap-oob="append"` targeting content div
4. **Chunks append directly** via HTMX OOB swaps - no JavaScript processing
5. **On completion**, trigger element fires `render-markdown` event
6. **Markdown renders** via minimal JavaScript handler

## Benefits

- ✅ **Pure HTMX for streaming** - No JavaScript message handling
- ✅ **Direct DOM updates** - HTMX handles all swapping via OOB
- ✅ **Clean separation** - Backend sends HTML, frontend just displays
- ✅ **Works for all message types** - Regular, /quote, system messages
- ✅ **More efficient** - No double buffering or data parsing

## Testing

1. Run server: `cargo run`
2. Navigate to `http://localhost:3000`
3. Send messages and observe:
   - Streaming text appears character by character
   - Markdown renders on completion
   - No JavaScript errors in console
   - Network tab shows HTML fragments in SSE
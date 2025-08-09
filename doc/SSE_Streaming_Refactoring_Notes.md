# SSE Streaming Refactoring Notes

## Current Implementation Status
**Date:** 2025-08-09  
**Status:** Working but uses JavaScript for accumulation

## Current Architecture

### Backend (Rust)
- Sends SSE events with HTML spans containing data attributes:
  ```html
  event: message
  data: <span class="msg-content" data-msg-id="{id}">{content}</span>
  
  event: complete
  data: <div data-complete-msg="{id}"></div>
  ```

### Frontend (HTML/HTMX/JS)
- HTMX SSE extension manages connection
- JavaScript accumulates chunks in a Map
- Markdown rendering triggered on complete event

## Why JavaScript is Currently Needed

1. **Text Accumulation**: HTMX can't accumulate streaming text chunks - it either replaces or appends DOM elements
2. **Markdown Rendering**: Need to call `marked.parse()` on the complete text
3. **Clean Text First**: Must show raw text while streaming, then render markdown when complete

## Potential Pure HTMX Approaches (Future Refactoring)

### Option 1: Server-Side Accumulation
**Backend changes:**
- Maintain accumulation buffer server-side per message
- Send complete accumulated text with each chunk
- Use `hx-swap-oob="innerHTML"` to replace entire content

**Pros:**
- True pure HTMX, no JavaScript needed
- Clean separation of concerns

**Cons:**
- Inefficient - sending all accumulated text with each chunk
- More server-side state management
- Higher bandwidth usage

### Option 2: Server-Side Markdown Rendering
**Backend changes:**
- Render markdown server-side
- Send HTML directly in SSE events
- Use `hx-swap-oob="beforeend"` to append HTML chunks

**Pros:**
- No client-side markdown rendering needed
- Could show formatted text as it streams

**Cons:**
- Can't show raw text during streaming (markdown partially rendered looks broken)
- Need Rust markdown parser (additional dependency)
- Harder to handle code blocks that span chunks

### Option 3: HTMX Extension
**Create custom HTMX extension:**
- Build an `sse-accumulate` extension
- Handle accumulation within HTMX's event system
- Trigger markdown rendering via HTMX events

**Pros:**
- Follows HTMX philosophy
- Reusable for other projects
- Clean integration

**Cons:**
- Requires writing and maintaining custom extension
- Still JavaScript, just packaged differently

## Current Trade-offs

The current implementation with minimal JavaScript is actually optimal because:

1. **Efficient**: Only sends new content chunks, not accumulated text
2. **Clean Display**: Shows raw text while streaming, formatted when complete  
3. **Simple Backend**: No server-side accumulation state
4. **Minimal JS**: ~50 lines of JavaScript only for accumulation and markdown

## Recommendations for Future

1. **Keep current architecture** unless bandwidth becomes an issue
2. **Consider server-side markdown** only if we need to support non-JS clients
3. **Document the JavaScript helpers** as essential infrastructure, not business logic
4. **Possible optimization**: Batch chunks server-side to reduce SSE events

## Code Metrics

- **JavaScript for SSE handling**: ~50 lines
- **JavaScript for UI helpers**: ~30 lines  
- **Pure HTMX elements**: Everything else
- **Backend complexity**: Simple stateless streaming

## Conclusion

While not 100% pure HTMX, the current solution is pragmatic and efficient. The JavaScript is minimal, focused, and handles only what HTMX cannot: text accumulation and markdown rendering. This is a reasonable trade-off for streaming markdown content.
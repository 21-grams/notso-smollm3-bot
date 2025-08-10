# Markdown Rendering Fix Documentation

## Problem Statement
Messages streamed via SSE were being incorrectly wrapped in `<pre><code>` tags instead of being rendered as proper markdown. This affected all messages including system messages and `/quote` command outputs.

### Example of Issue
**Expected Output:**
```html
🔴 <strong>Model not loaded</strong> Using FTS5 search...
```

**Actual Output (Incorrect):**
```html
<pre><code class="hljs language-vbnet">
🔴 **Model not loaded** Using FTS5 search...
</code></pre>
```

## Root Cause Analysis

The issue was caused by a chain of three interacting systems:

### 1. Rust HTML Generation with Indentation
The server generates HTML with readable indentation (`src/web/handlers/api.rs`):
```rust
r#"<div class="message assistant">
    <div class="message-bubble" 
         id="msg-{}-bubble"
         sse-swap="msg-{}"
         hx-swap="beforeend">
        <span class="thinking">Thinking...</span>
    </div>
</div>"#
```

### 2. HTMX Content Appending
- HTMX's `hx-swap="beforeend"` preserves the indentation context of the container
- Content streamed via SSE gets appended with the same 8-space indentation
- Result: `        🔴 **Model not loaded**...`

### 3. Markdown's Code Block Rule
- Markdown specification: Lines starting with 4+ spaces are treated as code blocks
- The 8-space indentation triggered this rule
- `marked.js` correctly wrapped the content in `<pre><code>` tags

## Solution Implementation

### Location
`src/web/templates/chat.html` - Lines 54-91 in the `htmx:sseBeforeMessage` event handler

### Code
```javascript
// Handle complete event - render markdown and highlight code
if (evt.detail.type === 'complete') {
    evt.preventDefault();
    
    const messageId = evt.detail.data;
    const bubble = document.getElementById(`msg-${messageId}-bubble`);
    
    if (bubble && !bubble.classList.contains('completed')) {
        // Remove thinking indicator
        const thinking = bubble.querySelector('.thinking');
        if (thinking) thinking.remove();
        
        // Get text content and clean up indentation
        const cleanedText = bubble.textContent
            .split('\n')           // Split into lines
            .map(line => line.trim())  // Remove leading/trailing whitespace
            .filter(line => line.length > 0)  // Remove empty lines
            .join('\n');           // Rejoin with newlines
        
        // Parse as markdown and replace bubble content
        bubble.innerHTML = marked.parse(cleanedText);
        
        // Apply highlighting only to actual code blocks with language specified
        bubble.querySelectorAll('pre code[class*="language-"]').forEach(block => {
            hljs.highlightElement(block);
        });
        
        bubble.classList.add('completed');
        document.getElementById('chat-messages').scrollTop = 
            document.getElementById('chat-messages').scrollHeight;
    }
}
```

### How It Works

1. **Text Extraction**: `bubble.textContent` gets the raw accumulated text from SSE streaming
2. **Line Processing**: 
   - Split text into lines
   - Trim whitespace from each line (removes the 8-space indentation)
   - Filter out empty lines
   - Join back with newlines
3. **Markdown Parsing**: `marked.parse()` converts markdown syntax to HTML
4. **Selective Highlighting**: Only apply syntax highlighting to actual code blocks with language classes
5. **Completion Marking**: Add `completed` class to prevent re-processing

## Why This Solution Works

- **Removes problematic indentation** that triggers markdown's code block rule
- **Preserves intentional markdown** like `**bold**`, `*italic*`, and real code blocks
- **Two-pass processing**: First markdown parsing, then selective syntax highlighting
- **Efficient**: Single DOM update with `innerHTML` replacement

## Files Modified

1. **`src/web/templates/chat.html`** 
   - Added markdown processing in complete event handler
   - Configured highlight.js to not auto-process content

2. **`src/web/static/js/chat.js`**
   - Removed conflicting markdown configuration
   - Now handles only SSE connection and UI interactions

## Testing

To verify the fix works:

1. **Test markdown rendering**:
   ```
   Send: "Test **bold** and *italic* and `inline code`"
   Expected: Test **bold** and *italic* and `inline code`
   ```

2. **Test code blocks**:
   ````
   Send: "Here's code:
   ```rust
   let x = 42;
   ```"
   Expected: Properly highlighted Rust code block
   ````

3. **Test `/quote` command**:
   ```
   Send: "/quote"
   Expected: Gospel text with H1 header and bold verse numbers
   ```

## Potential Issues & Limitations

1. **Emoji with markdown**: Some markdown parsers have issues with emojis directly adjacent to markdown syntax
   - Solution: Add space between emoji and markdown

2. **Multi-line content**: The current solution removes all leading whitespace
   - This is fine for messages but might affect intentionally indented content
   - If needed, could implement smarter dedenting that preserves relative indentation

## Alternative Approaches Considered

1. **Server-side fix**: Remove indentation in Rust HTML generation
   - Rejected: Would make code less readable
   
2. **Modify StreamingBuffer**: Strip whitespace during streaming
   - Rejected: Would affect all content, not just markdown

3. **CSS white-space property**: Use CSS to handle whitespace
   - Rejected: Wouldn't fix markdown parsing issue

## Conclusion

The solution elegantly handles the interaction between three systems (Rust HTML generation, HTMX streaming, and Markdown parsing) by cleaning the text at the right point - after accumulation but before markdown parsing. This maintains code readability in Rust while ensuring proper markdown rendering in the browser.
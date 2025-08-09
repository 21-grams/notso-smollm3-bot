// Enhanced chat functionality with markdown rendering and streaming

// Initialize marked.js for markdown (loaded from CDN)
marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function(code, lang) {
        // Use highlight.js if available
        if (typeof hljs !== 'undefined' && lang) {
            try {
                return hljs.highlight(code, { language: lang }).value;
            } catch (e) {}
        }
        return code;
    }
});

// Message streaming handler
class MessageStream {
    constructor(messageId) {
        this.messageId = messageId;
        this.contentBuffer = [];
        this.isThinking = false;
        this.contentElement = document.getElementById(`content-${messageId}`);
        this.renderTimer = null;
    }
    
    // Handle incoming SSE events
    handleEvent(event) {
        const data = JSON.parse(event.data);
        
        switch(event.type) {
            case 'thinking':
                this.handleThinking(data);
                break;
            case 'content':
                this.handleContent(data);
                break;
            case 'complete':
                this.handleComplete(data);
                break;
            case 'error':
                this.handleError(data);
                break;
        }
    }
    
    // Handle thinking mode content
    handleThinking(data) {
        const thinkingArea = document.getElementById(`thinking-${this.messageId}`);
        if (data.action === 'start') {
            thinkingArea.style.display = 'block';
            this.isThinking = true;
        } else if (data.action === 'append') {
            thinkingArea.querySelector('.thinking-content').innerHTML += data.html;
        } else if (data.action === 'end') {
            this.isThinking = false;
        }
    }
    
    // Handle content streaming
    handleContent(data) {
        if (data.action === 'append') {
            // Remove loading indicator on first content
            const loading = this.contentElement.querySelector('.loading-indicator');
            if (loading) {
                loading.remove();
            }
            
            // Add to buffer
            this.contentBuffer.push(data.html);
            
            // Debounced render for performance
            this.scheduleRender();
        } else if (data.action === 'replace') {
            this.contentElement.innerHTML = data.html;
        }
    }
    
    // Debounced markdown rendering
    scheduleRender() {
        if (this.renderTimer) {
            clearTimeout(this.renderTimer);
        }
        
        this.renderTimer = setTimeout(() => {
            this.renderMarkdown();
        }, 100); // Render every 100ms max
    }
    
    // Render accumulated content as markdown
    renderMarkdown() {
        const rawContent = this.contentBuffer.join('');
        const renderedHtml = marked.parse(rawContent);
        
        // Preserve scroll position
        const scrollPos = this.contentElement.scrollTop;
        
        // Update content with markdown
        this.contentElement.innerHTML = `
            <div class="markdown-body">
                ${renderedHtml}
            </div>
        `;
        
        // Restore scroll
        this.contentElement.scrollTop = scrollPos;
        
        // Highlight code blocks if highlight.js is available
        if (typeof hljs !== 'undefined') {
            this.contentElement.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }
    }
    
    // Handle completion
    handleComplete(data) {
        // Final render
        this.renderMarkdown();
        
        // Show actions and metadata
        document.getElementById(`actions-${this.messageId}`).style.display = 'flex';
        document.getElementById(`footer-${this.messageId}`).style.display = 'flex';
        
        // Update metadata if provided
        if (data.tokenCount) {
            document.querySelector(`#footer-${this.messageId} .token-count`).textContent = 
                `${data.tokenCount} tokens`;
        }
        if (data.generationTime) {
            document.querySelector(`#footer-${this.messageId} .generation-time`).textContent = 
                `${data.generationTime}ms`;
        }
        
        // Auto-scroll to bottom
        this.scrollToBottom();
    }
    
    // Handle errors
    handleError(data) {
        this.contentElement.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${data.message || 'Something went wrong'}
            </div>
        `;
    }
    
    // Scroll chat to bottom
    scrollToBottom() {
        const chatContainer = document.getElementById('messages');
        if (chatContainer) {
            chatContainer.scrollTo({
                top: chatContainer.scrollHeight,
                behavior: 'smooth'
            });
        }
    }
}

// Global message streams tracker
const activeStreams = new Map();

// HTMX event handlers
document.body.addEventListener('htmx:sseMessage', function(evt) {
    // Parse the SSE message
    const messageData = evt.detail;
    
    // Extract message ID from target
    const targetMatch = messageData.target?.match(/msg-([a-f0-9-]+)/);
    if (targetMatch) {
        const messageId = targetMatch[1];
        
        // Get or create stream handler
        if (!activeStreams.has(messageId)) {
            activeStreams.set(messageId, new MessageStream(messageId));
        }
        
        const stream = activeStreams.get(messageId);
        stream.handleEvent(messageData);
        
        // Clean up on complete
        if (messageData.type === 'complete') {
            activeStreams.delete(messageId);
        }
    }
});

// Copy message content
function copyMessage(messageId) {
    const content = document.querySelector(`#content-${messageId} .markdown-body`);
    if (content) {
        // Get plain text version
        const text = content.innerText || content.textContent;
        
        // Copy to clipboard
        navigator.clipboard.writeText(text).then(() => {
            // Show feedback
            const btn = event.target.closest('.copy');
            const originalText = btn.innerHTML;
            btn.innerHTML = '✓ Copied';
            setTimeout(() => {
                btn.innerHTML = originalText;
            }, 2000);
        });
    }
}

// Update timestamps periodically
setInterval(() => {
    document.querySelectorAll('.timestamp[data-time]').forEach(el => {
        const time = new Date(el.dataset.time);
        el.textContent = timeAgo(time);
    });
}, 60000); // Update every minute

// Time ago helper
function timeAgo(date) {
    const seconds = Math.floor((new Date() - date) / 1000);
    
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
}

// Chat functionality - handles SSE events and markdown rendering

// Configure marked for markdown parsing
marked.setOptions({ 
    breaks: true, 
    gfm: true, 
    sanitize: false 
});

// Track connection state
let sseConnectionActive = false;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

// Listen for SSE connection events
document.addEventListener('htmx:sseOpen', function(e) {
    console.log('[SSE] Connection opened');
    sseConnectionActive = true;
    reconnectAttempts = 0;
    updateConnectionStatus('connected');
});

document.addEventListener('htmx:sseError', function(e) {
    console.error('[SSE] Connection error:', e);
    sseConnectionActive = false;
    updateConnectionStatus('error');
    
    // Handle reconnection with exponential backoff
    if (reconnectAttempts < maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
        console.log(`[SSE] Reconnecting in ${delay}ms (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`);
        reconnectAttempts++;
        
        setTimeout(() => {
            // HTMX SSE extension should auto-reconnect
            // If not, we may need to manually trigger
            if (!sseConnectionActive) {
                console.log('[SSE] Attempting reconnection...');
                // The SSE extension handles reconnection automatically
            }
        }, delay);
    } else {
        updateConnectionStatus('failed');
        console.error('[SSE] Max reconnection attempts reached');
    }
});

document.addEventListener('htmx:sseClose', function(e) {
    console.log('[SSE] Connection closed');
    sseConnectionActive = false;
    updateConnectionStatus('disconnected');
});

// Update connection status indicator
function updateConnectionStatus(status) {
    const statusBadge = document.querySelector('.status-badge');
    if (!statusBadge) return;
    
    switch(status) {
        case 'connected':
            statusBadge.textContent = '🟢 Connected';
            statusBadge.style.color = 'var(--success-color, #10b981)';
            break;
        case 'error':
            statusBadge.textContent = '🟡 Reconnecting...';
            statusBadge.style.color = 'var(--warning-color, #f59e0b)';
            break;
        case 'disconnected':
            statusBadge.textContent = '🔴 Disconnected';
            statusBadge.style.color = 'var(--error-color, #ef4444)';
            break;
        case 'failed':
            statusBadge.textContent = '⚠️ Connection Failed';
            statusBadge.style.color = 'var(--error-color, #ef4444)';
            // Show user a message to refresh the page
            showErrorMessage('Connection lost. Please refresh the page to reconnect.');
            break;
    }
}

// Show error message to user
function showErrorMessage(message) {
    const messages = document.getElementById('chat-messages');
    if (messages) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'system-message error';
        errorDiv.innerHTML = `
            <div class="message-bubble">
                <strong>System:</strong> ${message}
                <button onclick="location.reload()" style="margin-left: 10px; padding: 5px 10px; background: #3b82f6; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Refresh Page
                </button>
            </div>
        `;
        messages.appendChild(errorDiv);
        messages.scrollTop = messages.scrollHeight;
    }
}

// Listen for SSE complete events to render markdown
document.addEventListener('sse:complete', function(e) {
    // Get the message ID from the event data
    const messageId = e.detail?.data || e.data;
    if (!messageId) return;
    
    console.log('[Markdown] Rendering for message:', messageId);
    
    // Find the content div for this message
    const contentDiv = document.querySelector(`#msg-${messageId}-content`);
    if (contentDiv && contentDiv.textContent) {
        // Remove loading indicator from parent bubble
        const bubble = contentDiv.parentElement;
        const loading = bubble.querySelector('.loading');
        if (loading) loading.remove();
        
        // Get the raw text and render as markdown
        const rawText = contentDiv.textContent;
        contentDiv.innerHTML = marked.parse(rawText);
        
        // Auto-scroll to show new content
        const messages = document.getElementById('chat-messages');
        if (messages) {
            messages.scrollTop = messages.scrollHeight;
        }
    }
});

// Remove loading indicators when content starts streaming
document.addEventListener('htmx:oobAfterSwap', function(e) {
    if (e.detail && e.detail.target) {
        const parent = e.detail.target.parentElement;
        if (parent) {
            const loading = parent.querySelector('.loading');
            if (loading) {
                loading.remove();
            }
        }
    }
});

// Auto-resize textarea as user types
const messageInput = document.getElementById('message-input');
if (messageInput) {
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
    
    // Submit on Enter (without Shift)
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            
            // Check if connection is active before submitting
            if (!sseConnectionActive) {
                showErrorMessage('Connection lost. Please refresh the page to send messages.');
                return;
            }
            
            document.getElementById('chat-form').requestSubmit();
        }
    });
}

// Form cleanup after submission
document.body.addEventListener('htmx:afterRequest', function(evt) {
    if (evt.detail && evt.detail.elt && evt.detail.elt.id === 'chat-form') {
        const messageInput = document.getElementById('message-input');
        if (messageInput) {
            messageInput.value = '';
            messageInput.style.height = 'auto';
            messageInput.focus();
        }
    }
});

// Auto-scroll when new messages are added
document.body.addEventListener('htmx:afterSwap', function(evt) {
    if (evt.detail && evt.detail.target && evt.detail.target.id === 'chat-messages') {
        const messages = document.getElementById('chat-messages');
        if (messages) {
            messages.scrollTop = messages.scrollHeight;
        }
    }
});

// Handle beforeunload to clean up SSE connection
window.addEventListener('beforeunload', function(e) {
    console.log('[SSE] Page unloading, connection will close');
    // The SSE connection will be closed automatically
});
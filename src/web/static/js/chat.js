// SmolLM3 Chat Application JavaScript

// Initialize HTMX SSE extension
document.addEventListener('DOMContentLoaded', function() {
    console.log('SmolLM3 Chat initialized');
    
    // Handle SSE events
    document.body.addEventListener('htmx:sseMessage', function(evt) {
        handleSSEMessage(evt.detail);
    });
    
    // Handle form submissions
    const chatForm = document.getElementById('chat-form');
    if (chatForm) {
        chatForm.addEventListener('htmx:afterRequest', function() {
            // Clear input after successful submission
            const input = chatForm.querySelector('input[name="message"]');
            if (input) input.value = '';
        });
    }
    
    // Handle thinking mode toggle
    const thinkingToggle = document.getElementById('thinking-toggle');
    if (thinkingToggle) {
        thinkingToggle.addEventListener('change', function() {
            toggleThinkingMode(this.checked);
        });
    }
});

// Handle SSE messages
function handleSSEMessage(detail) {
    const data = JSON.parse(detail.data);
    const messageId = data.message_id;
    
    switch(detail.type) {
        case 'thinking_start':
            showThinkingIndicator(messageId);
            break;
        case 'thinking':
            appendThinkingContent(messageId, data.content);
            break;
        case 'thinking_end':
            hideThinkingIndicator(messageId);
            break;
        case 'token':
            appendToken(messageId, data.content);
            break;
        case 'done':
            markMessageComplete(messageId);
            break;
        case 'error':
            handleError(messageId, data.error);
            break;
    }
}

// Show thinking indicator
function showThinkingIndicator(messageId) {
    const container = document.querySelector(`[data-message-id="${messageId}"]`);
    if (container) {
        const indicator = document.createElement('div');
        indicator.className = 'thinking-indicator';
        indicator.innerHTML = '🤔 Thinking...';
        container.appendChild(indicator);
    }
}

// Append thinking content
function appendThinkingContent(messageId, content) {
    const container = document.querySelector(`[data-message-id="${messageId}"] .thinking-content`);
    if (container) {
        container.innerHTML += content;
    }
}

// Hide thinking indicator
function hideThinkingIndicator(messageId) {
    const indicator = document.querySelector(`[data-message-id="${messageId}"] .thinking-indicator`);
    if (indicator) {
        indicator.innerHTML = '✨ Reasoning complete';
        setTimeout(() => indicator.style.display = 'none', 1000);
    }
}

// Append token to message
function appendToken(messageId, token) {
    const container = document.querySelector(`[data-message-id="${messageId}"] .message-text`);
    if (container) {
        container.innerHTML += token;
        // Auto-scroll to bottom
        const chatMessages = document.querySelector('.chat-messages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
}

// Mark message as complete
function markMessageComplete(messageId) {
    const container = document.querySelector(`[data-message-id="${messageId}"]`);
    if (container) {
        container.classList.add('complete');
        // Enable input for next message
        const input = document.querySelector('#chat-form input[name="message"]');
        if (input) input.disabled = false;
    }
}

// Handle errors
function handleError(messageId, error) {
    const container = document.querySelector(`[data-message-id="${messageId}"]`);
    if (container) {
        container.classList.add('error');
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = `Error: ${error}`;
        container.appendChild(errorDiv);
    }
}

// Toggle thinking mode
function toggleThinkingMode(enabled) {
    // Send request to server to update session
    fetch('/api/toggle-thinking', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ enabled: enabled })
    });
    
    // Update UI indicator
    const indicator = document.querySelector('.thinking-mode-status');
    if (indicator) {
        indicator.textContent = enabled ? 'Thinking Mode: ON' : 'Thinking Mode: OFF';
    }
}

// Utility function to copy message content
function copyMessage(messageId) {
    const content = document.querySelector(`[data-message-id="${messageId}"] .message-text`);
    if (content) {
        navigator.clipboard.writeText(content.textContent)
            .then(() => {
                // Show feedback
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = '✓ Copied';
                setTimeout(() => btn.textContent = originalText, 2000);
            });
    }
}

// Utility function to regenerate response
function regenerateResponse(messageId) {
    // Implementation for regenerating a response
    console.log('Regenerating response for:', messageId);
}

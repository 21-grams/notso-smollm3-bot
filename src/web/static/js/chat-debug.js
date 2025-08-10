// Chat functionality - handles SSE events and markdown rendering
// DEBUG VERSION - with extensive logging

// Configure marked for markdown parsing
marked.setOptions({ 
    breaks: true, 
    gfm: true, 
    sanitize: false,
    // Only highlight code blocks with language specified
    highlight: function(code, lang) {
        console.log('[Marked] Highlight called with lang:', lang, 'code length:', code.length);
        if (lang && typeof hljs !== 'undefined' && hljs.getLanguage(lang)) {
            try {
                const result = hljs.highlight(code, { language: lang });
                console.log('[Marked] Successfully highlighted as', lang);
                return result.value;
            } catch (e) {
                console.warn('[Marked] Highlight error:', e);
                return code;
            }
        }
        return code;
    }
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
            if (!sseConnectionActive) {
                console.log('[SSE] Attempting reconnection...');
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

// Listen for SSE complete events to render markdown
document.addEventListener('sse:complete', function(e) {
    console.log('[DEBUG] sse:complete event triggered');
    console.log('[DEBUG] Event detail:', e.detail);
    console.log('[DEBUG] Event data:', e.data);
    
    // Get the message ID from the event data
    const messageId = e.detail?.data || e.data;
    if (!messageId) {
        console.error('[DEBUG] No message ID in complete event');
        return;
    }
    
    console.log('[Markdown] Processing complete for message:', messageId);
    
    // Find the bubble for this message (not content div)
    const bubble = document.querySelector(`#msg-${messageId}-bubble`);
    console.log('[DEBUG] Found bubble:', bubble);
    
    if (bubble) {
        // Log current bubble state
        console.log('[DEBUG] Bubble innerHTML before processing:', bubble.innerHTML);
        console.log('[DEBUG] Bubble textContent:', bubble.textContent);
        
        // Check if bubble already has pre/code elements
        const existingPre = bubble.querySelector('pre');
        const existingCode = bubble.querySelector('code');
        console.log('[DEBUG] Existing <pre> elements:', existingPre);
        console.log('[DEBUG] Existing <code> elements:', existingCode);
        
        // Remove loading indicator from bubble
        const loading = bubble.querySelector('.loading');
        if (loading) {
            console.log('[DEBUG] Removing loading indicator');
            loading.remove();
        }
        
        // Get the raw text content
        const rawText = bubble.textContent || bubble.innerText;
        console.log('[DEBUG] Raw text to parse:', rawText);
        
        // Check if the text looks like it's already been processed
        if (bubble.querySelector('pre code[data-highlighted="yes"]')) {
            console.error('[DEBUG] Message already has highlighted code blocks! Skipping processing.');
            return;
        }
        
        // Parse markdown
        console.log('[DEBUG] Calling marked.parse()...');
        const formattedHtml = marked.parse(rawText);
        console.log('[DEBUG] Formatted HTML from marked:', formattedHtml);
        
        // Replace bubble content
        bubble.innerHTML = formattedHtml;
        bubble.classList.add('completed');
        
        // Apply syntax highlighting to code blocks only
        if (typeof hljs !== 'undefined') {
            const codeBlocks = bubble.querySelectorAll('pre code');
            console.log('[DEBUG] Found', codeBlocks.length, 'code blocks to highlight');
            
            codeBlocks.forEach((block, index) => {
                console.log(`[DEBUG] Code block ${index} classes:`, block.className);
                console.log(`[DEBUG] Code block ${index} content preview:`, block.textContent.substring(0, 50));
                
                // Only highlight blocks with language class
                if (block.className && block.className.includes('language-')) {
                    if (!block.dataset.highlighted) {
                        console.log(`[DEBUG] Highlighting block ${index} as`, block.className);
                        hljs.highlightElement(block);
                        block.dataset.highlighted = 'yes';
                    } else {
                        console.log(`[DEBUG] Block ${index} already highlighted`);
                    }
                } else {
                    console.log(`[DEBUG] Block ${index} has no language class, skipping highlight`);
                }
            });
        }
        
        // Auto-scroll to show new content
        const messages = document.getElementById('chat-messages');
        if (messages) {
            messages.scrollTop = messages.scrollHeight;
        }
    } else {
        // Try finding with different selector patterns
        console.error('[DEBUG] Bubble not found with standard selector');
        console.log('[DEBUG] Looking for alternative selectors...');
        
        // Log all elements with IDs containing the message ID
        const allElements = document.querySelectorAll(`[id*="${messageId}"]`);
        console.log('[DEBUG] Elements with message ID in their ID:', allElements);
        
        // Check if there's a content div instead
        const contentDiv = document.querySelector(`#msg-${messageId}-content`);
        if (contentDiv) {
            console.log('[DEBUG] Found content div instead of bubble:', contentDiv);
            console.log('[DEBUG] Content div innerHTML:', contentDiv.innerHTML);
        }
    }
});

// Also listen for HTMX SSE events to see what's happening
document.addEventListener('htmx:sseBeforeMessage', function(e) {
    console.log('[DEBUG] htmx:sseBeforeMessage event:', e.detail);
});

document.addEventListener('htmx:sseMessage', function(e) {
    console.log('[DEBUG] htmx:sseMessage event:', e.detail);
});

// Remove loading indicators when content starts streaming
document.addEventListener('htmx:oobAfterSwap', function(e) {
    console.log('[DEBUG] htmx:oobAfterSwap event:', e.detail);
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
    console.log('[DEBUG] htmx:afterRequest:', evt.detail);
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
    console.log('[DEBUG] htmx:afterSwap:', evt.detail);
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
});
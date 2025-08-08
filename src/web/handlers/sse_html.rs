//! SSE handler that sends pure HTML for HTMX to swap

use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use std::convert::Infallible;
use tokio_stream::StreamExt;

/// Generate SSE events that send HTML fragments for HTMX to swap directly
pub fn create_sse_html_stream(
    session_id: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    // Create a stream that yields HTML fragments
    async_stream::stream! {
        // Example: When user sends a message, we start streaming the response
        
        // 1. First, send the assistant message container
        yield Ok(Event::default()
            .event("message-start")
            .data(r#"<div class="message assistant" id="current-message">
                <div class="message-bubble" id="current-bubble"></div>
            </div>"#));
        
        // 2. Stream tokens as they come
        // Each token event sends a small HTML fragment that HTMX appends
        let tokens = ["Hello", " there!", " How", " can", " I", " help", " you", " today?"];
        for token in tokens {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            
            // Send HTML that will be appended to the current message bubble
            // Using hx-swap-oob (out of band) to target specific element
            yield Ok(Event::default()
                .event("token")
                .data(format!(
                    r#"<span hx-swap-oob="beforeend:#current-bubble">{}</span>"#,
                    token
                )));
        }
        
        // 3. Send completion event
        yield Ok(Event::default()
            .event("complete")
            .data(r#"<div hx-swap-oob="afterend:#current-message">
                <!-- Message complete marker -->
            </div>"#));
    }
}

/// Even simpler approach - send complete HTML chunks
pub fn create_simple_sse_stream(
    message: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        // Start with empty assistant message
        let message_id = uuid::Uuid::new_v4().to_string();
        
        yield Ok(Event::default()
            .event("message")
            .data(format!(r#"
                <div class="message assistant" id="msg-{}">
                    <div class="message-bubble"></div>
                </div>
            "#, message_id)));
        
        // Simulate streaming response
        let response_parts = [
            "I'm", "responding", "to", "your", "message:", 
            &format!("'{}'", message), "with", "a", "helpful", "answer."
        ];
        
        let mut accumulated = String::new();
        for part in response_parts {
            accumulated.push_str(part);
            accumulated.push(' ');
            tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
            
            // Send the complete message with all tokens so far
            // HTMX will replace the entire message bubble
            yield Ok(Event::default()
                .event("message")
                .data(format!(r#"
                    <div class="message assistant" id="msg-{}" hx-swap-oob="true">
                        <div class="message-bubble">{}</div>
                    </div>
                "#, message_id, html_escape::encode_text(&accumulated))));
        }
    }
}

//! API endpoint handlers

use crate::state::AppState;
// use crate::services::template::ChatTemplateService; // Currently unused
use crate::services::streaming::StreamingBuffer;
use crate::types::events::StreamEvent;
use axum::{
    extract::{State, Path, Form},
    response::{Html, sse::{Event, Sse, KeepAlive}},
    // http::StatusCode, // Currently unused
};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    session_id: String,
    message: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    success: bool,
    html: String,
}

/// Handle chat message submission (optimized for persistent SSE)
pub async fn send_message(
    State(state): State<AppState>,
    Form(msg): Form<ChatMessage>,
) -> Html<String> {
    let message_id = Uuid::now_v7().to_string();
    
    // Return HTML without SSE connection (using existing persistent connection)
    let html = format!(
        r#"<div class="message user">
            <div class="message-bubble">{}</div>
        </div>
        <div class="message assistant" id="msg-{}">
            <div class="message-bubble">
                <span class="loading">Thinking...</span>
            </div>
        </div>"#,
        html_escape::encode_text(&msg.message),
        message_id
    );
    
    // Clone values for the spawned task
    let session_id = msg.session_id.clone();
    let message = msg.message.clone();
    let state_clone = state.clone();
    let msg_id = message_id.clone();
    
    // Handle command or model generation in background
    tokio::spawn(async move {
        // Send a targeted event for this specific message
        let sender = {
            let mut sessions = state_clone.sessions.write().await;
            sessions.get_or_create_sender(&session_id)
        };
        
        // Send message start event with target ID
        let _ = sender.send(StreamEvent::Content(
            format!("<div hx-target='#msg-{}' hx-swap='innerHTML'></div>", msg_id)
        )).await;
        
        if message.starts_with("/quote") {
            // Use buffered streaming for quote
            let _ = stream_quote_buffered(state_clone, session_id).await;
        } else {
            // Regular model generation (also uses buffer)
            let _ = generate_response_buffered(state_clone, session_id, message).await;
        }
    });
    
    Html(html)
}

/// Implement persistent SSE endpoint
pub async fn stream_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Get or create the receiver for this session
    let receiver = {
        let mut sessions = state.sessions.write().await;
        sessions.take_receiver(&session_id)
            .expect("Receiver should be available for new SSE connection")
    };
    
    // Create stream from receiver
    let stream = ReceiverStream::new(receiver)
        .map(|event| {
            let sse_event = match event {
                StreamEvent::Content(text) => {
                    Event::default()
                        .event("message")
                        .data(text)
                }
                StreamEvent::Complete => {
                    Event::default()
                        .event("complete")
                        .data("done")
                }
                StreamEvent::Error(err) => {
                    Event::default()
                        .event("error")
                        .data(err)
                }
                _ => {
                    Event::default()
                        .event("status")
                        .data("processing")
                }
            };
            Ok(sse_event)
        });
    
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(30))
            .text("keep-alive")
    )
}

/// Stream quote content through the buffer
async fn stream_quote_buffered(
    state: AppState,
    session_id: String,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get session's event sender
    let sender = {
        let mut sessions = state.sessions.write().await;
        sessions.get_or_create_sender(&session_id)
    };
    
    // Create streaming buffer
    let mut buffer = StreamingBuffer::new(sender, session_id.clone());
    
    // The Gospel of John 1:1-14 text
    let scripture_text = r#"# Gospel of John 1:1-14
*Recovery Version*

**1** In the beginning was the Word, and the Word was with God, and the Word was God.

**2** He was in the beginning with God.

**3** All things came into being through Him, and apart from Him not one thing came into being which has come into being.

**4** In Him was life, and the life was the light of men.

**5** And the light shines in the darkness, and the darkness did not overcome it.

**6** There came a man sent from God, whose name was John.

**7** He came for a testimony that he might testify concerning the light, that all might believe through him.

**8** He was not the light, but came that he might testify concerning the light.

**9** This was the true light which, coming into the world, enlightens every man.

**10** He was in the world, and the world came into being through Him, yet the world did not know Him.

**11** He came to His own, yet those who were His own did not receive Him.

**12** But as many as received Him, to them He gave the authority to become children of God, to those who believe into His name,

**13** Who were begotten not of blood, nor of the will of the flesh, nor of the will of man, but of God.

**14** And the Word became flesh and tabernacled among us (and we beheld His glory, glory as of the only Begotten from the Father), full of grace and reality."#;
    
    // Split into words for token-like streaming
    let words: Vec<&str> = scripture_text.split_whitespace().collect();
    
    // Push words through buffer
    for word in words {
        buffer.push(&format!("{} ", word)).await?;
        
        // Small delay to simulate generation
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    
    // Signal completion
    buffer.complete().await?;
    
    Ok(())
}

/// Generate response through buffer (stub for now)
async fn generate_response_buffered(
    state: AppState,
    session_id: String,
    message: String,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get session's event sender
    let sender = {
        let mut sessions = state.sessions.write().await;
        sessions.get_or_create_sender(&session_id)
    };
    
    // Create streaming buffer
    let mut buffer = StreamingBuffer::new(sender, session_id.clone());
    
    // For now, just echo back with a simple response
    let response = format!("I received your message: '{}'. This is a stub response while the model is being integrated.", message);
    
    // Split into words and stream
    let words: Vec<&str> = response.split_whitespace().collect();
    
    for word in words {
        buffer.push(&format!("{} ", word)).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    // Complete
    buffer.complete().await?;
    
    Ok(())
}

/// Toggle thinking mode
pub async fn toggle_thinking(
    State(_state): State<AppState>,
) -> Html<String> {
    // This would update the session's thinking mode
    // For now, just return status
    Html("<span id='thinking-status'>ON</span>".to_string())
}

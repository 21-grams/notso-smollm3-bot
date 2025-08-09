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
    tracing::info!("Received message: '{}' for session: {}", msg.message, msg.session_id);
    
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
        tracing::debug!("Processing message in background task");
        
        if message.starts_with("/quote") {
            // Use buffered streaming for quote
            tracing::info!("Processing /quote command");
            let _ = stream_quote_buffered(state_clone, session_id, msg_id).await;
        } else {
            // Regular model generation (also uses buffer)
            tracing::info!("Processing regular message");
            let _ = generate_response_buffered(state_clone, session_id, message, msg_id).await;
        }
    });
    
    Html(html)
}

/// Implement persistent SSE endpoint
pub async fn stream_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    tracing::info!("SSE connection request for session: {}", session_id);
    
    // Ensure session exists and has a receiver
    let receiver = {
        let mut sessions = state.sessions.write().await;
        sessions.create_session(&session_id);  // This now handles reconnections
        sessions.take_receiver(&session_id)
            .expect("Receiver should be available after create_session")
    };
    
    // Create stream from receiver
    let stream = ReceiverStream::new(receiver)
        .map(|event| {
            tracing::debug!("Processing SSE event: {:?}", event);
            let sse_event = match event {
                StreamEvent::MessageContent { message_id, content } => {
                    // Send raw text/markdown - NO HTML escaping
                    Event::default()
                        .event("message")
                        .data(format!(
                            "{}|{}",
                            message_id,
                            content  // Raw content, no escaping
                        ))
                }
                StreamEvent::MessageComplete { message_id } => {
                    Event::default()
                        .event("complete")
                        .data(message_id)
                }
                StreamEvent::MessageError { message_id, error } => {
                    Event::default()
                        .event("message-error")
                        .data(format!(
                            "{}|{}",
                            message_id, error
                        ))
                }
                // Legacy events for compatibility
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
    message_id: String,
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("Starting quote streaming for message {}", message_id);
    
    // Get session's event sender
    let sender = {
        let mut sessions = state.sessions.write().await;
        sessions.get_or_create_sender(&session_id)
    };
    
    // Create streaming buffer with the provided message ID
    let mut buffer = StreamingBuffer::new(sender, message_id.clone());
    
    tracing::debug!("Created buffer, starting to stream quote");
    
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

/// Generate response through buffer
async fn generate_response_buffered(
    state: AppState,
    session_id: String,
    message: String,
    message_id: String,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get session's event sender
    let sender = {
        let mut sessions = state.sessions.write().await;
        sessions.get_or_create_sender(&session_id)
    };
    
    // Create streaming buffer with the provided message ID
    let mut buffer = StreamingBuffer::new(sender, message_id);
    
    // Check if model is available and generate accordingly
    let ml_service = state.model.read().await;
    match ml_service.as_ref() {
        Some(service) => {
            // Try to use the model
            if let Err(e) = service.generate_streaming(&message, &mut buffer).await {
                // Model failed, stream fallback message with markdown
                let error_msg = format!("⚠️ **Model generation failed**\n\n\
                                        Error: {}\n\n\
                                        Searching cached conversations with FTS5...\n\n\
                                        *(FTS5 search integration pending)*", e);
                for word in error_msg.split_whitespace() {
                    buffer.push(&format!("{} ", word)).await?;
                    tokio::time::sleep(Duration::from_millis(30)).await;
                }
            }
        }
        None => {
            // No model loaded, stream fallback message with markdown
            let fallback = "🔴 **Model not loaded**\n\n\
                           Using FTS5 search on cached conversations.\n\n\
                           *(This is a placeholder - FTS5 integration coming soon)*\n\n\
                           You can still use slash commands like `/quote` or `/status`.";
            for word in fallback.split_whitespace() {
                buffer.push(&format!("{} ", word)).await?;
                tokio::time::sleep(Duration::from_millis(30)).await;
            }
        }
    }
    
    // Complete the stream
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

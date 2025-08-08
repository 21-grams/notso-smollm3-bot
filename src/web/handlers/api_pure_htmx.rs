//! API endpoint handlers for pure HTMX SSE

use crate::state::AppState;
use crate::services::template::ChatTemplateService;
use axum::{
    extract::{State, Path, Form},
    response::{Html, sse::{Event, Sse}},
    http::StatusCode,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::Duration;
use tokio_stream::StreamExt;

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    session_id: String,
    message: String,
}

/// Handle chat message submission
pub async fn send_message(
    State(state): State<AppState>,
    Form(msg): Form<ChatMessage>,
) -> Html<String> {
    // Render user message immediately
    let user_html = format!(
        r#"<div class="message user">
            <div class="message-bubble">{}</div>
        </div>"#,
        html_escape::encode_text(&msg.message)
    );
    
    // Start generation in background
    let model = state.model.clone();
    let session_id = msg.session_id.clone();
    let message = msg.message.clone();
    let sessions = state.sessions.clone();
    
    tokio::spawn(async move {
        // Create a new assistant message container
        let message_id = uuid::Uuid::new_v4().to_string();
        
        // Send start of assistant message
        sessions.read().await
            .get(&session_id)
            .map(|session| {
                session.send_html_event(
                    "message-start",
                    format!(r#"<div class="message assistant" id="msg-{}">
                        <div class="message-bubble"></div>
                    </div>"#, message_id)
                );
            });
        
        // Generate response
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        
        // Start generation
        tokio::spawn(async move {
            let _ = model.generate_stream(&message, tx).await;
        });
        
        // Stream tokens as HTML updates
        while let Some(event) = rx.recv().await {
            if let Some(session) = sessions.read().await.get(&session_id) {
                match event {
                    crate::types::StreamEvent::Token { content } => {
                        // Send HTML fragment that will be appended to the message bubble
                        session.send_html_event(
                            "token",
                            format!(
                                r#"<span hx-swap-oob="beforeend:#msg-{} .message-bubble">{}</span>"#,
                                message_id,
                                html_escape::encode_text(&content)
                            )
                        );
                    }
                    crate::types::StreamEvent::Thinking { content } => {
                        // Send thinking indicator
                        session.send_html_event(
                            "thinking",
                            format!(
                                r#"<div class="thinking-indicator" hx-swap-oob="beforeend:#msg-{}">
                                    <span>🤔 {}</span>
                                </div>"#,
                                message_id,
                                html_escape::encode_text(&content)
                            )
                        );
                    }
                    crate::types::StreamEvent::Complete => {
                        // Send completion marker
                        session.send_html_event(
                            "complete",
                            format!(
                                r#"<span hx-swap-oob="afterend:#msg-{}">
                                    <div class="message-complete"></div>
                                </span>"#,
                                message_id
                            )
                        );
                    }
                    crate::types::StreamEvent::Error { message } => {
                        session.send_html_event(
                            "error",
                            format!(
                                r#"<div class="error-message" hx-swap-oob="beforeend:#msg-{}">
                                    Error: {}
                                </div>"#,
                                message_id,
                                html_escape::encode_text(&message)
                            )
                        );
                    }
                    _ => {}
                }
            }
        }
    });
    
    // Return user message HTML for immediate display
    Html(user_html)
}

/// Stream events for a session - now sending HTML fragments
pub async fn stream_events(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let sessions = state.sessions.read().await;
    let stream = sessions.get_html_stream(&session_id);
    
    let event_stream = stream.map(|html_event| {
        match html_event {
            Ok((event_name, html_content)) => {
                // Send HTML content directly - HTMX will swap it into the DOM
                Ok(Event::default()
                    .event(event_name)
                    .data(html_content))
            }
            Err(_) => Ok(Event::default()
                .event("error")
                .data("<div class='error'>Connection error</div>"))
        }
    });
    
    Sse::new(event_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(30))
            .text("keep-alive")
    )
}

/// Toggle thinking mode
pub async fn toggle_thinking(
    State(state): State<AppState>,
    Form(data): Form<std::collections::HashMap<String, String>>,
) -> StatusCode {
    // Update session thinking mode
    // Return 204 No Content since we're using hx-swap="none"
    StatusCode::NO_CONTENT
}

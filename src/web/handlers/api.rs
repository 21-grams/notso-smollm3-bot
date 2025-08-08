//! API endpoint handlers

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

/// Handle chat message submission
pub async fn send_message(
    State(state): State<AppState>,
    Form(msg): Form<ChatMessage>,
) -> Html<String> {
    // Check if message is a slash command
    if msg.message.starts_with("/quote") {
        // Return HTML that includes both user message and assistant response with SSE
        let message_id = uuid::Uuid::new_v4().to_string();
        
        let html = format!(
            r#"<div class="message user">
                <div class="message-bubble">{}</div>
            </div>
            <div class="message assistant" id="msg-{}">
                <div class="message-bubble" 
                     hx-ext="sse"
                     sse-connect="/api/stream/quote/{}/{}" 
                     sse-swap="message">
                    <!-- Content will be streamed here -->
                </div>
            </div>"#,
            html_escape::encode_text(&msg.message),
            message_id,
            msg.session_id,
            message_id
        );
        
        return Html(html);
    }
    // Render user message immediately
    let template_service = ChatTemplateService::new().unwrap();
    let user_html = template_service
        .render_user_message(&msg.message)
        .unwrap_or_default();
    
    // Start generation in background
    let model = state.model.clone();
    let session_id = msg.session_id.clone();
    let message = msg.message.clone();
    
    tokio::spawn(async move {
        let _ = model.generate_response(&session_id, &message).await;
    });
    
    // Return user message HTML for immediate display
    Html(user_html)
}

/// Stream events for a session
pub async fn stream_events(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let sessions = state.sessions.read().await;
    let stream = sessions.get_event_stream(&session_id);
    
    let event_stream = stream.map(|event| {
        match event {
            Ok(evt) => {
                let data = evt.to_sse_data();
                let event_type = evt.event_type();
                Ok(Event::default()
                    .event(event_type)
                    .data(data))
            }
            Err(_) => Ok(Event::default().data("error"))
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
) -> Html<String> {
    // This would update the session's thinking mode
    // For now, just return status
    Html("<span id='thinking-status'>ON</span>".to_string())
}

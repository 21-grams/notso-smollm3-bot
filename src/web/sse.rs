use crate::state::AppState;
use axum::{
    extract::{Path, State},
    response::sse::{Event, Sse},
};
use futures::stream::Stream;
use std::convert::Infallible;
use tokio_stream::StreamExt;

pub async fn stream_events(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    use tokio_stream::wrappers::BroadcastStream;
    use crate::types::events::StreamEvent;
    
    tracing::info!("📡 SSE connection established for session: {}", session_id);
    
    // Subscribe to broadcast channel
    let receiver = {
        let mut sessions = state.sessions.write().await;
        sessions.create_session(&session_id);  // Ensure session exists
        sessions.subscribe(&session_id)
            .expect("Session should exist after create_session")
    };
    
    // Convert StreamEvents to SSE Events
    let stream = BroadcastStream::new(receiver)
        .map(|result| {
            match result {
                Ok(event) => event,
                Err(_) => StreamEvent::KeepAlive  // Ignore lag errors
            }
        })
        .map(|event| {
            let sse_event = match event {
                StreamEvent::MessageContent { message_id, content } => {
                    Event::default()
                        .event("message-content")
                        .data(format!(
                            r#"<div hx-target='#msg-{}' hx-swap='beforeend'>{}</div>"#,
                            message_id,
                            html_escape::encode_text(&content)
                        ))
                }
                StreamEvent::MessageComplete { message_id } => {
                    Event::default()
                        .event("message-complete")
                        .data(format!(
                            r#"<script>document.querySelector('#msg-{} .loading').remove();</script>"#,
                            message_id
                        ))
                }
                StreamEvent::MessageError { message_id, error } => {
                    Event::default()
                        .event("message-error")
                        .data(format!(
                            r#"<div hx-target='#msg-{}' hx-swap='innerHTML'>
                                <div class='error-message'>{}</div>
                            </div>"#,
                            message_id, error
                        ))
                }
                StreamEvent::SessionExpired => {
                    Event::default()
                        .event("session-expired")
                        .data(r#"<div sse-close='true'>Session expired</div>"#)
                }
                _ => Event::default().event("keep-alive").data("")
            };
            Ok(sse_event)
        });
    
    Sse::new(stream)
}

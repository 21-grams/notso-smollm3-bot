//! SSE endpoint handler with smooth token streaming (simplified)

use axum::{
    response::sse::{Event, Sse},
    extract::{State, Path},
};
use futures::stream::Stream;
use std::convert::Infallible;
use crate::state::AppState;

/// Create a simple SSE stream for testing
pub async fn stream_smooth_sse(
    State(_state): State<AppState>,
    Path(session_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Create a simple test stream
    let stream = async_stream::stream! {
        // Send initial message
        yield Ok(Event::default()
            .event("message-start")
            .data(format!(
                r#"<div class="message assistant" id="msg-{}">
                    <div class="message-bubble">Starting response...</div>
                </div>"#,
                session_id
            )));
        
        // Simulate some tokens
        for i in 0..5 {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            yield Ok(Event::default()
                .event("token")
                .data(format!("Token {} ", i)));
        }
        
        // Send completion
        yield Ok(Event::default()
            .event("complete")
            .data("Response complete"));
    };
    
    Sse::new(stream)
}

/// Alternative simple implementation
pub async fn stream_time_based(
    State(_state): State<AppState>,
    Path(session_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        yield Ok(Event::default()
            .event("message")
            .data(format!("Session: {}", session_id)));
    };
    
    Sse::new(stream)
}

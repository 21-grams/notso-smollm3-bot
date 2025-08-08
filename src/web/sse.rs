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
    tracing::info!("📡 SSE connection established for session: {}", session_id);
    
    let receiver = state.sessions
        .read()
        .await
        .get_event_stream(&session_id);
    
    let stream = receiver
        .map(|event| {
            match event {
                Ok(evt) => Ok(Event::default()
                    .event(evt.event_type())
                    .data(evt.to_sse_data())),
                Err(e) => Ok(Event::default()
                    .event("error")
                    .data(format!("Stream error: {}", e)))
            }
        });
    
    Sse::new(stream)
}

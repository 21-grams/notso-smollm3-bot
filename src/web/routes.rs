use crate::state::AppState;
use axum::{
    routing::{get, post},
    Router,
};

pub fn create_routes(state: AppState) -> Router {
    Router::new()
        // Pages
        .route("/", get(super::handlers::chat::index))
        .route("/chat", get(super::handlers::chat::chat_page))
        
        // API endpoints
        .route("/api/chat", post(super::handlers::api::send_message))
        .route("/api/stream/{session_id}", get(super::handlers::api::stream_events))
        .route("/api/toggle-thinking", post(super::handlers::api::toggle_thinking))
        
        // Health check
        .route("/health", get(super::handlers::health::health_check))
        
        .with_state(state)
}

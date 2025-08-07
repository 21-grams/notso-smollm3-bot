use crate::state::AppState;
use axum::{
    routing::{get, post},
    Router,
};

pub fn create_routes(state: AppState) -> Router {
    Router::new()
        // Page routes
        .route("/", get(crate::web::handlers::chat::index))
        .route("/chat", get(crate::web::handlers::chat::chat_page))
        
        // API routes
        .route("/api/chat/send", post(crate::web::handlers::api::send_message))
        .route("/api/chat/thinking", post(crate::web::handlers::api::toggle_thinking))
        
        // SSE streaming
        .route("/sse/:session_id", get(crate::web::sse::stream_events))
        
        // Health check
        .route("/health", get(crate::web::handlers::health::health_check))
        
        .with_state(state)
}

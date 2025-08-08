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
        
        // Slash command endpoints
        .route("/api/reset-context", post(super::handlers::commands::reset_context))
        .route("/api/set-temperature", post(super::handlers::commands::set_temperature))
        .route("/api/model-info", get(super::handlers::commands::model_info))
        .route("/api/status", get(super::handlers::commands::system_status))
        .route("/api/export-chat", get(super::handlers::commands::export_chat))
        
        // Quote streaming endpoint
        .route("/api/stream/quote/{session_id}/{message_id}", get(super::handlers::quote::stream_quote))
        
        // Health check
        .route("/health", get(super::handlers::health::health_check))
        
        .with_state(state)
}

use crate::state::AppState;
use crate::services::template::TemplateEngine;
use axum::{
    extract::State,
    response::Html,
};
use uuid::Uuid;

pub async fn index(State(state): State<AppState>) -> Html<String> {
    let session_id = Uuid::new_v4().to_string();
    
    // Initialize session
    state.sessions.write().await.create_session(&session_id);
    
    // Render template
    let engine = TemplateEngine::new().expect("Failed to create template engine");
    let html = engine.render_chat_page(&session_id, state.config.thinking_mode_default)
        .unwrap_or_else(|e| {
            format!("<html><body><h1>Error: {}</h1></body></html>", e)
        });
    
    Html(html)
}

pub async fn chat_page(State(state): State<AppState>) -> Html<String> {
    index(State(state)).await
}

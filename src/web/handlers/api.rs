use crate::state::AppState;
use crate::types::{ChatRequest, ChatResponse};
use axum::{
    extract::{State, Form},
    response::Json,
};

pub async fn send_message(
    State(state): State<AppState>,
    Form(request): Form<ChatRequest>,
) -> Json<ChatResponse> {
    tracing::info!("📨 Received message: {}", request.message);
    
    // Get or create session
    let mut sessions = state.sessions.write().await;
    let session = sessions.get_or_create(&request.session_id);
    
    // Start generation in background
    let model = state.model.clone();
    let session_id = request.session_id.clone();
    let message = request.message.clone();
    
    tokio::spawn(async move {
        if let Err(e) = model.generate_response(&session_id, &message).await {
            tracing::error!("Generation error: {}", e);
        }
    });
    
    Json(ChatResponse {
        status: "processing".to_string(),
        session_id: request.session_id,
    })
}

pub async fn toggle_thinking(
    State(state): State<AppState>,
    Form(request): Form<serde_json::Value>,
) -> Json<serde_json::Value> {
    // Toggle thinking mode for session
    let session_id = request.get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    
    let mut sessions = state.sessions.write().await;
    if let Some(session) = sessions.get_mut(session_id) {
        session.toggle_thinking_mode();
        Json(serde_json::json!({
            "thinking_mode": session.thinking_mode
        }))
    } else {
        Json(serde_json::json!({
            "error": "Session not found"
        }))
    }
}

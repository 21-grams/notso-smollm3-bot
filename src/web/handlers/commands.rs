//! Slash command API endpoints

use crate::state::AppState;
use axum::{
    extract::{State, Json},
    response::{Json as JsonResponse},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct TemperatureRequest {
    temperature: f32,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    model: String,
    status: String,
    parameters: ModelParameters,
}

#[derive(Debug, Serialize)]
pub struct ModelParameters {
    temperature: f32,
    max_tokens: usize,
    thinking_mode: bool,
}

#[derive(Debug, Serialize)]
pub struct SystemStatus {
    server: String,
    model: String,
    sessions: usize,
    uptime: String,
    memory_usage: String,
}

/// Reset conversation context
pub async fn reset_context(
    State(_state): State<AppState>,
) -> StatusCode {
    // Clear session context
    // In a real implementation, you'd clear the specific session
    tracing::info!("Context reset requested");
    StatusCode::OK
}

/// Set model temperature
pub async fn set_temperature(
    State(_state): State<AppState>,
    Json(req): Json<TemperatureRequest>,
) -> StatusCode {
    if req.temperature < 0.0 || req.temperature > 1.0 {
        return StatusCode::BAD_REQUEST;
    }
    
    tracing::info!("Temperature set to: {}", req.temperature);
    // Store in session or config
    StatusCode::OK
}

/// Get model information
pub async fn model_info(
    State(state): State<AppState>,
) -> JsonResponse<ModelInfo> {
    let info = ModelInfo {
        model: "SmolLM3-3B-Q4_K_M".to_string(),
        status: if state.model.read().await.is_stub() { "Stub Mode" } else { "Active" }.to_string(),
        parameters: ModelParameters {
            temperature: 0.7,
            max_tokens: 2048,
            thinking_mode: state.config.thinking_mode_default,
        },
    };
    
    JsonResponse(info)
}

/// Get system status
pub async fn system_status(
    State(state): State<AppState>,
) -> JsonResponse<SystemStatus> {
    let sessions = state.sessions.read().await;
    
    let status = SystemStatus {
        server: "Running".to_string(),
        model: if state.model.read().await.is_stub() { "Stub Mode" } else { "SmolLM3-3B" }.to_string(),
        sessions: sessions.count(),
        uptime: format_uptime(),
        memory_usage: get_memory_usage(),
    };
    
    JsonResponse(status)
}

/// Export chat history
pub async fn export_chat(
    State(_state): State<AppState>,
) -> String {
    // In a real implementation, get the session ID from the request
    // and export that session's chat history
    "Chat export functionality not yet implemented".to_string()
}

// Helper functions
fn format_uptime() -> String {
    // Calculate uptime from process start
    "0d 0h 0m".to_string()
}

fn get_memory_usage() -> String {
    // Get current memory usage
    "0 MB".to_string()
}

// Extension trait for MLService
impl crate::services::ml::MLService {
    pub fn is_stub(&self) -> bool {
        // Check if running in stub mode
        // This would be implemented in the actual MLService
        true
    }
}

// Extension trait for SessionManager
impl crate::services::SessionManager {
    pub fn count(&self) -> usize {
        // Count active sessions
        // This would be implemented in the actual SessionManager
        0
    }
}

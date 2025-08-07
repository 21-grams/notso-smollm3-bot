use axum::Json;
use serde_json::json;

pub async fn health_check() -> Json<serde_json::Value> {
    Json(json!({
        "status": "healthy",
        "service": "notso-smollm3-bot",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

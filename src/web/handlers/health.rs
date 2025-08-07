//! Health check endpoint

use axum::{
    response::Json,
    http::StatusCode,
};
use serde_json::json;

pub async fn health_check() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "service": "notso-smollm3-bot",
            "version": env!("CARGO_PKG_VERSION"),
        }))
    )
}

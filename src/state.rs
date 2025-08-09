use crate::config::Config;
use crate::services::{SessionManager, MLService};
use crate::services::template::engine::TemplateEngine;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub model: Arc<RwLock<Option<MLService>>>,  // Option - can be None if model fails to load
    pub sessions: Arc<RwLock<SessionManager>>,
    pub templates: Arc<TemplateEngine>,
}

impl AppState {
    pub async fn new() -> Result<Self> {
        let config = Config::from_env()?;
        
        // Try to load ML service, but don't fail if it doesn't work
        let template_path = "models/smollm3_thinking_chat_template.jinja2".to_string();
        let ml_service = match MLService::new(
            &config.model_path,
            &config.tokenizer_path,
            &template_path,
            config.to_candle_device(),
        ) {
            Ok(service) => {
                tracing::info!("✅ Model loaded successfully");
                Some(service)
            }
            Err(e) => {
                tracing::warn!("⚠️ Model not available: {}", e);
                tracing::info!("🌐 Server will start without model inference");
                None
            }
        };
        
        let templates = TemplateEngine::new()?;
        
        Ok(Self {
            config: Arc::new(config),
            model: Arc::new(RwLock::new(ml_service)),
            sessions: Arc::new(RwLock::new(SessionManager::new())),
            templates: Arc::new(templates),
        })
    }
}

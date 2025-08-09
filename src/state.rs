use crate::config::Config;
use crate::services::{SessionManager, MLService};
use crate::services::template::engine::TemplateEngine;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub model: Arc<RwLock<MLService>>,
    pub sessions: Arc<RwLock<SessionManager>>,
    pub templates: Arc<TemplateEngine>,
}

impl AppState {
    pub async fn new() -> Result<Self> {
        let config = Config::from_env()?;
        
        // Initialize ML service with fallback to stub
        let ml_service = match MLService::new(
            &config.model_path,
            &config.tokenizer_path,
            &config.template_path,
            config.to_candle_device(),
        ) {
            Ok(service) => {
                tracing::info!("✅ Model loaded successfully");
                service
            }
            Err(e) => {
                tracing::warn!("⚠️ Model load failed: {}, using stub mode", e);
                MLService::new_stub_mode()
            }
        };
        
        let templates = TemplateEngine::new()?;  // Plan shows path arg but constructor takes 0
        
        Ok(Self {
            config: Arc::new(config),
            model: Arc::new(RwLock::new(ml_service)),
            sessions: Arc::new(RwLock::new(SessionManager::new())),
            templates: Arc::new(templates),
        })
    }
}

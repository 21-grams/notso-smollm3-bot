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
        tracing::info!("[STATE] Initializing AppState...");
        let config = Config::from_env()?;
        tracing::info!("[STATE] Config loaded:");
        tracing::info!("[STATE]   Model path: {}", config.model_path);
        tracing::info!("[STATE]   Device: {:?}", config.device);
        
        // Try to load ML service, but don't fail if it doesn't work
        // Tokenizer directory is the parent of the model file
        let tokenizer_dir = config.model_path
            .rsplit_once('/')
            .map(|(dir, _)| dir.to_string())
            .unwrap_or_else(|| "models".to_string());
        
        tracing::info!("[STATE] Tokenizer directory: {}", tokenizer_dir);
        tracing::info!("[STATE] Attempting to load ML service...");
        
        let ml_service = match MLService::new(
            &config.model_path,
            &tokenizer_dir,
            config.to_candle_device(),
        ) {
            Ok(service) => {
                tracing::info!("[STATE] ✅ Model loaded successfully");
                Some(service)
            }
            Err(e) => {
                tracing::error!("[STATE] ⚠️ Model loading failed: {}", e);
                tracing::info!("[STATE] 🌐 Server will start without model inference");
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

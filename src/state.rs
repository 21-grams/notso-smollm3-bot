use crate::config::Config;
use crate::services::ml::MLService;
use crate::services::SessionManager;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<MLService>,  // Using MLService instead of SmolLM3Model
    pub config: Arc<Config>,
    pub sessions: Arc<RwLock<SessionManager>>,
}

impl AppState {
    pub fn new(model: Arc<MLService>, config: Config) -> Self {
        Self {
            model,
            config: Arc::new(config),
            sessions: Arc::new(RwLock::new(SessionManager::new())),
        }
    }
}

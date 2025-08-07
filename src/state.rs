use crate::config::Config;
use crate::smollm3::SmolLM3Model;
use crate::services::SessionManager;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<SmolLM3Model>,
    pub config: Arc<Config>,
    pub sessions: Arc<RwLock<SessionManager>>,
}

impl AppState {
    pub fn new(model: Arc<SmolLM3Model>, config: Config) -> Self {
        Self {
            model,
            config: Arc::new(config),
            sessions: Arc::new(RwLock::new(SessionManager::new())),
        }
    }
}

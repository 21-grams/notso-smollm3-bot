//! Streaming service for real-time token generation

use crate::types::events::StreamEvent;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::mpsc::{UnboundedSender, UnboundedReceiver, unbounded_channel};

pub struct StreamingService {
    sessions: Arc<RwLock<HashMap<String, UnboundedSender<StreamEvent>>>>,
}

impl StreamingService {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn create_stream(&self, session_id: &str) -> UnboundedReceiver<StreamEvent> {
        let (tx, rx) = unbounded_channel();
        self.sessions.write().await.insert(session_id.to_string(), tx);
        rx
    }
    
    pub async fn send_event(&self, session_id: &str, event: StreamEvent) {
        if let Some(sender) = self.sessions.read().await.get(session_id) {
            let _ = sender.send(event);
        }
    }
    
    pub async fn close_stream(&self, session_id: &str) {
        self.sessions.write().await.remove(session_id);
    }
}

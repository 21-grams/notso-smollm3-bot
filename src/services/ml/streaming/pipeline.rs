//! Streaming pipeline for real-time generation

use super::{GenerationEvent, TokenBuffer};
use tokio::sync::mpsc::{UnboundedSender, UnboundedReceiver};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct StreamingPipeline {
    buffer: Arc<Mutex<TokenBuffer>>,
    event_sender: Option<UnboundedSender<GenerationEvent>>,
}

impl StreamingPipeline {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(TokenBuffer::default())),
            event_sender: None,
        }
    }
    
    /// Create with event channel
    pub fn with_channel() -> (Self, UnboundedReceiver<GenerationEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let pipeline = Self {
            buffer: Arc::new(Mutex::new(TokenBuffer::default())),
            event_sender: Some(tx),
        };
        (pipeline, rx)
    }
    
    /// Send generation event
    pub async fn send_event(&self, event: GenerationEvent) {
        if let Some(sender) = &self.event_sender {
            let _ = sender.send(event);
        }
    }
    
    /// Process token
    pub async fn process_token(&self, token: String, is_thinking: bool) {
        let mut buffer = self.buffer.lock().await;
        buffer.push(token.clone());
        
        let event = if is_thinking {
            GenerationEvent::ThinkingToken(token)
        } else {
            GenerationEvent::ResponseToken(token)
        };
        
        self.send_event(event).await;
    }
    
    /// Start generation
    pub async fn start(&self) {
        self.send_event(GenerationEvent::Start).await;
    }
    
    /// Complete generation
    pub async fn complete(&self) {
        self.send_event(GenerationEvent::Complete).await;
    }
    
    /// Handle error
    pub async fn error(&self, msg: String) {
        self.send_event(GenerationEvent::Error(msg)).await;
    }
    
    /// Get accumulated text
    pub async fn get_text(&self) -> String {
        self.buffer.lock().await.get_text().to_string()
    }
}

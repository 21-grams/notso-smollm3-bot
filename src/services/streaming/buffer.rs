use crate::types::events::StreamEvent;
use tokio::sync::broadcast;
use std::time::{Duration, Instant};
use anyhow::Result;

pub struct StreamingBuffer {
    sender: broadcast::Sender<StreamEvent>,
    buffer: String,
    token_count: usize,
    last_send: Instant,
    message_id: String,
}

impl StreamingBuffer {
    pub fn new(sender: broadcast::Sender<StreamEvent>, message_id: String) -> Self {
        Self {
            sender,
            buffer: String::new(),
            token_count: 0,
            last_send: Instant::now(),
            message_id,
        }
    }
    
    pub async fn push(&mut self, content: &str) -> Result<()> {
        self.buffer.push_str(content);
        self.token_count += 1;
        
        // Send if we have 10 tokens or 500ms elapsed
        if self.token_count >= 10 || self.last_send.elapsed() > Duration::from_millis(500) {
            self.flush().await?;
        }
        Ok(())
    }
    
    async fn flush(&mut self) -> Result<()> {
        if !self.buffer.is_empty() {
            tracing::debug!("Flushing buffer with {} chars for message {}", self.buffer.len(), self.message_id);
            let event = StreamEvent::MessageContent {
                message_id: self.message_id.clone(),
                content: self.buffer.clone(),
            };
            // Broadcast send doesn't require await
            let _ = self.sender.send(event);
            self.buffer.clear();
            self.token_count = 0;
            self.last_send = Instant::now();
        }
        Ok(())
    }
    
    pub async fn complete(&mut self) -> Result<()> {
        self.flush().await?;
        tracing::info!("Sending complete event for message {}", self.message_id);
        let _ = self.sender.send(StreamEvent::MessageComplete {
            message_id: self.message_id.clone(),
        });
        Ok(())
    }
    
    pub async fn error(&mut self, error: String) -> Result<()> {
        self.flush().await?;
        let _ = self.sender.send(StreamEvent::MessageError {
            message_id: self.message_id.clone(),
            error,
        });
        Ok(())
    }
}

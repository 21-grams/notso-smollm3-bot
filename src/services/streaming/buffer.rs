use crate::types::events::StreamEvent;
use tokio::sync::mpsc;
use std::time::{Duration, Instant};
use anyhow::Result;

pub struct StreamingBuffer {
    sender: mpsc::Sender<StreamEvent>,
    buffer: String,
    token_count: usize,
    last_send: Instant,
    message_id: String,
}

impl StreamingBuffer {
    pub fn new(sender: mpsc::Sender<StreamEvent>, message_id: String) -> Self {
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
        
        // Send if we have 10 tokens or 100ms elapsed
        if self.token_count >= 10 || self.last_send.elapsed() > Duration::from_millis(100) {
            self.flush().await?;
        }
        Ok(())
    }
    
    async fn flush(&mut self) -> Result<()> {
        if !self.buffer.is_empty() {
            let event = StreamEvent::MessageContent {
                message_id: self.message_id.clone(),
                content: self.buffer.clone(),
            };
            self.sender.send(event).await?;
            self.buffer.clear();
            self.token_count = 0;
            self.last_send = Instant::now();
        }
        Ok(())
    }
    
    pub async fn complete(&mut self) -> Result<()> {
        self.flush().await?;
        self.sender.send(StreamEvent::MessageComplete {
            message_id: self.message_id.clone(),
        }).await?;
        Ok(())
    }
    
    pub async fn error(&mut self, error: String) -> Result<()> {
        self.flush().await?;
        self.sender.send(StreamEvent::MessageError {
            message_id: self.message_id.clone(),
            error,
        }).await?;
        Ok(())
    }
}

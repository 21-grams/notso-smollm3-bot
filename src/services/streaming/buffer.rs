//! Unified streaming buffer for all output (model generation and commands)

use std::time::{Duration, Instant};
use tokio::sync::mpsc::Sender;
use crate::types::events::StreamEvent;

/// Streaming buffer that accumulates content and flushes based on thresholds
pub struct StreamingBuffer {
    session_id: String,
    buffer: Vec<String>,
    token_threshold: usize,
    time_threshold: Duration,
    last_flush: Instant,
    sender: Sender<StreamEvent>,
}

impl StreamingBuffer {
    /// Create a new streaming buffer
    pub fn new(session_id: String, sender: Sender<StreamEvent>) -> Self {
        Self {
            session_id,
            buffer: Vec::new(),
            token_threshold: 10,  // Keep at 10 tokens for reasonable batching
            time_threshold: Duration::from_millis(500),  // Changed from 100ms to 500ms
            last_flush: Instant::now(),
            sender,
        }
    }
    
    /// Create with custom thresholds
    pub fn with_thresholds(
        session_id: String,
        sender: Sender<StreamEvent>,
        token_threshold: usize,
        time_threshold: Duration,
    ) -> Self {
        Self {
            session_id,
            buffer: Vec::new(),
            token_threshold,
            time_threshold,
            last_flush: Instant::now(),
            sender,
        }
    }
    
    /// Add content to buffer (from model OR command)
    pub async fn push(&mut self, content: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.buffer.push(content.to_string());
        
        // Check if should flush
        if self.should_flush() {
            self.flush().await?;
        }
        
        Ok(())
    }
    
    /// Check if buffer should be flushed
    fn should_flush(&self) -> bool {
        self.buffer.len() >= self.token_threshold ||
        self.last_flush.elapsed() >= self.time_threshold
    }
    
    /// Flush buffered content
    pub async fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.buffer.is_empty() { 
            return Ok(()); 
        }
        
        // Join buffered content
        let content = self.buffer.join("");
        self.buffer.clear();
        self.last_flush = Instant::now();
        
        // Send via session event stream
        self.sender.send(StreamEvent::Content(content)).await?;
        
        Ok(())
    }
    
    /// Complete the stream
    pub async fn complete(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Flush any remaining content
        self.flush().await?;
        
        // Send completion event
        self.sender.send(StreamEvent::Complete).await?;
        
        Ok(())
    }
    
    /// Force flush with timeout check
    pub async fn flush_if_timeout(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.buffer.is_empty() && self.last_flush.elapsed() >= self.time_threshold {
            self.flush().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;
    
    #[tokio::test]
    async fn test_buffer_token_threshold() {
        let (tx, mut rx) = mpsc::channel(10);
        let mut buffer = StreamingBuffer::new("test".to_string(), tx);
        
        // Add tokens below threshold
        for i in 0..9 {
            buffer.push(&format!("token{} ", i)).await.unwrap();
        }
        
        // Should not have flushed yet
        assert!(rx.try_recv().is_err());
        
        // Add one more to hit threshold
        buffer.push("token10 ").await.unwrap();
        
        // Should have flushed
        let event = rx.recv().await.unwrap();
        match event {
            StreamEvent::Content(content) => {
                assert!(content.contains("token0"));
                assert!(content.contains("token9"));
            }
            _ => panic!("Expected Content event"),
        }
    }
}

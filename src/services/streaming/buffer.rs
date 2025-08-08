//! Token buffer for smooth streaming output (simplified version)

use std::time::{Duration, Instant};

/// Configuration for the streaming buffer
#[derive(Clone, Debug)]
pub struct StreamBufferConfig {
    /// Maximum time to hold tokens before flushing (milliseconds)
    pub flush_interval_ms: u64,
    /// Minimum tokens to accumulate before considering a flush
    pub min_tokens_for_flush: usize,
    /// Flush on punctuation marks for natural breaks
    pub flush_on_punctuation: bool,
    /// Flush on complete words (spaces)
    pub flush_on_word_boundary: bool,
}

impl Default for StreamBufferConfig {
    fn default() -> Self {
        Self {
            flush_interval_ms: 150,
            min_tokens_for_flush: 2,
            flush_on_punctuation: true,
            flush_on_word_boundary: true,
        }
    }
}

/// Simple token buffer
pub struct TokenStreamBuffer {
    config: StreamBufferConfig,
    buffer: String,
    last_flush: Instant,
    token_count: usize,
}

impl TokenStreamBuffer {
    pub fn new(config: StreamBufferConfig) -> Self {
        Self {
            config,
            buffer: String::new(),
            last_flush: Instant::now(),
            token_count: 0,
        }
    }
    
    pub fn add_token(&mut self, token: &str) -> Option<String> {
        self.buffer.push_str(token);
        self.token_count += 1;
        
        if self.should_flush() {
            return self.flush();
        }
        
        None
    }
    
    fn should_flush(&self) -> bool {
        if self.last_flush.elapsed() >= Duration::from_millis(self.config.flush_interval_ms) {
            return true;
        }
        
        if self.buffer.is_empty() {
            return false;
        }
        
        if self.token_count < self.config.min_tokens_for_flush {
            return false;
        }
        
        if self.config.flush_on_punctuation {
            if self.buffer.chars().any(|c| ".!?,;:".contains(c)) {
                return true;
            }
        }
        
        if self.config.flush_on_word_boundary {
            if self.buffer.ends_with(' ') || self.buffer.ends_with('\n') {
                return true;
            }
        }
        
        false
    }
    
    pub fn flush(&mut self) -> Option<String> {
        if !self.buffer.is_empty() {
            let content = self.buffer.clone();
            self.buffer.clear();
            self.token_count = 0;
            self.last_flush = Instant::now();
            return Some(content);
        }
        None
    }
    
    pub fn flush_remaining(&mut self) -> Option<String> {
        self.flush()
    }
}

/// Simple word buffer
pub struct WordBuffer {
    partial_word: String,
    flush_interval: Duration,
    last_flush: Instant,
}

impl WordBuffer {
    pub fn new(flush_interval_ms: u64) -> Self {
        Self {
            partial_word: String::new(),
            flush_interval: Duration::from_millis(flush_interval_ms),
            last_flush: Instant::now(),
        }
    }
    
    pub fn process_token(&mut self, token: &str) -> Option<String> {
        self.partial_word.push_str(token);
        
        let should_flush = self.partial_word.contains(' ') 
            || self.partial_word.contains('\n')
            || self.last_flush.elapsed() >= self.flush_interval;
        
        if should_flush && !self.partial_word.is_empty() {
            let content = self.partial_word.clone();
            self.partial_word.clear();
            self.last_flush = Instant::now();
            return Some(content);
        }
        
        None
    }
    
    pub fn flush_remaining(&mut self) -> Option<String> {
        if !self.partial_word.is_empty() {
            let content = self.partial_word.clone();
            self.partial_word.clear();
            return Some(content);
        }
        None
    }
}

/// Simplified HTMX stream processor
pub struct HtmxStreamProcessor {
    pub message_id: String,
    pub buffer: TokenStreamBuffer,
    pub accumulated_content: String,
}

impl HtmxStreamProcessor {
    pub fn new(message_id: String, config: StreamBufferConfig) -> Self {
        Self {
            message_id,
            buffer: TokenStreamBuffer::new(config),
            accumulated_content: String::new(),
        }
    }
}

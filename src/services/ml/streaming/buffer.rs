//! Token buffer for managing streaming generation

use std::collections::VecDeque;

pub struct TokenBuffer {
    buffer: VecDeque<String>,
    max_size: usize,
    current_text: String,
}

impl TokenBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_size),
            max_size,
            current_text: String::new(),
        }
    }
    
    /// Add token to buffer
    pub fn push(&mut self, token: String) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.current_text.push_str(&token);
        self.buffer.push_back(token);
    }
    
    /// Get buffered tokens
    pub fn get_tokens(&self) -> Vec<String> {
        self.buffer.iter().cloned().collect()
    }
    
    /// Get accumulated text
    pub fn get_text(&self) -> &str {
        &self.current_text
    }
    
    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.current_text.clear();
    }
    
    /// Check if buffer contains pattern
    pub fn contains(&self, pattern: &str) -> bool {
        self.current_text.contains(pattern)
    }
}

impl Default for TokenBuffer {
    fn default() -> Self {
        Self::new(100)
    }
}

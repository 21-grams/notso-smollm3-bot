//! Thinking mode detection and processing

use crate::services::ml::official::ThinkingTokens;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThinkingEvent {
    Start,
    Content(String),
    End,
}

pub struct ThinkingDetector {
    tokens: ThinkingTokens,
    in_thinking_mode: bool,
    thinking_buffer: String,
}

impl ThinkingDetector {
    pub fn new(tokens: ThinkingTokens) -> Self {
        Self {
            tokens,
            in_thinking_mode: false,
            thinking_buffer: String::new(),
        }
    }
    
    /// Process a token and detect thinking mode transitions
    pub fn process_token(&mut self, token_id: u32, token_text: &str) -> Option<ThinkingEvent> {
        if token_id == self.tokens.start {
            self.in_thinking_mode = true;
            self.thinking_buffer.clear();
            Some(ThinkingEvent::Start)
        } else if token_id == self.tokens.end {
            self.in_thinking_mode = false;
            let content = self.thinking_buffer.clone();
            self.thinking_buffer.clear();
            Some(ThinkingEvent::End)
        } else if self.in_thinking_mode {
            self.thinking_buffer.push_str(token_text);
            Some(ThinkingEvent::Content(token_text.to_string()))
        } else {
            None
        }
    }
    
    /// Check if currently in thinking mode
    pub fn is_thinking(&self) -> bool {
        self.in_thinking_mode
    }
    
    /// Check if text contains thinking markers
    pub fn contains_thinking_markers(&self, text: &str) -> bool {
        text.contains("<think>") || text.contains("</think>")
    }
    
    /// Get current thinking buffer
    pub fn get_thinking_buffer(&self) -> &str {
        &self.thinking_buffer
    }
    
    /// Reset detector state
    pub fn reset(&mut self) {
        self.in_thinking_mode = false;
        self.thinking_buffer.clear();
    }
}

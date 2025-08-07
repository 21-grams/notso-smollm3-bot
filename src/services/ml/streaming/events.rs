//! Event types for real-time streaming

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum GenerationEvent {
    /// Generation started
    Start,
    
    /// Thinking mode started
    ThinkingStart,
    
    /// Token generated during thinking
    ThinkingToken(String),
    
    /// Thinking mode ended
    ThinkingEnd,
    
    /// Regular response token
    ResponseToken(String),
    
    /// Tool call initiated
    ToolCall { name: String, args: String },
    
    /// Generation completed
    Complete,
    
    /// Error occurred
    Error(String),
}

impl GenerationEvent {
    /// Convert to SSE-compatible format
    pub fn to_sse_event(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
    
    /// Get event type name for SSE
    pub fn event_name(&self) -> &'static str {
        match self {
            GenerationEvent::Start => "start",
            GenerationEvent::ThinkingStart => "thinking_start",
            GenerationEvent::ThinkingToken(_) => "thinking_token",
            GenerationEvent::ThinkingEnd => "thinking_end",
            GenerationEvent::ResponseToken(_) => "response_token",
            GenerationEvent::ToolCall { .. } => "tool_call",
            GenerationEvent::Complete => "complete",
            GenerationEvent::Error(_) => "error",
        }
    }
}

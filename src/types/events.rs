use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEvent {
    Content(String),      // For buffered content
    Thinking(String),     // For thinking mode content
    Token(String),        // For individual tokens (if needed)
    Complete,             // Stream completion
    Error(String),        // Error messages
    Status(String),       // Status updates
}

impl StreamEvent {
    pub fn content(text: String) -> Self {
        Self::Content(text)
    }
    
    pub fn thinking(text: String) -> Self {
        Self::Thinking(text)
    }
    
    pub fn token(text: String) -> Self {
        Self::Token(text)
    }
    
    pub fn complete() -> Self {
        Self::Complete
    }
    
    pub fn error(message: String) -> Self {
        Self::Error(message)
    }
    
    pub fn status(message: String) -> Self {
        Self::Status(message)
    }
    
    pub fn event_type(&self) -> &str {
        match self {
            Self::Content(_) => "message",
            Self::Thinking(_) => "thinking",
            Self::Token(_) => "token",
            Self::Complete => "complete",
            Self::Error(_) => "error",
            Self::Status(_) => "status",
        }
    }
    
    pub fn to_sse_data(&self) -> String {
        match self {
            Self::Content(text) | Self::Thinking(text) | Self::Token(text) | 
            Self::Error(text) | Self::Status(text) => text.clone(),
            Self::Complete => "done".to_string(),
        }
    }
}

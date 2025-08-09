use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEvent {
    // Message-specific events (include message_id for targeting)
    MessageContent { message_id: String, content: String },
    MessageThinking { message_id: String, content: String },
    MessageComplete { message_id: String },
    MessageError { message_id: String, error: String },
    
    // Session-level events
    SessionStatus(String),
    SessionExpired,  // Only this would use sse-close
    
    // System events
    KeepAlive,
    
    // Legacy compatibility
    Content(String),
    Thinking(String),
    Token(String),
    Complete,
    Error(String),
    Status(String),
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
            Self::MessageContent { .. } | Self::Content(_) => "message",
            Self::MessageThinking { .. } | Self::Thinking(_) => "thinking",
            Self::MessageComplete { .. } | Self::Complete => "complete",
            Self::MessageError { .. } | Self::Error(_) => "error",
            Self::Token(_) => "token",
            Self::Status(_) | Self::SessionStatus(_) => "status",
            Self::SessionExpired => "session-expired",
            Self::KeepAlive => "keep-alive",
        }
    }
}

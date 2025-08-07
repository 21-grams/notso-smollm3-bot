use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEvent {
    Thinking { content: String },
    Token { content: String },
    Complete,
    Error { message: String },
}

impl StreamEvent {
    pub fn thinking(content: String) -> Self {
        Self::Thinking { content }
    }
    
    pub fn token(content: String) -> Self {
        Self::Token { content }
    }
    
    pub fn complete() -> Self {
        Self::Complete
    }
    
    pub fn error(message: String) -> Self {
        Self::Error { message }
    }
    
    pub fn event_type(&self) -> String {
        match self {
            Self::Thinking { .. } => "thinking",
            Self::Token { .. } => "token",
            Self::Complete => "complete",
            Self::Error { .. } => "error",
        }.to_string()
    }
    
    pub fn to_sse_data(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

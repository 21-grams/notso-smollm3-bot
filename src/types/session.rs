//! Session management types

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub messages: Vec<Message>,
    pub thinking_mode: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Session {
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            messages: Vec::new(),
            thinking_mode: false,
            created_at: now,
            updated_at: now,
        }
    }
    
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
        self.updated_at = Utc::now();
    }
    
    pub fn toggle_thinking_mode(&mut self) {
        self.thinking_mode = !self.thinking_mode;
        self.updated_at = Utc::now();
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: String,
    pub content: String,
    pub thinking_content: Option<String>,
    pub thinking_mode: bool,
    pub timestamp: DateTime<Utc>,
}

impl Message {
    pub fn new_user(content: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: "user".to_string(),
            content,
            thinking_content: None,
            thinking_mode: false,
            timestamp: Utc::now(),
        }
    }
    
    pub fn new_assistant(content: String, thinking_content: Option<String>, thinking_mode: bool) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: "assistant".to_string(),
            content,
            thinking_content,
            thinking_mode,
            timestamp: Utc::now(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub session_id: String,
    pub message: String,
    pub thinking_mode: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub message_id: String,
    pub session_id: String,
    pub content: String,
    pub thinking_content: Option<String>,
    pub thinking_mode: bool,
}

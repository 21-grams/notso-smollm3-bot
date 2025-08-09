use crate::types::events::StreamEvent;
use std::collections::HashMap;
use tokio::sync::{mpsc, broadcast};
use chrono::{DateTime, Utc};

pub struct SessionManager {
    sessions: HashMap<String, SessionState>,
}

pub struct SessionState {
    pub id: String,
    pub messages: Vec<Message>,
    pub thinking_mode: bool,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    // Use broadcast channel for multiple receivers
    pub event_sender: broadcast::Sender<StreamEvent>,
}

#[derive(Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

pub struct Session {
    pub id: String,
    pub thinking_mode: bool,
    pub messages: Vec<(String, String)>,
    event_sender: mpsc::Sender<StreamEvent>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }
    
    pub fn create_session(&mut self, session_id: &str) {
        if !self.sessions.contains_key(session_id) {
            // Create new session with broadcast channel
            let (tx, _rx) = broadcast::channel(100);
            
            let session = SessionState {
                id: session_id.to_string(),
                messages: Vec::new(),
                thinking_mode: false,
                created_at: Utc::now(),
                last_activity: Utc::now(),
                event_sender: tx,
            };
            
            self.sessions.insert(session_id.to_string(), session);
            tracing::debug!("Created new session: {}", session_id);
        } else {
            // Update last activity for existing session
            if let Some(session) = self.sessions.get_mut(session_id) {
                session.last_activity = Utc::now();
            }
        }
    }
    
    pub fn get_sender(&self, session_id: &str) -> Option<broadcast::Sender<StreamEvent>> {
        self.sessions.get(session_id).map(|s| s.event_sender.clone())
    }
    
    pub fn subscribe(&mut self, session_id: &str) -> Option<broadcast::Receiver<StreamEvent>> {
        self.sessions.get(session_id).map(|s| s.event_sender.subscribe())
    }
    
    pub fn get_or_create_sender(&mut self, session_id: &str) -> broadcast::Sender<StreamEvent> {
        self.create_session(session_id);
        self.get_sender(session_id).unwrap()
    }
    
    pub fn get(&self, id: &str) -> Option<&SessionState> {
        self.sessions.get(id)
    }
    
    pub fn get_mut(&mut self, id: &str) -> Option<&mut SessionState> {
        self.sessions.get_mut(id)
    }
    
    pub fn count(&self) -> usize {
        self.sessions.len()
    }
}

impl Session {
    pub fn toggle_thinking_mode(&mut self) {
        self.thinking_mode = !self.thinking_mode;
    }
    
    pub fn send_event(&self, event: StreamEvent) {
        let _ = self.event_sender.try_send(event);
    }
}

use crate::types::events::StreamEvent;
use std::collections::HashMap;
use tokio::sync::mpsc;
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
    // Single persistent channel per session
    pub event_sender: mpsc::Sender<StreamEvent>,
    pub event_receiver: Option<mpsc::Receiver<StreamEvent>>,
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
        // Always recreate the receiver if it's been taken
        if let Some(session) = self.sessions.get_mut(session_id) {
            if session.event_receiver.is_none() {
                // Receiver was taken, create a new channel
                let (tx, rx) = mpsc::channel(100);
                session.event_sender = tx;
                session.event_receiver = Some(rx);
                session.last_activity = Utc::now();
                tracing::debug!("Recreated SSE channel for existing session: {}", session_id);
            }
        } else {
            // Create new session
            let (tx, rx) = mpsc::channel(100);
            
            let session = SessionState {
                id: session_id.to_string(),
                messages: Vec::new(),
                thinking_mode: false,
                created_at: Utc::now(),
                last_activity: Utc::now(),
                event_sender: tx,
                event_receiver: Some(rx),
            };
            
            self.sessions.insert(session_id.to_string(), session);
            tracing::debug!("Created new session: {}", session_id);
        }
    }
    
    pub fn get_sender(&self, session_id: &str) -> Option<mpsc::Sender<StreamEvent>> {
        self.sessions.get(session_id).map(|s| s.event_sender.clone())
    }
    
    pub fn take_receiver(&mut self, session_id: &str) -> Option<mpsc::Receiver<StreamEvent>> {
        self.sessions.get_mut(session_id)?.event_receiver.take()
    }
    
    pub fn get_or_create_sender(&mut self, session_id: &str) -> mpsc::Sender<StreamEvent> {
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

use crate::types::StreamEvent;
use std::collections::HashMap;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

pub struct SessionManager {
    sessions: HashMap<String, Session>,
}

pub struct Session {
    pub id: String,
    pub thinking_mode: bool,
    pub messages: Vec<(String, String)>,
    event_sender: broadcast::Sender<StreamEvent>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }
    
    pub fn create_session(&mut self, id: &str) -> &Session {
        let (sender, _) = broadcast::channel(100);
        let session = Session {
            id: id.to_string(),
            thinking_mode: true,
            messages: Vec::new(),
            event_sender: sender,
        };
        
        self.sessions.insert(id.to_string(), session);
        self.sessions.get(id).unwrap()
    }
    
    pub fn get(&self, id: &str) -> Option<&Session> {
        self.sessions.get(id)
    }
    
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Session> {
        self.sessions.get_mut(id)
    }
    
    pub fn get_or_create(&mut self, id: &str) -> &mut Session {
        if !self.sessions.contains_key(id) {
            self.create_session(id);
        }
        self.sessions.get_mut(id).unwrap()
    }
    
    pub fn get_event_stream(&self, id: &str) -> BroadcastStream<StreamEvent> {
        if let Some(session) = self.sessions.get(id) {
            BroadcastStream::new(session.event_sender.subscribe())
        } else {
            let (sender, receiver) = broadcast::channel(1);
            drop(sender);
            BroadcastStream::new(receiver)
        }
    }
}

impl Session {
    pub fn toggle_thinking_mode(&mut self) {
        self.thinking_mode = !self.thinking_mode;
    }
    
    pub fn send_event(&self, event: StreamEvent) {
        let _ = self.event_sender.send(event);
    }
}

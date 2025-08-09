use crate::types::events::StreamEvent;
use std::collections::HashMap;
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio_stream::wrappers::BroadcastStream;

pub struct SessionManager {
    sessions: HashMap<String, Session>,
}

pub struct Session {
    pub id: String,
    pub thinking_mode: bool,
    pub messages: Vec<(String, String)>,
    event_sender: broadcast::Sender<StreamEvent>,
    mpsc_sender: mpsc::Sender<StreamEvent>,
    mpsc_receiver: Option<mpsc::Receiver<StreamEvent>>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }
    
    pub fn create_session(&mut self, id: &str) -> &Session {
        let (broadcast_sender, _) = broadcast::channel(100);
        let (mpsc_sender, mpsc_receiver) = mpsc::channel(100);
        
        let session = Session {
            id: id.to_string(),
            thinking_mode: false,
            messages: Vec::new(),
            event_sender: broadcast_sender,
            mpsc_sender,
            mpsc_receiver: Some(mpsc_receiver),
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
    
    /// Get event receiver for SSE streaming
    pub fn get_event_receiver(&mut self, id: &str) -> mpsc::Receiver<StreamEvent> {
        if let Some(session) = self.sessions.get_mut(id) {
            session.mpsc_receiver.take().unwrap_or_else(|| {
                let (tx, rx) = mpsc::channel(100);
                session.mpsc_sender = tx;
                rx
            })
        } else {
            // Create session if doesn't exist
            self.create_session(id);
            self.get_event_receiver(id)
        }
    }
    
    /// Get or create a sender for the streaming buffer
    pub fn get_or_create_sender(&mut self, id: &str) -> mpsc::Sender<StreamEvent> {
        let session = self.get_or_create(id);
        session.mpsc_sender.clone()
    }
}

impl Session {
    pub fn toggle_thinking_mode(&mut self) {
        self.thinking_mode = !self.thinking_mode;
    }
    
    pub fn send_event(&self, event: StreamEvent) {
        let _ = self.event_sender.send(event.clone());
        let _ = self.mpsc_sender.try_send(event);
    }
}

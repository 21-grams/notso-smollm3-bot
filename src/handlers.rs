use crate::AppState;
use axum::{
    extract::{Path, State, Form},
    response::{Html, Sse, sse::{Event, KeepAlive}},
    http::StatusCode,
};
use minijinja::context;
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::UnboundedReceiverStream;
use futures::stream::{self, Stream};
use std::convert::Infallible;
use std::time::Duration;
use uuid::Uuid;
use chrono::Utc;

#[derive(Clone, Debug)]
pub struct Session {
    pub id: String,
    pub messages: Vec<Message>,
    pub thinking_mode: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: String,
    pub content: String,
    pub thinking_content: Option<String>,
    pub thinking_mode: bool,
    pub timestamp: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub session_id: String,
    pub message: String,
    pub thinking_mode: Option<String>,
}

/// Render main chat page
pub async fn index_page(
    State(state): State<AppState>,
) -> Result<Html<String>, StatusCode> {
    let session_id = Uuid::new_v4().to_string();
    let session = Session {
        id: session_id.clone(),
        messages: vec![],
        thinking_mode: false,
    };
    
    // Store session
    state.sessions.lock().await.insert(session_id.clone(), session.clone());
    
    // Render template
    let html = state.templates
        .render("index.html", context! {
            session_id => session_id,
            messages => session.messages,
            thinking_mode => session.thinking_mode,
        })
        .await
        .map_err(|e| {
            tracing::error!("Template error: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    
    Ok(Html(html))
}

/// SSE connection endpoint - Axum 0.8 syntax
pub async fn sse_connect(
    Path(session_id): Path<String>,
    State(_state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    tracing::info!("📡 SSE connection established for session: {}", session_id);
    
    // Create a keep-alive stream
    let stream = stream::repeat_with(move || {
        Ok(Event::default().comment("keep-alive"))
    })
    .throttle(Duration::from_secs(30));
    
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive")
    )
}

/// Handle chat message submission - Updated for Axum 0.8
pub async fn send_message(
    State(state): State<AppState>,
    Form(payload): Form<ChatRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    // Get session
    let mut sessions = state.sessions.lock().await;
    let session = sessions.get_mut(&payload.session_id)
        .ok_or(StatusCode::NOT_FOUND)?;
    
    // Update thinking mode if provided
    if let Some(mode) = &payload.thinking_mode {
        session.thinking_mode = mode == "true";
    }
    
    // Create user message
    let user_message = Message {
        id: Uuid::new_v4().to_string(),
        role: "user".to_string(),
        content: payload.message.clone(),
        thinking_content: None,
        thinking_mode: false,
        timestamp: Utc::now().format("%H:%M").to_string(),
    };
    
    session.messages.push(user_message.clone());
    
    // Create bot message
    let bot_message_id = Uuid::new_v4().to_string();
    let bot_message = Message {
        id: bot_message_id.clone(),
        role: "assistant".to_string(),
        content: String::new(),
        thinking_content: if session.thinking_mode {
            Some(String::new())
        } else {
            None
        },
        thinking_mode: session.thinking_mode,
        timestamp: Utc::now().format("%H:%M").to_string(),
    };
    
    session.messages.push(bot_message.clone());
    
    let thinking_mode = session.thinking_mode;
    let session_id = payload.session_id.clone();
    drop(sessions); // Release lock before async work
    
    // Create channel for SSE
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    
    // Clone state for async task
    let state_clone = state.clone();
    let prompt = payload.message.clone();
    
    // Spawn generation task
    tokio::spawn(async move {
        // Render and send user message
        if let Ok(user_html) = state_clone.templates
            .render("components/message.html", context! {
                message => user_message,
                is_user => true,
            })
            .await
        {
            let _ = tx.send(Ok(Event::default()
                .event("message")
                .data(user_html)));
        }
        
        // Render and send bot message placeholder
        if let Ok(bot_html) = state_clone.templates
            .render("components/message.html", context! {
                message => bot_message,
                is_user => false,
                session_id => session_id,
            })
            .await
        {
            let _ = tx.send(Ok(Event::default()
                .event("message")
                .data(bot_html)));
        }
        
        // Generate response
        let mut model = state_clone.model.lock().await;
        
        // Format prompt (will use proper chat template in Phase 2)
        let formatted_prompt = format!("User: {}\nAssistant:", prompt);
        
        // Generate with streaming
        if let Err(e) = model.generate_stream_sse(
            &formatted_prompt,
            tx.clone(),
            bot_message_id,
            thinking_mode,
        ).await {
            tracing::error!("Generation error: {}", e);
            let _ = tx.send(Ok(Event::default()
                .event("error")
                .data(format!("Error: {}", e))));
        }
    });
    
    // Convert to stream
    let stream = UnboundedReceiverStream::new(rx);
    
    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive")
    ))
}

/// Status check endpoint
pub async fn status_check(
    State(state): State<AppState>,
) -> Html<String> {
    if state.model.try_lock().is_ok() {
        Html("Ready".to_string())
    } else {
        Html("Busy".to_string())
    }
}

/// Toggle thinking mode
pub async fn toggle_thinking(
    State(state): State<AppState>,
    Form(payload): Form<std::collections::HashMap<String, String>>,
) -> Result<Html<String>, StatusCode> {
    let session_id = payload.get("session_id")
        .ok_or(StatusCode::BAD_REQUEST)?;
    
    let mut sessions = state.sessions.lock().await;
    if let Some(session) = sessions.get_mut(session_id) {
        session.thinking_mode = !session.thinking_mode;
        
        let status = if session.thinking_mode {
            "🧠 Thinking enabled"
        } else {
            "💬 Thinking disabled"
        };
        
        Ok(Html(status.to_string()))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

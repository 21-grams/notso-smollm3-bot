//! Main entry point for SmolLM3 bot with proper Candle integration
//! Uses Axum 0.8 with new path parameter syntax

use axum::{
    routing::{get, post},
    Router,
    extract::{State, Form},
    response::sse::{Event, KeepAlive, Sse},
};
use candle_core::Device;
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration, convert::Infallible};
use tokio::sync::Mutex;
use tower_http::services::ServeDir;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod services;
mod config;
mod state;
mod types;
mod web;

use services::ml::{MLService, MLServiceBuilder};

/// Application state shared across requests
#[derive(Clone)]
struct AppState {
    /// ML service wrapped in Arc<Mutex> for thread safety
    ml_service: Arc<Mutex<MLService>>,
}

/// Chat message form data
#[derive(Debug, Deserialize)]
struct ChatMessageForm {
    message: String,
    session_id: String,
    enable_thinking: bool,
}

/// Chat response format
#[derive(Debug, Serialize)]
struct ChatResponse {
    response: String,
    session_id: String,
    thinking_mode: bool,
}

/// Health check endpoint
async fn health() -> &'static str {
    "SmolLM3 Bot is running!"
}

/// Handle chat messages with SSE streaming
async fn handle_chat(
    State(state): State<AppState>,
    Form(form): Form<ChatMessageForm>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    tracing::info!("📨 Received message: {} (session: {})", 
                  form.message, form.session_id);
    
    // Create SSE stream
    let stream = stream::unfold(
        (state.ml_service, form, 0),
        |(ml_service, form, mut index)| async move {
            // Only run once
            if index > 0 {
                return None;
            }
            index += 1;
            
            // Generate response
            match generate_response(ml_service, form).await {
                Ok(tokens) => {
                    // Stream tokens as SSE events
                    let events: Vec<_> = tokens.into_iter()
                        .map(|token| {
                            Ok(Event::default()
                                .data(token)
                                .event("token"))
                        })
                        .collect();
                    
                    // Add completion event
                    let mut all_events = events;
                    all_events.push(Ok(Event::default()
                        .event("done")
                        .data("Generation complete")));
                    
                    Some((stream::iter(all_events), (ml_service, ChatMessageForm {
                        message: String::new(),
                        session_id: String::new(),
                        enable_thinking: false,
                    }, index)))
                }
                Err(e) => {
                    tracing::error!("❌ Generation error: {}", e);
                    let error_event = Ok(Event::default()
                        .event("error")
                        .data(format!("Error: {}", e)));
                    Some((stream::once(async { error_event }), (ml_service, ChatMessageForm {
                        message: String::new(),
                        session_id: String::new(),
                        enable_thinking: false,
                    }, index)))
                }
            }
        }
    )
    .flatten();
    
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

/// Generate response using ML service
async fn generate_response(
    ml_service: Arc<Mutex<MLService>>,
    form: ChatMessageForm,
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    // Create chat messages
    let messages = vec![
        serde_json::json!({
            "role": "user",
            "content": form.message
        })
    ];
    
    // Lock ML service
    let mut service = ml_service.lock().await;
    
    // Apply chat template
    let prompt = service.apply_chat_template(messages, form.enable_thinking)?;
    
    tracing::info!("📝 Prompt: {}", prompt);
    
    // Generate response
    let tokens = service.generate(&prompt, 256, form.enable_thinking)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
    
    Ok(tokens)
}

/// Simple HTML interface
async fn chat_interface() -> &'static str {
    r#"
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmolLM3 Chat</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            #messages { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; }
            .message { margin: 10px 0; }
            .user { color: blue; }
            .assistant { color: green; }
            .thinking { color: gray; font-style: italic; }
            #input-area { display: flex; gap: 10px; }
            #message-input { flex: 1; padding: 10px; }
            button { padding: 10px 20px; }
            #status { margin-top: 10px; color: gray; }
        </style>
    </head>
    <body>
        <h1>SmolLM3 Chat Bot</h1>
        <div id="messages"></div>
        <div id="input-area">
            <input type="text" id="message-input" placeholder="Type your message..." />
            <label><input type="checkbox" id="thinking-mode" checked /> Thinking Mode</label>
            <button onclick="sendMessage()">Send</button>
        </div>
        <div id="status"></div>
        
        <script>
            const messagesDiv = document.getElementById('messages');
            const messageInput = document.getElementById('message-input');
            const statusDiv = document.getElementById('status');
            const thinkingMode = document.getElementById('thinking-mode');
            const sessionId = Math.random().toString(36).substring(7);
            
            function addMessage(content, className) {
                const msgDiv = document.createElement('div');
                msgDiv.className = 'message ' + className;
                msgDiv.textContent = content;
                messagesDiv.appendChild(msgDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;
                
                addMessage('You: ' + message, 'user');
                messageInput.value = '';
                statusDiv.textContent = 'Generating...';
                
                const formData = new FormData();
                formData.append('message', message);
                formData.append('session_id', sessionId);
                formData.append('enable_thinking', thinkingMode.checked);
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let assistantMessage = '';
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = line.substring(6);
                                if (data && data !== '[DONE]') {
                                    assistantMessage += data;
                                    // Update last assistant message
                                    const lastMsg = messagesDiv.lastElementChild;
                                    if (lastMsg && lastMsg.classList.contains('assistant')) {
                                        lastMsg.textContent = 'SmolLM3: ' + assistantMessage;
                                    } else {
                                        addMessage('SmolLM3: ' + assistantMessage, 'assistant');
                                    }
                                }
                            }
                        }
                    }
                    
                    statusDiv.textContent = 'Ready';
                } catch (error) {
                    statusDiv.textContent = 'Error: ' + error.message;
                    addMessage('Error: ' + error.message, 'error');
                }
            }
            
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    "#
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    tracing::info!("🚀 Starting SmolLM3 Bot Server");
    
    // Setup device (prefer CUDA, fallback to CPU)
    let device = Device::cuda_if_available(0)
        .unwrap_or(Device::Cpu);
    
    tracing::info!("🖥️ Using device: {:?}", device);
    
    // Build ML service
    let ml_service = MLServiceBuilder::default()
        .model_path("models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf")
        .tokenizer_path("models/tokenizer.json")
        .template_path("models/smollm3_thinking_chat_template.jinja2")
        .device(device)
        .temperature(0.9)
        .top_p(0.95)
        .seed(42)
        .build()?;
    
    // Create app state
    let state = AppState {
        ml_service: Arc::new(Mutex::new(ml_service)),
    };
    
    // Build router with Axum 0.8 syntax
    let app = Router::new()
        .route("/", get(chat_interface))
        .route("/health", get(health))
        .route("/chat", post(handle_chat))
        .nest_service("/static", ServeDir::new("static"))
        .with_state(state);
    
    // Start server
    let addr = "0.0.0.0:3000";
    tracing::info!("🌐 Server listening on http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

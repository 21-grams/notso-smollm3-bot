//! Template rendering with MiniJinja 2 for streaming chat

use minijinja::{Environment, Value, context};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use pulldown_cmark::{Parser, html};

/// Message role in conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Message data for rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageData {
    pub message_id: String,
    pub role: MessageRole,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub token_count: Option<usize>,
    pub generation_time: Option<u64>,
}

/// Template engine for chat UI
pub struct ChatTemplateEngine {
    env: Arc<RwLock<Environment<'static>>>,
}

impl ChatTemplateEngine {
    /// Initialize template engine with templates
    pub fn new() -> anyhow::Result<Self> {
        let mut env = Environment::new();
        
        // Load templates
        env.add_template("message/container", 
            include_str!("../../web/templates/components/message/container.html"))?;
        env.add_template("message/stream_chunk", 
            include_str!("../../web/templates/components/message/stream_chunk.html"))?;
        env.add_template("message/final", 
            include_str!("../../web/templates/components/message/final.html"))?;
        
        // Add custom filters
        env.add_filter("markdown", markdown_filter);
        env.add_filter("escape_html", html_escape_filter);
        env.add_filter("time_ago", time_ago_filter);
        
        Ok(Self {
            env: Arc::new(RwLock::new(env)),
        })
    }
    
    /// Render initial message container
    pub async fn render_message_container(
        &self,
        role: MessageRole,
        content: Option<&str>,
    ) -> anyhow::Result<String> {
        let message_id = Uuid::now_v7().to_string();
        let env = self.env.read().await;
        let tmpl = env.get_template("message/container")?;
        
        let html = tmpl.render(context! {
            message_id => message_id.clone(),
            role => role,
            content => content.unwrap_or(""),
            timestamp => Utc::now().to_rfc3339(),
        })?;
        
        Ok(html)
    }
    
    /// Render a streaming chunk (for SSE)
    pub async fn render_stream_chunk(
        &self,
        message_id: &str,
        content: &str,
        chunk_index: usize,
    ) -> anyhow::Result<StreamEvent> {
        let env = self.env.read().await;
        let tmpl = env.get_template("message/stream_chunk")?;
        
        let chunk_html = tmpl.render(context! {
            chunk_id => format!("{}-{}", message_id, chunk_index),
            content => content,
        })?;
        
        // Wrap in SSE event format
        Ok(StreamEvent {
            event_type: "content",
            target: format!("#content-{}", message_id),
            action: "append",
            data: chunk_html,
        })
    }
    
    /// Render final message with markdown
    pub async fn render_final_message(
        &self,
        message_id: &str,
        full_content: &str,
        token_count: usize,
        generation_time_ms: u64,
    ) -> anyhow::Result<String> {
        let env = self.env.read().await;
        let tmpl = env.get_template("message/final")?;
        
        // Convert markdown to HTML
        let rendered_markdown = markdown_to_html(full_content);
        
        let html = tmpl.render(context! {
            message_id => message_id,
            rendered_markdown => rendered_markdown,
            token_count => token_count,
            generation_time => generation_time_ms,
        })?;
        
        Ok(html)
    }
}

/// SSE Event structure for HTMX
#[derive(Debug, Serialize)]
pub struct StreamEvent {
    pub event_type: &'static str,
    pub target: String,
    pub action: &'static str,  // "append", "replace", "morph"
    pub data: String,
}

impl StreamEvent {
    /// Convert to SSE format
    pub fn to_sse(&self) -> String {
        format!(
            "event: {}\ndata: {}\n\n",
            self.event_type,
            serde_json::json!({
                "target": self.target,
                "action": self.action,
                "html": self.data
            })
        )
    }
}

/// Convert markdown to HTML
fn markdown_to_html(markdown: &str) -> String {
    let parser = Parser::new(markdown);
    let mut html_output = String::new();
    html::push_html(&mut html_output, parser);
    html_output
}

/// MiniJinja filter for markdown
fn markdown_filter(value: Value) -> Result<Value, minijinja::Error> {
    let markdown = value.as_str().ok_or_else(|| {
        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "expected string")
    })?;
    Ok(Value::from(markdown_to_html(markdown)))
}

/// HTML escape filter
fn html_escape_filter(value: Value) -> Result<Value, minijinja::Error> {
    let text = value.as_str().ok_or_else(|| {
        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "expected string")
    })?;
    Ok(Value::from(html_escape::encode_text(text).to_string()))
}

/// Time ago filter
fn time_ago_filter(value: Value) -> Result<Value, minijinja::Error> {
    // Simple time ago implementation
    Ok(Value::from("just now"))
}

/// Streaming response handler
pub struct StreamingResponseHandler {
    template_engine: Arc<ChatTemplateEngine>,
    message_id: String,
    buffer: String,
    chunk_index: usize,
    start_time: std::time::Instant,
    token_count: usize,
}

impl StreamingResponseHandler {
    pub fn new(
        template_engine: Arc<ChatTemplateEngine>,
        message_id: String,
    ) -> Self {
        Self {
            template_engine,
            message_id,
            buffer: String::new(),
            chunk_index: 0,
            start_time: std::time::Instant::now(),
            token_count: 0,
        }
    }
    
    /// Process a token and return SSE event if buffer is ready to flush
    pub async fn process_token(
        &mut self,
        token: &str,
    ) -> Option<StreamEvent> {
        self.buffer.push_str(token);
        self.token_count += 1;
        
        // Flush buffer based on conditions
        if self.should_flush() {
            let content = self.buffer.clone();
            self.buffer.clear();
            self.chunk_index += 1;
            
            self.template_engine
                .render_stream_chunk(&self.message_id, &content, self.chunk_index)
                .await
                .ok()
        } else {
            None
        }
    }
    
    /// Check if buffer should be flushed
    fn should_flush(&self) -> bool {
        // Flush on natural boundaries or size
        self.buffer.ends_with('\n') || 
        self.buffer.ends_with(". ") ||
        self.buffer.len() > 100 ||
        self.token_count % 10 == 0
    }
    
    /// Finalize the message
    pub async fn finalize(
        mut self,
        full_content: String,
    ) -> anyhow::Result<String> {
        let generation_time = self.start_time.elapsed().as_millis() as u64;
        
        self.template_engine
            .render_final_message(
                &self.message_id,
                &full_content,
                self.token_count,
                generation_time,
            )
            .await
    }
}

//! MiniJinja template engine wrapper

use minijinja::{Environment, Value, context, Error};
use std::path::Path;
use anyhow::Result;

pub struct TemplateEngine {
    env: Environment<'static>,
}

impl TemplateEngine {
    pub fn new() -> Result<Self> {
        let mut env = Environment::new();
        
        // Configure MiniJinja
        env.set_debug(cfg!(debug_assertions));
        
        // Load templates from src/web/templates directory
        let template_path = "src/web/templates";
        if Path::new(template_path).exists() {
            env.set_loader(minijinja::path_loader(template_path));
        } else {
            tracing::warn!("Template directory not found: {}", template_path);
        }
        
        // Add custom filters
        env.add_filter("datetime", format_datetime);
        env.add_filter("escape_html", escape_html);
        env.add_filter("truncate", truncate_text);
        
        Ok(Self { env })
    }
    
    /// Render a template with context
    pub fn render(&self, template_name: &str, ctx: Value) -> Result<String> {
        let template = self.env.get_template(template_name)?;
        Ok(template.render(ctx)?)
    }
    
    /// Add a template from string
    pub fn add_template(&mut self, name: &str, content: &str) -> Result<()> {
        self.env.add_template(name, content)?;
        Ok(())
    }
    
    /// Render chat page
    pub fn render_chat_page(&self, session_id: &str, thinking_mode: bool) -> Result<String> {
        self.render("chat.html", context! {
            session_id => session_id,
            thinking_mode => thinking_mode,
        })
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create template engine")
    }
}

// Custom filter functions
fn format_datetime(value: &Value) -> Result<String, Error> {
    let timestamp = value.as_str().ok_or_else(|| {
        Error::new(minijinja::ErrorKind::InvalidOperation, "expected string")
    })?;
    
    // For now, just return the timestamp as-is
    // Could be enhanced with proper formatting
    Ok(timestamp.to_string())
}

fn escape_html(value: &Value) -> Result<String, Error> {
    let text = value.as_str().ok_or_else(|| {
        Error::new(minijinja::ErrorKind::InvalidOperation, "expected string")
    })?;
    
    let escaped = text
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;");
    
    Ok(escaped)
}

fn truncate_text(value: &Value, length: Option<Value>) -> Result<String, Error> {
    let text = value.as_str().ok_or_else(|| {
        Error::new(minijinja::ErrorKind::InvalidOperation, "expected string")
    })?;
    
    let max_length = length
        .and_then(|v| v.as_u64())
        .unwrap_or(50) as usize;
    
    if text.len() <= max_length {
        Ok(text.to_string())
    } else {
        Ok(format!("{}...", &text[..max_length]))
    }
}

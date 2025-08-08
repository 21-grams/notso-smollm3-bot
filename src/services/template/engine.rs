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
    
    /// Add a template from string - requires owned strings for 'static lifetime
    pub fn add_template_owned(&mut self, name: String, content: String) -> Result<()> {
        // MiniJinja needs 'static strings, so we leak the memory
        // This is okay for templates as they're loaded once at startup
        let name_static: &'static str = Box::leak(name.into_boxed_str());
        let content_static: &'static str = Box::leak(content.into_boxed_str());
        self.env.add_template(name_static, content_static)?;
        Ok(())
    }
    
    /// Convenience method that clones the strings
    pub fn add_template(&mut self, name: &str, content: &str) -> Result<()> {
        self.add_template_owned(name.to_string(), content.to_string())
    }
    
    /// Render chat page
    pub fn render_chat_page(&self, session_id: &str, thinking_mode: bool) -> Result<String> {
        self.render("chat.html", context! {
            session_id => session_id,
            thinking_mode => thinking_mode,
            messages => Vec::<String>::new(),
        })
    }
    
    /// Render index page
    pub fn render_index(&self) -> Result<String> {
        self.render("index.html", context! {
            title => "NotSo-SmolLM3 Bot",
            version => env!("CARGO_PKG_VERSION"),
        })
    }
}

// Filter functions
fn format_datetime(value: &Value, _: &[Value]) -> Result<Value, Error> {
    if let Some(timestamp) = value.as_i64() {
        let dt = chrono::DateTime::from_timestamp(timestamp, 0)
            .ok_or_else(|| Error::new(minijinja::ErrorKind::InvalidOperation, "invalid timestamp"))?;
        Ok(Value::from(dt.format("%Y-%m-%d %H:%M:%S").to_string()))
    } else {
        Err(Error::new(minijinja::ErrorKind::InvalidOperation, "expected timestamp"))
    }
}

fn escape_html(value: &Value, _: &[Value]) -> Result<Value, Error> {
    let text = value.as_str().ok_or_else(|| {
        Error::new(minijinja::ErrorKind::InvalidOperation, "expected string")
    })?;
    
    let escaped = text
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;");
    
    Ok(Value::from(escaped))
}

fn truncate_text(value: &Value, length: &[Value]) -> Result<Value, Error> {
    let text = value.as_str().ok_or_else(|| {
        Error::new(minijinja::ErrorKind::InvalidOperation, "expected string")
    })?;
    
    let max_length = length
        .first()
        .and_then(|v| v.as_i64().map(|i| i as usize))
        .unwrap_or(50);
    
    if text.len() <= max_length {
        Ok(Value::from(text))
    } else {
        Ok(Value::from(format!("{}...", &text[..max_length])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_template_engine() -> Result<()> {
        let mut engine = TemplateEngine::new()?;
        engine.add_template("test", "Hello {{ name }}!")?;
        
        let result = engine.render("test", context! { name => "World" })?;
        assert_eq!(result, "Hello World!");
        
        Ok(())
    }
}

use minijinja::{Environment, Value, Error};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct TemplateEngine {
    env: Arc<RwLock<Environment<'static>>>,
}

impl TemplateEngine {
    pub fn new() -> anyhow::Result<Self> {
        let mut env = Environment::new();
        
        // Configure MiniJinja
        env.set_debug(cfg!(debug_assertions));
        
        // Load templates from filesystem
        env.set_loader(minijinja::path_loader("templates"));
        
        // Add custom filters
        env.add_filter("datetime", format_datetime);
        env.add_filter("escape_html", escape_html);
        
        Ok(Self {
            env: Arc::new(RwLock::new(env)),
        })
    }
    
    pub async fn render<S: Serialize>(
        &self,
        template_name: &str,
        context: S,
    ) -> anyhow::Result<String> {
        let env = self.env.read().await;
        let template = env.get_template(template_name)?;
        Ok(template.render(context)?)
    }
}

fn format_datetime(_env: &Environment, value: &Value) -> Result<String, Error> {
    let timestamp = value.as_str().ok_or_else(|| {
        Error::new(minijinja::ErrorKind::InvalidOperation, "expected string")
    })?;
    Ok(timestamp.to_string())
}

fn escape_html(_env: &Environment, value: &Value) -> Result<String, Error> {
    let text = value.as_str().ok_or_else(|| {
        Error::new(minijinja::ErrorKind::InvalidOperation, "expected string")
    })?;
    
    let escaped = text
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;");
    
    Ok(escaped)
}

//! Chat-specific template rendering

use super::TemplateEngine;
use minijinja::context;
use anyhow::Result;

pub struct ChatTemplateService {
    engine: TemplateEngine,
}

impl ChatTemplateService {
    pub fn new() -> Result<Self> {
        Ok(Self {
            engine: TemplateEngine::new()?,
        })
    }
    
    /// Render user message bubble
    pub fn render_user_message(&self, content: &str) -> Result<String> {
        let template = r#"
<div class="message user">
    <div class="message-bubble">{{ content }}</div>
</div>
"#;
        
        let mut engine = TemplateEngine::new()?;
        engine.add_template("user_message", template)?;
        engine.render("user_message", context! { content => content })
    }
    
    /// Render assistant message bubble
    pub fn render_assistant_message(&self, content: &str, is_thinking: bool) -> Result<String> {
        let template = r#"
<div class="message assistant">
    {%- if is_thinking -%}
    <div class="thinking-indicator">🤔 Thinking...</div>
    {%- endif -%}
    <div class="message-bubble">{{ content }}</div>
</div>
"#;
        
        let mut engine = TemplateEngine::new()?;
        engine.add_template("assistant_message", template)?;
        engine.render("assistant_message", context! { 
            content => content,
            is_thinking => is_thinking
        })
    }
}

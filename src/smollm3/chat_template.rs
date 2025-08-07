//! Chat template formatting for SmolLM3

use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn new() -> Self {
        let mut env = Environment::new();
        
        // SmolLM3 chat template
        let template = r#"
{%- if system_message -%}
<|im_start|>system
{{ system_message }}
<|im_end|>
{%- endif -%}

{%- for message in messages -%}
{%- if message.role == "user" -%}
<|im_start|>user
{{ message.content }}<|im_end|>
{%- elif message.role == "assistant" -%}
<|im_start|>assistant
{%- if thinking_mode and not message.content.contains("<think>") -%}
<think>
Let me think about this step by step.
</think>
{%- endif -%}
{{ message.content }}<|im_end|>
{%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
<|im_start|>assistant
{%- if thinking_mode -%}
<think>
{%- endif -%}
{%- endif -%}
"#;
        
        env.add_template("chat", template).unwrap();
        
        Self { env }
    }
    
    pub fn format(
        &self,
        messages: &[ChatMessage],
        system_message: Option<&str>,
        thinking_mode: bool,
        add_generation_prompt: bool,
    ) -> String {
        let template = self.env.get_template("chat").unwrap();
        
        template.render(context! {
            messages => messages,
            system_message => system_message,
            thinking_mode => thinking_mode,
            add_generation_prompt => add_generation_prompt,
        }).unwrap_or_default()
    }
    
    pub fn format_single_turn(&self, user_message: &str, thinking_start: u32) -> String {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: user_message.to_string(),
            }
        ];
        
        self.format(
            &messages,
            Some("You are SmolLM3, a helpful AI assistant."),
            true,  // Enable thinking by default
            true,  // Add generation prompt
        )
    }
}

impl Default for ChatTemplate {
    fn default() -> Self {
        Self::new()
    }
}

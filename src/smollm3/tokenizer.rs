use anyhow::Result;
use minijinja::{Environment, context};
use serde_json::Value;

pub struct SmolLM3Tokenizer {
    inner: Option<tokenizers::Tokenizer>,
    jinja: Option<Environment<'static>>,
    is_stub: bool,
}

impl SmolLM3Tokenizer {
    /// Create tokenizer from file
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)?;
        Ok(Self {
            inner: Some(tokenizer),
            jinja: None,
            is_stub: false,
        })
    }
    
    /// Create tokenizer with template support
    pub fn new(tokenizer_path: &str, template_path: &str) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)?;
        let mut jinja = Environment::new();
        let template_content = std::fs::read_to_string(template_path)?;
        jinja.add_template("chat", &template_content)?;
        
        Ok(Self {
            inner: Some(tokenizer),
            jinja: Some(jinja),
            is_stub: false,
        })
    }
    
    /// Create stub tokenizer for testing
    pub fn new_stub() -> Self {
        Self {
            inner: None,
            jinja: None,
            is_stub: true,
        }
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if self.is_stub {
            // Return mock token IDs
            Ok(text.chars().map(|_| 1u32).collect())
        } else if let Some(tokenizer) = &self.inner {
            let encoding = tokenizer.encode(text, false)?;
            Ok(encoding.get_ids().to_vec())
        } else {
            anyhow::bail!("Tokenizer not initialized")
        }
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        if self.is_stub {
            Ok("Mock decoded text".to_string())
        } else if let Some(tokenizer) = &self.inner {
            Ok(tokenizer.decode(ids, false)?)
        } else {
            anyhow::bail!("Tokenizer not initialized")
        }
    }
    
    /// Apply chat template to messages
    pub fn apply_chat_template(
        &self, 
        messages: Vec<Value>,
        enable_thinking: bool
    ) -> Result<String> {
        if self.is_stub {
            Ok("Mock prompt".to_string())
        } else if let Some(jinja) = &self.jinja {
            let ctx = context! {
                messages => messages,
                enable_thinking => enable_thinking,
                add_generation_prompt => true,
                system_message => "You are a helpful AI assistant.",
            };
            
            let template = jinja.get_template("chat")?;
            Ok(template.render(ctx)?)
        } else {
            // Fallback to simple concatenation
            let mut prompt = String::new();
            for msg in messages {
                if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                    prompt.push_str(content);
                    prompt.push_str("\n");
                }
            }
            Ok(prompt)
        }
    }
}

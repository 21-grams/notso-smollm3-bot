use tokenizers::Tokenizer;
use anyhow::Result;
use crate::config::SmolLM3Config;

/// SmolLM3 tokenizer wrapper
pub struct SmolLM3Tokenizer {
    tokenizer: Tokenizer,
    config: SmolLM3Config,
}

impl SmolLM3Tokenizer {
    /// Load tokenizer from file
    pub fn from_file(tokenizer_path: &str, config: SmolLM3Config) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        Ok(Self { tokenizer, config })
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, add_special_tokens)?;
        Ok(encoding.get_ids().to_vec())
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        Ok(self.tokenizer.decode(ids, skip_special_tokens)?)
    }
    
    /// Format chat messages with SmolLM3 template
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        thinking_mode: bool,
    ) -> Result<String> {
        let mut prompt = String::new();
        
        // System prompt
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str("You are SmolLM3, a helpful AI assistant created by Hugging Face.\n");
        if thinking_mode {
            prompt.push_str("You are in thinking mode. Use <think></think> tags for your reasoning process.\n");
        }
        prompt.push_str("<|im_end|>\n");
        
        // Message history
        for msg in messages {
            match msg.role.as_str() {
                "user" => {
                    prompt.push_str("<|im_start|>user\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n<|im_end|>\n");
                }
                "assistant" => {
                    prompt.push_str("<|im_start|>assistant\n");
                    
                    // Include thinking content if present
                    if thinking_mode {
                        if let Some(thinking) = &msg.thinking_content {
                            prompt.push_str(&self.config.thinking_tokens.start_text);
                            prompt.push_str(thinking);
                            prompt.push_str(&self.config.thinking_tokens.end_text);
                            prompt.push('\n');
                        }
                    }
                    
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n<|im_end|>\n");
                }
                _ => {}
            }
        }
        
        // Generation prompt
        if add_generation_prompt {
            prompt.push_str("<|im_start|>assistant\n");
            
            // Start with thinking tag if in thinking mode
            if thinking_mode {
                prompt.push_str(&self.config.thinking_tokens.start_text);
                prompt.push_str("\nLet me think about this...\n");
            }
        }
        
        Ok(prompt)
    }
    
    /// Check if token is a stop token
    pub fn is_stop_token(&self, token_id: u32) -> bool {
        // Common stop tokens for SmolLM3
        token_id == 128001 ||  // <|im_end|>
        token_id == 128009 ||  // <|eot_id|>
        token_id == 2         // EOS
    }
    
    /// Check if token is thinking start
    pub fn is_thinking_start(&self, token_id: u32) -> bool {
        token_id == self.config.thinking_tokens.start_id
    }
    
    /// Check if token is thinking end
    pub fn is_thinking_end(&self, token_id: u32) -> bool {
        token_id == self.config.thinking_tokens.end_id
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub thinking_content: Option<String>,
}

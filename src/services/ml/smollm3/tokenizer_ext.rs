//! Extended tokenizer functionality for SmolLM3

use tokenizers::Tokenizer;
use anyhow::Result;
use super::chat_template::{ChatTemplate, ChatMessage};
use minijinja::Environment;

pub struct SmolLM3TokenizerExt {
    tokenizer: Tokenizer,
    special_tokens: SpecialTokens,
}

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos: u32,           // <|begin_of_text|>
    pub eos: u32,           // <|end_of_text|>
    pub thinking_start: u32, // <think>
    pub thinking_end: u32,   // </think>
    pub tool_start: u32,     // <tool>
    pub tool_end: u32,       // </tool>
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos: 128000,
            eos: 128001,
            thinking_start: 128002,
            thinking_end: 128003,
            tool_start: 128004,
            tool_end: 128005,
        }
    }
}

impl SmolLM3TokenizerExt {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            special_tokens: SpecialTokens::default(),
        }
    }
    
    /// Create tokenizer from file
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self::new(tokenizer))
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_with_special(text, true)
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.decode_filtered(ids, false)
    }
    
    /// Encode with special token handling
    pub fn encode_with_special(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, add_special)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }
    
    /// Decode with special token filtering
    pub fn decode_filtered(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        if skip_special {
            let filtered: Vec<u32> = ids.iter()
                .filter(|&&id| !self.is_special_token(id))
                .copied()
                .collect();
            Ok(self.tokenizer.decode(&filtered, false)
                .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {}", e))?)
        } else {
            Ok(self.tokenizer.decode(ids, false)
                .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {}", e))?)
        }
    }
    
    fn is_special_token(&self, id: u32) -> bool {
        id == self.special_tokens.bos ||
        id == self.special_tokens.eos ||
        id == self.special_tokens.thinking_start ||
        id == self.special_tokens.thinking_end ||
        id == self.special_tokens.tool_start ||
        id == self.special_tokens.tool_end
    }
    
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
    
    /// Apply SmolLM3 chat template using the ChatTemplate struct
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        reasoning_mode: ReasoningMode,
    ) -> Result<String> {
        let template = ChatTemplate::new();
        let thinking_mode = reasoning_mode == ReasoningMode::Think;
        
        Ok(template.format(
            messages,
            Some("You are SmolLM3, a helpful AI assistant."),
            thinking_mode,
            add_generation_prompt,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReasoningMode {
    Think,
    NoThink,
}

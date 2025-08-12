//! SmolLM3 Tokenizer with processing pipeline
//! 
//! Implements a clean tokenizer with builder pattern for enforcing
//! processing order while maintaining simplicity.

use tokenizers::Tokenizer;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;
use chrono::Local;

/// Chat message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Special token IDs
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos: u32,           // <|begin_of_text|>
    pub eos: u32,           // <|end_of_text|>
    pub thinking_start: u32, // <think>
    pub thinking_end: u32,   // </think>
    pub pad: u32,           // <|finetune_right_pad_id|>
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos: 128000,
            eos: 128001,
            thinking_start: 128002,
            thinking_end: 128003,
            pad: 128004,
        }
    }
}

/// Tokenizer configuration from tokenizer_config.json
#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    pad_token_id: Option<u32>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<u32>,
    #[serde(default)]
    padding_side: Option<String>,
}

/// Main SmolLM3 tokenizer
pub struct SmolLM3Tokenizer {
    tokenizer: Tokenizer,
    special_tokens: SpecialTokens,
    padding_side: String,
    chat_template: minijinja::Environment<'static>,
}

impl SmolLM3Tokenizer {
    /// Load tokenizer from files at server startup
    pub fn from_files<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_path = model_dir.as_ref();
        
        // Load main tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e))?;
        
        // Load tokenizer config
        let config_path = model_path.join("tokenizer_config.json");
        let config_str = fs::read_to_string(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read {:?}: {}", config_path, e))?;
        let config: TokenizerConfig = serde_json::from_str(&config_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer_config.json: {}", e))?;
        
        // Load special tokens if available
        let special_tokens_path = model_path.join("special_tokens_map.json");
        let mut special_tokens = SpecialTokens::default();
        
        if special_tokens_path.exists() {
            let special_str = fs::read_to_string(&special_tokens_path)?;
            // Parse and update special_tokens if needed
            // For now, using defaults
        }
        
        // Update special tokens from config
        if let Some(bos) = config.bos_token_id {
            special_tokens.bos = bos;
        }
        if let Some(eos) = config.eos_token_id {
            special_tokens.eos = eos;
        }
        if let Some(pad) = config.pad_token_id {
            special_tokens.pad = pad;
        }
        
        // Configure padding
        let padding_side = config.padding_side.unwrap_or_else(|| "left".to_string());
        if let Some(padding) = tokenizer.get_padding_mut() {
            padding.pad_id = special_tokens.pad;
            padding.pad_token = "<|finetune_right_pad_id|>".to_string();
        }
        
        // Set up chat template with custom functions
        let mut chat_template = minijinja::Environment::new();
        
        // Register the strftime_now function
        chat_template.add_function("strftime_now", |format: String| {
            // Use chrono to get current local time and format it
            let now = Local::now();
            now.format(&format).to_string()
        });
        
        // Load the template
        let template = include_str!("../../../../templates/smollm3_official.j2");
        chat_template.add_template("chat", template)?;
        
        Ok(Self {
            tokenizer,
            special_tokens,
            padding_side,
            chat_template,
        })
    }
    
    /// Single entry point for processing user input
    pub fn process_input(
        &self,
        user_input: String,
        thinking_enabled: bool,
    ) -> Result<Vec<Vec<u32>>> {
        Pipeline::new(self, user_input)
            .with_thinking(thinking_enabled)
            .process()
    }
    
    /// Get special token IDs for the model
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
    
    /// Simple encode for compatibility
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encodings = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
        Ok(encodings.get_ids().to_vec())
    }
    
    /// Simple decode for compatibility
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }
    
    // Internal processing methods
    
    fn sanitize_input(&self, input: String) -> String {
        // TODO: Implement input sanitization
        // - Remove control characters
        // - Normalize whitespace
        // - Handle encoding issues
        input.trim().to_string()
    }
    
    fn filter_prompt(&self, input: String) -> Result<String> {
        // TODO: Implement malicious activity filtering
        // - Check for prompt injection attempts
        // - Filter harmful content
        // - Rate limiting checks
        
        // For now, just pass through
        Ok(input)
    }
    
    fn apply_chat_template(
        &self,
        input: String,
        conversation_history: Option<&[ChatMessage]>,
        thinking_enabled: bool,
    ) -> Result<String> {
        // Build messages
        let mut messages = conversation_history
            .map(|h| h.to_vec())
            .unwrap_or_default();
        
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: input,
        });
        
        // Prepare context matching the template's expectations
        let ctx = minijinja::context! {
            messages => messages,
            add_generation_prompt => true,
            enable_thinking => thinking_enabled,  // Changed from thinking_mode
            // System message is handled by template defaults
            // Tools not implemented yet
            xml_tools => false,
            python_tools => false,
            tools => false,
        };
        
        // Render template
        let template = self.chat_template.get_template("chat")?;
        let rendered = template.render(ctx)?;
        
        Ok(rendered)
    }
    
    fn encode_batch(&self, inputs: Vec<String>) -> Result<Vec<Vec<u32>>> {
        // Always use batch encoding, even for single input
        let encodings = self.tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| anyhow::anyhow!("Batch encoding failed: {}", e))?;
        
        // Extract token IDs
        let mut result = Vec::with_capacity(encodings.len());
        for encoding in encodings {
            result.push(encoding.get_ids().to_vec());
        }
        
        Ok(result)
    }
}

/// Pipeline builder for enforcing processing order
pub struct Pipeline<'a, S> {
    tokenizer: &'a SmolLM3Tokenizer,
    data: String,
    thinking_enabled: bool,
    conversation_history: Option<Vec<ChatMessage>>,
    _state: std::marker::PhantomData<S>,
}

// Pipeline states for type-safe ordering
struct Initial;
struct Configured;

impl<'a> Pipeline<'a, Initial> {
    fn new(tokenizer: &'a SmolLM3Tokenizer, input: String) -> Self {
        Pipeline {
            tokenizer,
            data: input,
            thinking_enabled: false,
            conversation_history: None,
            _state: std::marker::PhantomData,
        }
    }
    
    /// Configure thinking mode
    pub fn with_thinking(self, enabled: bool) -> Pipeline<'a, Configured> {
        Pipeline {
            tokenizer: self.tokenizer,
            data: self.data,
            thinking_enabled: enabled,
            conversation_history: self.conversation_history,
            _state: std::marker::PhantomData,
        }
    }
    
    /// Add conversation history
    pub fn with_history(mut self, history: Vec<ChatMessage>) -> Self {
        self.conversation_history = Some(history);
        self
    }
}

impl<'a> Pipeline<'a, Configured> {
    /// Process the pipeline - enforces that configuration happens before processing
    pub fn process(self) -> Result<Vec<Vec<u32>>> {
        // Step 1: Sanitize
        let sanitized = self.tokenizer.sanitize_input(self.data);
        tracing::debug!("After sanitization: {} chars", sanitized.len());
        
        // Step 2: Filter
        let filtered = self.tokenizer.filter_prompt(sanitized)?;
        tracing::debug!("After filtering: {} chars", filtered.len());
        
        // Step 3: Apply chat template
        let templated = self.tokenizer.apply_chat_template(
            filtered,
            self.conversation_history.as_deref(),
            self.thinking_enabled,
        )?;
        tracing::debug!("After template: {} chars", templated.len());
        
        // Step 4: Encode as batch
        let tokens = self.tokenizer.encode_batch(vec![templated])?;
        tracing::info!(
            "Tokenization complete: batch_size={}, seq_len={}", 
            tokens.len(),
            tokens.get(0).map(|t| t.len()).unwrap_or(0)
        );
        
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pipeline_ordering() {
        // This would fail to compile if trying to process without configuring:
        // let tokenizer = SmolLM3Tokenizer::from_files("models").unwrap();
        // let tokens = Pipeline::new(&tokenizer, "test".to_string())
        //     .process();  // ERROR: process() not available on Pipeline<Initial>
        
        // This compiles:
        // let tokens = Pipeline::new(&tokenizer, "test".to_string())
        //     .with_thinking(true)  // Returns Pipeline<Configured>
        //     .process();           // Now process() is available
    }
}

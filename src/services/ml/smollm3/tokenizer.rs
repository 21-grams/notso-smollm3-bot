//! SmolLM3 Tokenizer with processing pipeline
//! 
//! Implements a clean tokenizer with builder pattern for enforcing
//! processing order while maintaining simplicity.

use tokenizers::{Tokenizer, AddedToken};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::fs;
use chrono::Local;

/// Chat message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Special token IDs dynamically loaded from configuration
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos: u32,           // <|begin_of_text|> or <|im_start|>
    pub eos: u32,           // <|im_end|> (NOT <|end_of_text|>!)
    pub thinking_start: u32, // <think>
    pub thinking_end: u32,   // </think>
    pub pad: u32,           // <|finetune_right_pad_id|>
    pub im_start: u32,      // <|im_start|>
    pub im_end: u32,        // <|im_end|>
    // Set of all special token IDs for filtering
    pub special_ids: HashSet<u32>,
    // Reserved token range that should be filtered
    pub reserved_range: (u32, u32),  // (128009, 128255)
}

/// Full tokenizer configuration from tokenizer_config.json
#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    pad_token_id: Option<u32>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<u32>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    pad_token: Option<String>,
    #[serde(default)]
    padding_side: Option<String>,
    #[serde(default)]
    added_tokens_decoder: Option<HashMap<String, AddedTokenInfo>>,
    #[serde(default)]
    clean_up_tokenization_spaces: Option<bool>,
}

/// Added token information from config
#[derive(Debug, Deserialize)]
struct AddedTokenInfo {
    content: String,
    special: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    single_word: bool,
}

/// Main SmolLM3 tokenizer with dynamic configuration
pub struct SmolLM3Tokenizer {
    tokenizer: Tokenizer,
    special_tokens: SpecialTokens,
    padding_side: String,
    chat_template: minijinja::Environment<'static>,
    // Map of special token strings to IDs
    special_token_map: HashMap<String, u32>,
    // Whether to clean up tokenization spaces
    clean_up_spaces: bool,
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
        
        // Parse as both typed config and raw JSON for flexibility
        let config: TokenizerConfig = serde_json::from_str(&config_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer_config.json: {}", e))?;
        let config_json: Value = serde_json::from_str(&config_str)?;
        
        // Build special token map from added_tokens_decoder
        let mut special_token_map = HashMap::new();
        let mut special_ids = HashSet::new();
        let mut tokens_to_add = Vec::new();
        
        // Parse added_tokens_decoder to get all special tokens
        if let Some(decoder) = config_json.get("added_tokens_decoder").and_then(|v| v.as_object()) {
            for (id_str, token_info) in decoder {
                if let (Ok(id), Some(content)) = (
                    id_str.parse::<u32>(),
                    token_info.get("content").and_then(|v| v.as_str())
                ) {
                    special_token_map.insert(content.to_string(), id);
                    let is_special = token_info.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
                    if is_special {
                        special_ids.insert(id);
                    }
                    
                    // Create AddedToken for this token
                    let added_token = AddedToken {
                        content: content.to_string(),
                        single_word: token_info.get("single_word").and_then(|v| v.as_bool()).unwrap_or(false),
                        lstrip: token_info.get("lstrip").and_then(|v| v.as_bool()).unwrap_or(false),
                        rstrip: token_info.get("rstrip").and_then(|v| v.as_bool()).unwrap_or(false),
                        normalized: token_info.get("normalized").and_then(|v| v.as_bool()).unwrap_or(false),
                        special: is_special,
                    };
                    tokens_to_add.push(added_token);
                }
            }
        }
        
        // Add all special tokens to the tokenizer
        if !tokens_to_add.is_empty() {
            let num_added = tokenizer.add_special_tokens(&tokens_to_add);
            tracing::info!("[TOKENIZER] Added {} special tokens to tokenizer", num_added);
        }
        
        // Now verify tokens are properly registered and get their IDs
        let get_token_id = |token_str: &str| -> Option<u32> {
            tokenizer.token_to_id(token_str)
        };
        
        // Log vocabulary size after adding special tokens
        let vocab_size = tokenizer.get_vocab_size(true);
        tracing::info!("[TOKENIZER] Vocabulary size after adding special tokens: {}", vocab_size);
        
        // Get critical token IDs using token_to_id
        let im_start_id = get_token_id("<|im_start|>")
            .ok_or_else(|| anyhow::anyhow!("Missing <|im_start|> token"))?;
        let im_end_id = get_token_id("<|im_end|>")
            .ok_or_else(|| anyhow::anyhow!("Missing <|im_end|> token"))?;
        let begin_of_text_id = get_token_id("<|begin_of_text|>")
            .ok_or_else(|| anyhow::anyhow!("Missing <|begin_of_text|> token"))?;
        let thinking_start_id = get_token_id("<think>")
            .ok_or_else(|| anyhow::anyhow!("Missing <think> token"))?;
        let thinking_end_id = get_token_id("</think>")
            .ok_or_else(|| anyhow::anyhow!("Missing </think> token"))?;
        // Official SmolLM3 uses token 128004 for padding
        let _pad_id = 128004;  // Official pad_token_id from config.json
        
        // Try to verify it exists, fall back to official ID if not found
        let pad_id = get_token_id("<|finetune_right_pad_id|>")
            .or_else(|| get_token_id("<|pad|>"))
            .unwrap_or(128004);  // Use official pad token ID 128004
        
        // Verify critical tokens match expected IDs
        tracing::info!("[TOKENIZER] Token ID verification:");
        tracing::info!("  <|im_start|>: {} (expected 128011)", im_start_id);
        tracing::info!("  <|im_end|>: {} (expected 128012)", im_end_id);
        tracing::info!("  <|begin_of_text|>: {} (expected 128000)", begin_of_text_id);
        tracing::info!("  <think>: {} (expected 128002)", thinking_start_id);
        tracing::info!("  </think>: {} (expected 128003)", thinking_end_id);
        tracing::info!("  pad: {} (expected 128004)", pad_id);
        
        // Verify the tokenizer recognizes these as single tokens
        if let Ok(test_ids) = tokenizer.encode("<|im_start|>", false) {
            if test_ids.get_ids().len() == 1 && test_ids.get_ids()[0] == im_start_id {
                tracing::info!("[TOKENIZER] ✅ <|im_start|> encodes to single token");
            } else {
                tracing::warn!("[TOKENIZER] ⚠️ <|im_start|> splits into {} tokens: {:?}", 
                    test_ids.get_ids().len(), test_ids.get_ids());
            }
        }
        
        let special_tokens = SpecialTokens {
            // Use begin_of_text as BOS (standard for Llama models)
            bos: begin_of_text_id,
            // Use im_end as EOS (as specified in tokenizer_config.json)
            eos: im_end_id,
            thinking_start: thinking_start_id,
            thinking_end: thinking_end_id,
            pad: pad_id,
            im_start: im_start_id,
            im_end: im_end_id,
            special_ids: special_ids.clone(),
            reserved_range: (128009, 128255),  // Reserved special tokens that shouldn't appear in output
        };
        
        // Log loaded special tokens for debugging
        tracing::info!("[TOKENIZER] Loaded special tokens:");
        tracing::info!("  BOS: {} ({})", special_tokens.bos, 
            special_token_map.iter().find(|(_, &id)| id == special_tokens.bos)
                .map(|(s, _)| s.as_str()).unwrap_or("unknown"));
        tracing::info!("  EOS: {} ({})", special_tokens.eos,
            special_token_map.iter().find(|(_, &id)| id == special_tokens.eos)
                .map(|(s, _)| s.as_str()).unwrap_or("unknown"));
        tracing::info!("  Thinking: {}-{}", special_tokens.thinking_start, special_tokens.thinking_end);
        tracing::info!("  Im_start/end: {}/{}", special_tokens.im_start, special_tokens.im_end);
        tracing::info!("  Special token count: {}", special_ids.len());
        
        // Configure padding
        let padding_side = config.padding_side.unwrap_or_else(|| "left".to_string());
        if let Some(padding) = tokenizer.get_padding_mut() {
            padding.pad_id = special_tokens.pad;
            padding.pad_token = config.pad_token.unwrap_or_else(|| "<|finetune_right_pad_id|>".to_string());
        }
        
        // Set up chat template with custom functions
        let mut chat_template = minijinja::Environment::new();
        
        // Register the strftime_now function
        chat_template.add_function("strftime_now", |format: String| {
            let now = Local::now();
            now.format(&format).to_string()
        });
        
        // Load the template
        let template = include_str!("../../../../templates/smollm3_official.j2");
        chat_template.add_template("chat", template)?;
        
        let clean_up_spaces = config.clean_up_tokenization_spaces.unwrap_or(true);
        
        // Diagnostic: Test common word encoding/decoding
        tracing::info!("[TOKENIZER] Testing vocabulary alignment:");
        let test_words = vec!["Hello", "hello", "Hi", "hi", "How", "are", "you", "I", "am", "good"];
        for word in test_words {
            if let Ok(encoded) = tokenizer.encode(word, false) {
                let token_ids = encoded.get_ids();
                if !token_ids.is_empty() {
                    let decoded = tokenizer.decode(token_ids, false).unwrap_or("<decode_error>".to_string());
                    tracing::info!("  '{}' -> {:?} -> '{}'", word, token_ids, decoded);
                }
            }
        }
        
        Ok(Self {
            tokenizer,
            special_tokens,
            padding_side,
            chat_template,
            special_token_map,
            clean_up_spaces,
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
    
    /// Get a specific special token ID dynamically
    pub fn get_special_token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }
    
    /// Get token ID from string (wrapper for tokenizer.token_to_id)
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encodings = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
        Ok(encodings.get_ids().to_vec())
    }
    
    /// Decode token IDs to text with proper special token handling
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        // Filter out reserved special tokens that shouldn't appear in output
        let filtered_ids: Vec<u32> = ids.iter()
            .copied()
            .filter(|&id| {
                // Skip reserved token range
                if id >= self.special_tokens.reserved_range.0 && id <= self.special_tokens.reserved_range.1 {
                    tracing::trace!("Filtering reserved token: {}", id);
                    return false;
                }
                // Skip special tokens that shouldn't appear in output
                if self.special_tokens.special_ids.contains(&id) {
                    // Allow thinking tokens if they're part of the response
                    if id == self.special_tokens.thinking_start || id == self.special_tokens.thinking_end {
                        return true;
                    }
                    tracing::trace!("Filtering special token: {}", id);
                    return false;
                }
                true
            })
            .collect();
        
        // Decode with skip_special_tokens=true
        let decoded = self.tokenizer
            .decode(&filtered_ids, true)  // skip_special_tokens = true
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
        
        // Clean up if configured
        if self.clean_up_spaces {
            Ok(decoded.trim().to_string())
        } else {
            Ok(decoded)
        }
    }
    
    /// Decode for streaming - single token at a time
    pub fn decode_single(&self, token_id: u32) -> Result<String> {
        // Check if this is a token that should be filtered
        if token_id >= self.special_tokens.reserved_range.0 && token_id <= self.special_tokens.reserved_range.1 {
            tracing::trace!("Skipping reserved token in stream: {}", token_id);
            return Ok(String::new());
        }
        
        // Skip certain special tokens
        if self.special_tokens.special_ids.contains(&token_id) {
            // Check if it's a control token we should skip
            if token_id == self.special_tokens.eos || 
               token_id == self.special_tokens.bos ||
               token_id == self.special_tokens.pad ||
               token_id == self.special_tokens.im_start ||
               token_id == self.special_tokens.im_end {
                tracing::trace!("Skipping special token in stream: {}", token_id);
                return Ok(String::new());
            }
        }
        
        // Decode single token
        self.decode(&[token_id])
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
        // Use ChatML format with proper special tokens
        // Note: thinking mode is inverted - false means enable thinking
        let mut template = String::new();
        
        // Add conversation history if provided
        if let Some(history) = conversation_history {
            for msg in history {
                template.push_str("<|im_start|>");
                template.push_str(&msg.role);
                template.push('\n');
                template.push_str(&msg.content);
                template.push_str("<|im_end|>\n");
            }
        }
        
        // Add user message
        template.push_str("<|im_start|>user\n");
        template.push_str(&input);
        template.push_str("<|im_end|>\n");
        
        // Start assistant response (no newline - model expects to start generating immediately)
        template.push_str("<|im_start|>assistant");
        
        // Add thinking prefix if enabled
        if thinking_enabled {
            template.push_str("<think>\n");
        }
        
        tracing::info!("[TOKENIZER] Using ChatML template format");
        tracing::info!("[TOKENIZER] Thinking mode: {}", thinking_enabled);
        tracing::info!("[TOKENIZER] Template output (first 500 chars):\n{}", 
            if template.len() > 500 { &template[..500] } else { &template });
        
        Ok(template)
        
        /* DISABLED COMPLEX TEMPLATE
        // Build messages
        let mut messages = conversation_history
            .map(|h| h.to_vec())
            .unwrap_or_default();
        
        // Add user message
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: input.clone(),
        });
        
        // Prepare context for template
        // Note: The template handles system message extraction internally
        // We should NOT pass system_message or custom_instructions as they might be None
        // which causes "cannot perform containment check" errors in the template
        let ctx = minijinja::context! {
            // Core message data
            messages => messages,
            add_generation_prompt => true,
            
            // Thinking configuration
            enable_thinking => thinking_enabled,
            
            // Tool configuration (not implemented yet)
            xml_tools => false,
            python_tools => false,
            tools => false,
        };
        
        // Render template
        let template = self.chat_template.get_template("chat")?;
        let rendered = template.render(ctx)?;
        
        // Log template output for debugging
        tracing::debug!("[TEMPLATE] Rendered {} chars for input: '{}'", 
            rendered.len(), 
            if input.len() > 50 { &input[..50] } else { &input }
        );
        
        Ok(rendered)
        */
    }
    
    fn encode_batch(&self, inputs: Vec<String>) -> Result<Vec<Vec<u32>>> {
        // Log the input strings
        for (i, input) in inputs.iter().enumerate() {
            tracing::info!("[TOKENIZER] encode_batch input[{}] (first 200 chars): {}", 
                i, 
                if input.len() > 200 { &input[..200] } else { input }
            );
        }
        
        // Always use batch encoding, even for single input
        // IMPORTANT: add_special_tokens=false since we add them in template!
        let encodings = self.tokenizer
            .encode_batch(inputs, false)  // false because template adds special tokens
            .map_err(|e| anyhow::anyhow!("Batch encoding failed: {}", e))?;
        
        // Extract token IDs and log with token strings
        let mut result = Vec::with_capacity(encodings.len());
        for (i, encoding) in encodings.into_iter().enumerate() {
            let token_ids = encoding.get_ids().to_vec();
            
            // Log first 10 tokens with their string representations
            let preview_count = token_ids.len().min(10);
            let mut token_strs = Vec::new();
            for &id in &token_ids[..preview_count] {
                let token_str = self.tokenizer.id_to_token(id)
                    .unwrap_or_else(|| format!("<unknown_{}>", id));
                token_strs.push(format!("{}:'{}'", id, token_str));
            }
            
            tracing::info!("[TOKENIZER] encode_batch output[{}]: {} tokens", i, token_ids.len());
            tracing::info!("[TOKENIZER]   First {} tokens: [{}]", preview_count, token_strs.join(", "));
            result.push(token_ids);
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

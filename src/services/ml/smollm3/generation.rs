//! SmolLM3 generation pipeline with streaming

use crate::services::ml::official::OfficialSmolLM3Model;
use crate::types::events::StreamEvent;
use crate::services::streaming::StreamingBuffer;
use candle_transformers::generation::LogitsProcessor;
use candle_core::{Tensor, Device, Result};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::UnboundedSender;
use super::thinking::ThinkingDetector;
use super::kv_cache::KVCache;
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Serialize, Deserialize};

/// Internal ML-specific generation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenerationEvent {
    Start,
    ThinkingStart,
    ThinkingToken(String),
    ThinkingEnd,
    ResponseToken(String),
    ToolCall { name: String, args: String },
    Complete,
    Error(String),
}

impl GenerationEvent {
    /// Convert to StreamEvent for general streaming
    pub fn to_stream_event(&self) -> StreamEvent {
        match self {
            GenerationEvent::Start => StreamEvent::Status("Generation started".to_string()),
            GenerationEvent::ThinkingStart => StreamEvent::Thinking("<thinking>".to_string()),
            GenerationEvent::ThinkingToken(token) => StreamEvent::Thinking(token.clone()),
            GenerationEvent::ThinkingEnd => StreamEvent::Thinking("</thinking>".to_string()),
            GenerationEvent::ResponseToken(token) => StreamEvent::Token(token.clone()),
            GenerationEvent::ToolCall { name, args } => StreamEvent::Content(format!("[Tool: {} Args: {}]", name, args)),
            GenerationEvent::Complete => StreamEvent::Complete,
            GenerationEvent::Error(msg) => StreamEvent::Error(msg.clone()),
        }
    }
}

pub struct SmolLM3Generator {
    model: Arc<Mutex<Arc<OfficialSmolLM3Model>>>,  // Arc<Mutex<Arc>> for async sharing without Clone
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    thinking_detector: ThinkingDetector,
    kv_cache: KVCache,
    device: Device,  // Store device for tensor creation
}

impl SmolLM3Generator {
    pub fn new(
        model: Arc<OfficialSmolLM3Model>,
        tokenizer: Tokenizer,
        temperature: Option<f64>,
        top_p: Option<f64>,
    ) -> Self {
        let config = model.config().clone();
        let logits_processor = LogitsProcessor::new(
            42,           // seed
            temperature,  // temperature
            top_p,        // top_p
        );
        
        let device = model.device().clone();
        let kv_cache = KVCache::new(
            &config,  // config
            &device  // device
        ).expect("Failed to create KV cache");
        
        // Wrap the Arc model in a Mutex for async safety
        // We'll share the Arc instead of cloning the model
        let model_mutex = Arc::new(Mutex::new(model));
        
        Self {
            thinking_detector: ThinkingDetector::new(config.thinking_tokens),
            model: model_mutex,
            tokenizer,
            logits_processor,
            kv_cache,
            device,
        }
    }
    
    /// Generate with streaming support using StreamingBuffer
    pub async fn generate_with_buffer(
        &mut self,
        prompt: &str,
        buffer: &mut StreamingBuffer,
        max_tokens: usize,
    ) -> anyhow::Result<String> {
        tracing::info!("[GENERATION] Starting generate_with_buffer");
        tracing::info!("[GENERATION] Prompt length: {} chars, max_tokens: {}", prompt.len(), max_tokens);
        
        // 1. Tokenize input
        tracing::info!("[GENERATION] Tokenizing input...");
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
        let input_ids = encoding.get_ids().to_vec();
        tracing::info!("[GENERATION] Tokenized {} tokens", input_ids.len());
        tracing::debug!("[GENERATION] First 10 tokens: {:?}", &input_ids[..input_ids.len().min(10)]);
        
        // 2. Generation loop with streaming
        let tokens = input_ids.clone();
        let accumulated_text = String::new();
        
        tracing::info!("[GENERATION] Entering generation loop for {} steps", max_tokens);
        for step in 0..max_tokens {
            tracing::debug!("[GENERATION] Step {}/{}", step + 1, max_tokens);
            // Create input tensor
            tracing::trace!("[GENERATION] Creating input tensor for step {}", step);
            let _input_tensor = self.create_input_tensor(&tokens, step)?;
            tracing::trace!("[GENERATION] Input tensor created with shape: {:?}", _input_tensor.dims());
            
            // Forward pass with mutex lock
            tracing::debug!("[GENERATION] Attempting forward pass at step {}", step);
            #[allow(unused_variables)]
            let logits = {
                tracing::trace!("[GENERATION] Acquiring model lock...");
                let _model_arc = self.model.lock().await;
                tracing::trace!("[GENERATION] Model lock acquired");
                // We need to call forward on the model inside the Arc
                // This is a limitation - we can't mutate through Arc
                // For now, return an error to indicate this needs refactoring
                tracing::error!("[GENERATION] Arc<Model> limitation hit - needs refactoring");
                return Err(anyhow::anyhow!("Model forward pass needs refactoring for Arc<Model>"));
            };
            
            // Sample next token
            #[allow(unreachable_code)]
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            
            // Decode token to text
            let token_text = self.tokenizer.decode(&[next_token], false)
                .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {}", e))?;
            
            // Handle thinking mode
            if let Some(event) = self.thinking_detector.process_token(next_token, &token_text) {
                match event {
                    super::thinking::ThinkingEvent::Start => {
                        // Don't buffer thinking tokens, they're hidden
                        continue;
                    }
                    super::thinking::ThinkingEvent::Content(_) => {
                        continue; // Skip buffering thinking content
                    }
                    super::thinking::ThinkingEvent::End => {
                        continue; // Skip
                    }
                }
            }
            
            // Buffer token if not in thinking mode
            if !self.thinking_detector.is_thinking() {
                accumulated_text.push_str(&token_text);
                buffer.push(&token_text).await
                    .map_err(|e| anyhow::anyhow!("Buffer push error: {}", e))?;
            }
            
            // Check stop conditions
            if self.is_stop_token(next_token) {
                break;
            }
        }
        
        buffer.complete().await
            .map_err(|e| anyhow::anyhow!("Buffer complete error: {}", e))?;
        Ok(accumulated_text)
    }
    
    /// Generate with streaming support (legacy, uses direct sender)
    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        sender: UnboundedSender<StreamEvent>,
        max_tokens: usize,
    ) -> anyhow::Result<String> {
        tracing::info!("[GENERATION] Starting generate_stream (legacy)");
        tracing::info!("[GENERATION] Prompt length: {} chars, max_tokens: {}", prompt.len(), max_tokens);
        
        // 1. Tokenize input
        tracing::info!("[GENERATION] Tokenizing input...");
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
        let input_ids = encoding.get_ids().to_vec();
        tracing::info!("[GENERATION] Tokenized {} tokens", input_ids.len());
        
        // 2. Generation loop with streaming
        let tokens = input_ids.clone();
        let accumulated_text = String::new();
        
        for step in 0..max_tokens {
            // Create input tensor
            let _input_tensor = self.create_input_tensor(&tokens, step)?;
            
            // Forward pass with mutex lock
            #[allow(unused_variables)]
            let logits = {
                let _model_arc = self.model.lock().await;
                // We need to call forward on the model inside the Arc
                // This is a limitation - we can't mutate through Arc
                // For now, return an error to indicate this needs refactoring
                return Err(anyhow::anyhow!("Model forward pass needs refactoring for Arc<Model>"));
            };
            
            // Sample next token
            #[allow(unreachable_code)]
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            
            // Decode token to text
            let token_text = self.tokenizer.decode(&[next_token], false)
                .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {}", e))?;
            
            // Handle thinking mode
            if let Some(event) = self.thinking_detector.process_token(next_token, &token_text) {
                match event {
                    super::thinking::ThinkingEvent::Start => {
                        let _ = sender.send(StreamEvent::thinking("Starting to think...".to_string()));
                    }
                    super::thinking::ThinkingEvent::Content(content) => {
                        let _ = sender.send(StreamEvent::thinking(content));
                    }
                    super::thinking::ThinkingEvent::End => {
                        let _ = sender.send(StreamEvent::token("".to_string())); // Signal end of thinking
                    }
                }
                
                if !self.thinking_detector.is_thinking() {
                    continue; // Skip thinking tokens from output
                }
            }
            
            // Stream token if not in thinking mode
            if !self.thinking_detector.is_thinking() {
                accumulated_text.push_str(&token_text);
                let _ = sender.send(StreamEvent::token(token_text));
            }
            
            // Check stop conditions
            if self.is_stop_token(next_token) {
                break;
            }
        }
        
        let _ = sender.send(StreamEvent::complete());
        Ok(accumulated_text)
    }
    
    fn create_input_tensor(&self, tokens: &[u32], step: usize) -> Result<Tensor> {
        // Use stored device reference
        let device = &self.device;
        
        if step == 0 {
            // Prefill: entire sequence
            Tensor::new(tokens, device)?.unsqueeze(0)
        } else {
            // Generation: only last token
            Tensor::new(&[tokens[tokens.len()-1]], device)?.unsqueeze(0)
        }
    }
    
    fn is_stop_token(&self, token: u32) -> bool {
        // Common stop tokens
        token == 2 || // EOS
        token == 128001 || // SmolLM3 EOS
        token == 128009    // SmolLM3 EOT
    }
}

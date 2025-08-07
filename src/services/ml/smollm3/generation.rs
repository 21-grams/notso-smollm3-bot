//! SmolLM3 generation pipeline with streaming

use crate::services::ml::official::OfficialSmolLM3Model;
use crate::types::events::StreamEvent;
use candle_transformers::generation::LogitsProcessor;
use candle_core::{Tensor, Device, Result};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::UnboundedSender;
use super::thinking::ThinkingDetector;
use super::kv_cache::KVCache;

pub struct SmolLM3Generator {
    model: OfficialSmolLM3Model,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    thinking_detector: ThinkingDetector,
    kv_cache: KVCache,
}

impl SmolLM3Generator {
    pub fn new(
        model: OfficialSmolLM3Model,
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
        let kv_cache = KVCache::new(2048, device);  // Max sequence length for SmolLM3
        
        Self {
            thinking_detector: ThinkingDetector::new(config.thinking_tokens),
            model,
            tokenizer,
            logits_processor,
            kv_cache,
        }
    }
    
    /// Generate with streaming support
    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        sender: UnboundedSender<StreamEvent>,
        max_tokens: usize,
    ) -> anyhow::Result<String> {
        // 1. Tokenize input
        let encoding = self.tokenizer.encode(prompt, false)?;
        let input_ids = encoding.get_ids().to_vec();
        
        // 2. Generation loop with streaming
        let mut tokens = input_ids.clone();
        let mut accumulated_text = String::new();
        
        for step in 0..max_tokens {
            // Create input tensor
            let input_tensor = self.create_input_tensor(&tokens, step)?;
            
            // Forward pass
            let logits = self.model.forward(&input_tensor, step)?;
            
            // Sample next token
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            
            // Decode token to text
            let token_text = self.tokenizer.decode(&[next_token], false)?;
            
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
        let device = self.model.device();
        
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

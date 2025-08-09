//! ML Service that orchestrates SmolLM3 model with proper Llama integration

use candle_core::{Device, Result, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::path::Path;

use super::smollm3::{SmolLM3Tokenizer, SmolLM3KVCache};
use super::official::model::SmolLM3Model;

/// Main ML service for SmolLM3 inference
pub struct MLService {
    /// The SmolLM3 model wrapping quantized Llama
    model: SmolLM3Model,
    /// Tokenizer with thinking mode support
    tokenizer: SmolLM3Tokenizer,
    /// KV cache for efficient generation
    kv_cache: SmolLM3KVCache,
    /// Device for tensor operations
    device: Device,
    /// Logits processor for sampling
    logits_processor: LogitsProcessor,
}

impl MLService {
    /// Create a new ML service
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        tokenizer_path: P,
        template_path: P,
        device: Device,
    ) -> Result<Self> {
        tracing::info!("🚀 Initializing SmolLM3 ML Service");
        
        // Load model using our SmolLM3 wrapper
        let model = SmolLM3Model::from_gguf(model_path, &device)?;
        let config = model.config().clone();
        
        // Initialize tokenizer
        let tokenizer = SmolLM3Tokenizer::from_file(
            tokenizer_path.as_ref().to_str().unwrap()
        ).map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {}", e)))?;
        
        // Initialize KV cache for all layers
        let kv_cache = SmolLM3KVCache::new(
            config.base.num_hidden_layers,
            65536, // Max context length
            device.clone(),
        );
        
        // Setup logits processor for sampling
        let logits_processor = LogitsProcessor::new(
            42,           // seed
            Some(0.9),    // temperature
            Some(0.95),   // top_p
        );
        
        tracing::info!("✅ ML Service initialized successfully");
        
        Ok(Self {
            model,
            tokenizer,
            kv_cache,
            device,
            logits_processor,
        })
    }
    
    /// Generate text from a prompt
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        enable_thinking: bool,
    ) -> Result<Vec<String>> {
        tracing::info!("🔄 Starting generation with max_tokens={}", max_tokens);
        
        // Encode the prompt
        let tokens = self.tokenizer.encode(prompt)
            .map_err(|e| candle_core::Error::Msg(format!("Encoding error: {}", e)))?;
        
        let mut all_tokens = tokens.clone();
        let mut output_tokens = Vec::new();
        let mut in_thinking_mode = false;
        
        // Convert to tensor
        let mut input_ids = Tensor::new(tokens.as_slice(), &self.device)?
            .unsqueeze(0)?; // Add batch dimension
        
        // Generation loop
        for _step in 0..max_tokens {
            // Forward pass through model
            let logits = self.forward_with_cache(&input_ids)?;
            
            // Get the last token's logits
            let last_logits = logits.i((0, logits.dim(1)? - 1, ..))?;
            
            // Sample next token
            let next_token = self.logits_processor.sample(&last_logits)?;
            
            // Get config for special tokens
            let config = self.model.config();
            
            // Handle special tokens
            if next_token == config.think_token_id {
                in_thinking_mode = true;
                if enable_thinking {
                    output_tokens.push("<thinking>".to_string());
                }
                all_tokens.push(next_token);
                continue;
            } else if next_token == config.think_end_token_id {
                in_thinking_mode = false;
                if enable_thinking {
                    output_tokens.push("</thinking>".to_string());
                }
                all_tokens.push(next_token);
                continue;
            }
            
            // Check for EOS token
            if next_token == 128001 { // EOS token
                tracing::info!("🛑 EOS token detected, stopping generation");
                break;
            }
            
            // Decode and add to output
            let token_text = self.tokenizer.decode(&[next_token])
                .map_err(|e| candle_core::Error::Msg(format!("Decoding error: {}", e)))?;
            
            if !in_thinking_mode || enable_thinking {
                output_tokens.push(token_text);
            }
            
            all_tokens.push(next_token);
            
            // Update input for next iteration
            input_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
        }
        
        // Reset cache for next generation
        self.kv_cache.reset();
        
        tracing::info!("✅ Generated {} tokens", output_tokens.len());
        Ok(output_tokens)
    }
    
    /// Forward pass with KV caching
    fn forward_with_cache(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        // For now, use the model's forward method
        // In a full implementation, this would handle:
        // 1. KV cache management per layer
        // 2. NoPE layer special handling
        // 3. GQA attention with proper head expansion
        
        self.model.forward(input_ids, None, Some(&mut self.kv_cache))
    }
    
    /// Apply chat template to messages
    pub fn apply_chat_template(
        &self,
        messages: Vec<serde_json::Value>,
        enable_thinking: bool,
    ) -> std::result::Result<String, Box<dyn std::error::Error>> {
        self.tokenizer.apply_chat_template(messages, enable_thinking)
    }
    
    /// Get tokenizer reference
    pub fn tokenizer(&self) -> &SmolLM3Tokenizer {
        &self.tokenizer
    }
}

/// Builder pattern for MLService configuration
pub struct MLServiceBuilder {
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    template_path: Option<String>,
    device: Option<Device>,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
}

impl Default for MLServiceBuilder {
    fn default() -> Self {
        Self {
            model_path: None,
            tokenizer_path: None,
            template_path: None,
            device: None,
            temperature: 0.9,
            top_p: Some(0.95),
            seed: 42,
        }
    }
}

impl MLServiceBuilder {
    /// Set model path
    pub fn model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.model_path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }
    
    /// Set tokenizer path
    pub fn tokenizer_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.tokenizer_path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }
    
    /// Set template path
    pub fn template_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.template_path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }
    
    /// Set device
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }
    
    /// Set temperature
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }
    
    /// Set top_p
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }
    
    /// Set seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    
    /// Build the MLService
    pub fn build(self) -> Result<MLService> {
        let model_path = self.model_path
            .ok_or_else(|| candle_core::Error::Msg("Model path required".to_string()))?;
        let tokenizer_path = self.tokenizer_path
            .ok_or_else(|| candle_core::Error::Msg("Tokenizer path required".to_string()))?;
        let template_path = self.template_path
            .ok_or_else(|| candle_core::Error::Msg("Template path required".to_string()))?;
        let device = self.device
            .unwrap_or_else(|| Device::cuda_if_available(0).unwrap_or(Device::Cpu));
        
        MLService::new(model_path, tokenizer_path, template_path, device)
    }
}

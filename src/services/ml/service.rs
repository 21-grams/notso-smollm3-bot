//! ML Service that orchestrates SmolLM3 model with proper Llama integration

use candle_core::{Device, Result, Tensor, IndexOp};
use candle_transformers::generation::LogitsProcessor;
use std::path::Path;

use super::smollm3::{SmolLM3Tokenizer, SmolLM3KVCache};
use super::official::model::SmolLM3Model;
use super::smollm3::nope_model::NopeModel;

/// Model backend selection
enum ModelBackend {
    /// Standard ModelWeights (all layers use RoPE)
    Standard(SmolLM3Model),
    /// NoPE-aware model (selective RoPE)
    Nope(NopeModel),
}

/// Main ML service for SmolLM3 inference
pub struct MLService {
    /// The model backend
    model: ModelBackend,
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
    
    /// Generate with streaming support
    pub async fn generate_streaming(
        &self,
        _prompt: &str,
        _buffer: &mut crate::services::StreamingBuffer,
    ) -> anyhow::Result<()> {
        // TODO: Implement actual model inference here
        // For now, return an error to indicate model is not ready
        Err(anyhow::anyhow!("Model inference not yet implemented"))
    }
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        tokenizer_path: P,
        _template_path: P,
        device: Device,
    ) -> Result<Self> {
        Self::new_with_backend(model_path, tokenizer_path, _template_path, device, true)
    }
    
    pub fn new_with_backend<P: AsRef<Path>>(
        model_path: P,
        tokenizer_path: P,
        _template_path: P,
        device: Device,
        use_nope: bool,
    ) -> Result<Self> {
        tracing::info!("🚀 Initializing SmolLM3 ML Service");
        tracing::info!("  Backend: {}", if use_nope { "NoPE-aware" } else { "Standard" });
        
        // Load model based on backend selection
        let (model, config) = if use_nope {
            // Use NoPE-aware model
            let nope_model = NopeModel::from_gguf(model_path, &device)?;
            let config = nope_model.config().clone();
            tracing::info!("✅ NoPE model loaded with layers: {:?}", config.nope_layer_indices);
            (ModelBackend::Nope(nope_model), config)
        } else {
            // Use standard ModelWeights
            let std_model = SmolLM3Model::from_gguf(model_path, &device)?;
            let config = std_model.config().clone();
            tracing::info!("✅ Standard model loaded (RoPE on all layers)");
            (ModelBackend::Standard(std_model), config)
        };
        
        // Initialize tokenizer
        let tokenizer = SmolLM3Tokenizer::from_file(
            tokenizer_path.as_ref().to_str().unwrap()
        ).map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {}", e)))?;
        
        // Initialize KV cache for all layers
        let kv_cache = SmolLM3KVCache::new(
            config.base.num_hidden_layers,
            config.base.max_position_embeddings, // Use actual max context
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
        
        let prompt_len = tokens.len();
        let mut all_tokens = tokens.clone();
        let mut output_tokens = Vec::new();
        let mut in_thinking_mode = false;
        let mut position = 0; // Start at position 0
        
        tracing::debug!("Prompt has {} tokens", prompt_len);
        
        // Convert to tensor
        let input_ids = Tensor::new(tokens.as_slice(), &self.device)?
            .unsqueeze(0)?; // Add batch dimension
        
        // Process the prompt through the model
        tracing::debug!("📥 Processing prompt at position 0");
        let logits = match &mut self.model {
            ModelBackend::Standard(m) => m.forward(&input_ids, position, Some(&mut self.kv_cache))?,
            ModelBackend::Nope(m) => m.forward(&input_ids, position)?,
        };
        
        // After prompt, position jumps to prompt_len
        position = prompt_len;
        tracing::debug!("📍 Position after prompt: {}", position);
        
        // Get the last token's logits from prompt
        let mut last_logits = logits.i((0, logits.dim(1)? - 1, ..))?;
        
        // Generation loop
        for step in 0..max_tokens {
            // Sample next token
            let next_token = self.logits_processor.sample(&last_logits)?;
            
            // Get config for special tokens
            let config = match &self.model {
                ModelBackend::Standard(m) => m.config(),
                ModelBackend::Nope(m) => m.config(),
            };
            
            // Handle special tokens
            if next_token == config.think_token_id {
                in_thinking_mode = true;
                if enable_thinking {
                    output_tokens.push("<thinking>".to_string());
                }
                all_tokens.push(next_token);
            } else if next_token == config.think_end_token_id {
                in_thinking_mode = false;
                if enable_thinking {
                    output_tokens.push("</thinking>".to_string());
                }
                all_tokens.push(next_token);
            } else if next_token == 128001 { // EOS token
                tracing::info!("🛑 EOS token detected at step {}, stopping generation", step);
                break;
            } else {
                // Decode and add to output
                let token_text = self.tokenizer.decode(&[next_token])
                    .map_err(|e| candle_core::Error::Msg(format!("Decoding error: {}", e)))?;
                
                if !in_thinking_mode || enable_thinking {
                    output_tokens.push(token_text);
                }
                
                all_tokens.push(next_token);
            }
            
            // Forward pass for next token
            let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            
            tracing::trace!("Generating token {} at position {}", step + 1, position);
            let logits = match &mut self.model {
                ModelBackend::Standard(m) => m.forward(&next_input, position, Some(&mut self.kv_cache))?,
                ModelBackend::Nope(m) => m.forward(&next_input, position)?,
            };
            
            // Increment position for next token
            position += 1;
            
            // Get logits for sampling
            last_logits = logits.i((0, 0, ..))?; // Shape is [1, 1, vocab_size] for single token
        }
        
        // Reset cache for next generation
        self.kv_cache.reset();
        if let ModelBackend::Nope(m) = &mut self.model {
            m.reset_cache();
        }
        
        tracing::info!("✅ Generated {} tokens", output_tokens.len());
        Ok(output_tokens)
    }

    
    /// Apply chat template to messages
    pub fn apply_chat_template(
        &self,
        messages: Vec<serde_json::Value>,
        enable_thinking: bool,
    ) -> std::result::Result<String, Box<dyn std::error::Error>> {
        // Convert JSON messages to ChatMessage format
        let chat_messages: Vec<super::smollm3::ChatMessage> = messages
            .into_iter()
            .map(|msg| {
                let role = msg["role"].as_str().unwrap_or("user").to_string();
                let content = msg["content"].as_str().unwrap_or("").to_string();
                super::smollm3::ChatMessage { role, content }
            })
            .collect();
        
        let reasoning_mode = if enable_thinking {
            super::smollm3::ReasoningMode::Think
        } else {
            super::smollm3::ReasoningMode::NoThink
        };
        
        self.tokenizer.apply_chat_template(&chat_messages, true, reasoning_mode)
            .map_err(|e| e.into())
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

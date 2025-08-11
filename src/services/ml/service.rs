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
        &mut self,
        prompt: &str,
        buffer: &mut crate::services::StreamingBuffer,
    ) -> anyhow::Result<()> {
        tracing::info!("🔄 Starting streaming generation");
        
        // 1. Tokenize the prompt
        let tokens = self.tokenizer.encode(prompt)
            .map_err(|e| anyhow::anyhow!("Tokenizer encoding error: {}", e))?;
        
        if tokens.is_empty() {
            return Err(anyhow::anyhow!("Empty prompt after tokenization"));
        }
        
        let prompt_len = tokens.len();
        tracing::debug!("📝 Prompt tokenized: {} tokens", prompt_len);
        
        // 2. Initialize generation state
        let mut all_tokens = tokens.clone();
        let mut position = 0;
        let max_tokens = 512; // Default max generation length
        let mut in_thinking_mode = false;
        
        // Get special token IDs from config
        let (eos_token, think_token, think_end_token) = match &self.model {
            ModelBackend::Standard(m) => {
                let cfg = m.config();
                (128001u32, cfg.think_token_id, cfg.think_end_token_id)
            },
            ModelBackend::Nope(m) => {
                let cfg = m.config();
                (128001u32, cfg.think_token_id, cfg.think_end_token_id)
            }
        };
        
        // 3. Process the prompt (prefill phase)
        // Create input tensor - keep it 1D for embedding lookup
        let input_tensor = Tensor::new(tokens.as_slice(), &self.device)?;
        
        tracing::debug!("📥 Running prefill for {} tokens", prompt_len);
        let prompt_logits = match &mut self.model {
            ModelBackend::Standard(m) => {
                m.forward(&input_tensor, position, Some(&mut self.kv_cache))?
            },
            ModelBackend::Nope(m) => {
                m.forward(&input_tensor, position)?
            }
        };
        
        // Get last token's logits from the prompt
        // Handle different tensor shapes - logits should be [batch, seq_len, vocab_size]
        let mut last_logits = if prompt_logits.dims().len() == 3 {
            // [batch, seq_len, vocab_size] - take last position
            prompt_logits.i((0, prompt_logits.dim(1)? - 1, ..))?  
        } else if prompt_logits.dims().len() == 2 {
            // [seq_len, vocab_size] - take last position
            prompt_logits.i((prompt_logits.dim(0)? - 1, ..))?  
        } else {
            return Err(anyhow::anyhow!("Unexpected logits shape: {:?}", prompt_logits.dims()));
        };
        
        // Update position to end of prompt
        position = prompt_len;
        tracing::debug!("📍 Position after prefill: {}", position);
        
        // 4. Generation loop
        let mut generated_count = 0;
        let mut consecutive_newlines = 0;
        
        for step in 0..max_tokens {
            // 4a. Sample next token from logits
            let next_token = self.logits_processor.sample(&last_logits)?;
            all_tokens.push(next_token);
            
            // 4b. Check for stop conditions
            if next_token == eos_token {
                tracing::info!("🛑 EOS token detected at step {}", step);
                break;
            }
            
            // 4c. Handle thinking mode tokens
            if next_token == think_token {
                in_thinking_mode = true;
                tracing::debug!("🤔 Entering thinking mode");
                // Don't stream thinking start token
            } else if next_token == think_end_token {
                in_thinking_mode = false;
                tracing::debug!("💭 Exiting thinking mode");
                // Don't stream thinking end token
            } else if !in_thinking_mode {
                // 4d. Decode and stream the token (only if not in thinking mode)
                let token_text = self.tokenizer.decode(&[next_token])
                    .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))?;
                
                // Check for repeated newlines (potential endless generation)
                if token_text == "\n" {
                    consecutive_newlines += 1;
                    if consecutive_newlines > 3 {
                        tracing::warn!("Too many consecutive newlines, stopping");
                        break;
                    }
                } else {
                    consecutive_newlines = 0;
                }
                
                // Stream the token to the buffer
                buffer.push(&token_text).await
                    .map_err(|e| anyhow::anyhow!("Buffer push error: {}", e))?;
                
                generated_count += 1;
                
                // Yield occasionally to prevent blocking
                if generated_count % 10 == 0 {
                    tokio::task::yield_now().await;
                }
            }
            
            // 4e. Prepare next token input and run forward pass
            // Keep tensor 1D for embedding lookup
            let next_input = Tensor::new(&[next_token], &self.device)?;
            
            tracing::trace!("🔮 Generating token {} at position {}", step + 1, position);
            
            let logits = match &mut self.model {
                ModelBackend::Standard(m) => {
                    m.forward(&next_input, position, Some(&mut self.kv_cache))?
                },
                ModelBackend::Nope(m) => {
                    m.forward(&next_input, position)?
                }
            };
            
            // Update for next iteration
            last_logits = logits.i((0, 0, ..))?; // Shape [1, 1, vocab_size] -> [vocab_size]
            position += 1;
            
            // Safety check for position overflow
            if position >= 65536 { // Max context length
                tracing::warn!("Reached maximum context length");
                break;
            }
        }
        
        // 5. Complete the stream
        buffer.complete().await
            .map_err(|e| anyhow::anyhow!("Buffer completion error: {}", e))?;
        
        // 6. Reset caches for next generation
        self.kv_cache.reset();
        if let ModelBackend::Nope(m) = &mut self.model {
            m.reset_cache();
        }
        
        tracing::info!("✅ Generated {} visible tokens", generated_count);
        Ok(())
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
    
    /// Generate text from a prompt (non-streaming version)
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
        
        // Convert to tensor - keep 1D for NoPE model
        let input_ids = Tensor::new(tokens.as_slice(), &self.device)?;
        
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
            
            // Forward pass for next token - keep 1D for NoPE model
            let next_input = Tensor::new(&[next_token], &self.device)?;
            
            tracing::trace!("Generating token {} at position {}", step + 1, position);
            let logits = match &mut self.model {
                ModelBackend::Standard(m) => m.forward(&next_input, position, Some(&mut self.kv_cache))?,
                ModelBackend::Nope(m) => m.forward(&next_input, position)?,
            };
            
            // Increment position for next token
            position += 1;
            
            // Get logits for sampling
            // For single token, logits should be [1, 1, vocab_size] or [1, vocab_size]
            last_logits = if logits.dims().len() == 3 {
                logits.i((0, 0, ..))?  // [batch, 1, vocab_size] -> [vocab_size]
            } else if logits.dims().len() == 2 {
                logits.i((0, ..))?  // [1, vocab_size] -> [vocab_size]
            } else {
                return Err(anyhow::anyhow!("Unexpected logits shape: {:?}", logits.dims()));
            };
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
    use_nope: bool,
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
            use_nope: true, // Default to NoPE backend
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
    
    /// Set whether to use NoPE backend
    pub fn use_nope(mut self, use_nope: bool) -> Self {
        self.use_nope = use_nope;
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
        
        // Create service with specified backend
        let mut service = MLService::new_with_backend(
            model_path, 
            tokenizer_path, 
            template_path, 
            device,
            self.use_nope
        )?;
        
        // Update logits processor with builder settings
        service.logits_processor = LogitsProcessor::new(
            self.seed,
            Some(self.temperature),
            self.top_p,
        );
        
        Ok(service)
    }
}

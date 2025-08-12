//! ML Service that orchestrates SmolLM3 model with batch tokenization

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::path::Path;

use super::smollm3::{SmolLM3Tokenizer, KVCache};
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
    kv_cache: KVCache,
    /// Device for tensor operations
    device: Device,
    /// Logits processor for sampling
    logits_processor: LogitsProcessor,
}

impl MLService {
    /// Generate with streaming support using new tokenizer pipeline
    pub async fn generate_streaming(
        &mut self,
        user_input: &str,
        buffer: &mut crate::services::StreamingBuffer,
        thinking_enabled: bool,
    ) -> anyhow::Result<()> {
        let start_time = std::time::Instant::now();
        tracing::info!("[SERVICE] ===== Starting streaming generation =====");
        tracing::info!("[SERVICE] Input length: {} chars, thinking: {}", user_input.len(), thinking_enabled);
        
        // 1. Process input through tokenizer pipeline
        tracing::info!("[SERVICE] Processing input through tokenizer...");
        let token_batch = self.tokenizer
            .process_input(user_input.to_string(), thinking_enabled)
            .map_err(|e| anyhow::anyhow!("Tokenizer processing error: {}", e))?;
        
        // Extract tokens from batch (we know batch_size = 1)
        let tokens = token_batch.into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty token batch"))?;
        
        if tokens.is_empty() {
            return Err(anyhow::anyhow!("Empty prompt after tokenization"));
        }
        
        let prompt_len = tokens.len();
        tracing::info!("[SERVICE] Tokenization complete: {} tokens", prompt_len);
        tracing::debug!("[SERVICE] First 10 tokens: {:?}", &tokens[..tokens.len().min(10)]);
        
        // 2. Initialize generation state
        let mut all_tokens = tokens.clone();
        let mut position = 0;
        let max_tokens = 512;
        let mut in_thinking_mode = false;
        
        // Get special token IDs
        let special_tokens = self.tokenizer.special_tokens();
        let eos_token = special_tokens.eos;
        let think_token = special_tokens.thinking_start;
        let think_end_token = special_tokens.thinking_end;
        
        // 3. Process the prompt (prefill phase)
        // Create batch tensor [1, seq_len] for consistency
        tracing::info!("[SERVICE] Creating input tensor from {} tokens", tokens.len());
        let input_tensor = Tensor::new(&tokens[..], &self.device)?
            .unsqueeze(0)?;  // Add batch dimension
        
        tracing::info!("[SERVICE] Running prefill phase with tensor shape: {:?}", input_tensor.dims());
        tracing::info!("[SERVICE] Device: {:?}, Position: {}", self.device, position);
        
        let prefill_start = std::time::Instant::now();
        let prompt_logits = match &mut self.model {
            ModelBackend::Standard(m) => {
                tracing::info!("[SERVICE] Using Standard backend for forward pass");
                m.forward(&input_tensor, position, Some(&mut self.kv_cache))?
            },
            ModelBackend::Nope(m) => {
                tracing::info!("[SERVICE] Using NoPE backend for forward pass");
                m.forward(&input_tensor, position)?
            }
        };
        tracing::info!("[SERVICE] Prefill forward pass completed in {:?}", prefill_start.elapsed());
        tracing::info!("[SERVICE] Prompt logits shape: {:?}", prompt_logits.dims());
        
        // Get last token's logits
        // Expected shape: [batch, seq_len, vocab_size]
        tracing::debug!("[SERVICE] Extracting last token logits from position {}", prompt_logits.dim(1)? - 1);
        let mut last_logits = prompt_logits.i((0, prompt_logits.dim(1)? - 1, ..))?;
        tracing::debug!("[SERVICE] Last logits shape: {:?}", last_logits.dims());
        
        // Update position to end of prompt
        position = prompt_len;
        tracing::info!("[SERVICE] Position after prefill: {}", position);
        
        // 4. Generation loop
        let mut generated_count = 0;
        let mut consecutive_newlines = 0;
        
        tracing::info!("[SERVICE] Starting generation loop for max {} tokens", max_tokens);
        
        for step in 0..max_tokens {
            if step == 0 || step % 10 == 0 {
                tracing::info!("[SERVICE] Generation step {}/{}", step, max_tokens);
            }
            
            // 4a. Sample next token from logits
            tracing::trace!("[SERVICE] Sampling from logits with shape: {:?}", last_logits.dims());
            let next_token = self.logits_processor.sample(&last_logits)?;
            tracing::trace!("[SERVICE] Sampled token: {}", next_token);
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
            } else if next_token == think_end_token {
                in_thinking_mode = false;
                tracing::debug!("💭 Exiting thinking mode");
            } else if !in_thinking_mode {
                // 4d. Decode and stream the token
                let token_text = self.tokenizer.decode(&[next_token])
                    .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))?;
                
                // Check for repeated newlines
                if token_text == "\n" {
                    consecutive_newlines += 1;
                    if consecutive_newlines > 3 {
                        tracing::warn!("Too many consecutive newlines, stopping");
                        break;
                    }
                } else {
                    consecutive_newlines = 0;
                }
                
                // Stream the token
                buffer.push(&token_text).await
                    .map_err(|e| anyhow::anyhow!("Buffer push error: {}", e))?;
                
                generated_count += 1;
                
                if generated_count % 10 == 0 {
                    tokio::task::yield_now().await;
                }
            }
            
            // 4e. Prepare next token input
            // Keep batch dimension [1, 1]
            tracing::trace!("[SERVICE] Creating next input tensor for token {}", next_token);
            let next_input = Tensor::new(&[next_token], &self.device)?
                .unsqueeze(0)?;  // Add batch dimension
            
            tracing::debug!("[SERVICE] Running forward pass for step {} at position {}", step + 1, position);
            
            let step_start = std::time::Instant::now();
            let logits = match &mut self.model {
                ModelBackend::Standard(m) => {
                    tracing::trace!("[SERVICE] Using Standard backend");
                    m.forward(&next_input, position, Some(&mut self.kv_cache))?
                },
                ModelBackend::Nope(m) => {
                    tracing::trace!("[SERVICE] Using NoPE backend");
                    m.forward(&next_input, position)?
                }
            };
            tracing::trace!("[SERVICE] Forward pass for step {} completed in {:?}", step, step_start.elapsed());
            
            // Update for next iteration
            tracing::trace!("[SERVICE] Extracting logits for next iteration");
            last_logits = logits.i((0, 0, ..))?;
            position += 1;
            tracing::trace!("[SERVICE] Updated position to {}", position);
            
            if position >= 65536 {
                tracing::warn!("Reached maximum context length");
                break;
            }
        }
        
        // 5. Complete the stream
        tracing::info!("[SERVICE] Completing stream...");
        buffer.complete().await
            .map_err(|e| anyhow::anyhow!("Buffer completion error: {}", e))?;
        
        // 6. Reset caches for next generation
        tracing::debug!("[SERVICE] Resetting caches for next generation");
        self.kv_cache.reset();
        if let ModelBackend::Nope(m) = &mut self.model {
            m.reset_cache();
        }
        
        let total_time = start_time.elapsed();
        tracing::info!("[SERVICE] ✅ Generation complete!");
        tracing::info!("[SERVICE]   - Generated {} visible tokens", generated_count);
        tracing::info!("[SERVICE]   - Total time: {:?}", total_time);
        tracing::info!("[SERVICE]   - Tokens/sec: {:.2}", generated_count as f64 / total_time.as_secs_f64());
        Ok(())
    }
    
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        tokenizer_dir: P,
        device: Device,
    ) -> Result<Self> {
        Self::new_with_backend(model_path, tokenizer_dir, device, true)
    }
    
    pub fn new_with_backend<P: AsRef<Path>>(
        model_path: P,
        tokenizer_dir: P,
        device: Device,
        use_nope: bool,
    ) -> Result<Self> {
        tracing::info!("[ML_SERVICE] Initializing ML service");
        tracing::info!("[ML_SERVICE]   Model path: {:?}", model_path.as_ref());
        tracing::info!("[ML_SERVICE]   Tokenizer dir: {:?}", tokenizer_dir.as_ref());
        tracing::info!("[ML_SERVICE]   Device: {:?}", device);
        tracing::info!("[ML_SERVICE]   Use NoPE: {}", use_nope);
        
        // Load tokenizer from directory containing all config files
        tracing::info!("[ML_SERVICE] Loading tokenizer...");
        let tokenizer = SmolLM3Tokenizer::from_files(tokenizer_dir.as_ref())
            .map_err(|e| {
                tracing::error!("[ML_SERVICE] Tokenizer load failed: {}", e);
                candle_core::Error::Msg(format!("Tokenizer load failed: {}", e))
            })?;
        tracing::info!("[ML_SERVICE] Tokenizer loaded successfully");
        
        // Load model
        tracing::info!("[ML_SERVICE] Loading model...");
        let model = if use_nope {
            tracing::info!("[ML_SERVICE] Loading NoPE-aware model from GGUF");
            ModelBackend::Nope(NopeModel::from_gguf(&model_path, &device)?)
        } else {
            tracing::info!("[ML_SERVICE] Loading standard model from GGUF");
            ModelBackend::Standard(SmolLM3Model::from_gguf(&model_path, &device)?)
        };
        tracing::info!("[ML_SERVICE] Model loaded successfully");
        
        // Initialize KV cache
        // Get config for cache dimensions
        let (num_layers, max_length) = match &model {
            ModelBackend::Standard(m) => {
                let cfg = m.config();
                (cfg.base.num_hidden_layers, cfg.base.max_position_embeddings)
            },
            ModelBackend::Nope(m) => {
                let cfg = m.config();
                (cfg.base.num_hidden_layers, cfg.base.max_position_embeddings)
            }
        };
        let kv_cache = KVCache::new(num_layers, max_length, device.clone());
        
        // Create logits processor
        let logits_processor = LogitsProcessor::new(
            42,        // seed
            Some(0.8), // temperature
            Some(0.9), // top_p
        );
        
        Ok(Self {
            model,
            tokenizer,
            kv_cache,
            device,
            logits_processor,
        })
    }
}

/// Builder for MLService with fluent configuration
pub struct MLServiceBuilder {
    model_path: Option<String>,
    tokenizer_dir: Option<String>,
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
            tokenizer_dir: None,
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
    
    /// Set tokenizer directory path
    pub fn tokenizer_dir<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.tokenizer_dir = Some(path.as_ref().to_string_lossy().to_string());
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
        let tokenizer_dir = self.tokenizer_dir
            .ok_or_else(|| candle_core::Error::Msg("Tokenizer directory required".to_string()))?;
        let device = self.device
            .unwrap_or_else(|| Device::cuda_if_available(0).unwrap_or(Device::Cpu));
        
        // Create service with specified backend
        let mut service = MLService::new_with_backend(
            model_path, 
            tokenizer_dir, 
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

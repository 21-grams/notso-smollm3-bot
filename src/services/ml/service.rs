//! Optimized ML Service with batch prefill and efficient generation

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
    /// Optimized generate with streaming - batch prefill for entire prompt
    pub async fn generate_streaming(
        &mut self,
        user_input: &str,
        buffer: &mut crate::services::StreamingBuffer,
        thinking_enabled: bool,
    ) -> anyhow::Result<()> {
        let start_time = std::time::Instant::now();
        tracing::info!("[SERVICE] ===== Starting optimized streaming generation =====");
        tracing::info!("[SERVICE] Input: {} chars, thinking: {}", user_input.len(), thinking_enabled);
        
        // 1. Tokenize input - single batch processing
        let token_batch = self.tokenizer
            .process_input(user_input.to_string(), thinking_enabled)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
        
        let tokens = token_batch.into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty token batch"))?;
        
        if tokens.is_empty() {
            return Err(anyhow::anyhow!("Empty prompt after tokenization"));
        }
        
        let prompt_len = tokens.len();
        tracing::info!("[SERVICE] Prompt: {} tokens", prompt_len);
        
        // 2. Reset KV-cache for new generation
        self.kv_cache.reset();
        if let ModelBackend::Nope(m) = &mut self.model {
            m.reset_cache();
        }
        
        // 3. BATCH PREFILL - Process entire prompt at once
        let prefill_start = std::time::Instant::now();
        tracing::info!("[SERVICE] Starting batch prefill for {} tokens", prompt_len);
        
        // Create prompt tensor [batch=1, seq_len]
        let prompt_tensor = Tensor::new(&tokens[..], &self.device)?
            .unsqueeze(0)?
            .contiguous()?; // Ensure contiguous for optimal performance
        
        // Single forward pass for entire prompt
        let prompt_logits = match &mut self.model {
            ModelBackend::Standard(m) => {
                tracing::debug!("[SERVICE] Batch prefill with Standard backend");
                m.forward(&prompt_tensor, 0, Some(&mut self.kv_cache))?
            },
            ModelBackend::Nope(m) => {
                tracing::debug!("[SERVICE] Batch prefill with NoPE backend");
                m.forward(&prompt_tensor, 0)?
            }
        };
        
        // Update KV-cache position after prefill
        self.kv_cache.update_position(prompt_len);
        
        let prefill_time = prefill_start.elapsed();
        tracing::info!("[SERVICE] ✅ Batch prefill complete in {:?}", prefill_time);
        tracing::info!("[SERVICE]   Prefill speed: {:.2} tokens/sec", 
            prompt_len as f64 / prefill_time.as_secs_f64());
        
        // Get last token's logits for sampling
        let mut last_logits = prompt_logits.i((0, prompt_len - 1, ..))?;
        
        // 4. Initialize generation state
        let mut all_tokens = tokens.clone();
        let mut position = prompt_len;
        let max_tokens = 512;
        let mut in_thinking_mode = false;
        let mut generated_count = 0;
        let mut consecutive_newlines = 0;
        
        // Get special tokens
        let special_tokens = self.tokenizer.special_tokens();
        let eos_token = special_tokens.eos;
        let think_token = special_tokens.thinking_start;
        let think_end_token = special_tokens.thinking_end;
        
        // 5. GENERATION LOOP - Single token at a time
        tracing::info!("[SERVICE] Starting generation for up to {} tokens", max_tokens);
        let gen_start = std::time::Instant::now();
        
        for step in 0..max_tokens {
            // Sample next token
            let next_token = self.logits_processor.sample(&last_logits)?;
            all_tokens.push(next_token);
            
            // Check stop conditions
            if next_token == eos_token {
                tracing::info!("🛑 EOS token at step {}", step);
                break;
            }
            
            // Handle thinking mode
            if next_token == think_token {
                in_thinking_mode = true;
                tracing::debug!("🤔 Entering thinking mode");
            } else if next_token == think_end_token {
                in_thinking_mode = false;
                tracing::debug!("💭 Exiting thinking mode");
            } else if !in_thinking_mode {
                // Decode and stream visible tokens
                let token_text = self.tokenizer.decode(&[next_token])
                    .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
                
                // Check for excessive newlines
                if token_text == "\n" {
                    consecutive_newlines += 1;
                    if consecutive_newlines > 3 {
                        tracing::warn!("Too many newlines, stopping");
                        break;
                    }
                } else {
                    consecutive_newlines = 0;
                }
                
                // Stream token
                buffer.push(&token_text).await
                    .map_err(|e| anyhow::anyhow!("Stream error: {}", e))?;
                generated_count += 1;
                
                // Yield periodically for responsiveness
                if generated_count % 10 == 0 {
                    tokio::task::yield_now().await;
                }
            }
            
            // Single token forward pass
            let next_input = Tensor::new(&[next_token], &self.device)?
                .unsqueeze(0)?
                .contiguous()?;
            
            let logits = match &mut self.model {
                ModelBackend::Standard(m) => {
                    m.forward(&next_input, position, Some(&mut self.kv_cache))?
                },
                ModelBackend::Nope(m) => {
                    m.forward(&next_input, position)?
                }
            };
            
            // Update cache position
            self.kv_cache.update_position(position + 1);
            
            // Extract next token logits
            last_logits = logits.i((0, 0, ..))?;
            position += 1;
            
            // Check context limit
            if position >= self.kv_cache.max_len() {
                tracing::warn!("Reached max context length");
                break;
            }
            
            // Log progress
            if step > 0 && step % 50 == 0 {
                let elapsed = gen_start.elapsed();
                tracing::info!("[SERVICE] Generated {}/{} tokens @ {:.2} tok/s", 
                    step, max_tokens, step as f64 / elapsed.as_secs_f64());
            }
        }
        
        // 6. Complete stream
        buffer.complete().await
            .map_err(|e| anyhow::anyhow!("Buffer completion error: {}", e))?;
        
        // 7. Log final statistics
        let total_time = start_time.elapsed();
        let gen_time = gen_start.elapsed();
        
        tracing::info!("[SERVICE] ✅ Generation complete!");
        tracing::info!("[SERVICE] Statistics:");
        tracing::info!("[SERVICE]   - Total time: {:?}", total_time);
        tracing::info!("[SERVICE]   - Prefill: {:?} ({} tok/s)", 
            prefill_time, prompt_len as f64 / prefill_time.as_secs_f64());
        tracing::info!("[SERVICE]   - Generation: {:?} ({:.2} tok/s)", 
            gen_time, generated_count as f64 / gen_time.as_secs_f64());
        tracing::info!("[SERVICE]   - Generated {} visible tokens", generated_count);
        tracing::info!("[SERVICE]   - Total {} tokens processed", all_tokens.len());
        
        Ok(())
    }
    
    /// Create a new ML service with optimized settings
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        tokenizer_dir: P,
        device: Device,
    ) -> Result<Self> {
        Self::new_with_backend(model_path, tokenizer_dir, device, true)
    }
    
    /// Create with specific backend selection
    pub fn new_with_backend<P: AsRef<Path>>(
        model_path: P,
        tokenizer_dir: P,
        device: Device,
        use_nope: bool,
    ) -> Result<Self> {
        tracing::info!("[ML_SERVICE] Initializing optimized ML service");
        tracing::info!("[ML_SERVICE]   Model: {:?}", model_path.as_ref());
        tracing::info!("[ML_SERVICE]   Tokenizer: {:?}", tokenizer_dir.as_ref());
        tracing::info!("[ML_SERVICE]   Device: {:?}", device);
        tracing::info!("[ML_SERVICE]   Backend: {}", if use_nope { "NoPE" } else { "Standard" });
        
        // Load tokenizer
        let tokenizer = SmolLM3Tokenizer::from_files(tokenizer_dir.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {}", e)))?;
        
        // Load model with selected backend
        let (model, config) = if use_nope {
            tracing::info!("[ML_SERVICE] Loading NoPE-aware model");
            let nope_model = NopeModel::from_gguf(&model_path, &device)?;
            let cfg = nope_model.config().clone();
            (ModelBackend::Nope(nope_model), cfg)
        } else {
            tracing::info!("[ML_SERVICE] Loading standard model");
            let std_model = SmolLM3Model::from_gguf(&model_path, &device)?;
            let cfg = std_model.config().clone();
            (ModelBackend::Standard(std_model), cfg)
        };
        
        // Create pre-allocated KV-cache
        let kv_cache = KVCache::new(&config, &device)?;
        
        // Create logits processor with optimized settings
        let logits_processor = LogitsProcessor::new(
            42,        // seed
            Some(0.8), // temperature
            Some(0.9), // top_p
        );
        
        tracing::info!("[ML_SERVICE] ✅ Service initialized successfully");
        tracing::info!("[ML_SERVICE]   - {} layers", config.base.num_hidden_layers);
        tracing::info!("[ML_SERVICE]   - {} max context", config.base.max_position_embeddings);
        tracing::info!("[ML_SERVICE]   - GQA ratio: {}:{}", 
            config.base.num_attention_heads, config.base.num_key_value_heads);
        
        Ok(Self {
            model,
            tokenizer,
            kv_cache,
            device,
            logits_processor,
        })
    }
    
    /// Update sampling parameters
    pub fn set_sampling_params(&mut self, temperature: f64, top_p: Option<f64>, seed: u64) {
        self.logits_processor = LogitsProcessor::new(seed, Some(temperature), top_p);
    }
    
    /// Get device being used
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Check if using CUDA
    pub fn is_cuda(&self) -> bool {
        self.device.is_cuda()
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
    use_flash_attn: bool,
    yarn_factor: Option<f64>,
}

impl Default for MLServiceBuilder {
    fn default() -> Self {
        Self {
            model_path: None,
            tokenizer_dir: None,
            device: None,
            temperature: 0.8,
            top_p: Some(0.9),
            seed: 42,
            use_nope: true, // Default to optimized NoPE backend
            use_flash_attn: cfg!(feature = "flash-attn"),
            yarn_factor: None,
        }
    }
}

impl MLServiceBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set model path
    pub fn model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.model_path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }
    
    /// Set tokenizer directory
    pub fn tokenizer_dir<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.tokenizer_dir = Some(path.as_ref().to_string_lossy().to_string());
        self
    }
    
    /// Set device (auto-selects CUDA if available by default)
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }
    
    /// Auto-select best available device
    pub fn auto_device(mut self) -> Self {
        self.device = Some(Device::cuda_if_available(0).unwrap_or(Device::Cpu));
        self
    }
    
    /// Set temperature for sampling
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }
    
    /// Set top_p for nucleus sampling
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }
    
    /// Set random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    
    /// Use NoPE backend (recommended)
    pub fn use_nope(mut self, use_nope: bool) -> Self {
        self.use_nope = use_nope;
        self
    }
    
    /// Enable Flash Attention 2 (GPU only)
    pub fn use_flash_attention(mut self, enable: bool) -> Self {
        self.use_flash_attn = enable && cfg!(feature = "flash-attn");
        self
    }
    
    /// Enable YaRN scaling for extended context
    pub fn yarn_scaling(mut self, factor: f64) -> Self {
        self.yarn_factor = Some(factor);
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
        
        // Log configuration
        tracing::info!("[BUILDER] Building ML service with:");
        tracing::info!("[BUILDER]   Backend: {}", if self.use_nope { "NoPE" } else { "Standard" });
        tracing::info!("[BUILDER]   Device: {:?}", device);
        tracing::info!("[BUILDER]   Flash Attn: {}", self.use_flash_attn);
        if let Some(factor) = self.yarn_factor {
            tracing::info!("[BUILDER]   YaRN scaling: {}x", factor);
        }
        
        // Create service
        let mut service = MLService::new_with_backend(
            model_path,
            tokenizer_dir,
            device,
            self.use_nope
        )?;
        
        // Update sampling parameters
        service.set_sampling_params(self.temperature, self.top_p, self.seed);
        
        Ok(service)
    }
}

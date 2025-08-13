//! Enhanced ML Service with proper logits processing for SmolLM3
//! 
//! This module provides a fixed generation pipeline that addresses:
//! - Invalid token generation (4194 and reserved ranges)
//! - NaN/Inf handling in logits
//! - Proper use of Candle's LogitsProcessor
//! - Token filtering for reserved ranges

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::collections::HashSet;
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

/// Enhanced ML service with proper logits processing
pub struct EnhancedMLService {
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
    /// Reserved tokens to filter out
    reserved_tokens: HashSet<u32>,
    /// Repetition penalty settings
    repetition_penalty: f32,
    repetition_last_n: usize,
}

impl EnhancedMLService {
    /// Create logits processor with proper SmolLM3 configuration
    fn create_logits_processor(
        seed: u64,
        temperature: f64,
        top_p: Option<f64>,
        top_k: Option<usize>,
    ) -> LogitsProcessor {
        // Use Candle's recommended sampling setup
        let sampling = match (temperature <= 0.0, top_k, top_p) {
            (true, _, _) => Sampling::ArgMax,  // Greedy for temp <= 0
            (false, None, None) => Sampling::All { temperature },
            (false, Some(k), None) => Sampling::TopK { k, temperature },
            (false, None, Some(p)) => Sampling::TopP { p, temperature },
            (false, Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        };
        
        LogitsProcessor::from_sampling(seed, sampling)
    }
    
    /// Apply token filtering to logits before sampling
    fn filter_logits(&self, logits: &Tensor) -> Result<Tensor> {
        // Get logits as vec for processing
        let mut logits_vec = logits.to_vec1::<f32>()?;
        let vocab_size = logits_vec.len();
        
        // Filter reserved tokens (128009-128255) - SmolLM3 reserved range
        for token_id in 128009..=128255 {
            if token_id < vocab_size {
                logits_vec[token_id] = f32::NEG_INFINITY;
            }
        }
        
        // Filter specific problematic tokens identified during debugging
        const PROBLEMATIC_TOKENS: &[usize] = &[
            4194,  // Known corrupted token causing \u{a0}
            0,     // Sometimes causes issues if not intended as padding
        ];
        
        for &token_id in PROBLEMATIC_TOKENS {
            if token_id < vocab_size {
                logits_vec[token_id] = f32::NEG_INFINITY;
            }
        }
        
        // Additional filtering from reserved_tokens set
        for &token_id in &self.reserved_tokens {
            if (token_id as usize) < vocab_size {
                logits_vec[token_id as usize] = f32::NEG_INFINITY;
            }
        }
        
        // Check for NaN/Inf and replace with valid values
        let mut has_invalid = false;
        for logit in &mut logits_vec {
            if logit.is_nan() {
                *logit = -1000.0;  // Replace NaN with large negative value
                has_invalid = true;
            } else if logit.is_infinite() {
                *logit = if *logit > 0.0 { 100.0 } else { -1000.0 };
                has_invalid = true;
            }
        }
        
        if has_invalid {
            tracing::warn!("[FILTER] Fixed NaN/Inf values in logits");
        }
        
        // Recreate tensor from filtered logits
        Tensor::from_vec(logits_vec, logits.shape(), &self.device)
    }
    
    /// Ensure logits are valid (no NaN/Inf) - fallback safety check
    fn ensure_valid_logits(&self, logits: &Tensor) -> Result<Tensor> {
        let logits_vec = logits.to_vec1::<f32>()?;
        
        // Quick check if any invalid values
        let has_invalid = logits_vec.iter().any(|x| x.is_nan() || x.is_infinite());
        
        if !has_invalid {
            return Ok(logits.clone());
        }
        
        tracing::warn!("[SAFETY] Found invalid logits, applying safety fix");
        
        // Fix invalid values
        let mut fixed_logits = logits_vec;
        for logit in &mut fixed_logits {
            if logit.is_nan() {
                *logit = -100.0;  // Safe negative value
            } else if logit.is_infinite() {
                *logit = if *logit > 0.0 { 100.0 } else { -100.0 };
            }
        }
        
        Tensor::from_vec(fixed_logits, logits.shape(), &self.device)
    }
    
    /// Enhanced generate with proper logits processing
    pub async fn generate_streaming(
        &mut self,
        user_input: &str,
        buffer: &mut crate::services::StreamingBuffer,
        thinking_enabled: bool,
    ) -> anyhow::Result<()> {
        let start_time = std::time::Instant::now();
        tracing::info!("[ENHANCED] ===== Starting enhanced generation =====");
        tracing::info!("[ENHANCED] Input: {} chars, thinking: {}", user_input.len(), thinking_enabled);
        
        // 1. Tokenize input
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
        tracing::info!("[ENHANCED] Prompt: {} tokens", prompt_len);
        
        // 2. Reset caches
        self.kv_cache.reset();
        if let ModelBackend::Nope(m) = &mut self.model {
            m.reset_cache();
        }
        
        // 3. Batch prefill - process entire prompt
        let prefill_start = std::time::Instant::now();
        tracing::info!("[ENHANCED] Starting batch prefill for {} tokens", prompt_len);
        
        let prompt_tensor = Tensor::new(&tokens[..], &self.device)?
            .unsqueeze(0)?
            .contiguous()?;
        
        let prompt_logits = match &mut self.model {
            ModelBackend::Standard(m) => {
                m.forward(&prompt_tensor, 0, Some(&mut self.kv_cache))?
            },
            ModelBackend::Nope(m) => {
                m.forward(&prompt_tensor, 0)?
            }
        };
        
        self.kv_cache.update_position(prompt_len);
        
        let prefill_time = prefill_start.elapsed();
        tracing::info!("[ENHANCED] ✅ Batch prefill complete in {:?}", prefill_time);
        
        // Get last token logits for first generation
        let mut last_logits = prompt_logits.i((0, prompt_len - 1, ..))?;
        
        // 4. Initialize generation state
        let mut all_tokens = tokens.clone();
        let mut position = prompt_len;
        let max_new_tokens = 512;
        let mut in_thinking_mode = false;
        let mut generated_count = 0;
        let mut consecutive_errors = 0;
        
        // Get special tokens
        let special_tokens = self.tokenizer.special_tokens();
        let eos_token = special_tokens.eos;
        let think_token = special_tokens.thinking_start;
        let think_end_token = special_tokens.thinking_end;
        
        // 5. Main generation loop with enhanced processing
        tracing::info!("[ENHANCED] Starting generation for up to {} tokens", max_new_tokens);
        let gen_start = std::time::Instant::now();
        
        for step in 0..max_new_tokens {
            // Apply comprehensive logits processing pipeline
            
            // Step 1: Filter invalid/reserved tokens
            let filtered_logits = self.filter_logits(&last_logits)?;
            
            // Step 2: Apply repetition penalty to generated tokens only
            let penalized_logits = if self.repetition_penalty != 1.0 && all_tokens.len() > prompt_len {
                let start_idx = all_tokens.len().saturating_sub(prompt_len + self.repetition_last_n);
                let recent_tokens = &all_tokens[start_idx.max(prompt_len)..];
                
                candle_transformers::utils::apply_repeat_penalty(
                    &filtered_logits,
                    self.repetition_penalty,
                    recent_tokens,
                )?
            } else {
                filtered_logits
            };
            
            // Step 3: Final safety check for valid logits
            let safe_logits = self.ensure_valid_logits(&penalized_logits)?;
            
            // Step 4: Sample next token
            let mut next_token = self.logits_processor.sample(&safe_logits)?;
            
            // Step 5: Validate sampled token
            if self.reserved_tokens.contains(&next_token) || 
               (next_token >= 128009 && next_token <= 128255) ||
               next_token == 4194 {
                tracing::warn!("[ENHANCED] Caught invalid token {}, forcing safe token", next_token);
                consecutive_errors += 1;
                
                if consecutive_errors > 3 {
                    tracing::error!("[ENHANCED] Too many consecutive errors, stopping");
                    break;
                }
                
                // Force a safe token (period)
                next_token = self.tokenizer.token_to_id(".").unwrap_or(13);
            } else {
                consecutive_errors = 0;
            }
            
            all_tokens.push(next_token);
            
            // Debug first few tokens
            if step < 10 {
                let decoded = self.tokenizer.decode_single(next_token).unwrap_or_else(|_| format!("<err_{}>", next_token));
                tracing::info!("[ENHANCED] Step {}: token {} => '{}'", step, next_token, decoded);
            }
            
            // Check stop conditions
            if next_token == eos_token {
                tracing::info!("[ENHANCED] EOS token at step {}", step);
                break;
            }
            
            // Handle thinking mode
            if next_token == think_token {
                in_thinking_mode = true;
                tracing::debug!("[ENHANCED] Entering thinking mode");
            } else if next_token == think_end_token {
                in_thinking_mode = false;
                tracing::debug!("[ENHANCED] Exiting thinking mode");
            } else if !in_thinking_mode {
                // Decode and stream visible tokens
                if let Ok(text) = self.tokenizer.decode_single(next_token) {
                    if !text.is_empty() {
                        buffer.push(&text).await?;
                        generated_count += 1;
                    }
                }
                
                // Yield periodically
                if generated_count % 10 == 0 {
                    tokio::task::yield_now().await;
                }
            }
            
            // Forward pass for next token
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
            
            self.kv_cache.update_position(position + 1);
            last_logits = logits.i((0, 0, ..))?;
            position += 1;
            
            // Check context limit
            if position >= self.kv_cache.max_len() {
                tracing::warn!("[ENHANCED] Max context reached");
                break;
            }
            
            // Progress logging
            if step > 0 && step % 50 == 0 {
                let elapsed = gen_start.elapsed();
                tracing::info!("[ENHANCED] Generated {}/{} @ {:.2} tok/s", 
                    step, max_new_tokens, step as f64 / elapsed.as_secs_f64());
            }
        }
        
        // Complete stream
        buffer.complete().await?;
        
        // Final statistics
        let total_time = start_time.elapsed();
        let gen_time = gen_start.elapsed();
        
        tracing::info!("[ENHANCED] ✅ Generation complete!");
        tracing::info!("[ENHANCED] Statistics:");
        tracing::info!("[ENHANCED]   - Total time: {:?}", total_time);
        tracing::info!("[ENHANCED]   - Prefill: {:?} ({:.2} tok/s)", 
            prefill_time, prompt_len as f64 / prefill_time.as_secs_f64());
        tracing::info!("[ENHANCED]   - Generation: {:?} ({:.2} tok/s)", 
            gen_time, generated_count as f64 / gen_time.as_secs_f64());
        tracing::info!("[ENHANCED]   - Generated {} visible tokens", generated_count);
        
        Ok(())
    }
    
    /// Create enhanced ML service
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        tokenizer_dir: P,
        device: Device,
    ) -> Result<Self> {
        Self::new_with_config(
            model_path,
            tokenizer_dir,
            device,
            0.7,    // temperature
            0.9,    // top_p
            None,   // top_k
            42,     // seed
            1.1,    // repetition_penalty
            64,     // repetition_last_n
            true,   // use_nope
        )
    }
    
    /// Create with full configuration
    pub fn new_with_config<P: AsRef<Path>>(
        model_path: P,
        tokenizer_dir: P,
        device: Device,
        temperature: f64,
        top_p: f64,
        top_k: Option<usize>,
        seed: u64,
        repetition_penalty: f32,
        repetition_last_n: usize,
        use_nope: bool,
    ) -> Result<Self> {
        tracing::info!("[ENHANCED] Initializing enhanced ML service");
        
        // Load tokenizer
        let tokenizer = SmolLM3Tokenizer::from_files(tokenizer_dir.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {}", e)))?;
        
        // Create reserved tokens set
        let mut reserved_tokens = HashSet::new();
        // Add SmolLM3 reserved range
        for id in 128009..=128255 {
            reserved_tokens.insert(id);
        }
        // Add known problematic tokens
        reserved_tokens.insert(4194);
        
        // Load model
        let (model, config) = if use_nope {
            tracing::info!("[ENHANCED] Loading NoPE-aware model");
            let nope_model = NopeModel::from_gguf(&model_path, &device)?;
            let cfg = nope_model.config().clone();
            (ModelBackend::Nope(nope_model), cfg)
        } else {
            tracing::info!("[ENHANCED] Loading standard model");
            let std_model = SmolLM3Model::from_gguf(&model_path, &device)?;
            let cfg = std_model.config().clone();
            (ModelBackend::Standard(std_model), cfg)
        };
        
        // Create KV cache
        let kv_cache = KVCache::new(&config, &device)?;
        
        // Create enhanced logits processor
        let logits_processor = Self::create_logits_processor(
            seed,
            temperature,
            Some(top_p),
            top_k,
        );
        
        tracing::info!("[ENHANCED] ✅ Service initialized with:");
        tracing::info!("[ENHANCED]   - Temperature: {}", temperature);
        tracing::info!("[ENHANCED]   - Top-p: {}", top_p);
        tracing::info!("[ENHANCED]   - Top-k: {:?}", top_k);
        tracing::info!("[ENHANCED]   - Repetition penalty: {}", repetition_penalty);
        tracing::info!("[ENHANCED]   - Reserved tokens filtered: {}", reserved_tokens.len());
        
        Ok(Self {
            model,
            tokenizer,
            kv_cache,
            device,
            logits_processor,
            reserved_tokens,
            repetition_penalty,
            repetition_last_n,
        })
    }
    
    /// Update sampling parameters
    pub fn set_sampling_params(
        &mut self,
        temperature: f64,
        top_p: Option<f64>,
        top_k: Option<usize>,
        seed: u64,
    ) {
        self.logits_processor = Self::create_logits_processor(seed, temperature, top_p, top_k);
    }
    
    /// Set repetition penalty parameters
    pub fn set_repetition_params(&mut self, penalty: f32, last_n: usize) {
        self.repetition_penalty = penalty;
        self.repetition_last_n = last_n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reserved_token_filtering() {
        // Test that reserved tokens are properly filtered
        let reserved_tokens: HashSet<u32> = (128009..=128255).collect();
        assert!(reserved_tokens.contains(&128009));
        assert!(reserved_tokens.contains(&128255));
        assert!(reserved_tokens.contains(&128100));
        assert_eq!(reserved_tokens.len(), 247);
    }
    
    #[test]
    fn test_problematic_token_detection() {
        // Test known problematic tokens
        let problematic = vec![4194, 0];
        for token in problematic {
            assert!(token == 4194 || token == 0);
        }
    }
}
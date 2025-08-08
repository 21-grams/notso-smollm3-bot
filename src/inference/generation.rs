//! Core generation loop at the Candle level
//! 
//! This module implements the low-level generation loop that higher-level
//! services use to generate text. It handles the token-by-token generation
//! process, logit manipulation, and stopping criteria.

use candle_core::{Tensor, Device, Result, DType};
use tokenizers::Tokenizer;
use std::time::{Duration, Instant};

use crate::inference::candle::{
    tensor_ops::{SamplingOps, AttentionOps},
    kv_cache::KVCacheTensors,
};

/// Generation configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f64,
    /// Top-p value for nucleus sampling
    pub top_p: Option<f64>,
    /// Top-k value for top-k sampling
    pub top_k: Option<usize>,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Stop sequences
    pub stop_sequences: Vec<Vec<u32>>,
    /// EOS token ID
    pub eos_token_id: Option<u32>,
    /// Pad token ID
    pub pad_token_id: Option<u32>,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: Some(0.9),
            top_k: None,
            repetition_penalty: 1.0,
            stop_sequences: Vec::new(),
            eos_token_id: Some(2), // Standard EOS token
            pad_token_id: Some(0),  // Standard PAD token
            seed: None,
        }
    }
}

/// Core generation loop implementation
pub struct GenerationLoop<M> {
    model: M,
    device: Device,
    kv_cache: Option<KVCacheTensors>,
}

impl<M> GenerationLoop<M>
where
    M: GenerativeModel,
{
    /// Create new generation loop
    pub fn new(model: M, device: Device) -> Self {
        Self {
            model,
            device,
            kv_cache: None,
        }
    }
    
    /// Initialize KV cache if supported
    pub fn with_kv_cache(mut self, max_seq_len: usize) -> Self {
        self.kv_cache = Some(KVCacheTensors::new(
            max_seq_len,
            DType::F32,
            self.device.clone(),
        ));
        self
    }
    
    /// Generate tokens from input IDs
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        config: &GenerationConfig,
    ) -> Result<GenerationOutput> {
        let start_time = Instant::now();
        let mut generated_tokens = Vec::new();
        let mut logits_history = Vec::new();
        
        // Convert input to tensor
        let mut current_ids = Tensor::from_vec(
            input_ids.to_vec(),
            &[1, input_ids.len()],
            &self.device,
        )?;
        
        // Track token frequencies for repetition penalty
        let mut token_frequencies = std::collections::HashMap::new();
        for &token_id in input_ids {
            *token_frequencies.entry(token_id).or_insert(0) += 1;
        }
        
        // Generation loop
        for step in 0..config.max_tokens {
            // Forward pass
            let logits = if let Some(ref mut cache) = self.kv_cache {
                self.model.forward_with_cache(&current_ids, cache)?
            } else {
                self.model.forward(&current_ids)?
            };
            
            // Get last token logits
            let next_token_logits = Self::get_last_logits(&logits)?;
            
            // Apply repetition penalty
            let next_token_logits = if config.repetition_penalty != 1.0 {
                Self::apply_repetition_penalty(
                    &next_token_logits,
                    &token_frequencies,
                    config.repetition_penalty,
                )?
            } else {
                next_token_logits
            };
            
            // Store logits for analysis
            logits_history.push(next_token_logits.clone());
            
            // Sample next token
            let next_token = Self::sample_token(&next_token_logits, config)?;
            
            // Check stopping criteria
            if Self::should_stop(next_token, &generated_tokens, config) {
                break;
            }
            
            // Update state
            generated_tokens.push(next_token);
            *token_frequencies.entry(next_token).or_insert(0) += 1;
            
            // Prepare next input
            current_ids = Tensor::from_vec(
                vec![next_token],
                &[1, 1],
                &self.device,
            )?;
        }
        
        let generation_time = start_time.elapsed();
        
        let token_count = generated_tokens.len();
        Ok(GenerationOutput {
            tokens: generated_tokens,
            logits_history,
            generation_time,
            tokens_per_second: token_count as f32 / generation_time.as_secs_f32(),
        })
    }
    
    /// Generate with streaming callback
    pub fn generate_stream<F>(
        &mut self,
        input_ids: &[u32],
        config: &GenerationConfig,
        mut callback: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(StreamEvent) -> Result<StreamControl>,
    {
        let start_time = Instant::now();
        let mut generated_tokens = Vec::new();
        let mut logits_history = Vec::new();
        
        // Convert input to tensor
        let mut current_ids = Tensor::from_vec(
            input_ids.to_vec(),
            &[1, input_ids.len()],
            &self.device,
        )?;
        
        // Track token frequencies
        let mut token_frequencies = std::collections::HashMap::new();
        for &token_id in input_ids {
            *token_frequencies.entry(token_id).or_insert(0) += 1;
        }
        
        // Send start event
        if let StreamControl::Stop = callback(StreamEvent::Start)? {
            return Ok(GenerationOutput::empty());
        }
        
        // Generation loop
        for step in 0..config.max_tokens {
            let token_start = Instant::now();
            
            // Forward pass
            let logits = if let Some(ref mut cache) = self.kv_cache {
                self.model.forward_with_cache(&current_ids, cache)?
            } else {
                self.model.forward(&current_ids)?
            };
            
            // Get last token logits
            let next_token_logits = Self::get_last_logits(&logits)?;
            
            // Apply repetition penalty
            let next_token_logits = if config.repetition_penalty != 1.0 {
                Self::apply_repetition_penalty(
                    &next_token_logits,
                    &token_frequencies,
                    config.repetition_penalty,
                )?
            } else {
                next_token_logits
            };
            
            logits_history.push(next_token_logits.clone());
            
            // Sample next token
            let next_token = Self::sample_token(&next_token_logits, config)?;
            
            // Send token event
            let control = callback(StreamEvent::Token {
                token_id: next_token,
                logits: next_token_logits.clone(),
                step,
                latency: token_start.elapsed(),
            })?;
            
            if let StreamControl::Stop = control {
                break;
            }
            
            // Check stopping criteria
            if Self::should_stop(next_token, &generated_tokens, config) {
                break;
            }
            
            // Update state
            generated_tokens.push(next_token);
            *token_frequencies.entry(next_token).or_insert(0) += 1;
            
            // Prepare next input
            current_ids = Tensor::from_vec(
                vec![next_token],
                &[1, 1],
                &self.device,
            )?;
        }
        
        let generation_time = start_time.elapsed();
        
        // Send end event
        let token_count = generated_tokens.len();
        callback(StreamEvent::End {
            total_tokens: token_count,
            total_time: generation_time,
        })?;
        
        Ok(GenerationOutput {
            tokens: generated_tokens,
            logits_history,
            generation_time,
            tokens_per_second: token_count as f32 / generation_time.as_secs_f32(),
        })
    }
    
    /// Get logits for the last position
    fn get_last_logits(logits: &Tensor) -> Result<Tensor> {
        let seq_len = logits.dims()[1];
        logits.narrow(1, seq_len - 1, 1)?.squeeze(1)
    }
    
    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(
        logits: &Tensor,
        token_frequencies: &std::collections::HashMap<u32, usize>,
        penalty: f32,
    ) -> Result<Tensor> {
        let mut logits_vec = logits.to_vec1::<f32>()?;
        
        for (&token_id, &count) in token_frequencies.iter() {
            if count > 0 && (token_id as usize) < logits_vec.len() {
                let penalty_factor = penalty.powi(count as i32);
                logits_vec[token_id as usize] /= penalty_factor;
            }
        }
        
        Tensor::from_vec(logits_vec, logits.shape(), logits.device())
    }
    
    /// Sample token from logits
    fn sample_token(logits: &Tensor, config: &GenerationConfig) -> Result<u32> {
        if config.temperature == 0.0 {
            // Greedy decoding
            SamplingOps::greedy_sample(logits)
        } else if let Some(k) = config.top_k {
            // Top-k sampling
            SamplingOps::sample_top_k(logits, k, config.temperature)
        } else {
            // Temperature sampling with optional top-p
            SamplingOps::sample_with_temperature(logits, config.temperature, config.top_p)
        }
    }
    
    /// Check if generation should stop
    fn should_stop(
        token: u32,
        generated: &[u32],
        config: &GenerationConfig,
    ) -> bool {
        // Check EOS token
        if let Some(eos) = config.eos_token_id {
            if token == eos {
                return true;
            }
        }
        
        // Check stop sequences
        for stop_seq in &config.stop_sequences {
            if generated.len() >= stop_seq.len() - 1 {
                let mut matches = true;
                for i in 0..stop_seq.len() - 1 {
                    if generated[generated.len() - stop_seq.len() + 1 + i] != stop_seq[i] {
                        matches = false;
                        break;
                    }
                }
                if matches && token == stop_seq[stop_seq.len() - 1] {
                    return true;
                }
            }
        }
        
        false
    }
}

/// Trait for generative models
pub trait GenerativeModel {
    /// Forward pass without cache
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor>;
    
    /// Forward pass with KV cache
    fn forward_with_cache(
        &self,
        input_ids: &Tensor,
        cache: &mut KVCacheTensors,
    ) -> Result<Tensor>;
}

/// Generation output
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Generated token IDs
    pub tokens: Vec<u32>,
    /// Logits history for each generated token
    pub logits_history: Vec<Tensor>,
    /// Total generation time
    pub generation_time: Duration,
    /// Tokens per second
    pub tokens_per_second: f32,
}

impl GenerationOutput {
    /// Create empty output
    pub fn empty() -> Self {
        Self {
            tokens: Vec::new(),
            logits_history: Vec::new(),
            generation_time: Duration::ZERO,
            tokens_per_second: 0.0,
        }
    }
    
    /// Get generated text using tokenizer
    pub fn decode(&self, tokenizer: &Tokenizer) -> Result<String> {
        Ok(tokenizer
            .decode(&self.tokens, true)
            .map_err(|e| candle_core::Error::Msg(format!("Decoding error: {}", e)))?)
    }
}

/// Stream event for generation callback
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Generation started
    Start,
    /// Token generated
    Token {
        token_id: u32,
        logits: Tensor,
        step: usize,
        latency: Duration,
    },
    /// Generation ended
    End {
        total_tokens: usize,
        total_time: Duration,
    },
}

/// Stream control for callback
#[derive(Debug, Clone, Copy)]
pub enum StreamControl {
    /// Continue generation
    Continue,
    /// Stop generation
    Stop,
}

/// Beam search implementation for better quality generation
pub struct BeamSearch {
    beam_size: usize,
    length_penalty: f32,
}

impl BeamSearch {
    pub fn new(beam_size: usize, length_penalty: f32) -> Self {
        Self {
            beam_size,
            length_penalty,
        }
    }
    
    /// Run beam search generation
    pub fn generate<M>(
        &self,
        model: &M,
        input_ids: &[u32],
        max_length: usize,
        device: &Device,
    ) -> Result<Vec<BeamHypothesis>>
    where
        M: GenerativeModel,
    {
        let mut beams = vec![BeamHypothesis {
            tokens: input_ids.to_vec(),
            score: 0.0,
            is_complete: false,
        }];
        
        for _ in 0..max_length {
            let mut new_beams = Vec::new();
            
            for beam in &beams {
                if beam.is_complete {
                    new_beams.push(beam.clone());
                    continue;
                }
                
                // Get logits for this beam
                let input = Tensor::from_vec(
                    beam.tokens.clone(),
                    &[1, beam.tokens.len()],
                    device,
                )?;
                let logits = model.forward(&input)?;
                let last_logits = logits.narrow(1, logits.dims()[1] - 1, 1)?.squeeze(1)?;
                
                // Get top-k tokens
                let logits_vec = last_logits.to_vec1::<f32>()?;
                let mut indexed: Vec<_> = logits_vec.iter().enumerate()
                    .map(|(i, &l)| (i as u32, l))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed.truncate(self.beam_size);
                
                // Create new hypotheses
                for (token_id, score) in indexed {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(token_id);
                    
                    let length_penalty = ((new_tokens.len() as f32).powf(self.length_penalty))
                        / ((beam.tokens.len() as f32).powf(self.length_penalty));
                    
                    new_beams.push(BeamHypothesis {
                        tokens: new_tokens,
                        score: beam.score + score / length_penalty,
                        is_complete: token_id == 2, // EOS token
                    });
                }
            }
            
            // Keep top beams
            new_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            beams = new_beams;
            beams.truncate(self.beam_size);
            
            // Check if all beams are complete
            if beams.iter().all(|b| b.is_complete) {
                break;
            }
        }
        
        Ok(beams)
    }
}

/// Beam hypothesis for beam search
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    pub tokens: Vec<u32>,
    pub score: f32,
    pub is_complete: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock model for testing
    struct MockModel;
    
    impl GenerativeModel for MockModel {
        fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
            // Return random logits
            let vocab_size = 32000;
            let batch_size = input_ids.dims()[0];
            let seq_len = input_ids.dims()[1];
            
            Tensor::randn(
                0.0f32,
                1.0,
                &[batch_size, seq_len, vocab_size],
                input_ids.device(),
            )
        }
        
        fn forward_with_cache(
            &self,
            input_ids: &Tensor,
            _cache: &mut KVCacheTensors,
        ) -> Result<Tensor> {
            self.forward(input_ids)
        }
    }
    
    #[test]
    fn test_generation_loop() -> Result<()> {
        let device = Device::Cpu;
        let model = MockModel;
        let mut generator = GenerationLoop::new(model, device.clone());
        
        let input_ids = vec![1, 2, 3];
        let config = GenerationConfig {
            max_tokens: 10,
            temperature: 0.8,
            ..Default::default()
        };
        
        let output = generator.generate(&input_ids, &config)?;
        assert!(output.tokens.len() <= 10);
        assert!(output.tokens_per_second > 0.0);
        
        Ok(())
    }
}

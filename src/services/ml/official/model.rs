//! SmolLM3 model implementation using candle_transformers::models::quantized_llama
//! This wrapper adds SmolLM3-specific features while leveraging the official Llama implementation

use candle_core::{Device, Result, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_core::quantized::gguf_file::Content;
use std::path::Path;
use std::fs::File;

/// SmolLM3 model that wraps the official quantized Llama implementation
pub struct SmolLM3Model {
    /// Official Llama model weights
    weights: ModelWeights,
    /// Model configuration
    config: super::config::SmolLM3Config,
    /// Device for tensor operations
    device: Device,
    /// Layers that skip position encoding (NoPE)
    nope_layers: Vec<usize>,
}

impl SmolLM3Model {
    /// Load SmolLM3 model from GGUF file
    pub fn from_gguf<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        tracing::info!("🚀 Loading SmolLM3 model from GGUF");
        
        // Load GGUF with metadata mapping
        let content = super::gguf_loader::load_smollm3_gguf(&path, device)?;
        
        // Create configuration from GGUF metadata
        let config = super::config::SmolLM3Config::from_gguf(&content)?;
        
        tracing::info!("📐 Model config: {} layers, {} heads, {} KV heads",
                      config.base.num_hidden_layers,
                      config.base.num_attention_heads,
                      config.base.num_key_value_heads);
        
        // Load weights using official Llama loader
        let mut file = File::open(path)?;
        let weights = ModelWeights::from_gguf(content, &mut file, device)?;
        
        tracing::info!("✅ SmolLM3 model loaded successfully");
        
        let nope_layers = config.nope_layer_indices.clone();
        
        Ok(Self {
            weights,
            config,
            device: device.clone(),
            nope_layers,
        })
    }
    
    /// Forward pass through the model
    pub fn forward(
        &self,
        input_ids: &Tensor,
        _position_ids: Option<&Tensor>,
        _kv_cache: Option<&mut crate::smollm3::kv_cache::SmolLM3KVCache>,
    ) -> Result<Tensor> {
        // The official ModelWeights doesn't have a direct forward method
        // We'll use the forward_full implementation from llama_forward module
        
        // For now, return a placeholder
        // The actual implementation is in llama_forward.rs
        
        tracing::warn!("Forward pass needs full implementation - using placeholder");
        
        // Placeholder: return random logits of correct shape
        let batch_size = input_ids.dim(0)?;
        let seq_len = input_ids.dim(1)?;
        let vocab_size = self.config.base.vocab_size;
        
        Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, vocab_size], &self.device)
    }
    
    /// Check if a layer should skip position encoding (NoPE)
    pub fn is_nope_layer(&self, layer_idx: usize) -> bool {
        self.nope_layers.contains(&layer_idx)
    }
    
    /// Get model configuration
    pub fn config(&self) -> &super::config::SmolLM3Config {
        &self.config
    }
    
    /// Get the underlying weights for direct access if needed
    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// OfficialSmolLM3Model for backward compatibility
pub type OfficialSmolLM3Model = SmolLM3Model;

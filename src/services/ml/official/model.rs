//! SmolLM3 model implementation using candle_transformers::models::quantized_llama
//! This wrapper adds SmolLM3-specific features while leveraging the official Llama implementation

use candle_core::{Device, Result, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
// use candle_core::quantized::gguf_file::Content; // Currently unused
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
        &mut self,
        input_ids: &Tensor,
        position: usize,
        _kv_cache: Option<&mut crate::services::ml::smollm3::kv_cache::KVCache>,
    ) -> Result<Tensor> {
        tracing::debug!("🚀 Starting forward pass at position {}", position);
        
        // Use ModelWeights' forward method
        // Note: This applies RoPE to all layers (no NoPE support)
        let logits = self.weights.forward(input_ids, position)?;
        
        tracing::debug!("✅ Forward pass complete, logits shape: {:?}", logits.shape());
        
        Ok(logits)
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

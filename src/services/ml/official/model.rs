//! Official quantized_llama wrapper with SmolLM3 configuration

use candle_transformers::models::quantized_llama::{Llama, ModelWeights};
use candle_core::{Device, Tensor, Result};
use super::config::SmolLM3Config;

/// Wrapper around official Candle Llama model
pub struct OfficialSmolLM3Model {
    model: Llama,
    config: SmolLM3Config,
    device: Device,
}

impl OfficialSmolLM3Model {
    /// Load model using official Candle patterns
    pub async fn load(
        weights: &ModelWeights,
        config: SmolLM3Config,
        device: &Device,
    ) -> Result<Self> {
        let llama_config = config.to_llama_config();
        let model = Llama::load(weights, &llama_config, device)?;
        
        Ok(Self {
            model,
            config,
            device: device.clone(),
        })
    }
    
    /// Forward pass using official implementation
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position)
    }
    
    /// Forward without position (for prefill)
    pub fn forward_prefill(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids, 0)
    }
    
    /// Get model configuration
    pub fn config(&self) -> &SmolLM3Config {
        &self.config
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

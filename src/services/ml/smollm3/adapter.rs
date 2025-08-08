//! Bridge between official Candle and SmolLM3 features

use crate::services::ml::official::{OfficialSmolLM3Model, SmolLM3Config};
use candle_core::{Tensor, Result, Device};
use super::nope_layers::NopeHandler;
use super::thinking::ThinkingDetector;


pub struct SmolLM3Adapter {
    model: OfficialSmolLM3Model,  // Keep as owned for now
    nope_handler: NopeHandler,
    thinking_detector: ThinkingDetector,
    config: SmolLM3Config,
}

impl SmolLM3Adapter {
    pub fn new(model: OfficialSmolLM3Model) -> Self {
        let config = model.config().clone();
        
        Self {
            nope_handler: NopeHandler::new(config.nope_layers.clone()),
            thinking_detector: ThinkingDetector::new(config.thinking_tokens.clone()),
            model,
            config,
        }
    }
    
    /// Get a reference to the config (for creating generator)
    pub fn get_config(&self) -> &SmolLM3Config {
        &self.config
    }
    
    /// Get device (for creating generator)
    pub fn get_device(&self) -> &Device {
        self.model.device()
    }
    
    /// Forward pass with SmolLM3 extensions
    pub fn forward_with_extensions(
        &mut self,
        input_ids: &Tensor,
        position: usize,
    ) -> Result<Tensor> {
        // Check if current layer needs NoPE handling
        let layer_idx = position % self.config.base.n_layer;
        
        if self.nope_handler.should_skip_rope(layer_idx) {
            // Custom handling for NoPE layers
            self.forward_nope_layer(input_ids, layer_idx)
        } else {
            // Standard official forward pass
            self.model.forward(input_ids, position)
        }
    }
    
    fn forward_nope_layer(&mut self, input_ids: &Tensor, layer_idx: usize) -> Result<Tensor> {
        // For NoPE layers, we still use the standard forward but mark it for special handling
        // The actual RoPE skipping would need to be implemented at a lower level
        tracing::debug!("Processing NoPE layer {}", layer_idx);
        self.model.forward(input_ids, layer_idx)
    }
    
    pub fn config(&self) -> &SmolLM3Config {
        &self.config
    }
    
    pub fn device(&self) -> &Device {
        self.model.device()
    }
    
    pub fn thinking_detector(&self) -> &ThinkingDetector {
        &self.thinking_detector
    }
}

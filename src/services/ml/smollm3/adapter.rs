//! Bridge between official Candle and SmolLM3 features
//! 
//! NOTE: This adapter is currently not used in the main implementation.
//! It's kept for potential future use when bridging between different model backends.

use crate::services::ml::official::{OfficialSmolLM3Model, SmolLM3Config};
use candle_core::{Tensor, Result, Device};
use super::nope_layers::NopeHandler;
use super::thinking::ThinkingDetector;

pub struct SmolLM3Adapter {
    model: OfficialSmolLM3Model,
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
    
    /// Get a reference to the config
    pub fn get_config(&self) -> &SmolLM3Config {
        &self.config
    }
    
    /// Get device
    pub fn get_device(&self) -> &Device {
        self.model.device()
    }
    
    /// Forward pass with SmolLM3 extensions
    pub fn forward_with_extensions(
        &mut self,
        input_ids: &Tensor,
        position: usize,
    ) -> Result<Tensor> {
        // NOTE: This method is not currently compatible with the new model API
        // which expects position as usize instead of Option<&Tensor>
        // Keeping for reference but not actively used
        
        self.model.forward(input_ids, position, None)
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

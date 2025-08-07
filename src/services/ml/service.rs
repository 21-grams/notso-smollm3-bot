//! High-level ML service orchestrating all components

use super::official::{OfficialLoader, SmolLM3Config, OfficialSmolLM3Model};
use super::smollm3::{SmolLM3Adapter, SmolLM3Generator};
use super::streaming::GenerationEvent;
use candle_core::Device;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::UnboundedSender;
use std::sync::Arc;
use anyhow::Result;

pub struct MLService {
    adapter: SmolLM3Adapter,
    generator: SmolLM3Generator,
    config: SmolLM3Config,
    device: Device,
}

impl MLService {
    /// Initialize ML service with official foundation
    pub async fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        tracing::info!("🚀 Initializing ML service with official Candle foundation");
        
        // 1. Device detection
        let device = super::official::DeviceManager::detect_optimal_device()?;
        tracing::info!("🎮 Using device: {}", super::official::DeviceManager::device_info(&device));
        
        // 2. Load configuration
        let config = SmolLM3Config::default();
        
        // 3. Official GGUF loading
        OfficialLoader::validate_gguf(model_path)?;
        let weights = OfficialLoader::load_gguf(model_path, &device).await?;
        
        // 4. Create official model
        let official_model = OfficialSmolLM3Model::load(
            &weights,
            config.clone(),
            &device,
        ).await?;
        
        // 5. Create adapter with SmolLM3 extensions
        let adapter = SmolLM3Adapter::new(official_model);
        
        // 6. Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        
        // 7. Create generator
        let generator = SmolLM3Generator::new(
            adapter.model.clone(),  // Note: This needs adjustment for ownership
            tokenizer,
            Some(0.7),    // temperature
            Some(0.9),    // top_p
        );
        
        tracing::info!("✅ ML service initialized successfully");
        
        Ok(Self {
            adapter,
            generator,
            config,
            device,
        })
    }
    
    /// Generate response with streaming
    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        sender: UnboundedSender<GenerationEvent>,
    ) -> Result<String> {
        self.generator.generate_stream(prompt, sender, 512).await
    }
    
    /// Get model configuration
    pub fn config(&self) -> &SmolLM3Config {
        &self.config
    }
    
    /// Check if service is ready
    pub fn is_ready(&self) -> bool {
        true  // Can add more sophisticated checks
    }
}

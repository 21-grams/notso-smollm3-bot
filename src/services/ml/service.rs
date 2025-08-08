//! High-level ML service orchestrating all components

use super::official::{OfficialLoader, SmolLM3Config, OfficialSmolLM3Model};
use super::smollm3::SmolLM3Adapter;
use crate::types::events::StreamEvent;  // Use StreamEvent consistently
use candle_core::Device;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::UnboundedSender;
use std::sync::Arc;
use anyhow::Result;

pub struct MLService {
    config: SmolLM3Config,
    device: Device,
    tokenizer: Option<Tokenizer>,
    is_stub: bool,
    // We'll keep adapter and generator together for simplicity
    adapter: Option<SmolLM3Adapter>,
}

impl MLService {
    /// Create a stub ML service for testing without models
    pub async fn new_stub() -> Result<Self> {
        tracing::info!("🔌 Creating stub ML service (no model loaded)");
        
        let config = SmolLM3Config::default();
        let device = Device::Cpu;
        
        Ok(Self {
            config,
            device,
            tokenizer: None,
            is_stub: true,
            adapter: None,
        })
    }
    
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
        
        tracing::info!("✅ ML service initialized successfully");
        
        Ok(Self {
            config,
            device,
            tokenizer: Some(tokenizer),
            is_stub: false,
            adapter: Some(adapter),
        })
    }
    
    /// Generate response for a session (high-level API for handlers)
    pub async fn generate_response(
        &self,
        session_id: &str,
        message: &str,
    ) -> anyhow::Result<()> {
        tracing::info!("Generating response for session: {}", session_id);
        
        // For now, just log and return OK
        // Real implementation would use generate_stream internally
        tracing::info!("Message: {}", message);
        Ok(())
    }
    
    /// Generate response with streaming
    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        sender: UnboundedSender<StreamEvent>,
    ) -> Result<String> {
        if self.is_stub {
            // Stub mode - send mock events
            let _ = sender.send(StreamEvent::thinking("Thinking in stub mode...".to_string()));
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            
            let response = format!("Mock response to: {}", prompt);
            for word in response.split_whitespace() {
                let _ = sender.send(StreamEvent::token(format!("{} ", word)));
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }
            
            let _ = sender.send(StreamEvent::complete());
            Ok(response)
        } else {
            // For now, just return stub response even with model loaded
            // The actual generation requires more refactoring to handle the mutable model
            tracing::warn!("Real generation not yet implemented, using stub");
            
            let response = format!("Response to: {}", prompt);
            let _ = sender.send(StreamEvent::token(response.clone()));
            let _ = sender.send(StreamEvent::complete());
            Ok(response)
        }
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

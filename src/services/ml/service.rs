//! High-level ML service orchestrating all components

use super::official::{SmolLM3Config};
use crate::types::events::StreamEvent;
use candle_core::Device;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::UnboundedSender;
use std::sync::Arc;
use anyhow::Result;

pub struct MLService {
    config: SmolLM3Config,
    device: Device,
    tokenizer: Option<Arc<Tokenizer>>,
    is_stub: bool,
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
        })
    }
    
    /// Initialize ML service with official foundation
    pub async fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        tracing::info!("🚀 Initializing ML service with SmolLM3");
        
        // 1. Device detection
        let device = super::official::DeviceManager::detect_optimal_device()?;
        tracing::info!("🎮 Using device: {}", super::official::DeviceManager::device_info(&device));
        
        // 2. Check if model files exist
        if !std::path::Path::new(model_path).exists() {
            return Err(anyhow::anyhow!("Model file not found: {}", model_path));
        }
        
        if !std::path::Path::new(tokenizer_path).exists() {
            return Err(anyhow::anyhow!("Tokenizer file not found: {}", tokenizer_path));
        }
        
        // 3. Try to inspect GGUF metadata
        let inspection = super::official::inspect_gguf(model_path)?;
        inspection.print_report();
        
        if !inspection.has_llama_metadata {
            tracing::warn!("⚠️ Missing Llama metadata. The GGUF file may need conversion.");
            return Err(anyhow::anyhow!(
                "GGUF file missing required metadata. Model needs proper conversion for SmolLM3."
            ));
        }
        
        // 5. Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let tokenizer = Arc::new(tokenizer);
        
        // 6. For now, return stub mode since full model loading needs more work
        tracing::warn!("⚠️ Full model loading not yet implemented. Using stub mode.");
        
        Ok(Self {
            config: SmolLM3Config::default(),
            device,
            tokenizer: Some(tokenizer),
            is_stub: true,
        })
    }
    
    /// Generate response for a session (high-level API for handlers)
    pub async fn generate_response(
        &self,
        session_id: &str,
        message: &str,
    ) -> anyhow::Result<()> {
        tracing::info!("Generating response for session: {}", session_id);
        tracing::info!("Message: {}", message);
        
        if self.is_stub {
            tracing::info!("Running in stub mode - no actual generation");
        }
        
        Ok(())
    }
    
    /// Generate response with streaming
    pub async fn generate_stream(
        &self,
        prompt: &str,
        sender: UnboundedSender<StreamEvent>,
    ) -> Result<String> {
        if self.is_stub {
            // Stub mode - send mock events with more realistic responses
            let _ = sender.send(StreamEvent::thinking("Processing your message...".to_string()));
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            
            // Generate a contextual stub response based on the prompt
            let response = if prompt.to_lowercase().contains("hello") || prompt.to_lowercase().contains("hi") {
                "Hello! I'm SmolLM3 running in stub mode. The actual model is not loaded yet, but I'm here to help test the chat interface."
            } else if prompt.to_lowercase().contains("how are you") {
                "I'm functioning well in stub mode! The chat interface is working, though I'm not using the actual SmolLM3 model yet."
            } else if prompt.to_lowercase().contains("help") {
                "I'm currently running in stub mode, which means the real SmolLM3 model isn't loaded. This mode is useful for testing the chat interface and streaming functionality."
            } else {
                "I received your message, but I'm running in stub mode without the actual SmolLM3 model loaded. This is a test response to verify the streaming system works correctly."
            };
            
            // Stream the response word by word
            for word in response.split_whitespace() {
                let _ = sender.send(StreamEvent::token(format!("{} ", word)));
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }
            
            let _ = sender.send(StreamEvent::complete());
            return Ok(response.to_string());
        }
        
        // Real generation would go here when model is properly loaded
        Err(anyhow::anyhow!("Real model generation not yet implemented"))
    }
}

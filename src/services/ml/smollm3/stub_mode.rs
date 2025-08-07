use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;
use crate::types::events::GenerationEvent;
use tokio::time::{sleep, Duration};

/// Stub mode service for testing without models
pub struct StubModeService {
    enabled: bool,
}

impl StubModeService {
    pub fn new() -> Self {
        tracing::info!("📦 Creating SmolLM3 stub service (no model loaded)");
        Self { enabled: true }
    }
    
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Generate mock stream for UI testing
    pub async fn generate_mock_stream(
        &self,
        prompt: &str,
        sender: UnboundedSender<GenerationEvent>,
        thinking_mode: bool,
    ) -> Result<()> {
        tracing::info!("🎯 Mock generation for prompt: {}", prompt);
        
        // Send start event
        let _ = sender.send(GenerationEvent::Start);
        
        // Mock thinking mode if enabled
        if thinking_mode {
            let _ = sender.send(GenerationEvent::ThinkingStart);
            
            let thinking_steps = vec![
                "Analyzing the question...",
                "Considering different approaches...",
                "Formulating response...",
            ];
            
            for step in thinking_steps {
                sleep(Duration::from_millis(300)).await;
                let _ = sender.send(GenerationEvent::ThinkingToken(step.to_string()));
            }
            
            let _ = sender.send(GenerationEvent::ThinkingEnd);
            sleep(Duration::from_millis(200)).await;
        }
        
        // Mock response generation
        let mock_response = format!(
            "This is a mock response to your input: '{}'. \
             The SmolLM3 model is running in stub mode for UI testing. \
             Once the model is loaded, you'll see real AI-generated responses here.",
            prompt.chars().take(50).collect::<String>()
        );
        
        // Stream response tokens
        for word in mock_response.split_whitespace() {
            sleep(Duration::from_millis(50)).await;
            let _ = sender.send(GenerationEvent::ResponseToken(format!("{} ", word)));
        }
        
        // Send completion
        let _ = sender.send(GenerationEvent::Complete);
        
        Ok(())
    }
}

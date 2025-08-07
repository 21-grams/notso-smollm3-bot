use crate::config::Config;
use crate::inference::InferenceEngine;
use crate::smollm3::{config::SmolLM3Config, tokenizer::SmolLM3Tokenizer};
use crate::services::StreamingService;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct SmolLM3Model {
    engine: Arc<InferenceEngine>,
    tokenizer: Arc<Mutex<SmolLM3Tokenizer>>,
    config: SmolLM3Config,
    streaming_service: Arc<StreamingService>,
}

impl SmolLM3Model {
    pub async fn new(
        engine: Arc<InferenceEngine>,
        app_config: &Config,
    ) -> anyhow::Result<Self> {
        let config = SmolLM3Config::default();
        
        // Initialize tokenizer
        let tokenizer = if engine.is_stub() {
            SmolLM3Tokenizer::new_stub()
        } else {
            SmolLM3Tokenizer::from_file(&app_config.tokenizer_path)?
        };
        
        let streaming_service = Arc::new(StreamingService::new());
        
        Ok(Self {
            engine,
            tokenizer: Arc::new(Mutex::new(tokenizer)),
            config,
            streaming_service,
        })
    }
    
    pub async fn generate_response(
        &self,
        session_id: &str,
        message: &str,
    ) -> anyhow::Result<()> {
        tracing::info!("🤖 Generating response for session: {}", session_id);
        
        // Apply chat template
        let prompt = self.apply_chat_template(message).await?;
        
        // Tokenize
        let input_ids = self.tokenizer.lock().await.encode(&prompt)?;
        
        if self.engine.is_stub() {
            // Stub mode - send mock response
            self.generate_stub_response(session_id, message).await?;
        } else {
            // Real generation
            self.generate_with_model(session_id, input_ids).await?;
        }
        
        Ok(())
    }
    
    async fn apply_chat_template(&self, message: &str) -> anyhow::Result<String> {
        use crate::smollm3::chat_template::ChatTemplate;
        
        let template = ChatTemplate::new();
        Ok(template.format_single_turn(message, self.config.thinking_tokens.start_id))
    }
    
    async fn generate_stub_response(
        &self,
        session_id: &str,
        message: &str,
    ) -> anyhow::Result<()> {
        use crate::types::StreamEvent;
        
        // Send thinking events
        self.streaming_service.send_event(
            session_id,
            StreamEvent::thinking("Let me think about that...".to_string()),
        ).await;
        
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        
        // Send response
        let response = format!(
            "This is a stub response to: '{}'. The real SmolLM3 model is not loaded.",
            message
        );
        
        for word in response.split_whitespace() {
            self.streaming_service.send_event(
                session_id,
                StreamEvent::token(format!("{} ", word)),
            ).await;
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
        
        // Send completion
        self.streaming_service.send_event(
            session_id,
            StreamEvent::complete(),
        ).await;
        
        Ok(())
    }
    
    async fn generate_with_model(
        &self,
        session_id: &str,
        input_ids: Vec<u32>,
    ) -> anyhow::Result<()> {
        // TODO: Implement real generation with the model
        tracing::warn!("Real model generation not yet implemented");
        Ok(())
    }
}

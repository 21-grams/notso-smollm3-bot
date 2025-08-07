use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;
use axum::response::sse::Event;
use std::convert::Infallible;

// Phase 2: Import when ready
#[cfg(feature = "model")]
use {
    candle_core::Device,
    crate::{
        config::SmolLM3Config,
        inference::{InferenceEngine, GenerationEvent},
        tokenizer::ChatMessage,
    }
};

pub struct SmolLM3Service {
    // Phase 1: Stub mode
    is_stub: bool,
    
    // Phase 2: Real inference engine (when available)
    #[cfg(feature = "model")]
    engine: Option<InferenceEngine>,
    
    #[cfg(not(feature = "model"))]
    _phantom: std::marker::PhantomData<()>,
}

impl SmolLM3Service {
    /// Create a stub service for UI testing (Phase 1)
    pub fn new_stub() -> Result<Self> {
        tracing::info!("📦 Creating SmolLM3 stub service (no model loaded)");
        Ok(Self {
            is_stub: true,
            #[cfg(feature = "model")]
            engine: None,
            #[cfg(not(feature = "model"))]
            _phantom: std::marker::PhantomData,
        })
    }
    
    /// Load actual model (Phase 2)
    pub async fn new(_model_path: &str, _tokenizer_path: &str) -> Result<Self> {
        #[cfg(feature = "model")]
        {
            tracing::info!("🔧 Loading SmolLM3 model with official Candle");
            
            // Detect best available device
            let device = if candle_core::utils::cuda_is_available() {
                tracing::info!("🎮 CUDA available, using GPU");
                Device::new_cuda(0)?
            } else if candle_core::utils::metal_is_available() {
                tracing::info!("🎮 Metal available, using Apple GPU");
                Device::new_metal(0)?
            } else {
                tracing::info!("💻 Using CPU");
                Device::Cpu
            };
            
            // Load inference engine
            let engine = InferenceEngine::new(_model_path, _tokenizer_path, device).await?;
            
            return Ok(Self {
                is_stub: false,
                engine: Some(engine),
            });
        }
        
        #[cfg(not(feature = "model"))]
        {
            tracing::warn!("Model feature not enabled, using stub");
            Self::new_stub()
        }
    }
    
    /// Generate response with SSE streaming
    pub async fn generate_stream_sse(
        &mut self,
        prompt: &str,
        sender: UnboundedSender<Result<Event, Infallible>>,
        message_id: String,
        thinking_mode: bool,
    ) -> Result<()> {
        if self.is_stub {
            // Always use stub for now
            return self.generate_stub_response(prompt, sender, message_id, thinking_mode).await;
        }
        
        #[cfg(feature = "model")]
        if let Some(engine) = &mut self.engine {
            // Format prompt with chat template
            let messages = vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                    thinking_content: None,
                },
            ];
            
            let formatted_prompt = engine.tokenizer.apply_chat_template(
                &messages,
                true,  // add generation prompt
                thinking_mode,
            )?;
            
            // Generate with streaming
            let sender_clone = sender.clone();
            let message_id_clone = message_id.clone();
            
            engine.generate_stream(
                &formatted_prompt,
                512, // max_tokens
                move |event| {
                    match event {
                        GenerationEvent::Token(text) => {
                            let _ = sender_clone.send(Ok(Event::default()
                                .event(&format!("message-{}", message_id_clone))
                                .data(text)));
                        }
                        GenerationEvent::ThinkingStart => {
                            let _ = sender_clone.send(Ok(Event::default()
                                .event("thinking-start")
                                .data("")));
                        }
                        GenerationEvent::ThinkingToken(text) => {
                            let _ = sender_clone.send(Ok(Event::default()
                                .event(&format!("thinking-{}", message_id_clone))
                                .data(text)));
                        }
                        GenerationEvent::ThinkingEnd(_) => {
                            let _ = sender_clone.send(Ok(Event::default()
                                .event("thinking-end")
                                .data("")));
                        }
                        GenerationEvent::Complete(tokens_per_sec) => {
                            let _ = sender_clone.send(Ok(Event::default()
                                .event("complete")
                                .data(format!("{:.2} tok/s", tokens_per_sec))));
                        }
                    }
                },
            ).await?;
        }
        
        Ok(())
    }
    
    /// Stub generation for Phase 1 testing
    async fn generate_stub_response(
        &self,
        prompt: &str,
        sender: UnboundedSender<Result<Event, Infallible>>,
        message_id: String,
        thinking_mode: bool,
    ) -> Result<()> {
        tracing::info!("🤖 Stub generation for prompt: {}", prompt);
        
        // Simulate thinking mode if enabled
        if thinking_mode {
            let thinking_tokens = vec![
                "Let", " me", " think", " about", " this", "...",
                " The", " question", " seems", " to", " be", " about", "...",
            ];
            
            for token in thinking_tokens {
                let _ = sender.send(Ok(Event::default()
                    .event(&format!("thinking-{}", message_id))
                    .data(token)));
                
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }
            
            // Send thinking end
            let _ = sender.send(Ok(Event::default()
                .event("thinking-end")
                .data("")));
        }
        
        // Simulate main response
        let response_tokens = vec![
            "This", " is", " a", " stub", " response", ".",
            " When", " the", " actual", " SmolLM3", " model", " is",
            " loaded", ",", " you'll", " see", " real", " AI", "-generated",
            " responses", " using", " official", " Candle", " patterns", ".",
        ];
        
        for token in response_tokens {
            let _ = sender.send(Ok(Event::default()
                .event(&format!("message-{}", message_id))
                .data(token)));
            
            tokio::time::sleep(tokio::time::Duration::from_millis(80)).await;
        }
        
        // Send completion with fake metrics
        let _ = sender.send(Ok(Event::default()
            .event("complete")
            .data("12.5 tok/s (stub)")));
        
        Ok(())
    }
}

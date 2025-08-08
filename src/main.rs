use anyhow::Result;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod state;
mod web;
mod inference;
mod services;
mod types;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "notso_smollm3_bot=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("🚀 Starting NotSo-SmolLM3 Bot");

    // Load configuration
    let config = config::Config::from_env()?;
    
    // Initialize inference engine
    let inference_engine = Arc::new(
        inference::InferenceEngine::new(&config).await?
    );
    
    // Model and tokenizer paths
    let model_path = "models/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf";
    let tokenizer_path = "models/tokenizer.json";
    
    // Check if model files exist
    let use_real_model = std::path::Path::new(model_path).exists() 
        && std::path::Path::new(tokenizer_path).exists();
    
    // Initialize ML Service
    let ml_service = if use_real_model {
        tracing::info!("📦 Loading real SmolLM3 model from {}", model_path);
        tracing::info!("📝 Loading tokenizer from {}", tokenizer_path);
        
        // Try to load the real model, fall back to stub if it fails
        match services::ml::MLService::new(model_path, tokenizer_path).await {
            Ok(service) => {
                tracing::info!("✅ Successfully loaded SmolLM3 model");
                Arc::new(service)
            }
            Err(e) => {
                tracing::warn!("⚠️ Failed to load model: {}. Falling back to stub mode.", e);
                Arc::new(services::ml::MLService::new_stub().await?)
            }
        }
    } else {
        tracing::info!("🔌 Model files not found. Starting in stub mode.");
        tracing::info!("   Expected model: {}", model_path);
        tracing::info!("   Expected tokenizer: {}", tokenizer_path);
        Arc::new(services::ml::MLService::new_stub().await?)
    };
    
    // Create application state
    let app_state = state::AppState::new(ml_service, config);
    
    // Start web server
    tracing::info!("🌐 Starting web server on http://localhost:3000");
    web::start_server(app_state).await?;
    
    Ok(())
}

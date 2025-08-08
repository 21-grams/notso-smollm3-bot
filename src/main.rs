use anyhow::Result;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod state;
mod web;
mod inference;
// mod smollm3;  // Removed - functionality moved to services/ml/smollm3
mod services;
mod types;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "notso_smollm3_bot=debug,tower_http=debug".into()),
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
    
    // Initialize ML Service instead of SmolLM3 model directly
    // The ML service now handles all model operations
    let ml_service = Arc::new(
        services::ml::MLService::new_stub().await?  // Start with stub for testing
    );
    
    // Create application state
    let app_state = state::AppState::new(ml_service, config);
    
    // Start web server
    web::start_server(app_state).await?;
    
    Ok(())
}

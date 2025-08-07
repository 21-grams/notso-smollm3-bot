use anyhow::Result;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod state;
mod web;
mod inference;
mod smollm3;
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
    
    // Initialize SmolLM3 model
    let smollm3_model = Arc::new(
        smollm3::SmolLM3Model::new(inference_engine.clone(), &config).await?
    );
    
    // Create application state
    let app_state = state::AppState::new(smollm3_model, config);
    
    // Start web server
    web::start_server(app_state).await?;
    
    Ok(())
}

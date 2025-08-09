//! Minimal entry point for SmolLM3 Bot

use notso_smollm3_bot::{AppState, web};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    tracing::info!("🚀 Starting SmolLM3 Bot Server");
    
    // Create application state
    let state = AppState::new().await?;
    
    // Start server using lib components
    web::start_server(state).await?;
    
    Ok(())
}

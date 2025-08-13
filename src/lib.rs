// Core modules
pub mod config;
pub mod state;
pub mod services;
pub mod types;
pub mod web;

// Re-exports
pub use state::AppState;
pub use services::template::engine::TemplateEngine;
pub use services::ml::service::MLService;
pub use services::ml::enhanced_service::EnhancedMLService;

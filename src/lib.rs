// Core modules
pub mod config;
pub mod state;
pub mod services;
pub mod types;
pub mod web;
pub mod smollm3;
pub mod inference;

// Re-exports
pub use state::AppState;
pub use services::template::engine::TemplateEngine;
pub use services::ml::service::MLService;

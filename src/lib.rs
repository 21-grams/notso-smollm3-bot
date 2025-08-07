pub mod handlers;
pub mod model;
pub mod templates;

// New modules for Phase 2
pub mod config;
pub mod inference;
pub mod tokenizer;
pub mod cache;

pub use model::SmolLM3Service;
pub use templates::TemplateEngine;

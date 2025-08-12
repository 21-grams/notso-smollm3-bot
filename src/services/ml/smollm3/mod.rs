//! SmolLM3-specific extensions

mod adapter;
mod chat_template;  // Keep the module but don't export ChatMessage from here
mod thinking;
mod generation;
pub mod kv_cache;
mod nope_layers;
mod tokenizer;
mod stub_mode;
pub mod nope_model;

pub use adapter::SmolLM3Adapter;
pub use chat_template::ChatTemplate;  // Only export ChatTemplate, not ChatMessage
pub use tokenizer::{SmolLM3Tokenizer, SpecialTokens, ChatMessage};  // Export ChatMessage from tokenizer
pub use thinking::{ThinkingDetector, ThinkingEvent};
pub use generation::SmolLM3Generator;
pub use kv_cache::KVCache;
pub use nope_layers::NopeHandler;
pub use stub_mode::StubModeService;

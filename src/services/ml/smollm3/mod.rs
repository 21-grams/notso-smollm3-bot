//! SmolLM3-specific extensions

mod chat_template;
mod thinking;
mod generation;
pub mod kv_cache;
mod nope_layers;
mod tokenizer;
pub mod nope_model;

pub use chat_template::ChatTemplate;
pub use tokenizer::{SmolLM3Tokenizer, SpecialTokens, ChatMessage};
pub use thinking::{ThinkingDetector, ThinkingEvent};
pub use generation::SmolLM3Generator;
pub use kv_cache::KVCache;
pub use nope_layers::NopeHandler;

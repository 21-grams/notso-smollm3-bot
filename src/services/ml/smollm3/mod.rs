//! SmolLM3-specific extensions

mod adapter;
mod chat_template;
mod thinking;
mod generation;
mod kv_cache;
mod nope_layers;
mod tokenizer_ext;
mod stub_mode;

pub use adapter::SmolLM3Adapter;
pub use chat_template::{ChatTemplate, ChatMessage};
pub use thinking::{ThinkingDetector, ThinkingEvent};
pub use generation::SmolLM3Generator;
pub use kv_cache::KVCache;
pub use kv_cache::KVCache as SmolLM3KVCache; // Alias for compatibility
pub use nope_layers::NopeHandler;
pub use tokenizer_ext::{SmolLM3TokenizerExt, ReasoningMode};
pub use tokenizer_ext::SmolLM3TokenizerExt as SmolLM3Tokenizer; // Alias for compatibility
pub use stub_mode::StubModeService;

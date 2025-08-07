//! SmolLM3-specific extensions

mod adapter;
mod thinking;
mod generation;
mod kv_cache;
mod nope_layers;
mod tokenizer_ext;
mod stub_mode;

pub use adapter::SmolLM3Adapter;
pub use thinking::{ThinkingDetector, ThinkingEvent};
pub use generation::SmolLM3Generator;
pub use kv_cache::KVCache;
pub use nope_layers::NopeHandler;
pub use tokenizer_ext::{SmolLM3TokenizerExt, ChatMessage, ReasoningMode};
pub use stub_mode::StubModeService;

pub mod config;
pub mod device;
pub mod gguf_loader;
pub mod loader;
pub mod model;
pub mod quantized_model;

use serde::{Deserialize, Serialize};

pub use config::SmolLM3Config;
pub use device::get_device;
pub use gguf_loader::{load_smollm3_gguf, inspect_gguf};
pub use loader::OfficialLoader;
pub use model::{SmolLM3Model, OfficialSmolLM3Model};

/// Thinking mode token IDs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingTokens {
    pub start: u32,
    pub end: u32,
}

impl ThinkingTokens {
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }
    
    pub fn default_smollm3() -> Self {
        Self {
            start: 128002,  // <think> token
            end: 128003,     // </think> token
        }
    }
}

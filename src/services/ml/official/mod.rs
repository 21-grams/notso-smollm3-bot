//! Official Candle foundation components

mod model;
mod config;
mod loader;
mod device;

pub use model::OfficialSmolLM3Model;
pub use config::{SmolLM3Config, ThinkingTokens, ReasoningMode, ToolCallingConfig};
pub use loader::OfficialLoader;
pub use device::DeviceManager;

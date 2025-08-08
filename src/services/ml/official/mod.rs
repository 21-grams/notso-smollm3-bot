//! Official Candle foundation components

mod model;
mod config;
mod loader;
mod device;
mod gguf_loader;
mod quantized_model;

pub use model::OfficialSmolLM3Model;
pub use config::{SmolLM3Config, ThinkingTokens, ReasoningMode, ToolCallingConfig};
pub use loader::OfficialLoader;
pub use device::DeviceManager;
pub use gguf_loader::{SmolLM3GgufLoader, try_load_as_quantized_llama, inspect_gguf, GgufInspectionReport};
pub use quantized_model::QuantizedSmolLM3;

//! Streaming components for real-time generation

mod events;
mod buffer;
mod pipeline;

pub use events::GenerationEvent;
pub use buffer::TokenBuffer;
pub use pipeline::StreamingPipeline;

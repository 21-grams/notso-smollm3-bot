//! Unified streaming services

pub mod buffer;
pub mod sse_handler;
// service.rs is deprecated - functionality moved to SessionManager

pub use buffer::StreamingBuffer;

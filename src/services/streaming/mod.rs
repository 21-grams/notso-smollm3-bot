//! Streaming module for smooth token delivery

mod buffer;
mod sse_handler;

pub use buffer::{
    StreamBufferConfig,
    TokenStreamBuffer,
    HtmxStreamProcessor,
    WordBuffer,
};

pub use sse_handler::{
    stream_smooth_sse,
    stream_time_based,
};

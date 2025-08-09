mod session;
mod streaming_service;
pub mod streaming;  // Make public for ML modules to use
mod metrics;
pub mod ml;
pub mod template;  // Make template public for use in other modules

pub use session::{SessionManager, Session};
pub use streaming_service::StreamingService;
pub use streaming::{StreamingBuffer};  // Export for convenience
pub use metrics::{MetricsService, MetricsStats};
pub use ml::MLService;

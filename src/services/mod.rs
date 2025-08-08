mod session;
mod streaming;
mod metrics;
pub mod ml;
pub mod template;  // Make template public for use in other modules

pub use session::{SessionManager, Session};
pub use streaming::StreamingService;
pub use metrics::{MetricsService, MetricsStats};
pub use ml::MLService;

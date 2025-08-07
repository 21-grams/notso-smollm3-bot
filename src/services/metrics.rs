//! Metrics service for performance monitoring

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

pub struct MetricsService {
    tokens_generated: AtomicUsize,
    requests_processed: AtomicUsize,
    total_latency_ms: AtomicU64,
    start_time: Instant,
}

impl MetricsService {
    pub fn new() -> Self {
        Self {
            tokens_generated: AtomicUsize::new(0),
            requests_processed: AtomicUsize::new(0),
            total_latency_ms: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }
    
    pub fn record_tokens(&self, count: usize) {
        self.tokens_generated.fetch_add(count, Ordering::Relaxed);
    }
    
    pub fn record_request(&self, latency_ms: u64) {
        self.requests_processed.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ms.fetch_add(latency_ms, Ordering::Relaxed);
    }
    
    pub fn get_stats(&self) -> MetricsStats {
        let tokens = self.tokens_generated.load(Ordering::Relaxed);
        let requests = self.requests_processed.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ms.load(Ordering::Relaxed);
        let uptime_secs = self.start_time.elapsed().as_secs();
        
        MetricsStats {
            tokens_generated: tokens,
            requests_processed: requests,
            avg_latency_ms: if requests > 0 { total_latency / requests as u64 } else { 0 },
            tokens_per_second: if uptime_secs > 0 { tokens as f64 / uptime_secs as f64 } else { 0.0 },
            uptime_seconds: uptime_secs,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetricsStats {
    pub tokens_generated: usize,
    pub requests_processed: usize,
    pub avg_latency_ms: u64,
    pub tokens_per_second: f64,
    pub uptime_seconds: u64,
}

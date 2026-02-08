//! ALICE-CDN × ALICE-Analytics Bridge
//!
//! Streaming metrics for CDN delivery: unique clients (HLL), latency quantiles (DDSketch),
//! request frequency (CMS), anomaly detection.

use alice_analytics::prelude::*;

/// CDN delivery metrics collector.
pub struct CdnMetrics {
    /// Unique client estimation (HyperLogLog++).
    pub unique_clients: HyperLogLog,
    /// Latency quantile estimation (DDSketch, 1% relative error).
    pub latency_sketch: DDSketch,
    /// Request frequency per content-id (Count-Min Sketch).
    pub request_freq: CountMinSketch,
    /// Anomaly detection on latency.
    pub anomaly: MadDetector,
    /// Total requests counter.
    pub total_requests: u64,
}

impl CdnMetrics {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            unique_clients: HyperLogLog::new(),
            latency_sketch: DDSketch::new(0.01),
            request_freq: CountMinSketch::new(),
            anomaly: MadDetector::new(3.0),
            total_requests: 0,
        }
    }

    /// Record a content delivery event.
    pub fn record_delivery(&mut self, client_id: u64, content_id: u64, rtt_ms: f64) {
        self.unique_clients.insert(&client_id);
        self.latency_sketch.insert(rtt_ms);
        self.request_freq.insert(&content_id);
        self.anomaly.observe(rtt_ms);
        self.total_requests += 1;
    }

    /// Estimated unique client count.
    pub fn unique_client_count(&self) -> f64 {
        self.unique_clients.cardinality()
    }

    /// P99 latency estimation.
    pub fn p99_latency(&self) -> f64 {
        self.latency_sketch.quantile(0.99)
    }

    /// P50 (median) latency estimation.
    pub fn p50_latency(&self) -> f64 {
        self.latency_sketch.quantile(0.50)
    }

    /// Estimated request count for a content-id.
    pub fn content_frequency(&self, content_id: &u64) -> u64 {
        self.request_freq.estimate(content_id)
    }

    /// Check if a latency value is anomalous.
    pub fn is_latency_anomaly(&mut self, rtt_ms: f64) -> bool {
        self.anomaly.is_anomaly(rtt_ms)
    }

    /// Merge metrics from another node (distributed aggregation).
    pub fn merge(&mut self, other: &CdnMetrics) {
        self.unique_clients.merge(&other.unique_clients);
        self.latency_sketch.merge(&other.latency_sketch);
        self.request_freq.merge(&other.request_freq);
        self.total_requests += other.total_requests;
    }
}

impl Default for CdnMetrics {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdn_metrics() {
        let mut m = CdnMetrics::new();
        for i in 0..100 {
            m.record_delivery(i % 10, i % 5, 10.0 + (i as f64) * 0.1);
        }
        assert!(m.unique_client_count() > 5.0);
        assert!(m.p50_latency() > 0.0);
        assert!(m.p99_latency() > m.p50_latency());
        assert_eq!(m.total_requests, 100);
    }
}

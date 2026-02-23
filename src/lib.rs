//! ALICE-CDN: Decentralized Latency-Optimized Content Delivery - Scorched Earth Edition
//!
//! A high-performance CDN library implementing:
//! - **Vivaldi Network Coordinates**: Fused with SimdCoord, no conversion overhead
//! - **SIMD-Accelerated Math**: 32-byte aligned, integer-only operations
//! - **Spatial Indexing**: Compressed Octree with u32 indices (4x smaller)
//! - **MumHash**: Inline multiply-based hashing (faster than FNV-1a)
//! - **Maglev Consistent Hashing**: Zero-allocation O(1) lookup
//!
//! # Design Philosophy
//!
//! > "The best cache is the one closest to you."
//!
//! Traditional CDNs use centralized coordination for content placement.
//! ALICE-CDN enables fully decentralized, latency-optimized content delivery
//! without requiring a coordination server.
//!
//! # Core Components
//!
//! ## Vivaldi Network Coordinates
//!
//! Each node maintains a position in virtual 3D space + height that predicts
//! RTT to other nodes without direct measurement:
//!
//! ```text
//! RTT(A, B) ≈ ||pos_A - pos_B|| + height_A + height_B
//! ```
//!
//! ## Latency-Aware Rendezvous Hashing
//!
//! Combines consistent hashing with latency optimization:
//!
//! ```text
//! Score(Node) = hash_weight * Hash(Key, Node) + distance_weight / RTT(Client, Node)
//! ```
//!
//! ## Maglev Consistent Hashing
//!
//! Google's Maglev algorithm provides O(1) node lookup with even distribution
//! and minimal key redistribution when nodes change.
//!
//! # Example
//!
//! ```rust
//! use alice_cdn::{VivaldiCoord, ContentLocator, MaglevHash};
//!
//! // Create Vivaldi coordinates for nodes
//! let local = VivaldiCoord::new();
//! let node1 = VivaldiCoord::at(10.0, 0.0, 0.0, 5.0);
//! let node2 = VivaldiCoord::at(0.0, 20.0, 0.0, 3.0);
//!
//! // Predict RTT without direct measurement
//! let rtt = local.predict_rtt(&node1);
//!
//! // Create latency-aware content locator
//! let locator = ContentLocator::new(local);
//! let nodes = vec![(1u64, &node1), (2u64, &node2)];
//! let best = locator.find_best(12345, nodes);
//!
//! // Or use Maglev for pure O(1) consistent hashing
//! let maglev = MaglevHash::new(vec![1, 2, 3, 4, 5]);
//! let assigned_node = maglev.lookup(12345);
//! ```
//!
//! # Optimized Edition Features
//!
//! - **SIMD Coordinates**: 32-byte aligned, integer-only math
//! - **Spatial Index**: Octree with O(log N) nearest neighbor search
//! - **Zero-Allocation Maglev**: No N*M permutation table, O(M) memory
//! - **Batch Operations**: Cache-friendly bulk distance calculations
//!
//! # Performance Characteristics
//!
//! | Operation | Complexity | Notes |
//! |-----------|------------|-------|
//! | Vivaldi RTT prediction | O(1) | Integer-only math |
//! | Vivaldi coordinate update | O(1) | Spring model adjustment |
//! | Rendezvous find_best (linear) | O(n) | n = number of candidates |
//! | Rendezvous find_best (indexed) | O(log n + k) | k = candidate limit |
//! | Spatial nearest-k | O(log n + k) | Octree traversal |
//! | Maglev lookup | O(1) | Table lookup |
//! | Maglev rebuild | O(m) | m = table size |

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "analytics")]
pub mod analytics_bridge;
#[cfg(feature = "asp")]
pub mod asp_bridge;
#[cfg(feature = "cache")]
pub mod cache_bridge;
#[cfg(feature = "content_types")]
pub mod content_types;
#[cfg(feature = "crypto")]
pub mod crypto_bridge;
pub mod locator;
pub mod maglev;
#[cfg(feature = "sdf")]
pub mod sdf_cdn_bridge;
pub mod simd;
pub mod spatial;
pub mod vivaldi;

// Re-export main types
pub use locator::{ContentId, ContentLocator, IndexedLocator, NodeId, RendezvousHash, ScoredNode};
pub use maglev::{
    DistributionStats, MaglevHash, StaticMaglev, WeightedMaglev, DEFAULT_TABLE_SIZE,
    SMALL_TABLE_SIZE,
};
pub use simd::{batch_distances, find_nearest, isqrt, SimdCoord};
pub use spatial::{SpatialEntry, SpatialIndex, AABB};
pub use vivaldi::{Fixed, VivaldiCoord, VivaldiSystem};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::locator::{
        ContentId, ContentLocator, IndexedLocator, NodeId, RendezvousHash, ScoredNode,
    };
    pub use crate::maglev::{MaglevHash, StaticMaglev, WeightedMaglev};
    pub use crate::simd::{isqrt, SimdCoord};
    pub use crate::spatial::{SpatialEntry, SpatialIndex};
    pub use crate::vivaldi::{Fixed, VivaldiCoord, VivaldiSystem};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_vivaldi_locator() {
        // Setup: Local node and remote nodes with coordinates
        let local = VivaldiCoord::new();
        let node1_coord = VivaldiCoord::at(10.0, 0.0, 0.0, 5.0);
        let node2_coord = VivaldiCoord::at(50.0, 50.0, 0.0, 10.0);
        let node3_coord = VivaldiCoord::at(5.0, 5.0, 0.0, 2.0);

        // Create locator with strong distance preference
        let locator = ContentLocator::with_weights(local, 0.2, 0.8);

        // Find best node for content
        let nodes = vec![
            (1u64, &node1_coord),
            (2u64, &node2_coord),
            (3u64, &node3_coord),
        ];

        let best = locator.find_best(12345, nodes).unwrap();

        // With strong distance preference, node3 should often win (closest)
        assert!(best.id >= 1 && best.id <= 3);
        assert!(best.predicted_rtt.to_f64() > 0.0);
    }

    #[test]
    fn test_integration_maglev_consistency() {
        // Setup Maglev with some nodes
        let nodes: Vec<u64> = (1..=5).collect();
        let maglev = MaglevHash::new(nodes);

        // Verify consistent mapping
        let key = 12345u64;
        let node1 = maglev.lookup(key);
        let node2 = maglev.lookup(key);
        assert_eq!(node1, node2);

        // Check distribution stats
        let stats = maglev.distribution_stats();
        assert_eq!(stats.node_counts.len(), 5);
        assert!(stats.std_dev < 1000.0); // Reasonable distribution
    }

    #[test]
    fn test_integration_combined_routing() {
        // Scenario: Use Maglev for primary assignment, Vivaldi for replica selection

        // Primary assignment via Maglev
        let nodes: Vec<u64> = (1..=10).collect();
        let maglev = MaglevHash::new(nodes.clone());

        let content_id = 999u64;
        let _primary = maglev.lookup(content_id).unwrap();

        // Get replicas via Rendezvous
        let replicas = RendezvousHash::find_replicas(content_id, nodes.iter().copied(), 3);

        assert_eq!(replicas.len(), 3);

        // If we need to find closest replica, use Vivaldi
        let local = VivaldiCoord::new();
        let replica_coords: Vec<(u64, VivaldiCoord)> = replicas
            .iter()
            .map(|&id| (id, VivaldiCoord::at(id as f64 * 5.0, 0.0, 0.0, 2.0)))
            .collect();

        let locator = ContentLocator::new(local);
        let refs: Vec<_> = replica_coords.iter().map(|(id, c)| (*id, c)).collect();
        let (closest, rtt) = locator.find_closest(refs).unwrap();

        assert!(replicas.contains(&closest));
        assert!(rtt.to_f64() > 0.0);
    }

    #[test]
    fn test_weighted_maglev_distribution() {
        // Node 1 has weight 5, node 2 has weight 1
        let weighted = WeightedMaglev::new(vec![(1, 5), (2, 1)]);

        let mut count1 = 0;
        let mut count2 = 0;

        for key in 0..6000 {
            match weighted.lookup(key) {
                Some(1) => count1 += 1,
                Some(2) => count2 += 1,
                _ => {}
            }
        }

        // Node 1 should get roughly 5x traffic
        let ratio = count1 as f64 / count2 as f64;
        assert!(ratio > 3.0 && ratio < 7.0, "Ratio: {}", ratio);
    }

    #[test]
    fn test_simd_coord_integration() {
        let local = SimdCoord::from_f64(0.0, 0.0, 0.0, 5.0);
        let remote = SimdCoord::from_f64(10.0, 0.0, 0.0, 5.0);

        // Distance should be ~10
        let dist = local.distance(&remote) as f64 / (1 << 16) as f64;
        assert!((dist - 10.0).abs() < 1.0, "Distance: {}", dist);

        // RTT = 10 + 5 + 5 = 20
        let rtt = local.predict_rtt_ms(&remote);
        assert!((rtt - 20.0).abs() < 2.0, "RTT: {}", rtt);
    }

    #[test]
    fn test_spatial_index_integration() {
        let entries: Vec<_> = (1..=100)
            .map(|i| SpatialEntry {
                node_id: i as u64,
                coord: SimdCoord::from_f64(i as f64, 0.0, 0.0, 1.0),
            })
            .collect();

        let index = SpatialIndex::build(entries);
        assert_eq!(index.len(), 100);

        let query = SimdCoord::from_f64(50.0, 0.0, 0.0, 1.0);
        let nearest = index.find_nearest_k(&query, 5);

        assert_eq!(nearest.len(), 5);
        // Should find nodes near 50
        for (entry, _dist) in &nearest {
            assert!(entry.node_id >= 45 && entry.node_id <= 55);
        }
    }

    #[test]
    fn test_indexed_locator_integration() {
        let local = SimdCoord::from_f64(50.0, 50.0, 0.0, 1.0);
        let nodes: Vec<_> = (1..=1000)
            .map(|i| {
                let x = (i % 100) as f64;
                let y = (i / 10) as f64;
                (i as NodeId, SimdCoord::from_f64(x, y, 0.0, 1.0))
            })
            .collect();

        let locator = IndexedLocator::build(local, nodes, 0.3, 0.7);

        // Find best with limited candidates
        let result = locator.find_best(12345, 20);
        assert!(result.is_some());

        // Find top-3
        let top3 = locator.find_top_k(12345, 3, 30);
        assert_eq!(top3.len(), 3);
    }

    #[test]
    fn test_static_maglev() {
        let nodes: Vec<_> = (1..=10).collect();
        let maglev = StaticMaglev::build(&nodes, 1009);

        // Consistent lookup
        let idx1 = maglev.lookup(12345);
        let idx2 = maglev.lookup(12345);
        assert_eq!(idx1, idx2);
        assert!(idx1 < 10);
    }
}

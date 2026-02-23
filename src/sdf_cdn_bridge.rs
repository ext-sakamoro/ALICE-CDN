//! SDF-Aware CDN Routing Bridge
//!
//! Routes SDF content requests using spatial cell addressing.
//! Maps Morton code spatial cells to edge nodes via Maglev consistent hashing,
//! ensuring nearby spatial cells are served by the same edge nodes for cache
//! locality.
//!
//! Author: Moroya Sakamoto

use crate::locator::{ContentId, ContentLocator, NodeId};
use crate::maglev::MaglevHash;
use crate::vivaldi::VivaldiCoord;

/// Spatial cell identifier (from ALICE-DB Morton code)
pub type SpatialCell = u32;

/// SDF CDN router
///
/// Routes SDF requests to optimal edge nodes using:
/// 1. Maglev consistent hashing for primary assignment (O(1))
/// 2. Vivaldi coordinates for latency-optimized replica selection
pub struct SdfCdnRouter {
    /// Primary assignment via Maglev
    maglev: MaglevHash,
    /// Latency-aware locator for replica selection
    locator: ContentLocator,
    /// Number of replicas per spatial cell
    replica_count: usize,
}

impl SdfCdnRouter {
    /// Create a new SDF CDN router
    ///
    /// # Arguments
    /// * `edge_nodes` - List of edge node IDs
    /// * `local_coord` - Vivaldi coordinate of the requesting client/gateway
    /// * `replica_count` - Number of replicas per spatial cell
    pub fn new(edge_nodes: Vec<NodeId>, local_coord: VivaldiCoord, replica_count: usize) -> Self {
        let maglev = MaglevHash::new(edge_nodes);
        let locator = ContentLocator::new(local_coord);

        Self {
            maglev,
            locator,
            replica_count,
        }
    }

    /// Route an SDF request for a spatial cell to the best edge node
    ///
    /// Returns the primary node for this spatial cell (via Maglev).
    pub fn route_sdf_request(&self, spatial_cell: SpatialCell) -> Option<NodeId> {
        self.maglev.lookup(spatial_cell as u64)
    }

    /// Route with latency awareness
    ///
    /// Returns the best edge node considering both consistent hashing
    /// and network latency via Vivaldi coordinates.
    pub fn route_sdf_request_latency_aware(
        &self,
        spatial_cell: SpatialCell,
        node_coords: &[(NodeId, VivaldiCoord)],
    ) -> Option<NodeId> {
        if node_coords.is_empty() {
            return self.route_sdf_request(spatial_cell);
        }

        let refs: Vec<(NodeId, &VivaldiCoord)> =
            node_coords.iter().map(|(id, coord)| (*id, coord)).collect();

        self.locator
            .find_best(spatial_cell as ContentId, refs)
            .map(|scored| scored.id)
    }

    /// Find replica nodes for a spatial cell
    ///
    /// Returns up to `replica_count` nodes that should cache this cell's data.
    pub fn find_replicas(
        &self,
        spatial_cell: SpatialCell,
        node_coords: &[(NodeId, VivaldiCoord)],
    ) -> Vec<NodeId> {
        let refs: Vec<(NodeId, &VivaldiCoord)> =
            node_coords.iter().map(|(id, coord)| (*id, coord)).collect();

        self.locator
            .find_top_k(spatial_cell as ContentId, refs, self.replica_count)
            .into_iter()
            .map(|scored| scored.id)
            .collect()
    }

    /// Route a batch of spatial cells (e.g., for a spatial region query)
    ///
    /// Groups cells by their assigned edge node for efficient batching.
    pub fn route_batch(&self, cells: &[SpatialCell]) -> Vec<(NodeId, Vec<SpatialCell>)> {
        let mut groups: std::collections::HashMap<NodeId, Vec<SpatialCell>> =
            std::collections::HashMap::with_capacity(cells.len().min(32));

        for &cell in cells {
            if let Some(node) = self.route_sdf_request(cell) {
                groups
                    .entry(node)
                    .or_insert_with(|| Vec::with_capacity(cells.len() / 4))
                    .push(cell);
            }
        }

        groups.into_iter().collect()
    }
}

/// Routing statistics
#[derive(Debug, Clone, Default)]
pub struct SdfRoutingStats {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub avg_latency_ms: f64,
}

impl SdfRoutingStats {
    pub fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / self.total_requests as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdf_router_basic() {
        let nodes: Vec<NodeId> = (1..=5).collect();
        let local = VivaldiCoord::new();
        let router = SdfCdnRouter::new(nodes, local, 3);

        let node = router.route_sdf_request(12345);
        assert!(node.is_some());

        // Consistent routing
        let node2 = router.route_sdf_request(12345);
        assert_eq!(node, node2);
    }

    #[test]
    fn test_sdf_router_different_cells() {
        let nodes: Vec<NodeId> = (1..=10).collect();
        let local = VivaldiCoord::new();
        let router = SdfCdnRouter::new(nodes, local, 2);

        // Different cells may route to different nodes
        let node_a = router.route_sdf_request(100);
        let node_b = router.route_sdf_request(999999);
        // Both should be valid
        assert!(node_a.is_some());
        assert!(node_b.is_some());
    }

    #[test]
    fn test_route_batch_grouping() {
        let nodes: Vec<NodeId> = (1..=3).collect();
        let local = VivaldiCoord::new();
        let router = SdfCdnRouter::new(nodes, local, 2);

        let cells: Vec<SpatialCell> = (0..100).collect();
        let groups = router.route_batch(&cells);

        // All cells should be assigned
        let total_assigned: usize = groups.iter().map(|(_, cells)| cells.len()).sum();
        assert_eq!(total_assigned, 100);

        // Should have up to 3 groups (3 nodes)
        assert!(groups.len() <= 3);
    }

    #[test]
    fn test_routing_stats() {
        let stats = SdfRoutingStats {
            total_requests: 100,
            cache_hits: 85,
            cache_misses: 15,
            avg_latency_ms: 5.2,
        };
        assert!((stats.hit_rate() - 0.85).abs() < 0.01);
    }
}

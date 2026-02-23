//! ALICE-Streaming-Protocol bridge: ASP stream CDN routing
//!
//! Routes ASP video stream packets to optimal CDN edge nodes using
//! Vivaldi network coordinates for latency prediction and Maglev
//! consistent hashing for even load distribution.
//!
//! # Pipeline
//!
//! ```text
//! ASP stream_id → MaglevHash → primary edge node
//! ASP stream_id → Rendezvous + Vivaldi → closest replica
//! ```

use libasp::AspPacket;

use crate::{ContentLocator, MaglevHash, NodeId, RendezvousHash, VivaldiCoord};

/// ASP stream routing via CDN topology.
pub struct AspStreamRouter {
    maglev: MaglevHash,
    locator: ContentLocator,
}

impl AspStreamRouter {
    /// Create a new stream router with the given edge nodes and local coordinate.
    pub fn new(node_ids: Vec<NodeId>, local_coord: VivaldiCoord) -> Self {
        let maglev = MaglevHash::new(node_ids);
        let locator = ContentLocator::with_weights(local_coord, 0.3, 0.7);
        Self { maglev, locator }
    }

    /// Find the primary edge node for a stream using Maglev consistent hashing.
    ///
    /// Returns `None` if no nodes are available.
    pub fn primary_node(&self, stream_id: u64) -> Option<NodeId> {
        self.maglev.lookup(stream_id)
    }

    /// Find replica nodes for a stream using Rendezvous hashing.
    pub fn replica_nodes(&self, stream_id: u64, count: usize, all_nodes: &[NodeId]) -> Vec<NodeId> {
        RendezvousHash::find_replicas(stream_id, all_nodes.iter().copied(), count)
    }

    /// Find the closest replica node using Vivaldi coordinates.
    pub fn closest_replica(&self, replicas: &[(NodeId, VivaldiCoord)]) -> Option<NodeId> {
        let refs: Vec<_> = replicas.iter().map(|(id, c)| (*id, c)).collect();
        self.locator.find_closest(refs).map(|(id, _rtt)| id)
    }

    /// Route an ASP packet: returns (primary_node, closest_replica).
    ///
    /// Uses the packet's sequence number as the stream identifier.
    pub fn route_packet(
        &self,
        packet: &AspPacket,
        replicas_with_coords: &[(NodeId, VivaldiCoord)],
    ) -> (Option<NodeId>, Option<NodeId>) {
        let stream_id = packet.sequence() as u64;
        let primary = self.primary_node(stream_id);
        let closest = self.closest_replica(replicas_with_coords);
        (primary, closest)
    }
}

/// Estimate the CDN delivery overhead for an ASP packet.
///
/// Returns `(packet_bytes, estimated_rtt_ms)` for the closest node.
pub fn estimate_delivery(
    packet: &AspPacket,
    local: &VivaldiCoord,
    target: &VivaldiCoord,
) -> (usize, f64) {
    let bytes = packet.to_bytes().map(|b| b.len()).unwrap_or(0);
    let rtt = local.predict_rtt(target).to_f64();
    (bytes, rtt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_router_primary() {
        let nodes: Vec<NodeId> = (1..=5).collect();
        let local = VivaldiCoord::new();
        let router = AspStreamRouter::new(nodes, local);

        let primary = router.primary_node(12345);
        assert!(primary.is_some());

        // Consistent mapping
        let primary2 = router.primary_node(12345);
        assert_eq!(primary, primary2);
    }
}

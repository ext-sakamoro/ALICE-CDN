//! Latency-Aware Rendezvous Hashing (Scorched Earth Edition)
//!
//! **Algorithm**: Weighted Rendezvous Hashing combining hash score with Vivaldi distance.
//!
//! **Scorched Earth Optimizations**:
//! - **`MumHash`**: Inline multiply-based hash (faster than FNV-1a)
//! - **Fused `VivaldiCoord`**: Direct `SimdCoord` access, no conversion
//! - **Spatial Index**: O(log N) pre-filtering via Octree
//!
//! Score(Node) = Hash(Key, `NodeID`) / Distance(Client, Node)
//!
//! > "The best cache is the one closest to you."

use crate::simd::SimdCoord;
use crate::spatial::{SpatialEntry, SpatialIndex};
use crate::vivaldi::{Fixed, VivaldiCoord};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Node identifier
pub type NodeId = u64;

/// Content/Key identifier
pub type ContentId = u64;

/// MumHash-style fast inline hash
///
/// Based on multiply-and-mix technique (WyHash/MumHash family)
/// Faster than FNV-1a due to fewer operations and better pipelining
#[inline(always)]
const fn mum_hash(key: ContentId, node: NodeId) -> u64 {
    const K0: u64 = 0x517c_c1b7_2722_0a95;
    const K1: u64 = 0x9e37_79b9_7f4a_7c15;

    // Mix key and node with multiplies
    let a = key.wrapping_mul(K0);
    let b = node.wrapping_mul(K1);

    // Final mix
    let c = a ^ b;
    c.wrapping_mul(K0).wrapping_add(c.rotate_right(23))
}

/// Legacy FNV-1a hash (kept for compatibility)
#[inline]
const fn fnv1a_hash(key: ContentId, node: NodeId) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET;

    // Unrolled for key
    let kb = key.to_le_bytes();
    hash ^= kb[0] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= kb[1] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= kb[2] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= kb[3] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= kb[4] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= kb[5] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= kb[6] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= kb[7] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);

    // Unrolled for node
    let nb = node.to_le_bytes();
    hash ^= nb[0] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= nb[1] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= nb[2] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= nb[3] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= nb[4] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= nb[5] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= nb[6] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= nb[7] as u64;
    hash = hash.wrapping_mul(FNV_PRIME);

    hash
}

/// Candidate node with its score
#[derive(Clone, Debug)]
pub struct ScoredNode {
    /// Node identifier
    pub id: NodeId,
    /// Combined score (higher is better)
    pub score: Fixed,
    /// Predicted RTT to this node
    pub predicted_rtt: Fixed,
    /// Raw hash score (for debugging)
    pub hash_score: u64,
}

/// Locator for finding optimal nodes for content
pub struct ContentLocator {
    /// Local node's Vivaldi coordinate (fused with `SimdCoord`)
    local_coord: VivaldiCoord,
    /// Weight for hash component (0.0 - 1.0)
    hash_weight: Fixed,
    /// Weight for distance component (0.0 - 1.0)
    distance_weight: Fixed,
}

impl ContentLocator {
    /// Create new locator with default weights
    #[must_use]
    pub fn new(local_coord: VivaldiCoord) -> Self {
        Self {
            local_coord,
            hash_weight: Fixed::from_f64(0.5),
            distance_weight: Fixed::from_f64(0.5),
        }
    }

    /// Create locator with custom weights
    #[must_use]
    pub fn with_weights(local_coord: VivaldiCoord, hash_weight: f64, distance_weight: f64) -> Self {
        Self {
            local_coord,
            hash_weight: Fixed::from_f64(hash_weight),
            distance_weight: Fixed::from_f64(distance_weight),
        }
    }

    /// Update local coordinate (after Vivaldi update)
    pub const fn update_coord(&mut self, coord: VivaldiCoord) {
        self.local_coord = coord;
    }

    /// Calculate score for a single node using `MumHash`
    #[inline]
    #[must_use]
    pub fn score_node(
        &self,
        content_id: ContentId,
        node_id: NodeId,
        node_coord: &VivaldiCoord,
    ) -> ScoredNode {
        // Hash component using fast MumHash
        // Normalize to 0.0-1.0 range (hash >> 32 gives 0 to 2^32-1)
        let hash = mum_hash(content_id, node_id);
        let hash_normalized = Fixed(((hash >> 32) as i64 * Fixed::ONE.0) >> 32);

        // Distance component via fused SimdCoord
        let predicted_rtt = self.local_coord.predict_rtt(node_coord);
        let min_rtt = Fixed::from_f64(1.0);
        let distance_score = Fixed::ONE / predicted_rtt.max(min_rtt);

        // Combined score
        let score = self.hash_weight * hash_normalized + self.distance_weight * distance_score;

        ScoredNode {
            id: node_id,
            score,
            predicted_rtt,
            hash_score: hash,
        }
    }

    /// Find best node from candidates.
    ///
    /// Returns `None` if `candidates` is empty.
    #[must_use]
    pub fn find_best<'a, I>(&self, content_id: ContentId, candidates: I) -> Option<ScoredNode>
    where
        I: IntoIterator<Item = (NodeId, &'a VivaldiCoord)>,
    {
        candidates
            .into_iter()
            .map(|(id, coord)| self.score_node(content_id, id, coord))
            .max_by(|a, b| a.score.cmp(&b.score))
    }

    /// Find top-k best nodes from candidates.
    #[must_use]
    pub fn find_top_k<'a, I>(
        &self,
        content_id: ContentId,
        candidates: I,
        k: usize,
    ) -> Vec<ScoredNode>
    where
        I: IntoIterator<Item = (NodeId, &'a VivaldiCoord)>,
    {
        let mut scored: Vec<_> = candidates
            .into_iter()
            .map(|(id, coord)| self.score_node(content_id, id, coord))
            .collect();

        scored.sort_by(|a, b| b.score.cmp(&a.score));
        scored.truncate(k);
        scored
    }

    /// Find closest node (pure distance, no hash).
    ///
    /// Returns `None` if `candidates` is empty.
    #[must_use]
    pub fn find_closest<'a, I>(&self, candidates: I) -> Option<(NodeId, Fixed)>
    where
        I: IntoIterator<Item = (NodeId, &'a VivaldiCoord)>,
    {
        candidates
            .into_iter()
            .map(|(id, coord)| (id, self.local_coord.predict_rtt(coord)))
            .min_by(|a, b| a.1.cmp(&b.1))
    }

    /// Rank nodes by latency only.
    #[must_use]
    pub fn rank_by_latency<'a, I>(&self, candidates: I) -> Vec<(NodeId, Fixed)>
    where
        I: IntoIterator<Item = (NodeId, &'a VivaldiCoord)>,
    {
        let mut ranked: Vec<_> = candidates
            .into_iter()
            .map(|(id, coord)| (id, self.local_coord.predict_rtt(coord)))
            .collect();

        ranked.sort_by(|a, b| a.1.cmp(&b.1));
        ranked
    }

    /// Score a node with content-type-aware weight adjustment.
    ///
    /// For latency-sensitive types (ASDF, Mesh), `priority_weight` boosts
    /// distance importance: ASDF (priority=4) weights distance 4× higher
    /// and hash 4× lower, strongly preferring the closest node.
    #[cfg(feature = "content_types")]
    #[inline]
    pub fn score_node_typed(
        &self,
        content_id: ContentId,
        node_id: NodeId,
        node_coord: &VivaldiCoord,
        content_type: crate::content_types::ContentType,
    ) -> ScoredNode {
        let hash = mum_hash(content_id, node_id);
        let hash_normalized = Fixed(((hash >> 32) as i64 * Fixed::ONE.0) >> 32);

        let predicted_rtt = self.local_coord.predict_rtt(node_coord);
        let min_rtt = Fixed::from_f64(1.0);
        let distance_score = Fixed::ONE / predicted_rtt.max(min_rtt);

        let priority = content_type.priority_weight() as i64;
        let eff_hash_weight = Fixed(self.hash_weight.0 / priority.max(1));
        let eff_dist_weight = Fixed(self.distance_weight.0.saturating_mul(priority));

        let score = eff_hash_weight * hash_normalized + eff_dist_weight * distance_score;

        ScoredNode {
            id: node_id,
            score,
            predicted_rtt,
            hash_score: hash,
        }
    }

    /// Find the best node for the given content type.
    ///
    /// Equivalent to `find_best` but adjusts scoring weights based on
    /// `ContentType::priority_weight()`.
    ///
    /// Returns `None` if `candidates` is empty.
    #[cfg(feature = "content_types")]
    #[must_use]
    pub fn find_best_typed<'a, I>(
        &self,
        content_id: ContentId,
        candidates: I,
        content_type: crate::content_types::ContentType,
    ) -> Option<ScoredNode>
    where
        I: IntoIterator<Item = (NodeId, &'a VivaldiCoord)>,
    {
        candidates
            .into_iter()
            .map(|(id, coord)| self.score_node_typed(content_id, id, coord, content_type))
            .max_by(|a, b| a.score.cmp(&b.score))
    }

    /// Find top-k nodes using `ContentType::suggested_replicas()` as default k.
    ///
    /// ASDF → 5 replicas, Mesh → 3, Texture/Audio/Generic → 2.
    #[cfg(feature = "content_types")]
    #[must_use]
    pub fn find_top_k_typed<'a, I>(
        &self,
        content_id: ContentId,
        candidates: I,
        content_type: crate::content_types::ContentType,
    ) -> Vec<ScoredNode>
    where
        I: IntoIterator<Item = (NodeId, &'a VivaldiCoord)>,
    {
        let k = content_type.suggested_replicas();
        let mut scored: Vec<_> = candidates
            .into_iter()
            .map(|(id, coord)| self.score_node_typed(content_id, id, coord, content_type))
            .collect();

        scored.sort_by(|a, b| b.score.cmp(&a.score));
        scored.truncate(k);
        scored
    }
}

/// Indexed Locator with O(log N) spatial search
pub struct IndexedLocator {
    /// Local SIMD coordinate
    local_coord: SimdCoord,
    /// Spatial index
    spatial_index: SpatialIndex,
    /// Hash weight (scaled)
    hash_weight: i64,
    /// Distance weight (scaled)
    distance_weight: i64,
}

impl IndexedLocator {
    /// Build indexed locator from node list
    #[must_use]
    pub fn build(
        local: SimdCoord,
        nodes: Vec<(NodeId, SimdCoord)>,
        hash_weight: f64,
        distance_weight: f64,
    ) -> Self {
        let entries: Vec<SpatialEntry> = nodes
            .into_iter()
            .map(|(id, coord)| SpatialEntry { node_id: id, coord })
            .collect();

        let scale = 1 << 16;
        Self {
            local_coord: local,
            spatial_index: SpatialIndex::build(entries),
            hash_weight: (hash_weight * scale as f64) as i64,
            distance_weight: (distance_weight * scale as f64) as i64,
        }
    }

    /// Find best node using two-phase search with `MumHash`
    #[must_use]
    pub fn find_best(
        &self,
        content_id: ContentId,
        candidate_limit: usize,
    ) -> Option<(NodeId, i64)> {
        if self.spatial_index.is_empty() {
            return None;
        }

        let candidates = self
            .spatial_index
            .find_nearest_k(&self.local_coord, candidate_limit);

        let mut best_id = 0;
        let mut best_score = i64::MIN;

        for (entry, dist_sq) in candidates {
            // Fast MumHash
            let hash = mum_hash(content_id, entry.node_id);
            let hash_component = (hash >> 32) as i64;

            let dist_component = if dist_sq > 0 {
                (1i64 << 28) / dist_sq.max(1) // Reduced shift to prevent overflow
            } else {
                1i64 << 28
            };

            let score = self.hash_weight.saturating_mul(hash_component) / (1 << 16)
                + self.distance_weight.saturating_mul(dist_component) / (1 << 16);

            if score > best_score {
                best_score = score;
                best_id = entry.node_id;
            }
        }

        Some((best_id, best_score))
    }

    /// Find top-K nodes
    #[must_use]
    pub fn find_top_k(
        &self,
        content_id: ContentId,
        k: usize,
        candidate_limit: usize,
    ) -> Vec<(NodeId, i64)> {
        if self.spatial_index.is_empty() {
            return Vec::new();
        }

        let candidates = self
            .spatial_index
            .find_nearest_k(&self.local_coord, candidate_limit);

        let mut scored: Vec<(NodeId, i64)> = candidates
            .iter()
            .map(|(entry, dist_sq)| {
                let hash = mum_hash(content_id, entry.node_id);
                let hash_component = (hash >> 32) as i64;
                let dist_component = if *dist_sq > 0 {
                    (1i64 << 28) / (*dist_sq).max(1)
                } else {
                    1i64 << 28
                };
                let score = self.hash_weight.saturating_mul(hash_component) / (1 << 16)
                    + self.distance_weight.saturating_mul(dist_component) / (1 << 16);
                (entry.node_id, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(k);
        scored
    }

    /// Update local coordinate
    pub const fn update_local(&mut self, coord: SimdCoord) {
        self.local_coord = coord;
    }
}

/// Rendezvous hash for pure consistent hashing (no latency)
pub struct RendezvousHash;

impl RendezvousHash {
    /// Find owner using `MumHash`.
    ///
    /// Returns `None` if `nodes` is empty.
    #[must_use]
    pub fn find_owner<I>(content_id: ContentId, nodes: I) -> Option<NodeId>
    where
        I: IntoIterator<Item = NodeId>,
    {
        nodes
            .into_iter()
            .max_by_key(|&node_id| mum_hash(content_id, node_id))
    }

    /// Find top-k owners for replication.
    #[must_use]
    pub fn find_replicas<I>(content_id: ContentId, nodes: I, k: usize) -> Vec<NodeId>
    where
        I: IntoIterator<Item = NodeId>,
    {
        let mut scored: Vec<_> = nodes
            .into_iter()
            .map(|id| (id, mum_hash(content_id, id)))
            .collect();

        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(k);
        scored.into_iter().map(|(id, _)| id).collect()
    }

    /// Find owner using legacy FNV-1a (for compatibility).
    ///
    /// Returns `None` if `nodes` is empty.
    #[must_use]
    pub fn find_owner_fnv<I>(content_id: ContentId, nodes: I) -> Option<NodeId>
    where
        I: IntoIterator<Item = NodeId>,
    {
        nodes
            .into_iter()
            .max_by_key(|&node_id| fnv1a_hash(content_id, node_id))
    }

    /// Find replicas using `ContentType::suggested_replicas()` for automatic k.
    ///
    /// ASDF → 5, Mesh → 3, Texture/Audio/Generic → 2.
    #[cfg(feature = "content_types")]
    #[must_use]
    pub fn find_replicas_typed<I>(
        content_id: ContentId,
        nodes: I,
        content_type: crate::content_types::ContentType,
    ) -> Vec<NodeId>
    where
        I: IntoIterator<Item = NodeId>,
    {
        let k = content_type.suggested_replicas();
        Self::find_replicas(content_id, nodes, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nodes() -> Vec<(NodeId, VivaldiCoord)> {
        vec![
            (1, VivaldiCoord::at(0.0, 0.0, 0.0, 5.0)),
            (2, VivaldiCoord::at(10.0, 0.0, 0.0, 5.0)),
            (3, VivaldiCoord::at(0.0, 10.0, 0.0, 5.0)),
            (4, VivaldiCoord::at(100.0, 100.0, 0.0, 5.0)),
        ]
    }

    #[test]
    fn test_mum_hash_deterministic() {
        let h1 = mum_hash(12345, 1);
        let h2 = mum_hash(12345, 1);
        assert_eq!(h1, h2);

        let h3 = mum_hash(12345, 2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_fnv_hash_deterministic() {
        let h1 = fnv1a_hash(12345, 1);
        let h2 = fnv1a_hash(12345, 1);
        assert_eq!(h1, h2);

        let h3 = fnv1a_hash(12345, 2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_locator_find_best() {
        let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let locator = ContentLocator::new(local);
        let nodes = make_nodes();

        let refs: Vec<_> = nodes.iter().map(|(id, c)| (*id, c)).collect();
        let best = locator.find_best(12345, refs).unwrap();

        assert!((1..=4).contains(&best.id));
    }

    #[test]
    fn test_locator_distance_preference() {
        let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let locator = ContentLocator::with_weights(local, 0.1, 0.9);
        let nodes = make_nodes();

        let refs: Vec<_> = nodes.iter().map(|(id, c)| (*id, c)).collect();
        let best = locator.find_best(12345, refs).unwrap();

        // With strong distance preference, should pick node 1 (closest)
        assert_eq!(best.id, 1, "Expected closest node, got {}", best.id);
    }

    #[test]
    fn test_locator_find_top_k() {
        let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let locator = ContentLocator::new(local);
        let nodes = make_nodes();

        let refs: Vec<_> = nodes.iter().map(|(id, c)| (*id, c)).collect();
        let top2 = locator.find_top_k(12345, refs, 2);

        assert_eq!(top2.len(), 2);
        assert!(top2[0].score >= top2[1].score);
    }

    #[test]
    fn test_locator_find_closest() {
        let local = VivaldiCoord::at(5.0, 5.0, 0.0, 5.0);
        let locator = ContentLocator::new(local);
        let nodes = make_nodes();

        let refs: Vec<_> = nodes.iter().map(|(id, c)| (*id, c)).collect();
        let (closest_id, rtt) = locator.find_closest(refs).unwrap();

        assert!(closest_id <= 3, "Unexpected closest: {closest_id}");
        assert!(rtt.to_f64() < 50.0);
    }

    #[test]
    fn test_rendezvous_consistency() {
        let nodes: Vec<_> = (1..=10).collect();

        let owner1 = RendezvousHash::find_owner(12345, nodes.iter().copied());
        let owner2 = RendezvousHash::find_owner(12345, nodes.iter().copied());
        assert_eq!(owner1, owner2);
    }

    #[test]
    fn test_rendezvous_replicas() {
        let nodes: Vec<_> = (1..=10).collect();
        let replicas = RendezvousHash::find_replicas(12345, nodes.iter().copied(), 3);

        assert_eq!(replicas.len(), 3);
        assert_ne!(replicas[0], replicas[1]);
        assert_ne!(replicas[1], replicas[2]);
    }

    #[test]
    fn test_rendezvous_minimal_disruption() {
        let nodes1: Vec<_> = (1..=10).collect();
        let nodes2: Vec<_> = (1..=11).collect();

        let mut same_count = 0;
        for content_id in 0..1000 {
            let owner1 = RendezvousHash::find_owner(content_id, nodes1.iter().copied());
            let owner2 = RendezvousHash::find_owner(content_id, nodes2.iter().copied());
            if owner1 == owner2 {
                same_count += 1;
            }
        }

        let same_ratio = same_count as f64 / 1000.0;
        assert!(same_ratio > 0.85, "{}% stayed same", same_ratio * 100.0);
    }

    #[test]
    fn test_indexed_locator() {
        let local = SimdCoord::from_f64(0.0, 0.0, 0.0, 1.0);
        let nodes: Vec<_> = (1..=100)
            .map(|i| (i as NodeId, SimdCoord::from_f64(i as f64, 0.0, 0.0, 1.0)))
            .collect();

        let locator = IndexedLocator::build(local, nodes, 0.3, 0.7);

        let result = locator.find_best(12345, 10);
        assert!(result.is_some());

        let (best_id, _score) = result.unwrap();
        assert!(best_id <= 20, "Expected nearby node, got {best_id}");
    }

    #[test]
    fn test_indexed_locator_top_k() {
        let local = SimdCoord::from_f64(50.0, 0.0, 0.0, 1.0);
        let nodes: Vec<_> = (1..=100)
            .map(|i| (i as NodeId, SimdCoord::from_f64(i as f64, 0.0, 0.0, 1.0)))
            .collect();

        let locator = IndexedLocator::build(local, nodes, 0.5, 0.5);

        let top3 = locator.find_top_k(12345, 3, 20);
        assert_eq!(top3.len(), 3);
        assert!(top3[0].1 >= top3[1].1);
        assert!(top3[1].1 >= top3[2].1);
    }

    #[test]
    fn test_hash_distribution() {
        // Test MumHash distribution
        let mut counts = [0u32; 16];
        for i in 0..10000u64 {
            let h = mum_hash(i, 0);
            counts[(h >> 60) as usize] += 1;
        }

        // Each bucket should have roughly 10000/16 = 625
        for &count in &counts {
            assert!(count > 400 && count < 900, "Uneven: {count}");
        }
    }

    // --- Boundary tests ---

    #[test]
    fn test_find_best_empty_candidates() {
        let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let locator = ContentLocator::new(local);
        let empty: Vec<(NodeId, &VivaldiCoord)> = vec![];
        assert!(locator.find_best(12345, empty).is_none());
    }

    #[test]
    fn test_find_closest_empty_candidates() {
        let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let locator = ContentLocator::new(local);
        let empty: Vec<(NodeId, &VivaldiCoord)> = vec![];
        assert!(locator.find_closest(empty).is_none());
    }

    #[test]
    fn test_find_top_k_zero() {
        let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let locator = ContentLocator::new(local);
        let nodes = make_nodes();
        let refs: Vec<_> = nodes.iter().map(|(id, c)| (*id, c)).collect();
        let top0 = locator.find_top_k(12345, refs, 0);
        assert!(top0.is_empty());
    }

    #[test]
    fn test_find_top_k_exceeds_len() {
        let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let locator = ContentLocator::new(local);
        let nodes = make_nodes();
        let refs: Vec<_> = nodes.iter().map(|(id, c)| (*id, c)).collect();
        let top100 = locator.find_top_k(12345, refs, 100);
        assert_eq!(top100.len(), 4); // Only 4 nodes available
    }

    #[test]
    fn test_find_replicas_exceeds_nodes() {
        let nodes: Vec<u64> = vec![1, 2, 3];
        let replicas = RendezvousHash::find_replicas(42, nodes.iter().copied(), 10);
        assert_eq!(replicas.len(), 3); // Only 3 available
    }

    #[test]
    fn test_find_owner_empty() {
        let empty: Vec<u64> = vec![];
        assert!(RendezvousHash::find_owner(42, empty.iter().copied()).is_none());
    }

    #[test]
    fn test_find_owner_fnv_empty() {
        let empty: Vec<u64> = vec![];
        assert!(RendezvousHash::find_owner_fnv(42, empty.iter().copied()).is_none());
    }

    #[test]
    fn test_find_owner_deterministic_many_keys() {
        let nodes: Vec<u64> = (1..=20).collect();
        for key in 0..100u64 {
            let o1 = RendezvousHash::find_owner(key, nodes.iter().copied());
            let o2 = RendezvousHash::find_owner(key, nodes.iter().copied());
            assert_eq!(o1, o2, "Non-deterministic for key {key}");
        }
    }

    #[test]
    fn test_score_node_positive_score() {
        let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let locator = ContentLocator::new(local);
        let node = VivaldiCoord::at(10.0, 0.0, 0.0, 5.0);
        let scored = locator.score_node(42, 1, &node);
        assert!(scored.score.0 > 0, "Score should be positive");
        assert!(scored.predicted_rtt.to_f64() > 0.0);
    }

    #[test]
    fn test_rank_by_latency_sorted() {
        let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let locator = ContentLocator::new(local);
        let nodes = make_nodes();
        let refs: Vec<_> = nodes.iter().map(|(id, c)| (*id, c)).collect();
        let ranked = locator.rank_by_latency(refs);
        for i in 1..ranked.len() {
            assert!(ranked[i - 1].1 <= ranked[i].1, "Not sorted at index {i}");
        }
    }

    #[test]
    fn test_update_coord() {
        let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let mut locator = ContentLocator::new(local);
        let new_coord = VivaldiCoord::at(50.0, 50.0, 0.0, 5.0);
        locator.update_coord(new_coord);
        // No panic, coord updated
    }

    #[test]
    fn test_indexed_locator_empty() {
        let local = SimdCoord::from_f64(0.0, 0.0, 0.0, 1.0);
        let locator = IndexedLocator::build(local, vec![], 0.5, 0.5);
        assert!(locator.find_best(42, 10).is_none());
        assert!(locator.find_top_k(42, 3, 10).is_empty());
    }

    #[test]
    fn test_indexed_locator_update_local() {
        let local = SimdCoord::from_f64(0.0, 0.0, 0.0, 1.0);
        let nodes = vec![(1u64, SimdCoord::from_f64(10.0, 0.0, 0.0, 1.0))];
        let mut locator = IndexedLocator::build(local, nodes, 0.5, 0.5);
        locator.update_local(SimdCoord::from_f64(5.0, 0.0, 0.0, 1.0));
        assert!(locator.find_best(42, 10).is_some());
    }

    #[cfg(feature = "content_types")]
    mod typed_tests {
        use super::*;
        use crate::content_types::ContentType;

        #[test]
        fn test_typed_routing_prefers_closer_for_asdf() {
            let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
            let locator = ContentLocator::new(local); // default 0.5/0.5 weights

            let close_node = VivaldiCoord::at(1.0, 0.0, 0.0, 2.0);
            let far_node = VivaldiCoord::at(100.0, 100.0, 0.0, 10.0);

            let nodes = vec![(1u64, &close_node), (2u64, &far_node)];

            // With ASDF (priority=4), distance is boosted 4×
            let best = locator
                .find_best_typed(42, nodes, ContentType::Asdf)
                .unwrap();
            assert_eq!(best.id, 1, "ASDF should pick closest node, got {}", best.id);
        }

        #[test]
        fn test_typed_vs_untyped_asdf_boosts_distance() {
            let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
            let locator = ContentLocator::new(local);

            let node_coord = VivaldiCoord::at(10.0, 0.0, 0.0, 5.0);

            let untyped = locator.score_node(42, 1, &node_coord);
            let typed_asdf = locator.score_node_typed(42, 1, &node_coord, ContentType::Asdf);
            let typed_generic = locator.score_node_typed(42, 1, &node_coord, ContentType::Generic);

            // Generic (priority=1) should be close to untyped
            let diff = (typed_generic.score.0 - untyped.score.0).abs();
            assert!(
                diff < Fixed::from_f64(0.01).0,
                "Generic should match untyped"
            );

            // ASDF should have different score (distance boosted)
            assert_ne!(typed_asdf.score, untyped.score);
        }

        #[test]
        fn test_typed_replicas_count() {
            let nodes: Vec<u64> = (1..=10).collect();

            let asdf_replicas =
                RendezvousHash::find_replicas_typed(42, nodes.iter().copied(), ContentType::Asdf);
            assert_eq!(asdf_replicas.len(), 5, "ASDF should get 5 replicas");

            let mesh_replicas =
                RendezvousHash::find_replicas_typed(42, nodes.iter().copied(), ContentType::Mesh);
            assert_eq!(mesh_replicas.len(), 3, "Mesh should get 3 replicas");

            let generic_replicas = RendezvousHash::find_replicas_typed(
                42,
                nodes.iter().copied(),
                ContentType::Generic,
            );
            assert_eq!(generic_replicas.len(), 2, "Generic should get 2 replicas");
        }

        #[test]
        fn test_find_top_k_typed_uses_suggested_replicas() {
            let local = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
            let locator = ContentLocator::new(local);

            let nodes: Vec<(u64, VivaldiCoord)> = (1..=10)
                .map(|i| (i, VivaldiCoord::at(i as f64 * 5.0, 0.0, 0.0, 3.0)))
                .collect();
            let refs: Vec<_> = nodes.iter().map(|(id, c)| (*id, c)).collect();

            let asdf_top = locator.find_top_k_typed(42, refs.clone(), ContentType::Asdf);
            assert_eq!(asdf_top.len(), 5, "ASDF top-k should return 5");

            let generic_top = locator.find_top_k_typed(42, refs, ContentType::Generic);
            assert_eq!(generic_top.len(), 2, "Generic top-k should return 2");
        }
    }
}

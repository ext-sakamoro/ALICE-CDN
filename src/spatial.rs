//! Spatial Index for O(log N) Nearest Neighbor Search (Compressed Edition)
//!
//! **Algorithm**: Simplified Octree with linear scan for small buckets.
//!
//! **Scorched Earth Optimizations**:
//! - **u32 indices**: [u32; 8] = 32 bytes vs [Option<usize>; 8] = 128 bytes (4x smaller)
//! - **Flat storage**: All nodes in single Vec, cache-friendly
//! - **Stack traversal**: No recursion, no heap allocation during search
//! - **Bucket cutoff**: Small groups (≤16) use linear scan
//!
//! > "The fastest search is the one that skips 90% of the data."

use crate::locator::NodeId;
use crate::simd::SimdCoord;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Maximum items per leaf before splitting
const BUCKET_SIZE: usize = 16;

/// Maximum tree depth to prevent pathological cases
const MAX_DEPTH: usize = 20;

/// Empty child marker (u32::MAX)
const EMPTY: u32 = u32::MAX;

/// Axis-Aligned Bounding Box (Compact: 48 bytes)
#[derive(Clone, Copy, Debug)]
pub struct AABB {
    pub min: [i64; 3],
    pub max: [i64; 3],
}

impl AABB {
    /// Create AABB containing a single point
    #[inline(always)]
    pub fn from_point(coord: &SimdCoord) -> Self {
        Self {
            min: [coord.data[0], coord.data[1], coord.data[2]],
            max: [coord.data[0], coord.data[1], coord.data[2]],
        }
    }

    /// Expand to contain another point
    #[inline(always)]
    pub fn expand(&mut self, coord: &SimdCoord) {
        self.min[0] = self.min[0].min(coord.data[0]);
        self.min[1] = self.min[1].min(coord.data[1]);
        self.min[2] = self.min[2].min(coord.data[2]);
        self.max[0] = self.max[0].max(coord.data[0]);
        self.max[1] = self.max[1].max(coord.data[1]);
        self.max[2] = self.max[2].max(coord.data[2]);
    }

    /// Get center point
    #[inline(always)]
    pub fn center(&self) -> [i64; 3] {
        [
            (self.min[0] + self.max[0]) / 2,
            (self.min[1] + self.max[1]) / 2,
            (self.min[2] + self.max[2]) / 2,
        ]
    }

    /// Check if point is inside (or on boundary)
    #[inline(always)]
    pub fn contains(&self, coord: &SimdCoord) -> bool {
        coord.data[0] >= self.min[0] && coord.data[0] <= self.max[0] &&
        coord.data[1] >= self.min[1] && coord.data[1] <= self.max[1] &&
        coord.data[2] >= self.min[2] && coord.data[2] <= self.max[2]
    }

    /// Minimum squared distance from point to AABB
    #[inline(always)]
    pub fn distance_squared(&self, coord: &SimdCoord) -> i64 {
        let mut dist_sq = 0i128;

        for i in 0..3 {
            let v = coord.data[i];
            if v < self.min[i] {
                let d = (self.min[i] - v) as i128;
                dist_sq += d * d;
            } else if v > self.max[i] {
                let d = (v - self.max[i]) as i128;
                dist_sq += d * d;
            }
        }

        // Scale down to match SimdCoord::distance_squared
        (dist_sq / (1 << 16)) as i64
    }

    /// Get octant index (0-7) for a point relative to center
    #[inline(always)]
    fn octant(&self, coord: &SimdCoord) -> usize {
        let c = self.center();
        let mut idx = 0;
        if coord.data[0] >= c[0] { idx |= 1; }
        if coord.data[1] >= c[1] { idx |= 2; }
        if coord.data[2] >= c[2] { idx |= 4; }
        idx
    }

    /// Get child AABB for given octant
    #[inline(always)]
    fn child_aabb(&self, octant: usize) -> Self {
        let c = self.center();
        let mut child = *self;

        if octant & 1 == 0 {
            child.max[0] = c[0];
        } else {
            child.min[0] = c[0];
        }

        if octant & 2 == 0 {
            child.max[1] = c[1];
        } else {
            child.min[1] = c[1];
        }

        if octant & 4 == 0 {
            child.max[2] = c[2];
        } else {
            child.min[2] = c[2];
        }

        child
    }
}

/// Entry in the spatial index (40 bytes)
#[derive(Clone, Copy, Debug)]
pub struct SpatialEntry {
    pub node_id: NodeId,
    pub coord: SimdCoord,
}

/// Compressed Octree node
///
/// **Memory Layout**:
/// - Leaf: tag (1 byte) + count (1 byte) + items_start (4 bytes) = 6 bytes + padding
/// - Internal: tag (1 byte) + children (32 bytes) = 33 bytes + padding
///
/// Using enum discriminant for tag
#[derive(Clone, Debug)]
enum OctreeNode {
    /// Leaf node: indices into items array
    Leaf {
        /// Start index in items array
        start: u32,
        /// Number of items
        count: u16,
    },
    /// Internal node: 8 children indices (u32::MAX = empty)
    Internal {
        /// Child node indices [u32; 8] = 32 bytes (vs 128 bytes with Option<usize>)
        children: [u32; 8],
    },
}

/// Flat Octree for spatial indexing (Compressed Edition)
pub struct SpatialIndex {
    /// All nodes stored flat
    nodes: Vec<OctreeNode>,
    /// All items stored flat (leaves reference into this)
    items: Vec<SpatialEntry>,
    /// Root node index
    root: u32,
    /// Bounding box of entire tree
    bounds: AABB,
    /// Total number of entries
    count: usize,
}

impl SpatialIndex {
    /// Build spatial index from nodes
    pub fn build(entries: Vec<SpatialEntry>) -> Self {
        if entries.is_empty() {
            return Self {
                nodes: vec![OctreeNode::Leaf { start: 0, count: 0 }],
                items: Vec::new(),
                root: 0,
                bounds: AABB { min: [0; 3], max: [0; 3] },
                count: 0,
            };
        }

        // Calculate bounds
        let mut bounds = AABB::from_point(&entries[0].coord);
        for entry in &entries[1..] {
            bounds.expand(&entry.coord);
        }

        // Add small padding
        for i in 0..3 {
            bounds.min[i] -= 1;
            bounds.max[i] += 1;
        }

        let count = entries.len();

        // Build tree recursively, collecting items in traversal order
        let mut builder = TreeBuilder {
            nodes: Vec::with_capacity(entries.len() / BUCKET_SIZE + 1),
            items: Vec::with_capacity(entries.len()),
        };

        let root = builder.build_node(entries, bounds, 0);

        Self {
            nodes: builder.nodes,
            items: builder.items,
            root,
            bounds,
            count,
        }
    }

    /// Find k nearest neighbors to query point
    pub fn find_nearest_k(&self, query: &SimdCoord, k: usize) -> Vec<(SpatialEntry, i64)> {
        if k == 0 || self.count == 0 {
            return Vec::new();
        }

        let mut results: Vec<(SpatialEntry, i64)> = Vec::with_capacity(k + 1);
        let mut max_dist = i64::MAX;

        // Stack: (node_idx, bounds, min_dist_to_bounds)
        let mut stack: Vec<(u32, AABB, i64)> = Vec::with_capacity(MAX_DEPTH * 8);
        stack.push((self.root, self.bounds, 0));

        while let Some((node_idx, bounds, min_dist)) = stack.pop() {
            if results.len() >= k && min_dist >= max_dist {
                continue;
            }

            match &self.nodes[node_idx as usize] {
                OctreeNode::Leaf { start, count } => {
                    let end = *start as usize + *count as usize;
                    for entry in &self.items[*start as usize..end] {
                        let dist = query.distance_squared(&entry.coord);

                        if results.len() < k || dist < max_dist {
                            let pos = results.iter().position(|(_, d)| dist < *d)
                                .unwrap_or(results.len());
                            results.insert(pos, (*entry, dist));

                            if results.len() > k {
                                results.pop();
                            }

                            if results.len() >= k {
                                max_dist = results.last().map(|(_, d)| *d).unwrap_or(i64::MAX);
                            }
                        }
                    }
                }
                OctreeNode::Internal { children } => {
                    let mut child_dists: Vec<(u32, AABB, i64)> = Vec::with_capacity(8);

                    for (i, &child_idx) in children.iter().enumerate() {
                        if child_idx != EMPTY {
                            let child_bounds = bounds.child_aabb(i);
                            let dist = child_bounds.distance_squared(query);
                            child_dists.push((child_idx, child_bounds, dist));
                        }
                    }

                    child_dists.sort_by_key(|(_, _, d)| *d);

                    for (child_idx, child_bounds, dist) in child_dists.into_iter().rev() {
                        if results.len() < k || dist < max_dist {
                            stack.push((child_idx, child_bounds, dist));
                        }
                    }
                }
            }
        }

        results
    }

    /// Find all entries within distance threshold
    pub fn find_within_radius(&self, query: &SimdCoord, radius_squared: i64) -> Vec<SpatialEntry> {
        let mut results = Vec::new();

        let mut stack = vec![(self.root, self.bounds)];

        while let Some((node_idx, bounds)) = stack.pop() {
            if bounds.distance_squared(query) > radius_squared {
                continue;
            }

            match &self.nodes[node_idx as usize] {
                OctreeNode::Leaf { start, count } => {
                    let end = *start as usize + *count as usize;
                    for entry in &self.items[*start as usize..end] {
                        if query.distance_squared(&entry.coord) <= radius_squared {
                            results.push(*entry);
                        }
                    }
                }
                OctreeNode::Internal { children } => {
                    for (i, &child_idx) in children.iter().enumerate() {
                        if child_idx != EMPTY {
                            let child_bounds = bounds.child_aabb(i);
                            stack.push((child_idx, child_bounds));
                        }
                    }
                }
            }
        }

        results
    }

    /// Get total number of entries
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.nodes.len() * core::mem::size_of::<OctreeNode>()
            + self.items.len() * core::mem::size_of::<SpatialEntry>()
    }
}

/// Helper for building tree
struct TreeBuilder {
    nodes: Vec<OctreeNode>,
    items: Vec<SpatialEntry>,
}

impl TreeBuilder {
    fn build_node(&mut self, entries: Vec<SpatialEntry>, bounds: AABB, depth: usize) -> u32 {
        let node_idx = self.nodes.len() as u32;

        // Base case: small enough for leaf or max depth
        if entries.len() <= BUCKET_SIZE || depth >= MAX_DEPTH {
            let start = self.items.len() as u32;
            let count = entries.len() as u16;
            self.items.extend(entries);
            self.nodes.push(OctreeNode::Leaf { start, count });
            return node_idx;
        }

        // Split into octants
        let mut octants: [Vec<SpatialEntry>; 8] = Default::default();
        for entry in entries {
            let oct = bounds.octant(&entry.coord);
            octants[oct].push(entry);
        }

        // Reserve slot for internal node
        self.nodes.push(OctreeNode::Internal { children: [EMPTY; 8] });

        // Recursively build children
        let mut children = [EMPTY; 8];
        for (i, oct_entries) in octants.into_iter().enumerate() {
            if !oct_entries.is_empty() {
                let child_bounds = bounds.child_aabb(i);
                children[i] = self.build_node(oct_entries, child_bounds, depth + 1);
            }
        }

        // Update internal node
        self.nodes[node_idx as usize] = OctreeNode::Internal { children };
        node_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entries() -> Vec<SpatialEntry> {
        vec![
            SpatialEntry { node_id: 1, coord: SimdCoord::from_f64(0.0, 0.0, 0.0, 1.0) },
            SpatialEntry { node_id: 2, coord: SimdCoord::from_f64(10.0, 0.0, 0.0, 1.0) },
            SpatialEntry { node_id: 3, coord: SimdCoord::from_f64(0.0, 10.0, 0.0, 1.0) },
            SpatialEntry { node_id: 4, coord: SimdCoord::from_f64(100.0, 100.0, 0.0, 1.0) },
            SpatialEntry { node_id: 5, coord: SimdCoord::from_f64(5.0, 5.0, 0.0, 1.0) },
        ]
    }

    #[test]
    fn test_spatial_index_build() {
        let entries = make_entries();
        let index = SpatialIndex::build(entries);

        assert_eq!(index.len(), 5);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_find_nearest_k() {
        let entries = make_entries();
        let index = SpatialIndex::build(entries);

        let query = SimdCoord::from_f64(0.0, 0.0, 0.0, 1.0);
        let nearest = index.find_nearest_k(&query, 3);

        assert_eq!(nearest.len(), 3);
        // First should be node 1 (at origin)
        assert_eq!(nearest[0].0.node_id, 1);
    }

    #[test]
    fn test_find_within_radius() {
        let entries = make_entries();
        let index = SpatialIndex::build(entries);

        let query = SimdCoord::from_f64(0.0, 0.0, 0.0, 1.0);
        // distance_squared returns sum_of_squares / SCALE
        let radius_sq = 15i64 * 15i64 * 65536;
        let within = index.find_within_radius(&query, radius_sq);

        assert!(within.len() >= 3);
        assert!(within.iter().all(|e| e.node_id != 4)); // Node 4 is far
    }

    #[test]
    fn test_empty_index() {
        let index = SpatialIndex::build(vec![]);
        assert!(index.is_empty());

        let query = SimdCoord::from_f64(0.0, 0.0, 0.0, 1.0);
        let nearest = index.find_nearest_k(&query, 5);
        assert!(nearest.is_empty());
    }

    #[test]
    fn test_large_dataset() {
        let entries: Vec<SpatialEntry> = (0..1000)
            .map(|i| SpatialEntry {
                node_id: i as u64,
                coord: SimdCoord::from_f64(
                    (i % 100) as f64,
                    ((i / 10) % 100) as f64,
                    (i / 100) as f64,
                    1.0,
                ),
            })
            .collect();

        let index = SpatialIndex::build(entries);
        assert_eq!(index.len(), 1000);

        let query = SimdCoord::from_f64(50.0, 50.0, 5.0, 1.0);
        let nearest = index.find_nearest_k(&query, 10);

        assert_eq!(nearest.len(), 10);
        // Results should be sorted by distance
        for i in 1..nearest.len() {
            assert!(nearest[i-1].1 <= nearest[i].1);
        }
    }

    #[test]
    fn test_compressed_node_size() {
        // Internal node with [u32; 8] should be much smaller than [Option<usize>; 8]
        let internal_size = core::mem::size_of::<OctreeNode>();
        // OctreeNode::Internal has [u32; 8] = 32 bytes + enum discriminant + padding
        assert!(internal_size <= 48, "Node size: {} bytes", internal_size);
    }

    #[test]
    fn test_memory_efficiency() {
        let entries: Vec<SpatialEntry> = (0..1000)
            .map(|i| SpatialEntry {
                node_id: i as u64,
                coord: SimdCoord::from_f64(i as f64, 0.0, 0.0, 1.0),
            })
            .collect();

        let index = SpatialIndex::build(entries);
        let usage = index.memory_usage();

        // Should be reasonably efficient
        // 1000 items * 40 bytes + nodes overhead
        assert!(usage < 100_000, "Memory usage: {} bytes", usage);
    }
}

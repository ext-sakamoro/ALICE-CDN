//! Maglev Consistent Hashing (Zero-Allocation Edition)
//!
//! **Algorithm**: Google's Maglev load balancer hashing.
//!
//! **Optimizations**:
//! - **No permutation table**: Compute `(offset + i*skip) % M` on-the-fly
//! - **Compact lookup**: `Vec<u16>` instead of `Vec<Option<usize>>` (u16::MAX = empty)
//! - **Cache-friendly**: Single contiguous allocation for lookup table
//!
//! **Properties**:
//! - O(1) lookup via precomputed table
//! - Even distribution across nodes
//! - Minimal key redistribution on node changes
//! - Memory: O(M) instead of O(M*N)
//!
//! > "Memory bandwidth is the bottleneck."

use crate::locator::NodeId;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Default lookup table size (must be prime for good distribution)
pub const DEFAULT_TABLE_SIZE: usize = 65537;

/// Smaller table size for testing
pub const SMALL_TABLE_SIZE: usize = 17;

/// Empty slot marker
const EMPTY: u16 = u16::MAX;

/// Maglev hash function 1 (offset)
#[inline(always)]
fn hash1(key: u64, table_size: usize) -> usize {
    const MUL: u64 = 0x9E3779B97F4A7C15; // Golden ratio prime
    ((key.wrapping_mul(MUL)) as usize) % table_size
}

/// Maglev hash function 2 (skip)
/// Returns value in [1, table_size-1] to ensure coprimality with M
#[inline(always)]
fn hash2(key: u64, table_size: usize) -> usize {
    const MUL: u64 = 0x517CC1B727220A95; // Different prime
    ((key.wrapping_mul(MUL)) as usize) % (table_size - 1) + 1
}

/// Precomputed permutation parameters for a node
#[derive(Clone, Copy, Debug)]
struct NodeParams {
    /// Starting offset in permutation
    offset: usize,
    /// Step size in permutation
    skip: usize,
    /// Next probe index (during population)
    next: usize,
}

/// Maglev consistent hash lookup table (Zero-Allocation Edition)
pub struct MaglevHash {
    /// Lookup table: table[hash(key) % size] = node_index (u16::MAX = empty)
    table: Vec<u16>,
    /// Node list (maps index to NodeId)
    nodes: Vec<NodeId>,
    /// Table size (prime number)
    table_size: usize,
}

impl MaglevHash {
    /// Create new Maglev hash with default table size
    pub fn new(nodes: Vec<NodeId>) -> Self {
        Self::with_table_size(nodes, DEFAULT_TABLE_SIZE)
    }

    /// Create with custom table size (should be prime)
    pub fn with_table_size(nodes: Vec<NodeId>, table_size: usize) -> Self {
        assert!(nodes.len() <= EMPTY as usize, "Too many nodes (max {})", EMPTY);

        let mut maglev = Self {
            table: vec![EMPTY; table_size],
            nodes,
            table_size,
        };
        maglev.populate();
        maglev
    }

    /// Populate the lookup table using Maglev algorithm
    ///
    /// Zero-allocation: No N*M permutation table, just N*(offset,skip) pairs
    fn populate(&mut self) {
        if self.nodes.is_empty() {
            return;
        }

        let n = self.nodes.len();
        let m = self.table_size;

        // Precompute offset/skip for each node (only 2*N integers, not M*N)
        let mut params: Vec<NodeParams> = self
            .nodes
            .iter()
            .map(|&node_id| NodeParams {
                offset: hash1(node_id, m),
                skip: hash2(node_id, m),
                next: 0,
            })
            .collect();

        // Fill the table
        let mut filled = 0;
        while filled < m {
            for i in 0..n {
                // Find next unfilled slot for this node
                loop {
                    // Compute slot on-the-fly instead of looking up in permutation table
                    let slot = (params[i].offset + params[i].next * params[i].skip) % m;
                    params[i].next += 1;

                    if self.table[slot] == EMPTY {
                        self.table[slot] = i as u16;
                        filled += 1;
                        break;
                    }

                    // Safety: next will eventually find a slot (M is prime, skip is coprime)
                    if params[i].next >= m {
                        break;
                    }
                }
            }
        }
    }

    /// Lookup node for a key (O(1))
    #[inline(always)]
    pub fn lookup(&self, key: u64) -> Option<NodeId> {
        if self.nodes.is_empty() {
            return None;
        }

        let slot = hash1(key, self.table_size);
        let idx = self.table[slot];
        if idx == EMPTY {
            None
        } else {
            Some(self.nodes[idx as usize])
        }
    }

    /// Lookup returning node index (for advanced use)
    #[inline(always)]
    pub fn lookup_index(&self, key: u64) -> Option<usize> {
        if self.nodes.is_empty() {
            return None;
        }

        let slot = hash1(key, self.table_size);
        let idx = self.table[slot];
        if idx == EMPTY {
            None
        } else {
            Some(idx as usize)
        }
    }

    /// Lookup with explicit hash (for custom hashing)
    #[inline(always)]
    pub fn lookup_by_hash(&self, hash: u64) -> Option<NodeId> {
        if self.nodes.is_empty() {
            return None;
        }

        let slot = (hash as usize) % self.table_size;
        let idx = self.table[slot];
        if idx == EMPTY {
            None
        } else {
            Some(self.nodes[idx as usize])
        }
    }

    /// Add a node and rebuild table
    pub fn add_node(&mut self, node_id: NodeId) {
        if !self.nodes.contains(&node_id) {
            self.nodes.push(node_id);
            self.table.fill(EMPTY);
            self.populate();
        }
    }

    /// Remove a node and rebuild table
    pub fn remove_node(&mut self, node_id: NodeId) {
        if let Some(pos) = self.nodes.iter().position(|&id| id == node_id) {
            self.nodes.remove(pos);
            self.table.fill(EMPTY);
            self.populate();
        }
    }

    /// Get all nodes
    #[inline(always)]
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// Get table size
    #[inline(always)]
    pub fn table_size(&self) -> usize {
        self.table_size
    }

    /// Get memory usage in bytes
    #[inline(always)]
    pub fn memory_usage(&self) -> usize {
        self.table.len() * 2 + self.nodes.len() * 8
    }

    /// Check distribution evenness
    pub fn distribution_stats(&self) -> DistributionStats {
        if self.nodes.is_empty() {
            return DistributionStats {
                node_counts: vec![],
                min_count: 0,
                max_count: 0,
                std_dev: 0.0,
            };
        }

        let mut counts = vec![0usize; self.nodes.len()];
        for &idx in &self.table {
            if idx != EMPTY {
                counts[idx as usize] += 1;
            }
        }

        let min_count = *counts.iter().min().unwrap_or(&0);
        let max_count = *counts.iter().max().unwrap_or(&0);

        let mean = self.table_size as f64 / self.nodes.len() as f64;
        let variance: f64 = counts
            .iter()
            .map(|&c| (c as f64 - mean).powi(2))
            .sum::<f64>()
            / self.nodes.len() as f64;
        let std_dev = variance.sqrt();

        DistributionStats {
            node_counts: counts,
            min_count,
            max_count,
            std_dev,
        }
    }
}

/// Distribution statistics
#[derive(Debug)]
pub struct DistributionStats {
    /// Count per node
    pub node_counts: Vec<usize>,
    /// Minimum count
    pub min_count: usize,
    /// Maximum count
    pub max_count: usize,
    /// Standard deviation
    pub std_dev: f64,
}

/// Weighted Maglev for nodes with different capacities
pub struct WeightedMaglev {
    /// Inner Maglev hash
    inner: MaglevHash,
    /// Node weights (node_id -> weight)
    weights: Vec<(NodeId, u32)>,
}

impl WeightedMaglev {
    /// Create weighted Maglev
    ///
    /// Nodes with higher weight get more virtual entries
    pub fn new(weighted_nodes: Vec<(NodeId, u32)>) -> Self {
        // Expand nodes based on weight
        let expanded: Vec<NodeId> = weighted_nodes
            .iter()
            .flat_map(|&(id, weight)| {
                // Create virtual nodes: id * 1000 + i
                (0..weight).map(move |i| id * 1000 + i as u64)
            })
            .collect();

        Self {
            inner: MaglevHash::new(expanded),
            weights: weighted_nodes,
        }
    }

    /// Lookup node for key
    #[inline(always)]
    pub fn lookup(&self, key: u64) -> Option<NodeId> {
        self.inner.lookup(key).map(|virtual_id| virtual_id / 1000)
    }

    /// Get original weights
    #[inline(always)]
    pub fn weights(&self) -> &[(NodeId, u32)] {
        &self.weights
    }
}

/// Fast Maglev for single-use (no rebuild capability)
///
/// Even more memory efficient when you don't need add/remove
pub struct StaticMaglev {
    /// Lookup table (u16 indices)
    table: Vec<u16>,
    /// Table size
    table_size: usize,
}

impl StaticMaglev {
    /// Build from node IDs (nodes are indexed 0..N-1 in provided order)
    pub fn build(nodes: &[NodeId], table_size: usize) -> Self {
        assert!(nodes.len() <= EMPTY as usize);

        let mut table = vec![EMPTY; table_size];
        let n = nodes.len();

        if n == 0 {
            return Self { table, table_size };
        }

        // Inline populate without storing nodes
        let params: Vec<_> = nodes
            .iter()
            .map(|&id| (hash1(id, table_size), hash2(id, table_size)))
            .collect();

        let mut nexts = vec![0usize; n];
        let mut filled = 0;

        while filled < table_size {
            for i in 0..n {
                loop {
                    let slot = (params[i].0 + nexts[i] * params[i].1) % table_size;
                    nexts[i] += 1;

                    if table[slot] == EMPTY {
                        table[slot] = i as u16;
                        filled += 1;
                        break;
                    }

                    if nexts[i] >= table_size {
                        break;
                    }
                }
            }
        }

        Self { table, table_size }
    }

    /// Lookup returning node index
    #[inline(always)]
    pub fn lookup(&self, key: u64) -> usize {
        let slot = hash1(key, self.table_size);
        self.table[slot] as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maglev_basic() {
        let nodes = vec![1, 2, 3, 4, 5];
        let maglev = MaglevHash::with_table_size(nodes, SMALL_TABLE_SIZE);

        // Should return some node
        let node = maglev.lookup(12345);
        assert!(node.is_some());
        assert!(node.unwrap() >= 1 && node.unwrap() <= 5);
    }

    #[test]
    fn test_maglev_consistency() {
        let nodes = vec![1, 2, 3, 4, 5];
        let maglev = MaglevHash::with_table_size(nodes, SMALL_TABLE_SIZE);

        // Same key should always map to same node
        let node1 = maglev.lookup(12345);
        let node2 = maglev.lookup(12345);
        assert_eq!(node1, node2);
    }

    #[test]
    fn test_maglev_distribution() {
        let nodes: Vec<_> = (1..=5).collect();
        let maglev = MaglevHash::with_table_size(nodes, 1009); // Prime

        let stats = maglev.distribution_stats();

        // Each node should have roughly equal share
        let expected = 1009 / 5; // ~201

        for count in &stats.node_counts {
            let diff = (*count as i64 - expected as i64).abs();
            assert!(
                diff < expected as i64 / 2,
                "Uneven distribution: {} vs expected {}",
                count,
                expected
            );
        }
    }

    #[test]
    fn test_maglev_minimal_disruption() {
        let nodes1: Vec<_> = (1..=5).collect();
        let nodes2: Vec<_> = (1..=6).collect();

        let maglev1 = MaglevHash::with_table_size(nodes1, 1009);
        let maglev2 = MaglevHash::with_table_size(nodes2, 1009);

        // Count how many keys stayed with same node
        let mut same_count = 0;
        for key in 0..10000 {
            let node1 = maglev1.lookup(key);
            let node2 = maglev2.lookup(key);
            if node1 == node2 {
                same_count += 1;
            }
        }

        // Should have minimal disruption (roughly 5/6 ≈ 83% should stay same)
        let same_ratio = same_count as f64 / 10000.0;
        assert!(
            same_ratio > 0.75,
            "Too much disruption: only {:.1}% stayed same",
            same_ratio * 100.0
        );
    }

    #[test]
    fn test_maglev_add_remove() {
        let nodes = vec![1, 2, 3];
        let mut maglev = MaglevHash::with_table_size(nodes, SMALL_TABLE_SIZE);

        let key = 12345;
        let original = maglev.lookup(key);

        // Add a node
        maglev.add_node(4);
        let after_add = maglev.lookup(key);

        // Remove the added node
        maglev.remove_node(4);
        let after_remove = maglev.lookup(key);

        // After removing, should return to original
        assert_eq!(original, after_remove);

        // After add may or may not change (depends on key)
        assert!(after_add.is_some());
    }

    #[test]
    fn test_maglev_empty() {
        let maglev = MaglevHash::with_table_size(vec![], SMALL_TABLE_SIZE);
        assert_eq!(maglev.lookup(12345), None);
    }

    #[test]
    fn test_weighted_maglev() {
        // Node 1 has weight 3, node 2 has weight 1
        let weighted = WeightedMaglev::new(vec![(1, 3), (2, 1)]);

        // Count distribution over many keys
        let mut count1 = 0;
        let mut count2 = 0;

        for key in 0..10000 {
            match weighted.lookup(key) {
                Some(1) => count1 += 1,
                Some(2) => count2 += 1,
                _ => {}
            }
        }

        // Node 1 should get roughly 3x the traffic of node 2
        let ratio = count1 as f64 / count2 as f64;
        assert!(
            ratio > 2.0 && ratio < 4.0,
            "Expected ratio ~3, got {:.2}",
            ratio
        );
    }

    #[test]
    fn test_hash_functions() {
        // Test that hash functions produce good distribution
        let mut h1_counts = vec![0; 100];
        let mut h2_counts = vec![0; 100];

        for i in 0..10000u64 {
            h1_counts[hash1(i, 100)] += 1;
            h2_counts[hash2(i, 100)] += 1;
        }

        // Check distribution is relatively even
        for &count in &h1_counts {
            assert!(count > 50 && count < 150, "h1 uneven: {}", count);
        }
        // hash2 returns [1, table_size-1], so index 0 is always 0
        assert_eq!(h2_counts[0], 0, "h2 should never return 0");
        for &count in &h2_counts[1..] {
            // 10000 / 99 ≈ 101, allow wider range
            assert!(count > 50 && count < 160, "h2 uneven: {}", count);
        }
    }

    #[test]
    fn test_static_maglev() {
        let nodes: Vec<_> = (1..=5).collect();
        let static_maglev = StaticMaglev::build(&nodes, 1009);

        // Lookup should return valid index
        let idx = static_maglev.lookup(12345);
        assert!(idx < 5);

        // Consistency
        assert_eq!(static_maglev.lookup(12345), static_maglev.lookup(12345));
    }

    #[test]
    fn test_memory_usage() {
        let nodes: Vec<_> = (1..=100).collect();
        let maglev = MaglevHash::with_table_size(nodes, DEFAULT_TABLE_SIZE);

        // Should be roughly 2 * 65537 + 8 * 100 = ~131KB
        let usage = maglev.memory_usage();
        assert!(usage < 150_000, "Memory usage: {}", usage);
    }
}

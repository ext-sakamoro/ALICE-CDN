//! ALICE-CDN × ALICE-Cache Bridge
//!
//! Edge-node local caching for CDN content delivery.
//! Combines Vivaldi routing with predictive caching (Markov prefetch + TinyLFU).
//!
//! ```text
//! Client → CDN (Vivaldi route) → Cache (hit?) → Origin (miss)
//! ```

use crate::{ContentLocator, MaglevHash, VivaldiCoord};
use alice_cache::{AliceCache, CacheConfig};

/// CDN node with integrated local cache.
pub struct CachedCdnNode<V: Clone + Send + Sync + 'static> {
    pub node_id: u64,
    pub coord: VivaldiCoord,
    pub cache: AliceCache<u64, V>,
    pub maglev: MaglevHash,
}

/// Cache lookup result.
pub enum CdnCacheResult<V> {
    Hit(V),
    Miss,
}

impl<V: Clone + Send + Sync + 'static> CachedCdnNode<V> {
    /// Create a CDN node with local cache.
    pub fn new(
        node_id: u64,
        coord: VivaldiCoord,
        peer_ids: Vec<u64>,
        cache_capacity: usize,
    ) -> Self {
        let cache = AliceCache::with_config(CacheConfig {
            capacity: cache_capacity,
            num_nodes: peer_ids.len() as i32,
            node_id: node_id as u32,
            enable_oracle: true,
            ..Default::default()
        });
        let maglev = MaglevHash::new(peer_ids);
        Self {
            node_id,
            coord,
            cache,
            maglev,
        }
    }

    /// Fetch content: check cache first, return Miss if not found.
    pub fn fetch(&self, content_id: u64) -> CdnCacheResult<V> {
        match self.cache.get(&content_id) {
            Some(v) => CdnCacheResult::Hit(v),
            None => CdnCacheResult::Miss,
        }
    }

    /// Store content in local cache after origin fetch.
    pub fn store(&self, content_id: u64, value: V) {
        self.cache.put(content_id, value);
    }

    /// Check Markov prefetch: should we pre-fetch the next asset?
    pub fn should_prefetch(&self, current: &u64, next: &u64) -> bool {
        self.cache.should_prefetch(current, next)
    }

    /// Cache hit rate for this node.
    pub fn hit_rate(&self) -> f64 {
        self.cache.hit_rate()
    }

    /// O(1) Maglev routing: which peer owns this content?
    pub fn route(&self, content_id: u64) -> Option<u64> {
        self.maglev.lookup(content_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_cdn_node() {
        let coord = VivaldiCoord::at(0.0, 0.0, 0.0, 2.0);
        let node = CachedCdnNode::<Vec<u8>>::new(1, coord, vec![1, 2, 3], 100);

        // Miss on cold cache
        assert!(matches!(node.fetch(1001), CdnCacheResult::Miss));

        // Store and hit
        node.store(1001, vec![1, 2, 3]);
        assert!(matches!(node.fetch(1001), CdnCacheResult::Hit(_)));

        // Routing
        assert!(node.route(1001).is_some());
    }
}

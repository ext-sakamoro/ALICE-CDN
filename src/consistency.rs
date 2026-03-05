//! キャッシュ一貫性戦略 — バージョン管理付き退去ポリシー
//!
//! TTL・版数ベースのキャッシュ一貫性管理と、
//! LRU/LFU/ARC 退去ポリシーを提供する。

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, string::String, vec::Vec};
#[cfg(feature = "std")]
use std::collections::BTreeMap;

/// 退去ポリシー。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used。
    Lru,
    /// Least Frequently Used。
    Lfu,
    /// Adaptive Replacement Cache (LRU + LFU ハイブリッド)。
    Arc,
}

/// 一貫性設定。
#[derive(Debug, Clone, Copy)]
pub struct ConsistencyConfig {
    /// デフォルト TTL (秒)。0 は無期限。
    pub default_ttl: f64,
    /// 最大バージョン保持数。
    pub max_versions: u32,
    /// 退去ポリシー。
    pub policy: EvictionPolicy,
    /// 最大エントリー数。
    pub max_entries: usize,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            default_ttl: 300.0,
            max_versions: 3,
            policy: EvictionPolicy::Lru,
            max_entries: 10000,
        }
    }
}

/// バージョン付きキャッシュエントリー。
#[derive(Debug, Clone)]
struct VersionedEntry {
    /// バージョン番号。
    version: u64,
    /// 挿入時刻。
    inserted_at: f64,
    /// 最終アクセス時刻。
    last_access: f64,
    /// アクセス回数。
    access_count: u64,
    /// TTL (秒, 0 = 無期限)。
    ttl: f64,
    /// データサイズ (バイト)。
    size: usize,
}

impl VersionedEntry {
    fn is_expired(&self, now: f64) -> bool {
        self.ttl > 0.0 && (now - self.inserted_at) > self.ttl
    }
}

/// キャッシュ一貫性マネージャー。
#[derive(Debug)]
pub struct CacheConsistency {
    config: ConsistencyConfig,
    /// キー → エントリー。
    entries: BTreeMap<String, VersionedEntry>,
    /// グローバルバージョンカウンター。
    version_counter: u64,
}

impl CacheConsistency {
    /// 新しいマネージャーを作成。
    #[must_use]
    pub const fn new(config: ConsistencyConfig) -> Self {
        Self {
            config,
            entries: BTreeMap::new(),
            version_counter: 0,
        }
    }

    /// デフォルト設定で作成。
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ConsistencyConfig::default())
    }

    /// エントリーを登録/更新。バージョンを返す。
    pub fn put(&mut self, key: &str, size: usize, now: f64) -> u64 {
        self.version_counter += 1;
        let version = self.version_counter;

        let entry = VersionedEntry {
            version,
            inserted_at: now,
            last_access: now,
            access_count: 1,
            ttl: self.config.default_ttl,
            size,
        };

        self.entries.insert(key.into(), entry);

        // 容量超過時は退去
        while self.entries.len() > self.config.max_entries {
            self.evict_one(now);
        }

        version
    }

    /// カスタム TTL でエントリーを登録。
    pub fn put_with_ttl(&mut self, key: &str, size: usize, ttl: f64, now: f64) -> u64 {
        self.version_counter += 1;
        let version = self.version_counter;

        let entry = VersionedEntry {
            version,
            inserted_at: now,
            last_access: now,
            access_count: 1,
            ttl,
            size,
        };

        self.entries.insert(key.into(), entry);

        while self.entries.len() > self.config.max_entries {
            self.evict_one(now);
        }

        version
    }

    /// アクセス記録。有効なら `true`。
    pub fn access(&mut self, key: &str, now: f64) -> bool {
        if let Some(entry) = self.entries.get_mut(key) {
            if entry.is_expired(now) {
                self.entries.remove(key);
                return false;
            }
            entry.last_access = now;
            entry.access_count += 1;
            true
        } else {
            false
        }
    }

    /// エントリーのバージョンを取得。
    #[must_use]
    pub fn version(&self, key: &str) -> Option<u64> {
        self.entries.get(key).map(|e| e.version)
    }

    /// エントリーが有効か (存在 + 期限内)。
    #[must_use]
    pub fn is_valid(&self, key: &str, now: f64) -> bool {
        self.entries.get(key).is_some_and(|e| !e.is_expired(now))
    }

    /// バージョンが最新か (期待バージョン以上)。
    #[must_use]
    pub fn is_current(&self, key: &str, expected_version: u64) -> bool {
        self.entries
            .get(key)
            .is_some_and(|e| e.version >= expected_version)
    }

    /// 期限切れエントリーをパージ。パージ数を返す。
    pub fn invalidate_stale(&mut self, now: f64) -> usize {
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(now))
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired.len();
        for k in expired {
            self.entries.remove(&k);
        }
        count
    }

    /// 指定バージョン以前のエントリーを無効化。
    pub fn invalidate_before_version(&mut self, max_version: u64) -> usize {
        let stale: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.version < max_version)
            .map(|(k, _)| k.clone())
            .collect();
        let count = stale.len();
        for k in stale {
            self.entries.remove(&k);
        }
        count
    }

    /// エントリーを削除。
    pub fn remove(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }

    /// エントリー数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 空か。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 合計サイズ (バイト)。
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.entries.values().map(|e| e.size).sum()
    }

    /// ポリシーに従って1つ退去。
    fn evict_one(&mut self, now: f64) {
        // まず期限切れを探す
        let expired_key = self
            .entries
            .iter()
            .find(|(_, e)| e.is_expired(now))
            .map(|(k, _)| k.clone());
        if let Some(key) = expired_key {
            self.entries.remove(&key);
            return;
        }

        // ポリシーに従って退去
        let victim = match self.config.policy {
            EvictionPolicy::Lru => self
                .entries
                .iter()
                .min_by(|(_, a), (_, b)| {
                    a.last_access
                        .partial_cmp(&b.last_access)
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
                .map(|(k, _)| k.clone()),
            EvictionPolicy::Lfu => self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.access_count)
                .map(|(k, _)| k.clone()),
            EvictionPolicy::Arc => {
                // ARC 簡易: LRU と LFU のスコアを組み合わせ
                self.entries
                    .iter()
                    .min_by(|(_, a), (_, b)| {
                        let score_a = (a.access_count as f64).mul_add(0.5, a.last_access * 0.5);
                        let score_b = (b.access_count as f64).mul_add(0.5, b.last_access * 0.5);
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(core::cmp::Ordering::Equal)
                    })
                    .map(|(k, _)| k.clone())
            }
        };

        if let Some(key) = victim {
            self.entries.remove(&key);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_and_access() {
        let mut cc = CacheConsistency::with_defaults();
        let v = cc.put("k1", 100, 1.0);
        assert_eq!(v, 1);
        assert!(cc.access("k1", 2.0));
        assert_eq!(cc.len(), 1);
    }

    #[test]
    fn version_increments() {
        let mut cc = CacheConsistency::with_defaults();
        let v1 = cc.put("a", 10, 1.0);
        let v2 = cc.put("b", 10, 1.0);
        assert_eq!(v2, v1 + 1);
    }

    #[test]
    fn ttl_expiry() {
        let config = ConsistencyConfig {
            default_ttl: 5.0,
            ..Default::default()
        };
        let mut cc = CacheConsistency::new(config);
        cc.put("k1", 10, 1.0);
        assert!(cc.is_valid("k1", 3.0));
        assert!(!cc.is_valid("k1", 7.0));
    }

    #[test]
    fn access_expired_removes() {
        let config = ConsistencyConfig {
            default_ttl: 2.0,
            ..Default::default()
        };
        let mut cc = CacheConsistency::new(config);
        cc.put("k1", 10, 1.0);
        assert!(!cc.access("k1", 5.0));
        assert_eq!(cc.len(), 0);
    }

    #[test]
    fn invalidate_stale() {
        let config = ConsistencyConfig {
            default_ttl: 5.0,
            ..Default::default()
        };
        let mut cc = CacheConsistency::new(config);
        cc.put("a", 10, 1.0);
        cc.put("b", 10, 1.0);
        let purged = cc.invalidate_stale(7.0);
        assert_eq!(purged, 2);
        assert!(cc.is_empty());
    }

    #[test]
    fn invalidate_before_version() {
        let mut cc = CacheConsistency::with_defaults();
        cc.put("a", 10, 1.0); // v1
        cc.put("b", 10, 1.0); // v2
        cc.put("c", 10, 1.0); // v3
        let purged = cc.invalidate_before_version(3);
        assert_eq!(purged, 2);
        assert_eq!(cc.len(), 1);
    }

    #[test]
    fn is_current() {
        let mut cc = CacheConsistency::with_defaults();
        cc.put("k1", 10, 1.0); // v1
        assert!(cc.is_current("k1", 1));
        assert!(!cc.is_current("k1", 2));
    }

    #[test]
    fn eviction_lru() {
        let config = ConsistencyConfig {
            max_entries: 2,
            policy: EvictionPolicy::Lru,
            default_ttl: 0.0,
            ..Default::default()
        };
        let mut cc = CacheConsistency::new(config);
        cc.put("a", 10, 1.0);
        cc.put("b", 10, 2.0);
        cc.access("a", 3.0); // "a" は最近アクセス
        cc.put("c", 10, 4.0); // "b" が退去されるはず
        assert_eq!(cc.len(), 2);
        assert!(cc.version("a").is_some());
        assert!(cc.version("b").is_none());
    }

    #[test]
    fn eviction_lfu() {
        let config = ConsistencyConfig {
            max_entries: 2,
            policy: EvictionPolicy::Lfu,
            default_ttl: 0.0,
            ..Default::default()
        };
        let mut cc = CacheConsistency::new(config);
        cc.put("a", 10, 1.0);
        cc.put("b", 10, 1.0);
        // "a" を3回アクセス
        cc.access("a", 2.0);
        cc.access("a", 3.0);
        cc.access("a", 4.0);
        cc.put("c", 10, 5.0); // "b" が退去 (アクセス少ない)
        assert!(cc.version("a").is_some());
        assert!(cc.version("b").is_none());
    }

    #[test]
    fn custom_ttl() {
        let mut cc = CacheConsistency::with_defaults();
        cc.put_with_ttl("k1", 10, 1.0, 1.0); // TTL=1秒
        assert!(cc.is_valid("k1", 1.5));
        assert!(!cc.is_valid("k1", 3.0));
    }

    #[test]
    fn remove() {
        let mut cc = CacheConsistency::with_defaults();
        cc.put("k1", 10, 1.0);
        assert!(cc.remove("k1"));
        assert!(!cc.remove("k1"));
    }

    #[test]
    fn total_size() {
        let mut cc = CacheConsistency::with_defaults();
        cc.put("a", 100, 1.0);
        cc.put("b", 200, 1.0);
        assert_eq!(cc.total_size(), 300);
    }

    #[test]
    fn default_config() {
        let config = ConsistencyConfig::default();
        assert!((config.default_ttl - 300.0).abs() < 1e-10);
        assert_eq!(config.policy, EvictionPolicy::Lru);
    }
}

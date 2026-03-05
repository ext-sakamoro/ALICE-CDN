//! レプリカ配置最適化 — Vivaldi 座標ベースの分散コンテンツ配置
//!
//! コンテンツレプリカを複数エッジノードに最適配置し、
//! レイテンシ・負荷・地理的親和性を考慮した配置戦略を提供する。

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// 配置戦略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementStrategy {
    /// レイテンシ最適: クライアント群に最も近いノードを選択。
    LatencyOptimal,
    /// 負荷分散: ノード間の負荷を均等化。
    LoadBalanced,
    /// 地理分散: レプリカを地理的に離散させる。
    Geographic,
}

/// ノード情報 (配置計算用)。
#[derive(Debug, Clone)]
pub struct PlacementNode {
    /// ノードID。
    pub id: u64,
    /// Vivaldi 座標 (x, y, z)。
    pub coord: [i64; 3],
    /// 現在の負荷 (0–100)。
    pub load: u32,
    /// 最大容量 (レプリカ数)。
    pub capacity: u32,
    /// 現在のレプリカ数。
    pub replica_count: u32,
}

impl PlacementNode {
    /// 新しいノード情報を作成。
    #[must_use]
    pub const fn new(id: u64, coord: [i64; 3], capacity: u32) -> Self {
        Self {
            id,
            coord,
            load: 0,
            replica_count: 0,
            capacity,
        }
    }

    /// 受け入れ可能か。
    #[must_use]
    pub const fn can_accept(&self) -> bool {
        self.replica_count < self.capacity
    }

    /// 空き容量率 (0.0–1.0)。
    #[must_use]
    pub fn availability(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        f64::from(self.capacity - self.replica_count.min(self.capacity)) / f64::from(self.capacity)
    }
}

/// 配置結果。
#[derive(Debug, Clone)]
pub struct PlacementResult {
    /// 配置先ノードIDリスト。
    pub node_ids: Vec<u64>,
    /// 配置スコア (低いほど良い)。
    pub score: f64,
}

/// 2点間のユークリッド距離の二乗 (整数)。
const fn dist_sq(a: [i64; 3], b: [i64; 3]) -> i64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// レプリカ配置を計算。
///
/// `replica_count` 個のレプリカを `nodes` から選択する。
#[must_use]
pub fn compute_placement(
    nodes: &[PlacementNode],
    client_coords: &[[i64; 3]],
    replica_count: usize,
    strategy: PlacementStrategy,
) -> PlacementResult {
    if nodes.is_empty() || replica_count == 0 {
        return PlacementResult {
            node_ids: Vec::new(),
            score: f64::MAX,
        };
    }

    let available: Vec<&PlacementNode> = nodes.iter().filter(|n| n.can_accept()).collect();
    if available.is_empty() {
        return PlacementResult {
            node_ids: Vec::new(),
            score: f64::MAX,
        };
    }

    let count = replica_count.min(available.len());

    match strategy {
        PlacementStrategy::LatencyOptimal => {
            place_latency_optimal(&available, client_coords, count)
        }
        PlacementStrategy::LoadBalanced => place_load_balanced(&available, count),
        PlacementStrategy::Geographic => place_geographic(&available, count),
    }
}

/// レイテンシ最適配置: クライアント群との総距離が最小のノードを選択。
fn place_latency_optimal(
    nodes: &[&PlacementNode],
    clients: &[[i64; 3]],
    count: usize,
) -> PlacementResult {
    let mut scored: Vec<(u64, f64)> = nodes
        .iter()
        .map(|n| {
            let total_dist: f64 = if clients.is_empty() {
                0.0
            } else {
                clients
                    .iter()
                    .map(|c| (dist_sq(n.coord, *c) as f64).sqrt())
                    .sum()
            };
            (n.id, total_dist)
        })
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

    let node_ids: Vec<u64> = scored.iter().take(count).map(|(id, _)| *id).collect();
    let score = scored.iter().take(count).map(|(_, s)| *s).sum();

    PlacementResult { node_ids, score }
}

/// 負荷分散配置: 負荷の低いノードを優先。
fn place_load_balanced(nodes: &[&PlacementNode], count: usize) -> PlacementResult {
    let mut scored: Vec<(u64, f64)> = nodes
        .iter()
        .map(|n| {
            let score = (1.0 - n.availability()).mul_add(100.0, f64::from(n.load));
            (n.id, score)
        })
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

    let node_ids: Vec<u64> = scored.iter().take(count).map(|(id, _)| *id).collect();
    let score = scored.iter().take(count).map(|(_, s)| *s).sum();

    PlacementResult { node_ids, score }
}

/// 地理分散配置: レプリカ間の最小距離を最大化 (貪欲法)。
fn place_geographic(nodes: &[&PlacementNode], count: usize) -> PlacementResult {
    let mut selected: Vec<usize> = Vec::with_capacity(count);

    // 最初のノードは最も空いているものを選択
    let first = nodes
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.availability()
                .partial_cmp(&b.availability())
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .map_or(0, |(i, _)| i);
    selected.push(first);

    // 残りは既選択ノード群からの最小距離が最大のものを選択
    while selected.len() < count {
        let mut best_idx = 0;
        let mut best_min_dist = i64::MIN;

        for (i, node) in nodes.iter().enumerate() {
            if selected.contains(&i) {
                continue;
            }
            let min_dist = selected
                .iter()
                .map(|&si| dist_sq(node.coord, nodes[si].coord))
                .min()
                .unwrap_or(0);
            if min_dist > best_min_dist {
                best_min_dist = min_dist;
                best_idx = i;
            }
        }

        if selected.contains(&best_idx) {
            break; // 全ノード選択済み
        }
        selected.push(best_idx);
    }

    let node_ids: Vec<u64> = selected.iter().map(|&i| nodes[i].id).collect();
    let score = if selected.len() >= 2 {
        // スコア = レプリカ間最小距離の逆数 (大きいほど分散)
        let min_pair_dist = selected
            .iter()
            .enumerate()
            .flat_map(|(i, &a)| {
                selected[i + 1..]
                    .iter()
                    .map(move |&b| dist_sq(nodes[a].coord, nodes[b].coord))
            })
            .min()
            .unwrap_or(1);
        1.0 / (min_pair_dist as f64).sqrt().max(1.0)
    } else {
        0.0
    };

    PlacementResult { node_ids, score }
}

/// 再配置判定: 現在の配置の品質スコアを計算。
///
/// 返り値が `threshold` を超えたら `rebalance` すべき。
#[must_use]
pub fn placement_imbalance(nodes: &[PlacementNode]) -> f64 {
    if nodes.is_empty() {
        return 0.0;
    }
    let loads: Vec<f64> = nodes.iter().map(|n| f64::from(n.load)).collect();
    let mean = loads.iter().sum::<f64>() / loads.len() as f64;
    if mean < 1e-10 {
        return 0.0;
    }
    let variance = loads.iter().map(|l| (l - mean) * (l - mean)).sum::<f64>() / loads.len() as f64;
    variance.sqrt() / mean // 変動係数
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_nodes() -> Vec<PlacementNode> {
        vec![
            PlacementNode::new(1, [0, 0, 0], 10),
            PlacementNode::new(2, [100, 0, 0], 10),
            PlacementNode::new(3, [0, 100, 0], 10),
            PlacementNode::new(4, [100, 100, 0], 10),
        ]
    }

    #[test]
    fn latency_optimal_picks_closest() {
        let nodes = test_nodes();
        let clients = vec![[0, 0, 0], [10, 10, 0]];
        let result = compute_placement(&nodes, &clients, 1, PlacementStrategy::LatencyOptimal);
        assert_eq!(result.node_ids.len(), 1);
        assert_eq!(result.node_ids[0], 1); // 原点に最も近い
    }

    #[test]
    fn load_balanced_picks_lightest() {
        let mut nodes = test_nodes();
        nodes[0].load = 90;
        nodes[1].load = 10;
        nodes[2].load = 50;
        nodes[3].load = 30;
        let result = compute_placement(&nodes, &[], 1, PlacementStrategy::LoadBalanced);
        assert_eq!(result.node_ids[0], 2); // load=10 が最軽量
    }

    #[test]
    fn geographic_spreads_replicas() {
        let nodes = test_nodes();
        let result = compute_placement(&nodes, &[], 2, PlacementStrategy::Geographic);
        assert_eq!(result.node_ids.len(), 2);
        // 2つのレプリカは異なるノード
        assert_ne!(result.node_ids[0], result.node_ids[1]);
    }

    #[test]
    fn respects_capacity() {
        let mut nodes = test_nodes();
        nodes[0].replica_count = 10; // 満杯
        let result = compute_placement(&nodes, &[], 1, PlacementStrategy::LoadBalanced);
        assert!(!result.node_ids.contains(&1));
    }

    #[test]
    fn empty_nodes() {
        let result = compute_placement(&[], &[], 3, PlacementStrategy::LatencyOptimal);
        assert!(result.node_ids.is_empty());
    }

    #[test]
    fn replica_count_capped() {
        let nodes = test_nodes();
        let result = compute_placement(&nodes, &[], 100, PlacementStrategy::LatencyOptimal);
        assert_eq!(result.node_ids.len(), 4); // 最大4ノード
    }

    #[test]
    fn node_availability() {
        let mut node = PlacementNode::new(1, [0, 0, 0], 10);
        assert!((node.availability() - 1.0).abs() < 1e-10);
        node.replica_count = 5;
        assert!((node.availability() - 0.5).abs() < 1e-10);
        node.replica_count = 10;
        assert!((node.availability() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn node_zero_capacity() {
        let node = PlacementNode::new(1, [0, 0, 0], 0);
        assert!(!node.can_accept());
        assert!((node.availability() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn placement_imbalance_even() {
        let mut nodes = test_nodes();
        for n in &mut nodes {
            n.load = 50;
        }
        let imbalance = placement_imbalance(&nodes);
        assert!(imbalance < 0.01);
    }

    #[test]
    fn placement_imbalance_skewed() {
        let mut nodes = test_nodes();
        nodes[0].load = 100;
        nodes[1].load = 0;
        nodes[2].load = 0;
        nodes[3].load = 0;
        let imbalance = placement_imbalance(&nodes);
        assert!(imbalance > 1.0);
    }

    #[test]
    fn placement_imbalance_empty() {
        assert!((placement_imbalance(&[]) - 0.0).abs() < 1e-10);
    }
}

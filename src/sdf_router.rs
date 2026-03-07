//! SDF-aware ルーティング — 空間セルベースのエッジノード割り当て
//!
//! SDF シーンを空間セルに分割し、各セルのコンテンツを
//! 最適なエッジノードにルーティングする。
//! Vivaldi 座標との連携でレイテンシ最適化を実現。

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// 3次元空間セル (整数座標)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SdfCell {
    /// セル X 座標。
    pub x: i32,
    /// セル Y 座標。
    pub y: i32,
    /// セル Z 座標。
    pub z: i32,
    /// LOD レベル (0 = 最高解像度)。
    pub lod: u8,
}

impl SdfCell {
    /// 新しいセルを作成。
    #[must_use]
    pub const fn new(x: i32, y: i32, z: i32, lod: u8) -> Self {
        Self { x, y, z, lod }
    }

    /// ワールド座標からセルへ変換。`cell_size` はセルの一辺の長さ。
    #[must_use]
    pub fn from_world(wx: f64, wy: f64, wz: f64, cell_size: f64, lod: u8) -> Self {
        let scale = cell_size * f64::from(1u32 << u32::from(lod));
        Self {
            x: (wx / scale).floor() as i32,
            y: (wy / scale).floor() as i32,
            z: (wz / scale).floor() as i32,
            lod,
        }
    }

    /// セル間のチェビシェフ距離。
    #[must_use]
    pub fn chebyshev_distance(&self, other: &Self) -> u32 {
        let dx = (self.x - other.x).unsigned_abs();
        let dy = (self.y - other.y).unsigned_abs();
        let dz = (self.z - other.z).unsigned_abs();
        dx.max(dy).max(dz)
    }

    /// セル間のマンハッタン距離。
    #[must_use]
    pub const fn manhattan_distance(&self, other: &Self) -> u32 {
        let dx = (self.x - other.x).unsigned_abs();
        let dy = (self.y - other.y).unsigned_abs();
        let dz = (self.z - other.z).unsigned_abs();
        dx + dy + dz
    }

    /// セルの空間ハッシュ (ノード割り当て用)。
    #[must_use]
    pub fn spatial_hash(&self) -> u64 {
        // MurmurHash3 風のミキシング
        let mut h: u64 = 0x517c_c1b7_2722_0a95;
        h ^= self.x as u64;
        h = h.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h ^= self.y as u64;
        h = h.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h ^= self.z as u64;
        h = h.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h ^= u64::from(self.lod);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h
    }

    /// 隣接セル (26方向 + 自身) を列挙。
    #[must_use]
    pub fn neighbors_27(&self) -> Vec<Self> {
        let mut out = Vec::with_capacity(27);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    out.push(Self::new(self.x + dx, self.y + dy, self.z + dz, self.lod));
                }
            }
        }
        out
    }
}

/// エッジノードのセル担当情報。
#[derive(Debug, Clone)]
pub struct CellAssignment {
    /// ノードID。
    pub node_id: u64,
    /// 担当セルリスト。
    pub cells: Vec<SdfCell>,
    /// 現在の負荷 (0–100)。
    pub load: u32,
}

/// SDF ルーター — セルをエッジノードに割り当て。
#[derive(Debug)]
pub struct SdfRouter {
    /// ノードごとの担当情報。
    assignments: Vec<CellAssignment>,
}

impl SdfRouter {
    /// 新しいルーターを作成。
    #[must_use]
    pub fn new(node_ids: &[u64]) -> Self {
        Self {
            assignments: node_ids
                .iter()
                .map(|&id| CellAssignment {
                    node_id: id,
                    cells: Vec::new(),
                    load: 0,
                })
                .collect(),
        }
    }

    /// セルを担当ノードにルーティング。ハッシュベースで割り当て。
    #[must_use]
    pub fn route(&self, cell: &SdfCell) -> Option<u64> {
        if self.assignments.is_empty() {
            return None;
        }
        let hash = cell.spatial_hash();
        let idx = (hash as usize) % self.assignments.len();
        Some(self.assignments[idx].node_id)
    }

    /// 負荷を考慮したルーティング。負荷の低いノードを優先。
    #[must_use]
    pub fn route_load_aware(&self, cell: &SdfCell) -> Option<u64> {
        if self.assignments.is_empty() {
            return None;
        }

        let hash = cell.spatial_hash();
        // 上位2候補から負荷の低い方を選択
        let idx1 = (hash as usize) % self.assignments.len();
        let idx2 = ((hash >> 32) as usize) % self.assignments.len();

        let a1 = &self.assignments[idx1];
        let a2 = &self.assignments[idx2];

        if a1.load <= a2.load {
            Some(a1.node_id)
        } else {
            Some(a2.node_id)
        }
    }

    /// ノードの負荷を更新。
    pub fn update_load(&mut self, node_id: u64, load: u32) {
        if let Some(a) = self.assignments.iter_mut().find(|a| a.node_id == node_id) {
            a.load = load;
        }
    }

    /// セルの親和性スコアを計算。同じノードに隣接セルが多いほど高い。
    #[must_use]
    pub fn cell_affinity(&self, cell: &SdfCell) -> f64 {
        if self.assignments.is_empty() {
            return 0.0;
        }

        let Some(primary) = self.route(cell) else {
            return 0.0;
        };

        let neighbors = cell.neighbors_27();
        let same_node_count = neighbors
            .iter()
            .filter(|n| self.route(n) == Some(primary))
            .count();

        // 27隣接中、同じノードに割り当てられた比率
        same_node_count as f64 / 27.0
    }

    /// ノード数。
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.assignments.len()
    }

    /// 各ノードの負荷を取得。
    #[must_use]
    pub fn loads(&self) -> Vec<(u64, u32)> {
        self.assignments
            .iter()
            .map(|a| (a.node_id, a.load))
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_from_world() {
        let cell = SdfCell::from_world(15.0, 25.0, -5.0, 10.0, 0);
        assert_eq!(cell.x, 1);
        assert_eq!(cell.y, 2);
        assert_eq!(cell.z, -1);
        assert_eq!(cell.lod, 0);
    }

    #[test]
    fn cell_from_world_lod() {
        // LOD 1 → cell_size = 10 * 2 = 20
        let cell = SdfCell::from_world(15.0, 25.0, -5.0, 10.0, 1);
        assert_eq!(cell.x, 0); // 15/20 = 0.75 → floor = 0
        assert_eq!(cell.y, 1); // 25/20 = 1.25 → floor = 1
    }

    #[test]
    fn chebyshev_distance() {
        let a = SdfCell::new(0, 0, 0, 0);
        let b = SdfCell::new(3, 5, 2, 0);
        assert_eq!(a.chebyshev_distance(&b), 5);
    }

    #[test]
    fn manhattan_distance() {
        let a = SdfCell::new(0, 0, 0, 0);
        let b = SdfCell::new(3, 5, 2, 0);
        assert_eq!(a.manhattan_distance(&b), 10);
    }

    #[test]
    fn spatial_hash_deterministic() {
        let cell = SdfCell::new(10, 20, 30, 0);
        assert_eq!(cell.spatial_hash(), cell.spatial_hash());
    }

    #[test]
    fn spatial_hash_different_cells() {
        let a = SdfCell::new(0, 0, 0, 0);
        let b = SdfCell::new(1, 0, 0, 0);
        assert_ne!(a.spatial_hash(), b.spatial_hash());
    }

    #[test]
    fn neighbors_27_count() {
        let cell = SdfCell::new(5, 5, 5, 0);
        assert_eq!(cell.neighbors_27().len(), 27);
    }

    #[test]
    fn route_basic() {
        let router = SdfRouter::new(&[1, 2, 3]);
        let cell = SdfCell::new(0, 0, 0, 0);
        let node = router.route(&cell).unwrap();
        assert!([1, 2, 3].contains(&node));
    }

    #[test]
    fn route_consistent() {
        let router = SdfRouter::new(&[1, 2, 3, 4, 5]);
        let cell = SdfCell::new(10, 20, 30, 0);
        let n1 = router.route(&cell);
        let n2 = router.route(&cell);
        assert_eq!(n1, n2);
    }

    #[test]
    fn route_empty() {
        let router = SdfRouter::new(&[]);
        assert!(router.route(&SdfCell::new(0, 0, 0, 0)).is_none());
    }

    #[test]
    fn route_load_aware() {
        let mut router = SdfRouter::new(&[1, 2]);
        router.update_load(1, 90);
        router.update_load(2, 10);
        // 負荷考慮ルーティングが値を返すことを検証
        let cell = SdfCell::new(0, 0, 0, 0);
        assert!(router.route_load_aware(&cell).is_some());
    }

    #[test]
    fn cell_affinity_single_node() {
        let router = SdfRouter::new(&[1]);
        let cell = SdfCell::new(0, 0, 0, 0);
        // 全セルが同じノードなのでaffinity = 1.0
        assert!((router.cell_affinity(&cell) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn node_count_and_loads() {
        let mut router = SdfRouter::new(&[10, 20, 30]);
        assert_eq!(router.node_count(), 3);
        router.update_load(20, 50);
        let loads = router.loads();
        assert_eq!(loads.len(), 3);
        assert!(loads.iter().any(|&(id, l)| id == 20 && l == 50));
    }
}

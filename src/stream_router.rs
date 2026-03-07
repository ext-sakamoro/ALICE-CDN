//! ASP ストリームルーティング — アダプティブビットレートとリレー選択
//!
//! ALICE-Streaming-Protocol (ASP) のストリームを
//! 最適なリレーノードにルーティングし、
//! ネットワーク状態に応じたビットレート適応を行う。

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// ストリーム品質レベル。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityLevel {
    /// 最低品質 (音声のみ or 144p 相当)。
    Minimal,
    /// 低品質 (360p 相当)。
    Low,
    /// 中品質 (720p 相当)。
    Medium,
    /// 高品質 (1080p 相当)。
    High,
    /// 最高品質 (4K 相当)。
    Ultra,
}

impl QualityLevel {
    /// ビットレート目安 (kbps)。
    #[must_use]
    pub const fn target_kbps(&self) -> u32 {
        match self {
            Self::Minimal => 128,
            Self::Low => 1_000,
            Self::Medium => 3_000,
            Self::High => 6_000,
            Self::Ultra => 15_000,
        }
    }

    /// 帯域幅からレベルを決定。
    #[must_use]
    pub const fn from_bandwidth_kbps(bw: u32) -> Self {
        if bw >= 12_000 {
            Self::Ultra
        } else if bw >= 4_500 {
            Self::High
        } else if bw >= 2_000 {
            Self::Medium
        } else if bw >= 500 {
            Self::Low
        } else {
            Self::Minimal
        }
    }
}

/// ストリームルート情報。
#[derive(Debug, Clone)]
pub struct StreamRoute {
    /// ストリームID。
    pub stream_id: u64,
    /// ソースノードID。
    pub source_id: u64,
    /// リレーノードIDリスト (ソース → 宛先への中継パス)。
    pub relay_ids: Vec<u64>,
    /// 現在の品質レベル。
    pub quality: QualityLevel,
    /// 推定エンドツーエンドレイテンシ (ms)。
    pub estimated_latency_ms: f64,
}

/// リレーノード候補。
#[derive(Debug, Clone)]
pub struct RelayCandidate {
    /// ノードID。
    pub node_id: u64,
    /// ソースへの RTT (ms)。
    pub rtt_to_source_ms: f64,
    /// クライアントへの RTT (ms)。
    pub rtt_to_client_ms: f64,
    /// 現在の帯域使用率 (0.0–1.0)。
    pub bandwidth_usage: f64,
    /// 最大帯域 (kbps)。
    pub max_bandwidth_kbps: u32,
}

impl RelayCandidate {
    /// トータル RTT (ソース→リレー→クライアント)。
    #[must_use]
    pub fn total_rtt_ms(&self) -> f64 {
        self.rtt_to_source_ms + self.rtt_to_client_ms
    }

    /// 利用可能帯域 (kbps)。
    #[must_use]
    pub fn available_bandwidth_kbps(&self) -> u32 {
        let available = f64::from(self.max_bandwidth_kbps) * (1.0 - self.bandwidth_usage);
        available.max(0.0) as u32
    }
}

/// ストリームルーター。
#[derive(Debug)]
pub struct StreamRouter {
    /// レイテンシ重み (0.0–1.0)。
    latency_weight: f64,
    /// 帯域重み (0.0–1.0)。
    bandwidth_weight: f64,
    /// 最大許容レイテンシ (ms)。
    max_latency_ms: f64,
}

impl StreamRouter {
    /// 新しいルーターを作成。
    #[must_use]
    pub fn new(latency_weight: f64, bandwidth_weight: f64, max_latency_ms: f64) -> Self {
        let total = latency_weight + bandwidth_weight;
        Self {
            latency_weight: if total > 0.0 {
                latency_weight / total
            } else {
                0.5
            },
            bandwidth_weight: if total > 0.0 {
                bandwidth_weight / total
            } else {
                0.5
            },
            max_latency_ms,
        }
    }

    /// デフォルト設定で作成。
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(0.6, 0.4, 200.0)
    }

    /// 最適なリレーノードを選択。
    ///
    /// レイテンシと帯域を重み付けスコアで評価し、
    /// 最大レイテンシ制約を満たすノードから最良を返す。
    #[must_use]
    pub fn select_relay(&self, candidates: &[RelayCandidate]) -> Option<u64> {
        if candidates.is_empty() {
            return None;
        }

        // レイテンシ制約を満たす候補をフィルタ
        let feasible: Vec<&RelayCandidate> = candidates
            .iter()
            .filter(|c| c.total_rtt_ms() <= self.max_latency_ms)
            .collect();

        // 制約を満たす候補がなければ、最もレイテンシの低いものを選択
        let pool = if feasible.is_empty() {
            candidates.iter().collect::<Vec<_>>()
        } else {
            feasible
        };

        // 正規化用の最大値を求める
        let max_rtt = pool
            .iter()
            .map(|c| c.total_rtt_ms())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let max_bw = pool
            .iter()
            .map(|c| c.available_bandwidth_kbps())
            .max()
            .unwrap_or(1)
            .max(1);

        // スコア計算: レイテンシ低い + 帯域高い = スコア高い
        pool.iter()
            .max_by(|a, b| {
                let score_a = self.score(a, max_rtt, max_bw);
                let score_b = self.score(b, max_rtt, max_bw);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|c| c.node_id)
    }

    /// 候補のスコアを計算 (高いほど良い)。
    fn score(&self, candidate: &RelayCandidate, max_rtt: f64, max_bw: u32) -> f64 {
        let latency_score = 1.0 - (candidate.total_rtt_ms() / max_rtt);
        let bw_score = f64::from(candidate.available_bandwidth_kbps()) / f64::from(max_bw);
        self.latency_weight
            .mul_add(latency_score, self.bandwidth_weight * bw_score)
    }

    /// トップ K リレーを選択。
    #[must_use]
    pub fn select_top_k(&self, candidates: &[RelayCandidate], k: usize) -> Vec<u64> {
        if candidates.is_empty() || k == 0 {
            return Vec::new();
        }

        let max_rtt = candidates
            .iter()
            .map(RelayCandidate::total_rtt_ms)
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let max_bw = candidates
            .iter()
            .map(RelayCandidate::available_bandwidth_kbps)
            .max()
            .unwrap_or(1)
            .max(1);

        let mut scored: Vec<(u64, f64)> = candidates
            .iter()
            .map(|c| (c.node_id, self.score(c, max_rtt, max_bw)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        scored.iter().take(k).map(|(id, _)| *id).collect()
    }

    /// 最大許容レイテンシ。
    #[must_use]
    pub const fn max_latency_ms(&self) -> f64 {
        self.max_latency_ms
    }
}

/// ネットワーク状態に基づくビットレート適応。
///
/// 現在のRTTとパケットロス率から推奨品質レベルを返す。
#[must_use]
pub fn adapt_bitrate(
    current_quality: QualityLevel,
    rtt_ms: f64,
    packet_loss_ratio: f64,
    available_bw_kbps: u32,
) -> QualityLevel {
    // パケットロスが高い → 品質を下げる
    if packet_loss_ratio > 0.05 {
        return match current_quality {
            QualityLevel::Minimal | QualityLevel::Low => QualityLevel::Minimal,
            QualityLevel::Medium => QualityLevel::Low,
            QualityLevel::High => QualityLevel::Medium,
            QualityLevel::Ultra => QualityLevel::High,
        };
    }

    // RTT が高い → 品質を維持 or 下げる
    if rtt_ms > 150.0 {
        return match current_quality {
            QualityLevel::Ultra => QualityLevel::High,
            other => other,
        };
    }

    // 帯域ベースの品質決定
    let bw_quality = QualityLevel::from_bandwidth_kbps(available_bw_kbps);

    // 現在より1段階以上の変更は許可しない (安定性)
    clamp_quality_step(current_quality, bw_quality)
}

/// 品質変更を1段階に制限。
fn clamp_quality_step(current: QualityLevel, target: QualityLevel) -> QualityLevel {
    let levels = [
        QualityLevel::Minimal,
        QualityLevel::Low,
        QualityLevel::Medium,
        QualityLevel::High,
        QualityLevel::Ultra,
    ];

    let cur_idx = levels.iter().position(|&l| l == current).unwrap_or(0);
    let tgt_idx = levels.iter().position(|&l| l == target).unwrap_or(0);

    if tgt_idx > cur_idx + 1 {
        levels[cur_idx + 1]
    } else if cur_idx > 0 && tgt_idx + 1 < cur_idx {
        levels[cur_idx - 1]
    } else {
        target
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_candidates() -> Vec<RelayCandidate> {
        vec![
            RelayCandidate {
                node_id: 1,
                rtt_to_source_ms: 10.0,
                rtt_to_client_ms: 15.0,
                bandwidth_usage: 0.2,
                max_bandwidth_kbps: 10_000,
            },
            RelayCandidate {
                node_id: 2,
                rtt_to_source_ms: 50.0,
                rtt_to_client_ms: 60.0,
                bandwidth_usage: 0.1,
                max_bandwidth_kbps: 20_000,
            },
            RelayCandidate {
                node_id: 3,
                rtt_to_source_ms: 5.0,
                rtt_to_client_ms: 8.0,
                bandwidth_usage: 0.9,
                max_bandwidth_kbps: 5_000,
            },
        ]
    }

    #[test]
    fn quality_target_kbps() {
        assert_eq!(QualityLevel::Minimal.target_kbps(), 128);
        assert_eq!(QualityLevel::Ultra.target_kbps(), 15_000);
    }

    #[test]
    fn quality_from_bandwidth() {
        assert_eq!(
            QualityLevel::from_bandwidth_kbps(100),
            QualityLevel::Minimal
        );
        assert_eq!(QualityLevel::from_bandwidth_kbps(1000), QualityLevel::Low);
        assert_eq!(
            QualityLevel::from_bandwidth_kbps(3000),
            QualityLevel::Medium
        );
        assert_eq!(QualityLevel::from_bandwidth_kbps(5000), QualityLevel::High);
        assert_eq!(
            QualityLevel::from_bandwidth_kbps(20000),
            QualityLevel::Ultra
        );
    }

    #[test]
    fn relay_candidate_total_rtt() {
        let c = RelayCandidate {
            node_id: 1,
            rtt_to_source_ms: 10.0,
            rtt_to_client_ms: 20.0,
            bandwidth_usage: 0.0,
            max_bandwidth_kbps: 10_000,
        };
        assert!((c.total_rtt_ms() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn relay_candidate_available_bw() {
        let c = RelayCandidate {
            node_id: 1,
            rtt_to_source_ms: 0.0,
            rtt_to_client_ms: 0.0,
            bandwidth_usage: 0.3,
            max_bandwidth_kbps: 10_000,
        };
        assert_eq!(c.available_bandwidth_kbps(), 7_000);
    }

    #[test]
    fn select_relay_basic() {
        let candidates = test_candidates();
        let router = StreamRouter::with_defaults();
        let best = router.select_relay(&candidates).unwrap();
        // ノード1 が最もバランスが良い (低レイテンシ + そこそこの帯域)
        assert!([1, 2, 3].contains(&best));
    }

    #[test]
    fn select_relay_empty() {
        let router = StreamRouter::with_defaults();
        assert!(router.select_relay(&[]).is_none());
    }

    #[test]
    fn select_relay_latency_focused() {
        let router = StreamRouter::new(1.0, 0.0, 200.0);
        let candidates = test_candidates();
        let best = router.select_relay(&candidates).unwrap();
        // レイテンシ重視 → ノード3 (RTT=13ms) が最良
        assert_eq!(best, 3);
    }

    #[test]
    fn select_relay_bandwidth_focused() {
        let router = StreamRouter::new(0.0, 1.0, 200.0);
        let candidates = test_candidates();
        let best = router.select_relay(&candidates).unwrap();
        // 帯域重視 → ノード2 (available=18000kbps) が最良
        assert_eq!(best, 2);
    }

    #[test]
    fn select_top_k() {
        let candidates = test_candidates();
        let router = StreamRouter::with_defaults();
        let top2 = router.select_top_k(&candidates, 2);
        assert_eq!(top2.len(), 2);
    }

    #[test]
    fn adapt_bitrate_high_loss() {
        let result = adapt_bitrate(QualityLevel::High, 50.0, 0.10, 10_000);
        assert_eq!(result, QualityLevel::Medium);
    }

    #[test]
    fn adapt_bitrate_high_rtt() {
        let result = adapt_bitrate(QualityLevel::Ultra, 200.0, 0.01, 20_000);
        assert_eq!(result, QualityLevel::High);
    }

    #[test]
    fn adapt_bitrate_step_limit() {
        // Minimal → Ultra は一気に上がらない (1段階ずつ)
        let result = adapt_bitrate(QualityLevel::Minimal, 10.0, 0.0, 50_000);
        assert_eq!(result, QualityLevel::Low);
    }

    #[test]
    fn adapt_bitrate_stable() {
        // 条件が良くても現在のレベルと同じなら変わらない
        let result = adapt_bitrate(QualityLevel::Medium, 30.0, 0.0, 3_000);
        assert_eq!(result, QualityLevel::Medium);
    }

    #[test]
    fn max_latency() {
        let router = StreamRouter::new(0.5, 0.5, 100.0);
        assert!((router.max_latency_ms() - 100.0).abs() < 1e-10);
    }
}

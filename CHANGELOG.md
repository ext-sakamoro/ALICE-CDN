# Changelog

All notable changes to ALICE-CDN will be documented in this file.

## [Unreleased]

### Changed
- 依存: alice-crypto を 0.7 に上げた (`std` の API は同じ、0.7 は module ごとの feature を足した版)
- 依存: alice-analytics / alice-cache / libasp / alice-crypto を隣の path でなく各 repository の main branch と版で引く (Cargo.lock が commit を記録する) CI は隣を中身なしの stub に置き換えるのをやめ、bridge の feature (`analytics,cache,asp,crypto`) の test と clippy を足した (これまで bridge は CI で一度も compile されなかった) `scripts/preflight.sh` を同じ step で作り直した `cache_bridge.rs` の使わない import を外した
- **Breaking:** `CdnMetrics::merge` は `Result<(), MergeError>` を返す (alice-analytics の `Mergeable::merge` に合わせる) 数が `u64::MAX` を超える合算と、latency の sketch の精度 (alpha) が違う合算は断り、metrics を変えない (各部分を写しで合算し、全部できた時だけ入れ替える) `analytics` feature の import は alice-analytics から消えた `prelude` でなく `sketch` / `anomaly` から取る (これまで `analytics` feature は alice-analytics の main に対して build できなかった) 試験 `analytics_bridge` の 2 本
- Licence: `AGPL-3.0-only OR LicenseRef-Commercial` → `LicenseRef-Proprietary`
- License の SPDX 識別子を非推奨の `AGPL-3.0` から `AGPL-3.0-only` に (`Cargo.toml` の `AGPL-3.0-only OR LicenseRef-Commercial`、README の badge と表記) 条文 (`LICENSE-AGPL` / `LICENSE-COMMERCIAL.md`) と利用条件は不変
- CI: `ci.yml` は `ci/**` の branch と手動実行でも走る main の run は打ち切らず、`ci/**` の古い run は新しい run が来たら打ち切る
- License: `AGPL-3.0` → `AGPL-3.0 OR LicenseRef-Commercial` (2026-09-27)

## [0.2.0] - 2026-02-23

### Added
- `vivaldi` — `VivaldiCoord` 3D+height network coordinates, integer-only RTT prediction, spring-model updates
- `simd` — `SimdCoord` 32-byte aligned SIMD coordinate type, batch distance calculations
- `locator` — `ContentLocator` latency-aware rendezvous hashing, `RendezvousHash` consistent placement
- `maglev` — `MaglevHash` Google Maglev O(1) consistent hashing with minimal redistribution
- `spatial` — Compressed octree spatial index, O(log N) nearest-k search, u32 indices
- `content_types` — (feature `content_types`) ASDF/Mesh/Texture/Audio content type awareness
- `analytics_bridge` — (feature `analytics`) ALICE-Analytics delivery metrics
- `cache_bridge` — (feature `cache`) ALICE-Cache edge-node caching integration
- `asp_bridge` — (feature `asp`) ALICE-Streaming-Protocol stream routing
- `crypto_bridge` — (feature `crypto`) ALICE-Crypto content encryption (DRM, signed payloads)
- `sdf_cdn_bridge` — (feature `sdf`) SDF-aware CDN routing (spatial cell to edge node)
- `no_std` support with `alloc` fallback
- 142 unit tests (125 base + 13 content_types + 4 sdf)

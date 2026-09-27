# Changelog

All notable changes to ALICE-CDN will be documented in this file.

## [Unreleased]

### Changed
- **License: `AGPL-3.0` → `AGPL-3.0 OR LicenseRef-Commercial` (dual-licensed、2026-09-27)** AGPL 側の条件は変更なし (既存 AGPL 利用者への影響ゼロ)、商用という選択肢が追加されただけ SPDX が AGPL 単独だと cargo-deny / FOSSA / SBOM に「商用オプションなし」と見えるため宣言を dual に 変更点: SPDX / `LICENSE` → `LICENSE-AGPL` rename / `LICENSE-COMMERCIAL.md` (商用トリガー 6 条件 = クローズド製品・商用 SaaS・エッジ・ファームウェア配布・plugin 再配布・プラットフォーム NDA・保証、社内利用は AGPL 側で無償と明記) / README の選択肢表 商用窓口は法人 `contact@extoria.co.jp`

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

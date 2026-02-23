# Changelog

All notable changes to ALICE-CDN will be documented in this file.

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
- 53 unit tests

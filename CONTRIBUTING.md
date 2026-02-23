# Contributing to ALICE-CDN

## Build

```bash
cargo build
```

## Test

```bash
cargo test --lib --tests
```

Note: examples may have compilation issues independent of the library.

## Lint

```bash
cargo clippy --lib --tests -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Optional Features

```bash
# ALICE-Cache integration
cargo build --features cache

# ALICE-Analytics metrics
cargo build --features analytics

# ASP stream routing
cargo build --features asp

# ALICE-Crypto DRM/signed payloads
cargo build --features crypto
```

## Design Constraints

- **Integer-only Vivaldi**: coordinates use fixed-point `i64` arithmetic — no floating-point in hot paths.
- **SIMD alignment**: `SimdCoord` is `repr(C, align(32))` for AVX2-width loads.
- **Compressed octree**: `[u32; 8]` children (32 bytes) instead of `[Option<usize>; 8]` (128 bytes).
- **MumHash**: inline multiply-based hash replaces FNV-1a for higher throughput.
- **Maglev O(1)**: constant-time lookup with even distribution and minimal key redistribution.
- **`no_std` compatible**: core library works with `alloc` only.

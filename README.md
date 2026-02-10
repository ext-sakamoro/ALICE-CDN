# ALICE-CDN

**Decentralized Latency-Optimized Content Delivery - Scorched Earth Edition**

A high-performance Rust library for building decentralized CDN infrastructure with latency-aware content routing.

## Features

- **Vivaldi Network Coordinates**: Decentralized RTT prediction without centralized measurement
- **SIMD-Accelerated Coordinates**: 32-byte aligned, integer-only math for maximum throughput
- **Spatial Indexing**: Compressed Octree with O(log n) nearest neighbor search
- **Latency-Aware Rendezvous Hashing**: Content routing with MumHash (faster than FNV-1a)
- **Maglev Consistent Hashing**: Google's O(1) lookup algorithm with minimal disruption
- **Fixed-Point Arithmetic**: Deterministic cross-platform calculations
- **Zero Dependencies**: Core functionality requires no external crates

## Design Philosophy

> "The best cache is the one closest to you."

Traditional CDNs rely on centralized coordination for content placement. ALICE-CDN enables fully decentralized, latency-optimized content delivery where each node can independently make optimal routing decisions.

## Quick Start

```rust
use alice_cdn::{VivaldiCoord, ContentLocator, MaglevHash};

// Vivaldi: Predict RTT without direct measurement
let local = VivaldiCoord::new();
let remote = VivaldiCoord::at(10.0, 20.0, 0.0, 5.0);
let predicted_rtt = local.predict_rtt(&remote);

// Latency-Aware Routing: Find best node for content
let locator = ContentLocator::new(local);
let nodes = vec![(1u64, &remote)];
let best = locator.find_best(content_id, nodes);

// Maglev: O(1) consistent hashing
let maglev = MaglevHash::new(vec![1, 2, 3, 4, 5]);
let node = maglev.lookup(key);
```

## Core Components

### Vivaldi Network Coordinates

Each node maintains a position in virtual 3D space + height:

```
RTT(A, B) ≈ ||pos_A - pos_B|| + height_A + height_B
```

The spring model updates coordinates based on observed RTT samples:

```rust
let mut system = VivaldiSystem::new();

// Update with RTT sample
system.observe(&remote_coord, measured_rtt_ms);

// Predict RTT to any node
let predicted = system.predict_rtt(&other_coord);
```

**Properties**:
- Converges to stable coordinates
- Height models "last-mile" latency (WiFi, etc.)
- Error estimate adapts learning rate
- Fused with SimdCoord for zero-cost SIMD access

### SimdCoord (SIMD-Accelerated Coordinates)

Low-level coordinate representation optimized for SIMD operations:

```rust
use alice_cdn::SimdCoord;

// 32-byte aligned, integer-only math
let a = SimdCoord::from_f64(0.0, 0.0, 0.0, 5.0);
let b = SimdCoord::from_f64(10.0, 20.0, 0.0, 5.0);

// Fast distance calculation (Newton-Raphson isqrt)
let dist = a.distance(&b);
let rtt_ms = a.predict_rtt_ms(&b);

// Batch operations for cache efficiency
let targets: Vec<SimdCoord> = /* ... */;
let nearest_idx = find_nearest(&a, &targets);
```

**Properties**:
- 32-byte aligned for AVX2 / cache line efficiency
- Integer-only pipeline (no floating point in hot paths)
- Newton-Raphson integer square root
- Serialization to 32 bytes

### Spatial Index (Compressed Octree)

O(log n) nearest neighbor search for large node sets:

```rust
use alice_cdn::{SpatialIndex, SpatialEntry, SimdCoord};

// Build index from nodes
let entries: Vec<SpatialEntry> = nodes.iter()
    .map(|(id, coord)| SpatialEntry { node_id: *id, coord: *coord })
    .collect();
let index = SpatialIndex::build(entries);

// Find k nearest neighbors
let query = SimdCoord::from_f64(50.0, 50.0, 0.0, 1.0);
let nearest_5 = index.find_nearest_k(&query, 5);

// Find all nodes within radius
let within_radius = index.find_within_radius(&query, max_distance_squared);
```

**Properties**:
- Compressed u32 indices (4x smaller than `Option<usize>`)
- O(log n + k) nearest-k search
- Efficient for 1000+ nodes

### Latency-Aware Rendezvous Hashing

Combines consistent hashing with latency optimization using MumHash:

```
Score(Node) = hash_weight × Hash(Key, Node) + distance_weight / RTT
```

```rust
// Strong distance preference (0.2 hash, 0.8 distance)
let locator = ContentLocator::with_weights(local, 0.2, 0.8);

// Find top-3 nodes for replication
let top3 = locator.find_top_k(content_id, candidates, 3);

// Pure rendezvous hashing (no latency)
let owner = RendezvousHash::find_owner(content_id, nodes);
```

**IndexedLocator** for large-scale deployments:

```rust
use alice_cdn::IndexedLocator;

// Build spatial index + locator
let locator = IndexedLocator::build(local_coord, nodes, 0.3, 0.7);

// Find best with limited candidates (fast)
let best = locator.find_best(content_id, 20);  // Only evaluate 20 nearest
```

### Maglev Consistent Hashing

Google's Maglev algorithm for O(1) load balancing:

```rust
let maglev = MaglevHash::new(nodes);

// O(1) lookup
let node = maglev.lookup(key);

// Check distribution evenness
let stats = maglev.distribution_stats();
println!("Std dev: {}", stats.std_dev);

// Weighted nodes
let weighted = WeightedMaglev::new(vec![(1, 3), (2, 1)]); // 3:1 ratio
```

**Properties**:
- O(1) lookup via precomputed table
- Even distribution across nodes
- Minimal key redistribution on node changes (~1/n keys move)
- Zero-allocation lookup (table uses `Vec<u16>`)

## Memory Layout

Scorched Earth Edition optimizes memory layout for cache efficiency:

| Structure | Size | Alignment | Notes |
|-----------|------|-----------|-------|
| `SimdCoord` | 32 bytes | 32 bytes | AVX2 ready, `[i64; 4]` |
| `VivaldiCoord` | 48 bytes | 32 bytes | Fused SimdCoord + error |
| `OctreeNode` | 36 bytes | 4 bytes | `[u32; 8]` children (vs 128 bytes with `Option<usize>`) |
| `MaglevHash` | O(m) | - | `Vec<u16>` table, no N×M permutation storage |

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Vivaldi RTT prediction | O(1) | Integer-only math |
| Vivaldi update | O(1) | Spring model adjustment |
| SimdCoord distance | O(1) | Newton-Raphson isqrt |
| Spatial nearest-k | O(log n + k) | Octree traversal |
| Spatial within-radius | O(log n + m) | m = results |
| Rendezvous find_best | O(n) | n = candidates |
| Rendezvous find_best (indexed) | O(log n + k) | k = candidate limit |
| Maglev lookup | O(1) | Table lookup |
| Maglev rebuild | O(m) | m = table size |

## Fixed-Point Arithmetic

All calculations use fixed-point arithmetic for deterministic results:

```rust
use alice_cdn::Fixed;

let a = Fixed::from_f64(3.14159);
let b = Fixed::from_f64(2.0);

let sum = a + b;
let product = a * b;
let sqrt = a.sqrt();
```

This ensures identical results across platforms (no floating-point variance).

## Use Cases

1. **Edge Computing**: Route requests to nearest edge node
2. **P2P CDN**: Decentralized content delivery without coordination server
3. **Game Servers**: Match players to lowest-latency servers
4. **Distributed Caching**: Consistent cache assignment with latency awareness
5. **Load Balancing**: Maglev-style load distribution

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
alice-cdn = "0.1"
```

## Related Projects

- [ALICE-Cache](https://github.com/ext-sakamoro/ALICE-Cache) - High-performance probabilistic cache
- [ALICE-Queue](https://github.com/ext-sakamoro/ALICE-Queue) - Deterministic zero-copy message log
- [ALICE-Auth](https://github.com/ext-sakamoro/ALICE-Auth) - Zero-copy authentication library

## SDF Asset Delivery Network Integration

ALICE-CDN serves as the routing layer in the SDF asset delivery pipeline, achieving **200-800x bandwidth reduction** vs glTF by delivering mathematical descriptions instead of polygon meshes.

```
Client Request (asset_id + VivaldiCoord)
    │
    ▼
┌──────────────────────────────────────┐
│  ALICE-CDN (Vivaldi Routing)     ◀── this crate
│  ・VivaldiCoord → nearest node (RTT)  │
│  ・IndexedLocator: O(log n + k)       │
│  ・Rendezvous hash + distance weight  │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  ALICE-Cache (Markov Prefetch)        │
│  ・SharedOracle: lock-free prediction │
└──────────┬───────────────────────────┘
           │ cache miss → origin
           ▼
┌──────────────────────────────────────┐
│  ALICE-SDF (ASDF Binary Format)       │
│  ・80 bytes (sphere) vs 20 KB (glTF)  │
└──────────────────────────────────────┘
```

Related: [ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF) | [ALICE-Cache](https://github.com/ext-sakamoro/ALICE-Cache)

## Content Type Awareness (ASDF Detection)

Enable with `--features content_types` for type-aware content routing. ALICE-CDN can detect ASDF binary format from raw bytes and adjust routing priority accordingly.

```toml
[dependencies]
alice-cdn = { path = "../ALICE-CDN", features = ["content_types"] }
```

### Content Types

| Type | Priority | Replicas | Latency Sensitive |
|------|----------|----------|-------------------|
| ASDF (3D geometry) | 10 | 3 | Yes |
| Mesh | 8 | 2 | Yes |
| Texture | 5 | 2 | No |
| Audio | 3 | 1 | No |
| Generic | 1 | 1 | No |

### Usage

```rust
use alice_cdn::content_types::{ContentType, AsdfMetadata};

// Auto-detect from raw bytes
let content_type = ContentType::detect(&data);
println!("Type: {:?}, Priority: {}", content_type, content_type.priority_weight());

// Parse ASDF header metadata (16 bytes)
if data.len() >= 16 {
    let header: [u8; 16] = data[..16].try_into().unwrap();
    if let Some(meta) = AsdfMetadata::parse(&header) {
        println!("ASDF v{}, {} nodes, CRC32: 0x{:08X}",
            meta.version, meta.node_count, meta.crc32);
    }
}

// Typed content IDs (top 4 bits = type, bottom 60 bits = raw ID)
let typed_id = alice_cdn::content_types::typed_content_id(raw_id, ContentType::Asdf);
```

## Cross-Crate Bridges

ALICE-CDN connects to other ALICE ecosystem crates via feature-gated bridge modules:

| Bridge | Feature | Target Crate | Description |
|--------|---------|--------------|-------------|
| `asp_bridge` | `asp` | [ALICE-Streaming-Protocol](../ALICE-Streaming-Protocol) | ASP packet routing via Vivaldi coordinates + Maglev consistent hashing |
| `crypto_bridge` | `crypto` | [ALICE-Crypto](../ALICE-Crypto) | Content encryption for DRM, signed payloads, per-asset key derivation |

### ASP Bridge (feature: `asp`)

Routes ASP streaming packets to optimal CDN edge nodes using Vivaldi latency prediction and Maglev consistent hashing. Enables latency-aware live video stream delivery.

```toml
[dependencies]
alice-cdn = { path = "../ALICE-CDN", features = ["asp"] }
```

```rust
use alice_cdn::asp_bridge::{AspStreamRouter, estimate_delivery};
use alice_cdn::{VivaldiCoord, ContentLocator, MaglevHash};

// Build router with Maglev + Vivaldi
let nodes = vec![1u64, 2, 3, 4, 5];
let router = AspStreamRouter::new(nodes.clone(), local_coord);

// Route by stream key
let primary = router.primary_node(stream_key);
let replicas = router.replica_nodes(stream_key, 3);

// Find closest replica
let closest = router.closest_replica(stream_key, 3);

// Route an ASP packet (uses sequence number as key)
let node = router.route_packet(&asp_packet);

// Estimate delivery cost
let (hops, latency_ms) = estimate_delivery(&packet, &local_coord, &target_coord);
```

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

### Commercial Licensing

For commercial use without AGPL obligations, please contact:

**Extoria**
- GitHub: [@ext-sakamoro](https://github.com/ext-sakamoro)

We offer flexible commercial licensing options for businesses that need to:
- Use ALICE-CDN in proprietary/closed-source applications
- Distribute ALICE-CDN without source code disclosure
- Integrate into SaaS products without AGPL compliance burden

## Author

Moroya Sakamoto ([@ext-sakamoro](https://github.com/ext-sakamoro))

## References

- [Vivaldi: A Decentralized Network Coordinate System](https://pdos.csail.mit.edu/papers/vivaldi:sigcomm/paper.pdf)
- [Maglev: A Fast and Reliable Software Network Load Balancer](https://research.google/pubs/pub44824/)
- [Rendezvous Hashing](https://en.wikipedia.org/wiki/Rendezvous_hashing)

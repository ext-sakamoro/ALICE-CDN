//! Content Routing with Maglev Hashing Example
//!
//! Demonstrates consistent hashing for CDN node selection.
//!
//! ```bash
//! cargo run --example content_routing
//! ```

use alice_cdn::prelude::*;
use alice_cdn::{ContentId, NodeId, SMALL_TABLE_SIZE};

fn main() {
    println!("=== Maglev Consistent Hashing Demo ===\n");

    // Create a Maglev hash with 5 backend nodes
    let nodes: Vec<u64> = vec![100, 200, 300, 400, 500];
    let maglev = MaglevHash::new(nodes.clone());

    println!("Nodes: {:?}", nodes);
    println!("\nContent -> Node mapping:");

    for key in [1u64, 42, 100, 999, 12345, 67890] {
        if let Some(node) = maglev.lookup(key) {
            println!("  Content {:>5} -> Node {}", key, node);
        }
    }

    // Distribution stats
    let stats = maglev.distribution_stats();
    println!("\nDistribution stats:");
    println!("  Min load: {}", stats.min_count);
    println!("  Max load: {}", stats.max_count);
    println!("  Std dev:  {:.2}", stats.std_dev);

    // --- Weighted Maglev ---
    println!("\n=== Weighted Maglev (heterogeneous capacity) ===\n");

    let weighted_nodes = vec![
        (100u64, 3), // Node 100: weight 3 (3x capacity)
        (200, 1),    // Node 200: weight 1
        (300, 2),    // Node 300: weight 2
    ];
    let weighted = WeightedMaglev::new(weighted_nodes);

    let mut counts = std::collections::HashMap::new();
    for key in 0..10000u64 {
        if let Some(node) = weighted.lookup(key) {
            *counts.entry(node).or_insert(0u32) += 1;
        }
    }

    println!("Traffic distribution (10000 keys):");
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by_key(|(node, _)| **node);
    for (node, count) in sorted {
        println!("  Node {:>3}: {:>5} keys ({:.1}%)", node, count, *count as f64 / 100.0);
    }

    // --- Rendezvous Hashing ---
    println!("\n=== Rendezvous Hash (Replica Selection) ===\n");

    let cdn_nodes: Vec<NodeId> = (0..8).collect();
    let content = ContentId(0xDEADBEEF);
    let replicas = RendezvousHash::find_replicas(content, &cdn_nodes, 3);
    println!("Content 0xDEADBEEF -> Replicas: {:?}", replicas);
}

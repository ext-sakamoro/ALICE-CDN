//! Vivaldi Network Coordinates Example
//!
//! Simulates a network of nodes predicting RTT via spring-force model.
//!
//! ```bash
//! cargo run --example vivaldi_network
//! ```

use alice_cdn::find_nearest;
use alice_cdn::prelude::*;

fn main() {
    println!("=== Vivaldi Network Coordinates Demo ===\n");

    // Create nodes at known positions
    let tokyo = SimdCoord::from_f64(35.6, 139.7, 0.0, 1.0);
    let london = SimdCoord::from_f64(51.5, -0.1, 0.0, 2.0);
    let new_york = SimdCoord::from_f64(40.7, -74.0, 0.0, 1.5);
    let sydney = SimdCoord::from_f64(-33.9, 151.2, 0.0, 3.0);

    println!("Nodes:");
    println!("  Tokyo:    ({}, {})", 35.6, 139.7);
    println!("  London:   ({}, {})", 51.5, -0.1);
    println!("  New York: ({}, {})", 40.7, -74.0);
    println!("  Sydney:   ({}, {})", -33.9, 151.2);

    // Predict RTT between pairs
    println!("\nPredicted RTT (distance units):");
    println!("  Tokyo    <-> London:   {}", tokyo.distance(&london));
    println!("  Tokyo    <-> New York: {}", tokyo.distance(&new_york));
    println!("  Tokyo    <-> Sydney:   {}", tokyo.distance(&sydney));
    println!("  London   <-> New York: {}", london.distance(&new_york));

    // Find nearest node to a query point
    let candidates = [tokyo, london, new_york, sydney];
    let labels = ["Tokyo", "London", "New York", "Sydney"];
    let query = SimdCoord::from_f64(48.8, 2.3, 0.0, 1.0); // Paris

    println!("\nNearest to Paris (48.8, 2.3):");
    if let Some(idx) = find_nearest(&query, &candidates) {
        println!(
            "  Nearest: {} (distance: {})",
            labels[idx],
            query.distance(&candidates[idx])
        );
    }

    // Rank all nodes by distance
    let mut ranked: Vec<(usize, i64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, query.distance_squared(c)))
        .collect();
    ranked.sort_by_key(|&(_, d)| d);
    for (rank, (i, dist_sq)) in ranked.iter().enumerate() {
        println!("  {}. {} (dist²: {})", rank + 1, labels[*i], dist_sq);
    }
}

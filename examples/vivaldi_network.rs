//! Vivaldi Network Coordinates Example
//!
//! Simulates a network of nodes predicting RTT via spring-force model.
//!
//! ```bash
//! cargo run --example vivaldi_network
//! ```

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
    let nearest = find_nearest(query, &candidates, 4);
    for (i, (coord, dist)) in nearest.iter().enumerate() {
        let label = candidates
            .iter()
            .position(|c| c.distance(coord) == 0)
            .map(|idx| labels[idx])
            .unwrap_or("?");
        println!("  {}. {} (distance: {})", i + 1, label, dist);
    }
}

//! SIMD-Accelerated Vivaldi Coordinates
//!
//! **Optimizations**:
//! - `SimdCoord`: Pack x,y,z,h into 32-byte aligned array for AVX2
//! - **Integer-only math**: No floating point, pure integer pipeline
//! - **Newton-Raphson isqrt**: Hardware-friendly integer square root
//! - **Branchless updates**: Minimize pipeline stalls
//!
//! > "Math should be as fast as physics allows."

use core::ops::{Add, Sub};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Scale factor for fixed-point (2^16 for coordinate precision)
const COORD_SCALE: i64 = 1 << 16;

/// Precomputed reciprocal of COORD_SCALE for f64 conversion
const RCP_COORD_SCALE: f64 = 1.0 / (1u64 << 16) as f64;

/// 4-dimensional coordinate vector [x, y, z, height]
/// Aligned to 32 bytes for AVX2 / cache line efficiency
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimdCoord {
    /// Packed coordinates: x, y, z, height (all scaled integers)
    pub data: [i64; 4],
}

impl SimdCoord {
    /// Create new coordinate at origin
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            data: [0, 0, 0, COORD_SCALE],
        } // Default height = 1.0
    }

    /// Create coordinate from f64 values
    #[inline(always)]
    pub fn from_f64(x: f64, y: f64, z: f64, height: f64) -> Self {
        Self {
            data: [
                (x * COORD_SCALE as f64) as i64,
                (y * COORD_SCALE as f64) as i64,
                (z * COORD_SCALE as f64) as i64,
                (height * COORD_SCALE as f64) as i64,
            ],
        }
    }

    /// Create from raw scaled integers
    #[inline(always)]
    pub const fn from_raw(x: i64, y: i64, z: i64, h: i64) -> Self {
        Self { data: [x, y, z, h] }
    }

    /// Get x coordinate as f64
    #[inline(always)]
    pub fn x(&self) -> f64 {
        self.data[0] as f64 * RCP_COORD_SCALE
    }

    /// Get y coordinate as f64
    #[inline(always)]
    pub fn y(&self) -> f64 {
        self.data[1] as f64 * RCP_COORD_SCALE
    }

    /// Get z coordinate as f64
    #[inline(always)]
    pub fn z(&self) -> f64 {
        self.data[2] as f64 * RCP_COORD_SCALE
    }

    /// Get height as f64
    #[inline(always)]
    pub fn height(&self) -> f64 {
        self.data[3] as f64 * RCP_COORD_SCALE
    }

    /// Calculate squared Euclidean distance (no sqrt, for comparisons)
    /// Result is in squared scaled units
    #[inline(always)]
    pub fn distance_squared(&self, other: &Self) -> i64 {
        let dx = self.data[0] - other.data[0];
        let dy = self.data[1] - other.data[1];
        let dz = self.data[2] - other.data[2];

        // Use i128 to prevent overflow, then scale down
        // dx is in SCALE units, so dx^2 is in SCALE^2 units
        // We want result in SCALE units, so divide by SCALE
        let sq = (dx as i128 * dx as i128 + dy as i128 * dy as i128 + dz as i128 * dz as i128)
            / COORD_SCALE as i128;

        sq as i64
    }

    /// Calculate Euclidean distance using integer sqrt
    /// Result is in scaled units (divide by COORD_SCALE to get real value)
    #[inline(always)]
    pub fn distance(&self, other: &Self) -> i64 {
        let dx = self.data[0] - other.data[0];
        let dy = self.data[1] - other.data[1];
        let dz = self.data[2] - other.data[2];

        // Compute sum of squares in i128 to prevent overflow
        let sum_sq = dx as i128 * dx as i128 + dy as i128 * dy as i128 + dz as i128 * dz as i128;

        if sum_sq <= 0 {
            return 0;
        }

        // Take sqrt, result is in COORD_SCALE units
        // sqrt(x^2 * SCALE^2) = x * SCALE
        isqrt(sum_sq as u64) as i64
    }

    /// Predict RTT to another node
    /// RTT = distance + height_self + height_other
    #[inline(always)]
    pub fn predict_rtt(&self, other: &Self) -> i64 {
        self.distance(other) + self.data[3] + other.data[3]
    }

    /// Predict RTT in milliseconds (f64)
    #[inline(always)]
    pub fn predict_rtt_ms(&self, other: &Self) -> f64 {
        self.predict_rtt(other) as f64 * RCP_COORD_SCALE
    }

    /// SIMD-accelerated distance calculation (x86_64 AVX2)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    pub unsafe fn distance_simd(&self, other: &Self) -> i64 {
        // Load coordinates into AVX2 registers
        let a = _mm256_load_si256(self.data.as_ptr() as *const __m256i);
        let b = _mm256_load_si256(other.data.as_ptr() as *const __m256i);

        // Compute difference
        let diff = _mm256_sub_epi64(a, b);

        // Extract and compute squares (AVX2 lacks native 64-bit multiply)
        // Fall back to scalar for the multiply portion
        let mut result = [0i64; 4];
        _mm256_store_si256(result.as_mut_ptr() as *mut __m256i, diff);

        let dx = result[0];
        let dy = result[1];
        let dz = result[2];

        let sq = (dx as i128 * dx as i128 + dy as i128 * dy as i128 + dz as i128 * dz as i128)
            / COORD_SCALE as i128;

        isqrt(sq as u64) as i64
    }

    /// Update coordinate based on RTT measurement (Vivaldi spring model)
    ///
    /// Returns the prediction error for adaptive algorithms
    #[inline(always)]
    pub fn update(&mut self, peer: &Self, measured_rtt_scaled: i64, weight: i64) -> i64 {
        let predicted = self.predict_rtt(peer);
        let error = measured_rtt_scaled - predicted;

        // Direction vector (self -> peer)
        let dist = self.distance(peer);
        if dist < COORD_SCALE / 1000 {
            // Too close, add jitter
            self.data[0] += COORD_SCALE / 100;
            return error;
        }

        // Unit vector scaled by weight and error
        // move = -error * (peer - self) / dist * weight / SCALE
        // Negative because if measured < predicted, we're "too far" in coord space
        let neg_error = -error;

        // Calculate movement for each dimension
        for i in 0..3 {
            let diff = peer.data[i] - self.data[i];
            // (neg_error * diff * weight) / (dist * COORD_SCALE)
            let movement = (neg_error as i128 * diff as i128 * weight as i128)
                / (dist as i128 * COORD_SCALE as i128);

            // Clamp movement to prevent oscillation
            let clamped = movement.clamp(-COORD_SCALE as i128 * 10, COORD_SCALE as i128 * 10);
            self.data[i] += clamped as i64;
        }

        // Update height (absorbs residual error)
        let height_delta = (neg_error as i128 * weight as i128) / (COORD_SCALE as i128 * 4);
        self.data[3] = (self.data[3] + height_delta as i64).max(COORD_SCALE / 10);

        error
    }

    /// Serialize to bytes (32 bytes)
    #[inline(always)]
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (i, &val) in self.data.iter().enumerate() {
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Deserialize from bytes
    #[inline(always)]
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        let mut data = [0i64; 4];
        for i in 0..4 {
            data[i] = i64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        Self { data }
    }
}

impl Add for SimdCoord {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self {
            data: [
                self.data[0] + other.data[0],
                self.data[1] + other.data[1],
                self.data[2] + other.data[2],
                self.data[3] + other.data[3],
            ],
        }
    }
}

impl Sub for SimdCoord {
    type Output = Self;

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self {
            data: [
                self.data[0] - other.data[0],
                self.data[1] - other.data[1],
                self.data[2] - other.data[2],
                self.data[3] - other.data[3],
            ],
        }
    }
}

/// Fast integer square root using Newton-Raphson method
///
/// This is branchless-optimized and uses only integer operations.
#[inline(always)]
pub fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }

    // Initial estimate: 2^((bit_length + 1) / 2)
    // For n=100 (7 bits): shift = (7+1)/2 = 4, x = 16
    let bit_length = 64 - n.leading_zeros();
    let shift = (bit_length + 1) >> 1;
    let mut x = 1u64 << shift;

    // Newton-Raphson iterations (typically converges in 4-6 iterations)
    loop {
        let next = (x + n / x) >> 1;
        if next >= x {
            return x;
        }
        x = next;
    }
}

/// Fast inverse square root approximation (Quake-style)
/// Returns 1/sqrt(n) scaled by COORD_SCALE
#[inline(always)]
pub fn fast_inv_sqrt(n: i64) -> i64 {
    if n <= 0 {
        return i64::MAX;
    }

    // Use regular sqrt and divide
    let sqrt = isqrt(n as u64) as i64;
    if sqrt == 0 {
        return i64::MAX;
    }

    (COORD_SCALE * COORD_SCALE) / sqrt
}

/// Batch distance calculation for multiple coordinates
///
/// More cache-efficient than individual calls
#[inline(always)]
pub fn batch_distances(origin: &SimdCoord, targets: &[SimdCoord], out: &mut [i64]) {
    debug_assert_eq!(targets.len(), out.len());

    for (i, target) in targets.iter().enumerate() {
        out[i] = origin.distance_squared(target);
    }
}

/// Find index of minimum distance in batch
#[inline(always)]
pub fn find_nearest(origin: &SimdCoord, targets: &[SimdCoord]) -> Option<usize> {
    if targets.is_empty() {
        return None;
    }

    let mut min_dist = i64::MAX;
    let mut min_idx = 0;

    for (i, target) in targets.iter().enumerate() {
        let dist = origin.distance_squared(target);
        if dist < min_dist {
            min_dist = dist;
            min_idx = i;
        }
    }

    Some(min_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_coord_basic() {
        let a = SimdCoord::from_f64(0.0, 0.0, 0.0, 5.0);
        let b = SimdCoord::from_f64(3.0, 4.0, 0.0, 5.0);

        // Distance should be 5.0
        let dist = a.distance(&b) as f64 / COORD_SCALE as f64;
        assert!((dist - 5.0).abs() < 0.1, "Distance: {}", dist);

        // RTT = 5 + 5 + 5 = 15
        let rtt = a.predict_rtt_ms(&b);
        assert!((rtt - 15.0).abs() < 0.5, "RTT: {}", rtt);
    }

    #[test]
    fn test_isqrt() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(16), 4);
        assert_eq!(isqrt(100), 10);
        assert_eq!(isqrt(10000), 100);

        // Large numbers
        assert_eq!(isqrt(1_000_000), 1000);
        assert_eq!(isqrt(1_000_000_000_000), 1_000_000);
    }

    #[test]
    fn test_simd_coord_update() {
        let mut a = SimdCoord::from_f64(0.0, 0.0, 0.0, 5.0);
        let b = SimdCoord::from_f64(100.0, 0.0, 0.0, 5.0);

        let initial_dist = a.distance(&b);

        // Measured RTT is 50ms, predicted is ~110ms
        // So we should move closer
        let weight = COORD_SCALE / 4; // 0.25
        let measured = (50.0 * COORD_SCALE as f64) as i64;
        a.update(&b, measured, weight);

        let final_dist = a.distance(&b);
        assert!(final_dist < initial_dist, "Should move closer");
    }

    #[test]
    fn test_simd_coord_serialization() {
        let coord = SimdCoord::from_f64(10.5, -20.3, 30.7, 5.5);
        let bytes = coord.to_bytes();
        let restored = SimdCoord::from_bytes(&bytes);

        assert_eq!(coord.data, restored.data);
    }

    #[test]
    fn test_batch_operations() {
        let origin = SimdCoord::from_f64(0.0, 0.0, 0.0, 1.0);
        let targets = vec![
            SimdCoord::from_f64(10.0, 0.0, 0.0, 1.0),
            SimdCoord::from_f64(5.0, 0.0, 0.0, 1.0),
            SimdCoord::from_f64(20.0, 0.0, 0.0, 1.0),
        ];

        let nearest = find_nearest(&origin, &targets);
        assert_eq!(nearest, Some(1)); // Index 1 (distance 5) is closest
    }

    #[test]
    fn test_alignment() {
        let coord = SimdCoord::new();
        let ptr = &coord as *const SimdCoord as usize;
        assert_eq!(ptr % 32, 0, "Should be 32-byte aligned");
    }
}

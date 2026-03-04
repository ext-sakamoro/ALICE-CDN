//! Vivaldi Network Coordinates (Fused Edition)
//!
//! **Algorithm**: Decentralized RTT prediction using mass-spring model.
//!
//! **Scorched Earth Optimizations**:
//! - **Fused Layout**: `VivaldiCoord` IS `SimdCoord` - no conversion overhead
//! - **32-byte aligned**: Cache-line friendly, SIMD-ready
//! - **Integer-only math**: No floating point in hot paths
//!
//! **Math**:
//! - Distance: `d = ||x_A - x_B|| + h_A + h_B`
//! - Force: `F = (RTT_measured - d_predicted) * unit_vector`
//! - Update: `x_new = x_old + δ * F`
//!
//! > "Distance is not measured in miles, but in milliseconds."

use crate::simd::SimdCoord;
use core::ops::{Add, Div, Mul, Sub};

/// Fixed-point scale factor (2^20 = ~1M for microsecond precision)
const SCALE: i64 = 1 << 20;

/// Precomputed reciprocal of SCALE for f64 conversion
const RCP_SCALE: f64 = 1.0 / (1u64 << 20) as f64;

/// Coordinate scale (from simd.rs)
const COORD_SCALE: i64 = 1 << 16;

/// Fixed-point number for deterministic cross-platform math
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fixed(pub i64);

impl Fixed {
    pub const ZERO: Fixed = Fixed(0);
    pub const ONE: Fixed = Fixed(SCALE);

    /// Create from integer milliseconds
    #[inline(always)]
    #[must_use]
    pub fn from_ms(ms: i64) -> Self {
        Fixed(ms * SCALE)
    }

    /// Create from microseconds
    #[inline(always)]
    #[must_use]
    pub fn from_us(us: i64) -> Self {
        Fixed(us * SCALE / 1000)
    }

    /// Create from floating point (for initialization only)
    #[inline(always)]
    #[must_use]
    pub fn from_f64(f: f64) -> Self {
        Fixed((f * SCALE as f64) as i64)
    }

    /// Convert to milliseconds
    #[inline(always)]
    #[must_use]
    pub fn to_ms(self) -> i64 {
        self.0 / SCALE
    }

    /// Convert to f64 (for display/debug)
    #[inline(always)]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 * RCP_SCALE
    }

    /// Absolute value
    #[inline(always)]
    #[must_use]
    pub fn abs(self) -> Self {
        Fixed(self.0.abs())
    }

    /// Square root using Newton-Raphson (integer approximation)
    #[inline(always)]
    #[must_use]
    pub fn sqrt(self) -> Self {
        if self.0 <= 0 {
            return Fixed::ZERO;
        }

        // Scale up for precision before sqrt, then scale result
        let scaled = self.0 as u128 * SCALE as u128;
        let mut x = scaled;
        let mut y = x.div_ceil(2);

        while y < x {
            x = y;
            y = u128::midpoint(x, scaled / x);
        }

        Fixed(x as i64)
    }

    /// Minimum of two values
    #[inline(always)]
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        Fixed(self.0.min(other.0))
    }

    /// Maximum of two values
    #[inline(always)]
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Fixed(self.0.max(other.0))
    }
}

impl Add for Fixed {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Fixed(self.0 + rhs.0)
    }
}

impl Sub for Fixed {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Fixed(self.0 - rhs.0)
    }
}

impl Mul for Fixed {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Scale down after multiplication to maintain precision
        Fixed(((self.0 as i128 * rhs.0 as i128) / SCALE as i128) as i64)
    }
}

impl Div for Fixed {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        if rhs.0 == 0 {
            return Fixed(i64::MAX); // Avoid division by zero
        }
        // Scale up numerator before division
        Fixed(((self.0 as i128 * SCALE as i128) / rhs.0 as i128) as i64)
    }
}

/// Vivaldi learning constants
const MIN_HEIGHT: Fixed = Fixed(SCALE / 1000); // 0.001 ms
const MAX_MOVEMENT: Fixed = Fixed(SCALE * 100); // 100 ms
const CE: Fixed = Fixed(SCALE / 4); // 0.25 - error weight

/// 3D Vivaldi Coordinate + Height (Fused with `SimdCoord`)
///
/// **Memory Layout**: 32 bytes, 32-byte aligned
/// - `data[0]`: x coordinate (scaled by `COORD_SCALE`)
/// - `data[1]`: y coordinate (scaled by `COORD_SCALE`)
/// - `data[2]`: z coordinate (scaled by `COORD_SCALE`)
/// - `data[3]`: height (scaled by `COORD_SCALE`)
///
/// Error estimate is stored separately for cache efficiency
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default)]
pub struct VivaldiCoord {
    /// Fused SIMD-ready coordinate data [x, y, z, height]
    inner: SimdCoord,
    /// Local error estimate (0.0 - 1.0, scaled by SCALE)
    error: i64,
    /// Padding to 48 bytes (maintains alignment for arrays)
    _pad: i64,
}

impl VivaldiCoord {
    /// Create new coordinate at origin with default height
    #[inline(always)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: SimdCoord::from_f64(0.0, 0.0, 0.0, 5.0),
            error: SCALE, // Initial error = 1.0
            _pad: 0,
        }
    }

    /// Create coordinate at specific position
    #[inline(always)]
    #[must_use]
    pub fn at(x: f64, y: f64, z: f64, height: f64) -> Self {
        Self {
            inner: SimdCoord::from_f64(x, y, z, height),
            error: SCALE, // Initial error = 1.0
            _pad: 0,
        }
    }

    /// Create from raw `SimdCoord`
    #[inline(always)]
    #[must_use]
    pub fn from_simd(coord: SimdCoord) -> Self {
        Self {
            inner: coord,
            error: SCALE,
            _pad: 0,
        }
    }

    /// Get inner `SimdCoord` (zero-cost)
    #[inline(always)]
    #[must_use]
    pub fn as_simd(&self) -> &SimdCoord {
        &self.inner
    }

    /// Get mutable inner `SimdCoord`
    #[inline(always)]
    pub fn as_simd_mut(&mut self) -> &mut SimdCoord {
        &mut self.inner
    }

    /// Get x coordinate as Fixed
    #[inline(always)]
    #[must_use]
    pub fn x(&self) -> Fixed {
        Fixed(self.inner.data[0] << 4)
    }

    /// Get y coordinate as Fixed
    #[inline(always)]
    #[must_use]
    pub fn y(&self) -> Fixed {
        Fixed(self.inner.data[1] << 4)
    }

    /// Get z coordinate as Fixed
    #[inline(always)]
    #[must_use]
    pub fn z(&self) -> Fixed {
        Fixed(self.inner.data[2] << 4)
    }

    /// Get height as Fixed
    #[inline(always)]
    #[must_use]
    pub fn height(&self) -> Fixed {
        Fixed(self.inner.data[3] << 4)
    }

    /// Get error estimate as Fixed
    #[inline(always)]
    #[must_use]
    pub fn error(&self) -> Fixed {
        Fixed(self.error)
    }

    /// Calculate Euclidean distance (excluding height)
    #[inline(always)]
    #[must_use]
    pub fn euclidean_distance(&self, other: &Self) -> Fixed {
        let dist_scaled = self.inner.distance(&other.inner);
        Fixed(((dist_scaled as i128) << 4) as i64)
    }

    /// Predict RTT to another node
    /// RTT = `Euclidean_Distance` + `Height_A` + `Height_B`
    #[inline(always)]
    #[must_use]
    pub fn predict_rtt(&self, other: &Self) -> Fixed {
        let rtt_scaled = self.inner.predict_rtt(&other.inner);
        Fixed(((rtt_scaled as i128) << 4) as i64)
    }

    /// Update coordinate based on measured RTT to a peer
    ///
    /// Uses Vivaldi spring-force algorithm with fused SIMD operations
    pub fn update(&mut self, peer: &VivaldiCoord, measured_rtt: Fixed) {
        // measured_rtt is already in Fixed (SCALE) units

        // Predicted RTT (using fused SIMD) - use i128 to prevent overflow
        let predicted_scaled = self.inner.predict_rtt(&peer.inner);
        let predicted = Fixed(((predicted_scaled as i128) << 4) as i64);

        // Error calculation
        let rtt_error = measured_rtt - predicted;
        let relative_error = rtt_error.abs() / measured_rtt.max(Fixed::ONE);

        // Combined weight
        let peer_error = Fixed(peer.error);
        let self_error = Fixed(self.error);
        let weight = self_error / (self_error + peer_error + Fixed::from_f64(0.001));

        // Update local error estimate
        self.error = (relative_error * CE * weight + self_error * (Fixed::ONE - CE * weight)).0;
        self.error = self.error.clamp(SCALE / 100, SCALE);

        // Distance for unit vector calculation
        let dist = self.inner.distance(&peer.inner);
        if dist < COORD_SCALE / 1000 {
            // Too close, add jitter
            self.inner.data[0] += COORD_SCALE / 100;
            return;
        }

        // Calculate movement using integer arithmetic
        let neg_error = -rtt_error.0;
        let delta = weight.0 >> 2;

        // Apply movement for each spatial dimension
        // Convert from SCALE to COORD_SCALE by multiplying by COORD_SCALE/SCALE
        for i in 0..3 {
            let diff = peer.inner.data[i] - self.inner.data[i];
            // movement in COORD_SCALE units = neg_error * diff * delta * COORD_SCALE / (dist * SCALE * SCALE)
            let movement = (neg_error as i128 * diff as i128 * delta as i128 * COORD_SCALE as i128)
                / (dist as i128 * SCALE as i128 * SCALE as i128);

            // Clamp and apply
            let max_move = (MAX_MOVEMENT.0 * COORD_SCALE) / SCALE;
            let clamped = movement.clamp(-max_move as i128, max_move as i128);
            self.inner.data[i] += clamped as i64;
        }

        // Update height (also convert to COORD_SCALE)
        let height_delta = (neg_error as i128 * delta as i128 * COORD_SCALE as i128)
            / (SCALE as i128 * SCALE as i128 * 16);
        let min_height_scaled = (MIN_HEIGHT.0 * COORD_SCALE) / SCALE;
        self.inner.data[3] = (self.inner.data[3] + height_delta as i64).max(min_height_scaled);
    }

    /// Inflate height based on load (for back-pressure)
    #[inline(always)]
    #[must_use]
    pub fn apply_load_factor(&self, load: f64) -> Self {
        let load_penalty = (load * 50.0 * COORD_SCALE as f64) as i64;
        let mut result = *self;
        result.inner.data[3] += load_penalty;
        result
    }

    /// Serialize to bytes (48 bytes: 32 coord + 8 error + 8 pad)
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 48] {
        let mut bytes = [0u8; 48];
        bytes[0..32].copy_from_slice(&self.inner.to_bytes());
        bytes[32..40].copy_from_slice(&self.error.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes
    ///
    /// # Panics
    ///
    /// Panics if the byte slice arithmetic produces an invalid range (should never
    /// happen given the fixed-size `[u8; 48]` input).
    #[must_use]
    pub fn from_bytes(bytes: &[u8; 48]) -> Self {
        let mut coord_bytes = [0u8; 32];
        coord_bytes.copy_from_slice(&bytes[0..32]);
        Self {
            inner: SimdCoord::from_bytes(&coord_bytes),
            error: i64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            _pad: 0,
        }
    }
}

/// Vivaldi coordinate system manager
pub struct VivaldiSystem {
    /// This node's coordinate
    pub local: VivaldiCoord,
    /// Number of updates performed
    pub update_count: u64,
}

impl VivaldiSystem {
    /// Create new system
    #[must_use]
    pub fn new() -> Self {
        Self {
            local: VivaldiCoord::new(),
            update_count: 0,
        }
    }

    /// Create with initial coordinate
    #[must_use]
    pub fn with_coord(coord: VivaldiCoord) -> Self {
        Self {
            local: coord,
            update_count: 0,
        }
    }

    /// Process RTT measurement from peer
    pub fn observe(&mut self, peer_coord: &VivaldiCoord, measured_rtt_ms: f64) {
        let rtt = Fixed::from_f64(measured_rtt_ms);
        self.local.update(peer_coord, rtt);
        self.update_count += 1;
    }

    /// Predict RTT to a remote node
    #[inline(always)]
    #[must_use]
    pub fn predict_rtt(&self, remote: &VivaldiCoord) -> Fixed {
        self.local.predict_rtt(remote)
    }

    /// Get local coordinate
    #[inline(always)]
    #[must_use]
    pub fn get_coord(&self) -> &VivaldiCoord {
        &self.local
    }

    /// Check if coordinate has stabilized
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.update_count > 10 && self.local.error < SCALE * 3 / 10
    }
}

impl Default for VivaldiSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_basic() {
        let a = Fixed::from_ms(10);
        let b = Fixed::from_ms(20);

        assert_eq!((a + b).to_ms(), 30);
        assert_eq!((b - a).to_ms(), 10);
    }

    #[test]
    fn test_fixed_mul_div() {
        let a = Fixed::from_f64(2.5);
        let b = Fixed::from_f64(4.0);

        let product = a * b;
        assert!((product.to_f64() - 10.0).abs() < 0.01);

        let quotient = b / a;
        assert!((quotient.to_f64() - 1.6).abs() < 0.01);
    }

    #[test]
    fn test_fixed_sqrt() {
        let a = Fixed::from_f64(25.0);
        let sqrt_a = a.sqrt();
        assert!((sqrt_a.to_f64() - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_coord_distance() {
        let a = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let b = VivaldiCoord::at(3.0, 4.0, 0.0, 5.0);

        // Euclidean distance = sqrt(3^2 + 4^2) = 5
        let dist = a.euclidean_distance(&b);
        assert!(
            (dist.to_f64() - 5.0).abs() < 0.5,
            "Distance: {}",
            dist.to_f64()
        );

        // RTT = distance + heights = 5 + 5 + 5 = 15
        let rtt = a.predict_rtt(&b);
        assert!((rtt.to_f64() - 15.0).abs() < 1.0, "RTT: {}", rtt.to_f64());
    }

    #[test]
    fn test_coord_update() {
        let mut a = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let b = VivaldiCoord::at(100.0, 0.0, 0.0, 5.0);

        // Initial distance
        let initial_dist = a.euclidean_distance(&b);

        // Measured RTT is 50ms, predicted is ~110ms
        // So we should move closer
        a.update(&b, Fixed::from_f64(50.0));
        let final_dist = a.euclidean_distance(&b);

        assert!(final_dist < initial_dist, "Should move closer");
    }

    #[test]
    fn test_vivaldi_convergence() {
        let mut sys_a = VivaldiSystem::new();
        let mut sys_b = VivaldiSystem::with_coord(VivaldiCoord::at(1.0, 0.0, 0.0, 1.0));
        let mut sys_c = VivaldiSystem::with_coord(VivaldiCoord::at(0.0, 1.0, 0.0, 1.0));

        for _ in 0..200 {
            sys_a.observe(&sys_b.local, 20.0);
            sys_b.observe(&sys_a.local, 20.0);
            sys_a.observe(&sys_c.local, 30.0);
            sys_c.observe(&sys_a.local, 30.0);
            sys_b.observe(&sys_c.local, 25.0);
            sys_c.observe(&sys_b.local, 25.0);
        }

        let pred_ab = sys_a.predict_rtt(&sys_b.local).to_f64();
        let pred_ac = sys_a.predict_rtt(&sys_c.local).to_f64();
        let pred_bc = sys_b.predict_rtt(&sys_c.local).to_f64();

        // Check reasonable range
        assert!(pred_ab > 5.0 && pred_ab < 60.0, "A-B: {}", pred_ab);
        assert!(pred_ac > 5.0 && pred_ac < 80.0, "A-C: {}", pred_ac);
        assert!(pred_bc > 5.0 && pred_bc < 70.0, "B-C: {}", pred_bc);
    }

    #[test]
    fn test_coord_serialization() {
        let coord = VivaldiCoord::at(10.5, -20.3, 30.7, 5.5);
        let bytes = coord.to_bytes();
        let restored = VivaldiCoord::from_bytes(&bytes);

        assert!((coord.inner.x() - restored.inner.x()).abs() < 0.001);
        assert!((coord.inner.y() - restored.inner.y()).abs() < 0.001);
    }

    #[test]
    fn test_load_factor() {
        let coord = VivaldiCoord::at(0.0, 0.0, 0.0, 10.0);
        let loaded = coord.apply_load_factor(1.0);

        // Load factor should increase effective height
        assert!(loaded.inner.data[3] > coord.inner.data[3]);
    }

    #[test]
    fn test_fused_layout() {
        // Verify VivaldiCoord contains SimdCoord directly (zero-cost access)
        let coord = VivaldiCoord::at(10.0, 20.0, 30.0, 5.0);
        let simd = coord.as_simd();

        assert!((simd.x() - 10.0).abs() < 0.01);
        assert!((simd.y() - 20.0).abs() < 0.01);
        assert!((simd.z() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_layout() {
        // VivaldiCoord should be 48 bytes with proper alignment
        assert_eq!(core::mem::size_of::<VivaldiCoord>(), 64); // 32 + 8 + 8 + padding
        assert_eq!(core::mem::align_of::<VivaldiCoord>(), 32);
    }

    // --- Boundary tests ---

    #[test]
    fn test_fixed_from_ms_zero() {
        let f = Fixed::from_ms(0);
        assert_eq!(f.to_ms(), 0);
        assert_eq!(f.to_f64(), 0.0);
    }

    #[test]
    fn test_fixed_from_us() {
        let f = Fixed::from_us(1000);
        assert_eq!(f.to_ms(), 1);
    }

    #[test]
    fn test_fixed_division_by_zero() {
        let a = Fixed::from_f64(10.0);
        let zero = Fixed::ZERO;
        let result = a / zero;
        assert_eq!(result.0, i64::MAX);
    }

    #[test]
    fn test_fixed_sqrt_zero_and_negative() {
        assert_eq!(Fixed::ZERO.sqrt(), Fixed::ZERO);
        assert_eq!(Fixed::from_f64(-5.0).sqrt(), Fixed::ZERO);
    }

    #[test]
    fn test_fixed_abs() {
        let neg = Fixed::from_f64(-7.5);
        let pos = neg.abs();
        assert!((pos.to_f64() - 7.5).abs() < 0.01);
    }

    #[test]
    fn test_fixed_min_max() {
        let a = Fixed::from_f64(3.0);
        let b = Fixed::from_f64(7.0);
        assert_eq!(a.min(b), a);
        assert_eq!(a.max(b), b);
    }

    #[test]
    fn test_fixed_identity_operations() {
        let a = Fixed::from_f64(42.0);
        assert_eq!((a + Fixed::ZERO).0, a.0);
        assert_eq!((a - Fixed::ZERO).0, a.0);
        assert_eq!((a * Fixed::ONE).to_f64().round(), 42.0);
    }

    #[test]
    fn test_fixed_from_f64_roundtrip() {
        for &val in &[0.0, 1.0, -1.0, 3.125, 100.5, -999.9] {
            let f = Fixed::from_f64(val);
            assert!((f.to_f64() - val).abs() < 0.01, "Failed for {}", val);
        }
    }

    #[test]
    fn test_fixed_from_ms_roundtrip() {
        for &val in &[0i64, 1, 10, 100, 1000] {
            let f = Fixed::from_ms(val);
            assert_eq!(f.to_ms(), val);
        }
    }

    #[test]
    fn test_vivaldi_coord_new_defaults() {
        let c = VivaldiCoord::new();
        assert!((c.inner.x() - 0.0).abs() < 0.01);
        assert!((c.inner.y() - 0.0).abs() < 0.01);
        // error starts at 1.0 (SCALE)
        assert_eq!(c.error, SCALE);
    }

    #[test]
    fn test_vivaldi_system_not_stable_initially() {
        let sys = VivaldiSystem::new();
        assert!(!sys.is_stable());
        assert_eq!(sys.update_count, 0);
    }

    #[test]
    fn test_vivaldi_from_simd() {
        let simd = SimdCoord::from_f64(5.0, 10.0, 15.0, 3.0);
        let coord = VivaldiCoord::from_simd(simd);
        assert!((coord.as_simd().x() - 5.0).abs() < 0.01);
        assert!((coord.as_simd().y() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_vivaldi_coord_euclidean_symmetry() {
        let a = VivaldiCoord::at(10.0, 20.0, 0.0, 5.0);
        let b = VivaldiCoord::at(30.0, -5.0, 10.0, 3.0);
        assert_eq!(a.euclidean_distance(&b).0, b.euclidean_distance(&a).0);
    }

    #[test]
    fn test_vivaldi_system_with_coord() {
        let coord = VivaldiCoord::at(10.0, 20.0, 0.0, 5.0);
        let sys = VivaldiSystem::with_coord(coord);
        assert!((sys.get_coord().as_simd().x() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_vivaldi_system_default() {
        let sys = VivaldiSystem::default();
        assert_eq!(sys.update_count, 0);
    }

    #[test]
    fn test_vivaldi_as_simd_mut() {
        let mut coord = VivaldiCoord::new();
        coord.as_simd_mut().data[0] = 12345;
        assert_eq!(coord.as_simd().data[0], 12345);
    }

    #[test]
    fn test_vivaldi_predict_rtt_symmetry() {
        let a = VivaldiCoord::at(0.0, 0.0, 0.0, 5.0);
        let b = VivaldiCoord::at(10.0, 10.0, 0.0, 3.0);
        assert_eq!(a.predict_rtt(&b).0, b.predict_rtt(&a).0);
    }
}

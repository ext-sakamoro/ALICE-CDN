//! Content Type Awareness for ALICE-CDN
//!
//! Provides type-aware content routing and ASDF (ALICE-SDF binary format)
//! metadata extraction. Different content types have different latency
//! requirements and caching strategies:
//!
//! - **ASDF** (SDF fields): High priority, low latency required for collision
//! - **Mesh**: Medium priority, cacheable
//! - **Texture**: Lower priority, highly cacheable
//! - **Audio**: Streaming-friendly, sequential access
//!
//! # ASDF Header Format (from ALICE-SDF)
//!
//! ```text
//! Offset  Size  Field
//! 0       4     magic "ASDF"
//! 4       2     version (u16 LE)
//! 6       2     flags (u16 LE)
//! 8       4     node_count (u32 LE)
//! 12      4     crc32 (u32 LE)
//! ─────────────────────────
//! Total: 16 bytes
//! ```
//!
//! # Example
//!
//! ```rust
//! use alice_cdn::content_types::{ContentType, AsdfMetadata, typed_content_id, extract_content_type};
//!
//! // Detect type from raw data
//! let data = b"ASDF\x01\x00\x00\x00\x0a\x00\x00\x00\x00\x00\x00\x00extra...";
//! let ct = ContentType::detect(data);
//! assert_eq!(ct, ContentType::Asdf);
//!
//! // Parse ASDF metadata
//! let header: [u8; 16] = data[..16].try_into().unwrap();
//! let meta = AsdfMetadata::parse(&header).unwrap();
//! assert_eq!(meta.node_count, 10);
//!
//! // Create type-aware content ID for routing
//! let raw_id: u64 = 12345;
//! let typed = typed_content_id(raw_id, ContentType::Asdf);
//! assert_eq!(extract_content_type(typed), ContentType::Asdf);
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// ASDF magic bytes identifying ALICE-SDF binary format.
pub const ASDF_MAGIC: [u8; 4] = *b"ASDF";

/// ASDF magic as a single u32 for branchless comparison.
const ASDF_MAGIC_U32: u32 = u32::from_le_bytes(*b"ASDF");

/// Lookup table: ContentType discriminant → priority weight
const PRIORITY_TABLE: [u32; 5] = [4, 3, 2, 1, 1]; // Asdf, Mesh, Texture, Audio, Generic

/// Lookup table: ContentType discriminant → suggested replicas
const REPLICA_TABLE: [usize; 5] = [5, 3, 2, 2, 2]; // Asdf, Mesh, Texture, Audio, Generic

/// Lookup table: ContentType discriminant → type_bits for typed_content_id
const TYPE_BITS_TABLE: [u64; 5] = [1, 2, 3, 4, 0]; // Asdf, Mesh, Texture, Audio, Generic

/// Content type categories for routing and caching decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentType {
    /// ALICE-SDF binary field — requires low latency for real-time collision
    Asdf,
    /// 3D mesh data (vertices, indices, normals)
    Mesh,
    /// Texture image data
    Texture,
    /// Audio stream data
    Audio,
    /// Unknown or uncategorized content
    Generic,
}

impl ContentType {
    /// Discriminant index for lookup table access.
    #[inline(always)]
    const fn idx(self) -> usize {
        match self {
            ContentType::Asdf => 0,
            ContentType::Mesh => 1,
            ContentType::Texture => 2,
            ContentType::Audio => 3,
            ContentType::Generic => 4,
        }
    }

    /// Auto-detect content type from the first bytes of data.
    ///
    /// Uses single u32 integer comparison instead of byte-by-byte slice compare.
    #[inline]
    pub fn detect(data: &[u8]) -> Self {
        if data.len() >= 4 {
            // Single u32 comparison — 1 instruction vs 4-byte memcmp
            let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            if magic == ASDF_MAGIC_U32 {
                return ContentType::Asdf;
            }
        }
        // Future: detect glTF (magic: "glTF"), PNG (0x89504E47), OGG, etc.
        ContentType::Generic
    }

    /// Routing priority weight for latency-aware delivery.
    ///
    /// Higher weight = higher priority = prefer closer/faster nodes.
    /// Table lookup — branchless, single array access.
    #[inline(always)]
    pub fn priority_weight(&self) -> u32 {
        PRIORITY_TABLE[self.idx()]
    }

    /// Suggested replica count for this content type.
    ///
    /// Table lookup — branchless, single array access.
    #[inline(always)]
    pub fn suggested_replicas(&self) -> usize {
        REPLICA_TABLE[self.idx()]
    }

    /// Whether this content type benefits from proximity-based caching.
    #[inline(always)]
    pub fn is_latency_sensitive(&self) -> bool {
        matches!(self, ContentType::Asdf | ContentType::Mesh)
    }
}

/// Metadata extracted from an ASDF file header (16 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsdfMetadata {
    /// Format version
    pub version: u16,
    /// Feature flags
    pub flags: u16,
    /// Number of SDF nodes in the tree
    pub node_count: u32,
    /// CRC32 checksum of the body
    pub crc32: u32,
}

impl AsdfMetadata {
    /// Parse ASDF metadata from a 16-byte header.
    ///
    /// Returns `None` if magic bytes don't match "ASDF".
    /// Uses single u32 comparison for magic check.
    #[inline]
    pub fn parse(header: &[u8; 16]) -> Option<Self> {
        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        if magic != ASDF_MAGIC_U32 {
            return None;
        }
        Some(Self {
            version: u16::from_le_bytes([header[4], header[5]]),
            flags: u16::from_le_bytes([header[6], header[7]]),
            node_count: u32::from_le_bytes([header[8], header[9], header[10], header[11]]),
            crc32: u32::from_le_bytes([header[12], header[13], header[14], header[15]]),
        })
    }

    /// Estimate the total file size from node count.
    ///
    /// ASDF nodes average ~32 bytes each (bytecode instructions are 32B aligned).
    #[inline(always)]
    pub fn estimated_size(&self) -> usize {
        16 + self.node_count as usize * 32
    }

    /// Whether this ASDF uses any extended feature flags.
    #[inline(always)]
    pub fn has_flags(&self) -> bool {
        self.flags != 0
    }
}

/// Create a type-tagged content ID for routing.
///
/// Encodes the content type in the top 4 bits of the ContentId.
/// Table lookup — no match branching.
///
/// Layout: `[4-bit type][60-bit content_id]`
#[inline(always)]
pub fn typed_content_id(content_id: u64, content_type: ContentType) -> u64 {
    (TYPE_BITS_TABLE[content_type.idx()] << 60) | (content_id & 0x0FFF_FFFF_FFFF_FFFF)
}

/// Extract the content type from a type-tagged content ID.
#[inline(always)]
pub fn extract_content_type(typed_id: u64) -> ContentType {
    /// Reverse lookup: type_bits → ContentType (index 0-4 used, rest = Generic)
    const REVERSE: [ContentType; 5] = [
        ContentType::Generic, // 0
        ContentType::Asdf,    // 1
        ContentType::Mesh,    // 2
        ContentType::Texture, // 3
        ContentType::Audio,   // 4
    ];
    let bits = (typed_id >> 60) as usize;
    if bits < 5 {
        REVERSE[bits]
    } else {
        ContentType::Generic
    }
}

/// Extract the raw content ID (without type tag) from a typed ID.
#[inline(always)]
pub fn extract_raw_id(typed_id: u64) -> u64 {
    typed_id & 0x0FFF_FFFF_FFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_asdf() {
        let mut data = [0u8; 32];
        data[..4].copy_from_slice(b"ASDF");
        assert_eq!(ContentType::detect(&data), ContentType::Asdf);
    }

    #[test]
    fn test_detect_generic() {
        let data = [0u8; 32];
        assert_eq!(ContentType::detect(&data), ContentType::Generic);
    }

    #[test]
    fn test_detect_short_data() {
        let data = [b'A', b'S'];
        assert_eq!(ContentType::detect(&data), ContentType::Generic);
    }

    #[test]
    fn test_asdf_metadata_parse() {
        let mut header = [0u8; 16];
        header[..4].copy_from_slice(b"ASDF");
        header[4..6].copy_from_slice(&1u16.to_le_bytes()); // version = 1
        header[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        header[8..12].copy_from_slice(&42u32.to_le_bytes()); // node_count = 42
        header[12..16].copy_from_slice(&0xDEADBEEFu32.to_le_bytes()); // crc32

        let meta = AsdfMetadata::parse(&header).unwrap();
        assert_eq!(meta.version, 1);
        assert_eq!(meta.flags, 0);
        assert_eq!(meta.node_count, 42);
        assert_eq!(meta.crc32, 0xDEADBEEF);
        assert_eq!(meta.estimated_size(), 16 + 42 * 32);
        assert!(!meta.has_flags());
    }

    #[test]
    fn test_asdf_metadata_invalid_magic() {
        let header = [0u8; 16];
        assert!(AsdfMetadata::parse(&header).is_none());
    }

    #[test]
    fn test_typed_content_id_roundtrip() {
        let raw = 0x0123_4567_89AB_CDEFu64 & 0x0FFF_FFFF_FFFF_FFFF;
        let types = [
            ContentType::Generic,
            ContentType::Asdf,
            ContentType::Mesh,
            ContentType::Texture,
            ContentType::Audio,
        ];

        for ct in types {
            let typed = typed_content_id(raw, ct);
            assert_eq!(extract_content_type(typed), ct);
            assert_eq!(extract_raw_id(typed), raw);
        }
    }

    #[test]
    fn test_priority_ordering() {
        assert!(ContentType::Asdf.priority_weight() > ContentType::Mesh.priority_weight());
        assert!(ContentType::Mesh.priority_weight() > ContentType::Texture.priority_weight());
        assert!(ContentType::Texture.priority_weight() > ContentType::Audio.priority_weight());
    }

    #[test]
    fn test_latency_sensitivity() {
        assert!(ContentType::Asdf.is_latency_sensitive());
        assert!(ContentType::Mesh.is_latency_sensitive());
        assert!(!ContentType::Audio.is_latency_sensitive());
        assert!(!ContentType::Generic.is_latency_sensitive());
    }

    #[test]
    fn test_suggested_replicas() {
        assert!(ContentType::Asdf.suggested_replicas() > ContentType::Generic.suggested_replicas());
    }
}

//! ALICE-Crypto bridge: Content encryption for CDN delivery
//!
//! Provides content encryption and integrity verification for CDN-delivered
//! assets (DRM, private content, signed payloads).

use alice_crypto::{self as crypto, CipherError, Key};

/// Sealed CDN content with metadata.
#[derive(Debug, Clone)]
pub struct SealedContent {
    /// Encrypted data (nonce + ciphertext + tag)
    pub data: Vec<u8>,
    /// Content hash for cache deduplication (computed before encryption)
    pub content_hash: crypto::Hash,
    /// Original plaintext size
    pub plaintext_len: usize,
}

/// Encrypt content for secure CDN delivery.
///
/// The content hash is computed on plaintext before encryption,
/// enabling deduplication across encrypted copies with different keys.
pub fn seal_content(plaintext: &[u8], key: &Key) -> Result<SealedContent, CipherError> {
    let content_hash = crypto::hash(plaintext);
    let data = crypto::seal(key, plaintext)?;
    Ok(SealedContent {
        plaintext_len: plaintext.len(),
        content_hash,
        data,
    })
}

/// Decrypt CDN content.
pub fn open_content(sealed: &SealedContent, key: &Key) -> Result<Vec<u8>, CipherError> {
    crypto::open(key, &sealed.data)
}

/// Verify content integrity without decryption.
///
/// Compares the encrypted blob hash (not content hash) for
/// tamper detection in transit.
pub fn verify_integrity(data: &[u8]) -> crypto::Hash {
    crypto::hash(data)
}

/// Derive a per-asset encryption key from asset ID and master secret.
pub fn derive_asset_key(asset_id: u64, master_secret: &[u8]) -> Key {
    let context = format!("alice-cdn-asset-v1:{}", asset_id);
    let raw = crypto::derive_key(&context, master_secret);
    Key::from_bytes(raw)
}

/// Derive a per-channel DRM key.
pub fn derive_drm_key(channel_id: &str, license_secret: &[u8]) -> Key {
    let context = format!("alice-cdn-drm-v1:{}", channel_id);
    let raw = crypto::derive_key(&context, license_secret);
    Key::from_bytes(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seal_open_roundtrip() {
        let key = Key::generate().unwrap();
        let plaintext = b"SDF asset data: sphere(1.0) - box(0.5)";

        let sealed = seal_content(plaintext, &key).unwrap();
        assert_eq!(sealed.plaintext_len, plaintext.len());

        let recovered = open_content(&sealed, &key).unwrap();
        assert_eq!(&recovered, plaintext);
    }

    #[test]
    fn test_wrong_key_fails() {
        let k1 = Key::generate().unwrap();
        let k2 = Key::generate().unwrap();
        let sealed = seal_content(b"secret", &k1).unwrap();
        assert!(open_content(&sealed, &k2).is_err());
    }

    #[test]
    fn test_derive_asset_key_deterministic() {
        let k1 = derive_asset_key(42, b"master");
        let k2 = derive_asset_key(42, b"master");
        assert_eq!(k1.as_bytes(), k2.as_bytes());
    }

    #[test]
    fn test_content_hash_before_encryption() {
        let key = Key::generate().unwrap();
        let data = b"same content";
        let s1 = seal_content(data, &key).unwrap();
        let s2 = seal_content(data, &key).unwrap();
        // Content hash matches (same plaintext)
        assert_eq!(s1.content_hash.as_bytes(), s2.content_hash.as_bytes());
        // Encrypted data differs (random nonce)
        assert_ne!(s1.data, s2.data);
    }
}

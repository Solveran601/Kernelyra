//! Bounded file-signature classification for the format-intelligence layer.
//!
//! This module deliberately receives only a small prefix.  It never parses an
//! untrusted container or allocates according to a value declared in a file.

pub const UNKNOWN: u32 = 0;
pub const PARQUET: u32 = 1;
pub const SQLITE: u32 = 2;
pub const NUMPY: u32 = 3;
pub const HDF5: u32 = 4;
pub const ZIP: u32 = 5;
pub const GZIP: u32 = 6;
pub const PDF: u32 = 7;
pub const PNG: u32 = 8;
pub const JPEG: u32 = 9;
pub const GIF: u32 = 10;
pub const RIFF: u32 = 11;
pub const FLAC: u32 = 12;
pub const OGG: u32 = 13;
pub const MP4: u32 = 14;
pub const MATROSKA: u32 = 15;
pub const JSON: u32 = 16;
pub const DELIMITED_TEXT: u32 = 17;

pub fn classify(bytes: &[u8]) -> u32 {
    let prefix = &bytes[..bytes.len().min(4096)];
    if prefix.starts_with(b"PAR1") {
        PARQUET
    } else if prefix.starts_with(b"SQLite format 3\0") {
        SQLITE
    } else if prefix.starts_with(b"\x93NUMPY") {
        NUMPY
    } else if prefix.starts_with(b"\x89HDF\r\n\x1a\n") {
        HDF5
    } else if prefix.starts_with(b"PK\x03\x04") || prefix.starts_with(b"PK\x05\x06") {
        ZIP
    } else if prefix.starts_with(&[0x1f, 0x8b]) {
        GZIP
    } else if prefix.starts_with(b"%PDF-") {
        PDF
    } else if prefix.starts_with(b"\x89PNG\r\n\x1a\n") {
        PNG
    } else if prefix.starts_with(&[0xff, 0xd8, 0xff]) {
        JPEG
    } else if prefix.starts_with(b"GIF87a") || prefix.starts_with(b"GIF89a") {
        GIF
    } else if prefix.starts_with(b"fLaC") {
        FLAC
    } else if prefix.starts_with(b"OggS") {
        OGG
    } else if prefix.starts_with(b"\x1a\x45\xdf\xa3") {
        MATROSKA
    } else if prefix.len() >= 12 && &prefix[4..8] == b"ftyp" {
        MP4
    } else if prefix.len() >= 12 && &prefix[..4] == b"RIFF" {
        RIFF
    } else {
        classify_text(prefix)
    }
}

fn classify_text(prefix: &[u8]) -> u32 {
    let trimmed = prefix.iter().copied().skip_while(|byte| byte.is_ascii_whitespace());
    let Some(first) = trimmed.into_iter().next() else {
        return UNKNOWN;
    };
    if matches!(first, b'{' | b'[') {
        return JSON;
    }
    if !prefix.iter().all(|byte| byte.is_ascii() || *byte >= 0x80) {
        return UNKNOWN;
    }
    let has_line = prefix.contains(&b'\n') || prefix.contains(&b'\r');
    let has_separator = prefix.contains(&b',') || prefix.contains(&b'\t') || prefix.contains(&b';');
    if has_line && has_separator {
        DELIMITED_TEXT
    } else {
        UNKNOWN
    }
}

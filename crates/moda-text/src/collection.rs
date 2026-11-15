use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) struct CollectionFace {
    pub source: PathBuf,
    pub index: u32,
    pub metadata: Vec<u8>,
    cache_dir: PathBuf,
}

pub(crate) fn descriptor(path: &Path) -> Result<CollectionFace, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    Ok(CollectionFace {
        source: value["source"]
            .as_str()
            .ok_or("missing collection source")?
            .into(),
        index: value["index"]
            .as_u64()
            .and_then(|index| u32::try_from(index).ok())
            .ok_or("invalid collection index")?,
        cache_dir: value["cacheDir"]
            .as_str()
            .ok_or("missing collection cache")?
            .into(),
        metadata: value["metadata"]
            .as_array()
            .ok_or("missing collection metadata")?
            .iter()
            .map(|byte| {
                byte.as_u64()
                    .and_then(|byte| u8::try_from(byte).ok())
                    .ok_or("invalid metadata byte")
            })
            .collect::<Result<_, _>>()?,
    })
}

fn word(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let bytes = bytes.get(offset..offset + 4).ok_or("truncated font")?;
    Ok(u32::from_be_bytes(bytes.try_into().unwrap()))
}

fn read_at(file: &mut std::fs::File, offset: u64, size: usize) -> Result<Vec<u8>, String> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| error.to_string())?;
    let mut bytes = vec![0; size];
    file.read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

fn checksum(bytes: &[u8]) -> u32 {
    bytes.chunks(4).fold(0u32, |sum, chunk| {
        let mut padded = [0; 4];
        padded[..chunk.len()].copy_from_slice(chunk);
        sum.wrapping_add(u32::from_be_bytes(padded))
    })
}

fn extract(file: &mut std::fs::File, index: u32) -> Result<Vec<u8>, String> {
    let header = read_at(file, 0, 12)?;
    if &header[..4] != b"ttcf" || index >= word(&header, 8)? {
        return Err("invalid collection face".into());
    }
    let offset = word(&read_at(file, 12 + u64::from(index) * 4, 4)?, 0)?;
    let header = read_at(file, u64::from(offset), 12)?;
    let count = u16::from_be_bytes([header[4], header[5]]) as usize;
    let records = read_at(file, u64::from(offset) + 12, count * 16)?;
    let mut tables = Vec::new();
    for record in records.chunks_exact(16) {
        let tag: [u8; 4] = record[..4].try_into().unwrap();
        if &tag == b"DSIG" {
            continue;
        }
        let mut bytes = read_at(
            file,
            u64::from(word(record, 8)?),
            word(record, 12)? as usize,
        )?;
        if &tag == b"head" && bytes.len() >= 12 {
            bytes[8..12].fill(0);
        }
        tables.push((tag, bytes));
    }
    tables.sort_by_key(|(tag, _)| *tag);
    let count = tables.len();
    let selector = if count == 0 { 0 } else { count.ilog2() as u16 };
    let search_range = (1u16 << selector) * 16;
    let mut output = vec![0; 12 + count * 16];
    output[..4].copy_from_slice(&header[..4]);
    output[4..6].copy_from_slice(&(count as u16).to_be_bytes());
    output[6..8].copy_from_slice(&search_range.to_be_bytes());
    output[8..10].copy_from_slice(&selector.to_be_bytes());
    output[10..12].copy_from_slice(&((count as u16) * 16 - search_range).to_be_bytes());
    let mut head = None;
    for (position, (tag, bytes)) in tables.iter().enumerate() {
        let record = 12 + position * 16;
        let offset = output.len();
        output[record..record + 4].copy_from_slice(tag);
        output[record + 4..record + 8].copy_from_slice(&checksum(bytes).to_be_bytes());
        output[record + 8..record + 12].copy_from_slice(&(offset as u32).to_be_bytes());
        output[record + 12..record + 16].copy_from_slice(&(bytes.len() as u32).to_be_bytes());
        if tag == b"head" && bytes.len() >= 12 {
            head = Some(offset);
        }
        output.extend_from_slice(bytes);
        output.resize((output.len() + 3) & !3, 0);
    }
    if let Some(head) = head {
        let adjustment = 0xb1b0afbau32.wrapping_sub(checksum(&output));
        output[head + 8..head + 12].copy_from_slice(&adjustment.to_be_bytes());
    }
    Ok(output)
}

pub(crate) fn read(path: &Path) -> Result<Vec<u8>, String> {
    let face = descriptor(path)?;
    let mut source = std::fs::File::open(&face.source).map_err(|error| error.to_string())?;
    let stat = source.metadata().map_err(|error| error.to_string())?;
    let modified = stat
        .modified()
        .map_err(|error| error.to_string())?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    #[cfg(not(unix))]
    let changed = String::new();
    #[cfg(unix)]
    let changed = {
        use std::os::unix::fs::MetadataExt;
        (i128::from(stat.ctime()) * 1_000_000_000 + i128::from(stat.ctime_nsec())).to_string()
    };
    let input = format!(
        "sfnt-v1\n{}\n{}\n{}\n{}\n{}",
        face.source.display(),
        face.index,
        stat.len(),
        modified,
        changed
    );
    let hash = input.bytes().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    let cached = face.cache_dir.join(format!("{hash:016x}.ttf"));
    if let Ok(bytes) = std::fs::read(&cached) {
        return Ok(bytes);
    }
    let bytes = extract(&mut source, face.index)?;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    if std::fs::create_dir_all(&face.cache_dir).is_ok() {
        let temporary = cached.with_extension(format!(
            "{}.{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        if std::fs::write(&temporary, &bytes).is_ok() {
            let _ = std::fs::rename(&temporary, &cached);
        }
        let _ = std::fs::remove_file(&temporary);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cosmic_text::fontdb;

    fn fixture() -> (PathBuf, PathBuf, Vec<u8>) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "moda-collection-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let font = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../moda-engine/tests/fixtures/fonts/inter/Inter_24pt-Regular.ttf"),
        )
        .unwrap();
        let mut relocated = font.clone();
        let count = u16::from_be_bytes([font[4], font[5]]) as usize;
        for position in 0..count {
            let offset = 12 + position * 16 + 8;
            let absolute = word(&font, offset).unwrap() + 20;
            relocated[offset..offset + 4].copy_from_slice(&absolute.to_be_bytes());
        }
        let mut collection = b"ttcf\0\x01\0\0\0\0\0\x02\0\0\0\x14\0\0\0\x14".to_vec();
        collection.extend_from_slice(&relocated);
        let source = root.join("font.ttc");
        std::fs::write(&source, collection).unwrap();
        let descriptor = root.join("font.font-face");
        std::fs::write(
            &descriptor,
            serde_json::to_vec(&serde_json::json!({
                "source": source, "index": 1, "cacheDir": root.join("cache"), "metadata": font,
            }))
            .unwrap(),
        )
        .unwrap();
        (root, descriptor, font)
    }

    #[test]
    fn lazy_cache_preserves_tables_and_invalidates_changed_source() {
        let (root, descriptor, font) = fixture();
        assert!(!root.join("cache").exists());
        let bytes = read(&descriptor).unwrap();
        assert_eq!(checksum(&bytes), 0xb1b0afba);
        let count = u16::from_be_bytes([font[4], font[5]]) as usize;
        for record in font[12..12 + count * 16].chunks_exact(16) {
            if &record[..4] == b"head" || &record[..4] == b"DSIG" {
                continue;
            }
            let output_count = u16::from_be_bytes([bytes[4], bytes[5]]) as usize;
            let output_record = bytes[12..12 + output_count * 16]
                .chunks_exact(16)
                .find(|output| output[..4] == record[..4])
                .unwrap();
            let original_offset = word(record, 8).unwrap() as usize;
            let output_offset = word(output_record, 8).unwrap() as usize;
            let length = word(record, 12).unwrap() as usize;
            assert_eq!(
                &bytes[output_offset..output_offset + length],
                &font[original_offset..original_offset + length]
            );
        }
        let cached = std::fs::read_dir(root.join("cache"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let modified = std::fs::metadata(&cached).unwrap().modified().unwrap();
        assert_eq!(read(&descriptor).unwrap(), bytes);
        assert_eq!(
            std::fs::metadata(&cached).unwrap().modified().unwrap(),
            modified
        );
        let source = root.join("font.ttc");
        let mut changed = std::fs::read(&source).unwrap();
        changed.extend_from_slice(&[0; 4]);
        std::fs::write(&source, changed).unwrap();
        assert_eq!(read(&descriptor).unwrap(), bytes);
        assert_eq!(std::fs::read_dir(root.join("cache")).unwrap().count(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn degraded_collection_registration_preserves_the_selected_face_without_extracting() {
        let (root, descriptor, font) = fixture();
        let manifest = serde_json::json!({"systemFonts": false, "keys": {"Inter-400-normal": {"family": "Inter", "ttfPath": descriptor}}});
        let mut eager = fontdb::Database::new();
        let eager_ids = eager.load_font_source(fontdb::Source::Binary(std::sync::Arc::new(font)));
        assert_eq!(eager.face(eager_ids[0]).unwrap().index, 0);
        let host = crate::degraded::build_catalog_engine_from_snapshot(&root, &manifest);
        assert!(host
            .unresolved_catalog_keys(&["Inter-400-normal".into()])
            .is_empty());
        assert!(host.has_fonts());
        assert!(!root.join("cache").exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}

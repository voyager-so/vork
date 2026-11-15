//! Byte-capped LRU cache (std-only, no extra deps).
//!
//! Backs the wasm `TextHost`'s decoded RGBA caches (atlas pages ~7-17 MB each,
//! emoji ~65 KB). Those previously grew unbounded — every decoded page stayed
//! resident for the host's lifetime, pinning the never-shrinking wasm
//! linear-memory arena. Bounding them by total bytes keeps the resident
//! working set close to what a frame actually samples (the GPU already holds
//! the uploaded texture; the CPU copy is only needed to (re)upload).
//!
//! `V: Clone` — our values are `Arc<…>` / `Option<Arc<…>>`, so a clone is a
//! refcount bump, not a byte copy.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

struct Entry<V> {
    value: V,
    byte_size: u64,
    /// Monotonic access stamp; the smallest is the least-recently-used.
    access: u64,
}

pub struct ByteLruCache<K: Eq + Hash + Clone, V: Clone> {
    map: HashMap<K, Entry<V>>,
    byte_cap: u64,
    /// Hard cap on entry COUNT, independent of bytes. Needed because some values
    /// are charged ~0 bytes (e.g. the emoji cache's missing-glyph `None`
    /// placeholders): without it those entries never trip the byte cap, so the
    /// map would grow unbounded and the O(N) victim scan in `evict_to_cap` would
    /// get progressively slower. Bounds both memory and worst-case scan cost.
    max_entries: usize,
    current_bytes: u64,
    next_access: u64,
}

impl<K: Eq + Hash + Clone, V: Clone> ByteLruCache<K, V> {
    /// `byte_cap` bounds total decoded bytes; `max_entries` bounds entry count
    /// (so 0-byte entries can't accumulate without limit). Eviction stops at the
    /// FIRST limit hit, but never empties the cache below one entry.
    pub fn new(byte_cap: u64, max_entries: usize) -> Self {
        Self {
            map: HashMap::new(),
            byte_cap,
            max_entries: max_entries.max(1),
            current_bytes: 0,
            next_access: 0,
        }
    }

    /// Look up a key, promoting it to most-recently-used. Returns a clone of
    /// the value (refcount bump for our `Arc` payloads). `Borrow`-generic like
    /// `HashMap::get` so hot-path callers look up a `String` key by `&str`
    /// without allocating a `String` per lookup.
    pub fn get<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let stamp = self.next_access;
        let hit = self.map.get_mut(key).map(|e| {
            e.access = stamp;
            e.value.clone()
        });
        if hit.is_some() {
            self.next_access += 1;
        }
        hit
    }

    /// Insert (or replace) a key with its decoded byte size, then evict the
    /// least-recently-used entries until at/under BOTH the byte cap and the
    /// entry-count cap. The just-inserted entry is the most-recently-used, so it
    /// is never the eviction victim; a single entry larger than the whole cap is
    /// still kept (we must be able to serve it) — eviction never empties the
    /// cache below one entry.
    pub fn insert(&mut self, key: K, value: V, byte_size: u64) {
        let stamp = self.next_access;
        self.next_access += 1;
        if let Some(old) = self.map.insert(
            key,
            Entry {
                value,
                byte_size,
                access: stamp,
            },
        ) {
            self.current_bytes = self.current_bytes.saturating_sub(old.byte_size);
        }
        self.current_bytes = self.current_bytes.saturating_add(byte_size);
        self.evict_to_cap();
    }

    fn evict_to_cap(&mut self) {
        while (self.current_bytes > self.byte_cap || self.map.len() > self.max_entries)
            && self.map.len() > 1
        {
            let victim = self
                .map
                .iter()
                .min_by_key(|(_, e)| e.access)
                .map(|(k, _)| k.clone());
            let Some(k) = victim else { break };
            if let Some(e) = self.map.remove(&k) {
                self.current_bytes = self.current_bytes.saturating_sub(e.byte_size);
            }
        }
    }

    pub fn current_bytes(&self) -> u64 {
        self.current_bytes
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_lru_past_cap() {
        // cap = 2 entries worth (10 bytes each, cap 20).
        let mut c: ByteLruCache<&str, u32> = ByteLruCache::new(20, usize::MAX);
        c.insert("a", 1, 10);
        c.insert("b", 2, 10);
        assert_eq!(c.len(), 2);
        assert_eq!(c.current_bytes(), 20);
        // Inserting a 3rd evicts the LRU ("a").
        c.insert("c", 3, 10);
        assert_eq!(c.len(), 2);
        assert!(c.current_bytes() <= 20);
        assert!(c.get(&"a").is_none());
        assert_eq!(c.get(&"b"), Some(2));
        assert_eq!(c.get(&"c"), Some(3));
    }

    #[test]
    fn get_promotes_recency() {
        let mut c: ByteLruCache<&str, u32> = ByteLruCache::new(20, usize::MAX);
        c.insert("a", 1, 10);
        c.insert("b", 2, 10);
        // Touch "a" so "b" becomes the LRU.
        assert_eq!(c.get(&"a"), Some(1));
        c.insert("c", 3, 10);
        // "b" evicted, "a" survived.
        assert!(c.get(&"b").is_none());
        assert_eq!(c.get(&"a"), Some(1));
        assert_eq!(c.get(&"c"), Some(3));
    }

    #[test]
    fn replace_same_key_updates_bytes_without_double_count() {
        let mut c: ByteLruCache<&str, u32> = ByteLruCache::new(1000, usize::MAX);
        c.insert("a", 1, 100);
        c.insert("a", 2, 40);
        assert_eq!(c.len(), 1);
        assert_eq!(c.current_bytes(), 40);
        assert_eq!(c.get(&"a"), Some(2));
    }

    #[test]
    fn keeps_one_entry_larger_than_cap() {
        let mut c: ByteLruCache<&str, u32> = ByteLruCache::new(10, usize::MAX);
        c.insert("huge", 1, 1000);
        // Can't evict the only entry — we must be able to serve it.
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(&"huge"), Some(1));
        // A second insert evicts down to just the newest.
        c.insert("also-huge", 2, 1000);
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(&"also-huge"), Some(2));
    }

    #[test]
    fn saturating_byte_accounting() {
        let mut c: ByteLruCache<&str, u32> = ByteLruCache::new(u64::MAX, usize::MAX);
        c.insert("a", 1, u64::MAX);
        c.insert("a", 2, 1); // replace; old huge size subtracted without underflow
        assert_eq!(c.current_bytes(), 1);
    }

    #[test]
    fn bounds_entry_count_for_zero_byte_values() {
        // Mirrors the emoji cache's missing-glyph `None` placeholders: charged 0
        // bytes, so the byte cap never evicts them. The entry-count cap must.
        let mut c: ByteLruCache<u32, u32> = ByteLruCache::new(u64::MAX, 3);
        for k in 0..100u32 {
            c.insert(k, 0, 0); // 0-byte entries — only max_entries can bound these
        }
        assert_eq!(c.current_bytes(), 0);
        assert_eq!(c.len(), 3, "0-byte entries must be bounded by max_entries");
        // The 3 most-recently-inserted survive; older ones are evicted.
        assert!(c.get(&99).is_some());
        assert!(c.get(&0).is_none());
    }

    #[test]
    fn max_entries_floored_at_one() {
        // A degenerate 0 cap must not wedge eviction (the `> 1` guard keeps one).
        let mut c: ByteLruCache<u32, u32> = ByteLruCache::new(u64::MAX, 0);
        c.insert(1, 1, 0);
        c.insert(2, 2, 0);
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(&2), Some(2));
    }
}

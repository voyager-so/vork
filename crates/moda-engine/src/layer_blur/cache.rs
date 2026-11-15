//! The retained engine's layer-capture cache: filtered capture results (a
//! layer blur's, a shadow's) kept across frames under their content key
//! (`LayerBlurGroup::cache_key`, `residency::capture_key`), so a frame whose
//! captured nodes are unchanged composites the stored result instead of
//! re-rendering and re-filtering every capture.
//!
//! The key covers everything the result's pixels are made of — the captured
//! nodes' content, placement, opacity and clip, the capture's region, size
//! and filters (the camera's zoom bucket and capture window reach it through
//! these), and the scene-wide resource generations — so there is no explicit
//! invalidation: an edit, a drag preview, a zoom past a bucket or a font load
//! simply asks for a new key, and entries nobody asks for age out.
//!
//! Bounded by a byte budget (the results' texels: RGBA8, or RGBA16F after a
//! uniform blur), evicting the least recently used entries first. One engine's device only: dropped with the
//! engine's retained GPU state (`Engine::clear_retained_gpu`).
//!
//! Stand-ins (scale RFC stage C): each result is also remembered under its
//! capture's identity (`LayerBlurGroup::identity_key`, what it draws without
//! the zoom step or window) and composite id. When a zoom crosses a capture
//! zoom step every capture on screen asks for a new key at once; the live lane
//! renders a few per frame (`Engine::set_capture_frame_budget`) and composites
//! the previous step's result, over its own region, for the rest until theirs
//! is ready — no frame renders every capture again.

use std::collections::HashMap;

/// Default byte budget of the cached results.
pub const DEFAULT_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

struct Entry<V> {
    view: V,
    /// What the entry is charged: its texels and its stand-in rows ([`row_bytes`]).
    bytes: u64,
    last_used: u64,
    /// The canvas region the result covers (a stand-in composites over it).
    region: [f64; 4],
    /// The part of `bytes` charged for the stand-in rows that name this entry. An entry
    /// owns its rows: they leave with it.
    rows: u64,
}

/// Filtered capture results by content key, LRU within a byte budget. `V` is the result
/// (a texture view; the bookkeeping is tested with a plain value).
pub struct CaptureCache<V = wgpu::TextureView> {
    entries: HashMap<u64, Entry<V>>,
    bytes: u64,
    budget: u64,
    clock: u64,
    /// (identity key, composite id) → the content key of its latest result. Every row
    /// names a held entry, which is charged for it, so the byte budget bounds this map
    /// with the entries. The rows live here only.
    stand_ins: HashMap<(u64, String), u64>,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Composites drawn from an earlier zoom step's result.
    pub stand_in_hits: u64,
}

impl<V> Default for CaptureCache<V> {
    fn default() -> Self {
        CaptureCache {
            entries: HashMap::new(),
            bytes: 0,
            budget: DEFAULT_BUDGET_BYTES,
            clock: 0,
            stand_ins: HashMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
            stand_in_hits: 0,
        }
    }
}

impl CaptureCache<wgpu::TextureView> {
    /// Keep `view` (a `width × height` RGBA result) under `key`, evicting the
    /// least recently used entries past the budget. A result larger than the
    /// whole budget is not kept.
    pub fn put(&mut self, key: u64, view: wgpu::TextureView, width: u32, height: u32) {
        self.put_with(key, view, width, height, [0.0; 4], None);
    }

    /// [`Self::put`], also remembering the result as the stand-in of
    /// `identity` (identity key, composite id), over canvas `region`.
    pub fn put_with(
        &mut self,
        key: u64,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
        region: [f64; 4],
        identity: Option<(u64, &str)>,
    ) {
        let texel = view
            .texture()
            .format()
            .block_copy_size(None)
            .unwrap_or(4);
        let bytes = u64::from(width) * u64::from(height) * u64::from(texel);
        self.keep(key, view, bytes, region, identity);
    }
}

impl<V: Clone> CaptureCache<V> {
    /// The stored result for `key`, marked as just used.
    pub fn get(&mut self, key: u64) -> Option<V> {
        self.clock += 1;
        match self.entries.get_mut(&key) {
            Some(e) => {
                e.last_used = self.clock;
                self.hits += 1;
                Some(e.view.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Keep a result of `texels` bytes under `key` as the latest of `identity`, then evict
    /// down to the budget. An entry is charged its texels and its stand-in rows; one that
    /// alone is past the whole budget is not kept.
    fn keep(&mut self, key: u64, view: V, texels: u64, region: [f64; 4], identity: Option<(u64, &str)>) {
        if texels > self.budget {
            return;
        }
        // The same key is the same result: the rows that named the old entry name this one.
        let mut rows = self.entries.remove(&key).map_or(0, |old| {
            self.bytes -= old.bytes;
            old.rows
        });
        if let Some((id, node)) = identity {
            let row = (id, node.to_string());
            let charge = row_bytes(&row);
            match self.stand_ins.insert(row, key) {
                Some(previous) if previous == key => {}
                previous => {
                    // The row moves here from the entry that held this identity's earlier result.
                    if let Some(earlier) = previous.and_then(|k| self.entries.get_mut(&k)) {
                        earlier.rows -= charge;
                        earlier.bytes -= charge;
                        self.bytes -= charge;
                    }
                    rows += charge;
                }
            }
        }
        let bytes = texels + rows;
        if bytes > self.budget {
            self.stand_ins.retain(|_, named| *named != key);
            return;
        }
        self.clock += 1;
        self.entries.insert(key, Entry { view, bytes, last_used: self.clock, region, rows });
        self.bytes += bytes;
        self.evict_to_budget(Some(key));
    }

    /// Drop least recently used entries, never `keep`, until the budget holds what is left.
    fn evict_to_budget(&mut self, keep: Option<u64>) {
        while self.bytes > self.budget {
            let Some((&oldest, _)) = self
                .entries
                .iter()
                .filter(|(k, _)| Some(**k) != keep)
                .min_by_key(|(_, e)| e.last_used)
            else {
                break;
            };
            self.remove(oldest);
            self.evictions += 1;
        }
    }

    /// Drop one entry and the stand-in rows that name it.
    fn remove(&mut self, key: u64) {
        if let Some(entry) = self.entries.remove(&key) {
            self.bytes -= entry.bytes;
            if entry.rows > 0 {
                self.stand_ins.retain(|_, named| *named != key);
            }
        }
    }

    /// The latest result kept for this capture identity and composite (another
    /// zoom step's, typically), with the canvas region it covers.
    pub fn stand_in(&mut self, identity: u64, node_id: &str) -> Option<(V, [f64; 4])> {
        let key = *self.stand_ins.get(&(identity, node_id.to_string()))?;
        let e = self.entries.get_mut(&key)?;
        self.clock += 1;
        e.last_used = self.clock;
        self.stand_in_hits += 1;
        Some((e.view.clone(), e.region))
    }

    /// [`Self::stand_in`] when the result kept for this identity covers `needed` (a canvas rect `[x, y, w, h]`):
    /// a stand-in drawn over its own region leaves the rest of `needed` without the composite.
    pub fn stand_in_covering(
        &mut self,
        identity: u64,
        node_id: &str,
        needed: [f64; 4],
    ) -> Option<(V, [f64; 4])> {
        let key = *self.stand_ins.get(&(identity, node_id.to_string()))?;
        let region = self.entries.get(&key)?.region;
        if !covers(region, needed) {
            return None;
        }
        self.stand_in(identity, node_id)
    }

    /// Change the byte budget (evicting down to it).
    pub fn set_budget(&mut self, budget: u64) {
        self.budget = budget;
        self.evict_to_budget(None);
    }

    /// Drop every entry (the counters stay).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stand_ins.clear();
        self.bytes = 0;
    }

    /// (entries, bytes) held; the bytes are the results' texels and their stand-in rows.
    pub fn resident(&self) -> (usize, u64) {
        (self.entries.len(), self.bytes)
    }

    /// Stand-in rows held: at most one per (identity, composite) whose latest result is held.
    pub fn stand_ins(&self) -> usize {
        self.stand_ins.len()
    }
}

/// What one stand-in row is charged: the map's key with the composite id's text, and the
/// content key it maps to.
fn row_bytes(row: &(u64, String)) -> u64 {
    (std::mem::size_of::<(u64, String)>() + row.1.len() + std::mem::size_of::<u64>()) as u64
}

/// Whether canvas rect `outer` holds `inner` (`[x, y, w, h]`), to a millionth of a canvas unit; an empty `inner` is
/// held by anything.
fn covers(outer: [f64; 4], inner: [f64; 4]) -> bool {
    const EPS: f64 = 1e-6;
    let [ox, oy, ow, oh] = outer;
    let [ix, iy, iw, ih] = inner;
    if !(iw > 0.0 && ih > 0.0) {
        return true;
    }
    ix >= ox - EPS && iy >= oy - EPS && ix + iw <= ox + ow + EPS && iy + ih <= oy + oh + EPS
}

#[cfg(test)]
mod tests {
    use super::{covers, row_bytes, CaptureCache};

    #[test]
    fn the_budget_bounds_the_stand_in_rows_with_the_entries() {
        // audit-2d 14: every put added a row and only a later lookup of that exact identity removed it.
        let row = |name: &str| row_bytes(&(0, name.to_string()));
        let rows_match = |cache: &CaptureCache<u32>| {
            // Every row names a held entry, and each entry is charged exactly its rows.
            let mut named = std::collections::HashMap::<u64, u64>::new();
            for (row, key) in &cache.stand_ins {
                assert!(cache.entries.contains_key(key));
                *named.entry(*key).or_default() += row_bytes(row);
            }
            for (key, entry) in &cache.entries {
                assert_eq!(entry.rows, named.get(key).copied().unwrap_or(0));
                assert!(entry.bytes >= entry.rows);
            }
            let charged: u64 = cache.entries.values().map(|e| e.bytes).sum();
            assert_eq!(charged, cache.bytes);
            assert!(cache.bytes <= cache.budget);
        };
        // Many captures whose identity changes every time: only the held results keep a row.
        let one = 1 + row("layer");
        let mut cache = CaptureCache::<u32>::default();
        cache.set_budget(3 * one);
        for n in 0..10_000u64 {
            cache.keep(n, n as u32, 1, [0.0; 4], Some((1_000_000 + n, "layer")));
            assert!(cache.resident().0 <= 3 && cache.stand_ins() <= 3, "at {n}");
        }
        rows_match(&cache);
        assert_eq!((cache.resident(), cache.stand_ins()), ((3, 3 * one), 3));
        assert_eq!(cache.stand_in(1_009_999, "layer").map(|(v, _)| v), Some(9_999));
        assert_eq!(cache.stand_in(1_000_000, "layer"), None, "an evicted result stands in for nothing");
        // One result that many identities or composites name is charged for every row, so the
        // rows cannot outgrow the budget under one held key either.
        let mut cache = CaptureCache::<u32>::default();
        cache.set_budget(1 + 8 * row("layer"));
        for n in 0..10_000u64 {
            cache.keep(1, 42, 1, [0.0; 4], Some((n, "layer")));
            assert!(cache.stand_ins() <= 8, "at {n}");
            rows_match(&cache);
        }
        let mut cache = CaptureCache::<u32>::default();
        cache.set_budget(1 + 8 * row("layer-0000"));
        for n in 0..10_000u64 {
            cache.keep(1, 42, 1, [0.0; 4], Some((7, &format!("layer-{:04}", n % 10_000))));
            assert!(cache.stand_ins() <= 8, "at {n}");
            rows_match(&cache);
        }
        // Rows that move through many held results are charged once, where they are, and
        // the results they left hold nothing for them.
        let mut cache = CaptureCache::<u32>::default();
        cache.set_budget(64 * 4 + 32 * row("a"));
        for key in 0..64u64 {
            for identity in 0..32u64 {
                cache.keep(key, 42, 4, [0.0; 4], Some((identity, "a")));
            }
            rows_match(&cache);
        }
        assert_eq!((cache.resident(), cache.stand_ins()), ((64, 64 * 4 + 32 * row("a")), 32));
        assert!(cache.entries.iter().all(|(key, entry)| (entry.rows == 0) == (*key != 63)));
        // One identity's newer result takes the row; evicting the older result leaves it.
        let mut cache = CaptureCache::<u32>::default();
        cache.set_budget(2 + 2 * row("a"));
        cache.keep(1, 10, 1, [0.0; 4], Some((7, "a")));
        cache.keep(2, 20, 1, [0.0; 4], Some((7, "a")));
        assert_eq!(cache.stand_ins(), 1);
        assert_eq!(cache.resident(), (2, 2 + row("a")), "the moved row is charged once");
        cache.keep(3, 30, 1 + row("a"), [0.0; 4], None);
        assert_eq!(cache.get(1), None, "the oldest result left");
        assert_eq!(cache.stand_in(7, "a").map(|(v, _)| v), Some(20));
        rows_match(&cache);
        // The same key again keeps its rows; two composites may share one result.
        let mut cache = CaptureCache::<u32>::default();
        cache.set_budget(1 + 2 * row("a"));
        cache.keep(2, 20, 1, [0.0; 4], Some((7, "a")));
        cache.keep(2, 20, 1, [0.0; 4], Some((7, "b")));
        cache.keep(2, 20, 1, [0.0; 4], Some((7, "a")));
        assert_eq!((cache.resident(), cache.stand_ins()), ((1, 1 + 2 * row("a")), 2));
        rows_match(&cache);
        // A result past the whole budget is not kept and takes no row.
        cache.keep(4, 40, 2 + 2 * row("a"), [0.0; 4], Some((8, "c")));
        assert_eq!(cache.stand_in(8, "c"), None);
        assert_eq!(cache.stand_ins(), 2);
        // Lowering the budget and clearing take the rows with the entries.
        cache.set_budget(0);
        assert_eq!((cache.resident(), cache.stand_ins()), ((0, 0), 0));
        cache.set_budget(1 + row("d"));
        cache.keep(5, 50, 1, [0.0; 4], Some((9, "d")));
        assert_eq!(cache.stand_ins(), 1);
        cache.clear();
        assert_eq!((cache.resident(), cache.stand_ins()), ((0, 0), 0));
    }

    #[test]
    fn a_region_covers_what_lies_inside_it_and_nothing_past_it() {
        let r = [10.0, 20.0, 100.0, 50.0];
        assert!(covers(r, [10.0, 20.0, 100.0, 50.0]));
        assert!(covers(r, [30.0, 30.0, 10.0, 10.0]));
        assert!(covers(r, [10.0 - 1e-9, 20.0, 100.0, 50.0]), "within rounding");
        assert!(!covers(r, [9.0, 20.0, 10.0, 10.0]));
        assert!(!covers(r, [100.0, 60.0, 20.0, 5.0]));
        assert!(covers(r, [500.0, 500.0, 0.0, 0.0]), "nothing shown is covered");
    }
}

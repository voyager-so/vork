//! T9 (perf): content-addressed GPU buffer cache for vector-text node-LOCAL
//! arenas.
//!
//! Background. During a drag in a text-heavy scene, every camera frame misses
//! the prepared-frame packet (the plan `Arc` identity changes each pointermove),
//! so [`crate::text_draw::TextDraw::prepare`] re-runs and re-uploads the big
//! node-local text GPU arenas (`verts`, `indices`, per-node curve storage) EVERY
//! frame — even though a pure move/transform only changes each node's model
//! matrix (which lives in the SMALL per-batch `uniforms` arena). The node-local
//! geometry is byte-identical across a transform-only edit, so re-uploading it is
//! pure waste (~1.5 MB/frame), and it shows up as intermittent multi-hundred-ms
//! frame spikes.
//!
//! This cache content-addresses those node-local buffers so a re-prepare reuses
//! the already-uploaded GPU buffer instead of re-creating + re-uploading it. The
//! small `uniforms` arena (model matrices) is ALWAYS rebuilt/uploaded, so a
//! transform-only drag uploads only the tiny per-frame matrices.
//!
//! It mirrors the residency [`crate::residency::PlanCaches`] tess/text LRU
//! conventions: a byte budget, LRU eviction that is NEVER correctness-bearing (a
//! miss just rebuilds + re-uploads identical bytes), a per-`prepare` frame tick,
//! and `*_bytes` accounting plus hit/miss/eviction counters surfaced in the
//! residency stats JSON (`vec_arena_*`, following the `tess_*` / `text_*`
//! conventions). It is generic over the stored value so the accounting/LRU logic
//! is unit-testable WITHOUT a GPU; production stores `Arc<wgpu::Buffer>` handles,
//! where a hit is an `Arc` pointer-bump (never a byte copy).

use std::collections::HashMap;

/// GPU-buffer byte budget for the vector-text arena cache (T9). The cached
/// buffers are the node-LOCAL text geometry (`verts` + `indices` arenas and the
/// per-node curve-point storage) — byte-identical across a transform-only drag,
/// so during a drag the working set is a SINGLE scene's arenas, held once. Sized
/// to retain a dense deck's arenas (the ~1.5 MB verts/indices pair plus a few MB
/// of curve storage) with comfortable headroom and no thrash, while staying
/// bounded for the coarse wasm memory guard. Like the residency budgets, eviction
/// is purely a memory bound — an evicted entry just rebuilds + re-uploads
/// byte-identical data on the next miss.
pub(crate) const VEC_ARENA_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// Snapshot of the cache counters, flat-merged into the residency stats JSON as
/// `vec_arena_*` keys (mirrors the `tess_*` / `text_*` counter conventions).
#[derive(Clone, Copy, Default)]
pub struct VecArenaStats {
    pub entries: u64,
    pub bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

struct Entry<V> {
    value: V,
    /// Source-byte length the buffer was built from. A hit requires BOTH the
    /// content key AND this length to match — a cheap, strong guard so a false
    /// hit would require a full key collision at an identical length.
    content_len: usize,
    /// GPU-buffer byte size charged against the budget.
    bytes: usize,
    /// Last-touched tick (LRU stamp); the smallest is least-recently-used.
    tick: u64,
}

/// Content-addressed LRU cache of reusable GPU buffers, keyed by a 64-bit hash
/// of the source bytes (+ a role discriminator folded in by the caller, so a
/// VERTEX buffer can never be served for an INDEX/STORAGE slot).
pub(crate) struct VecArenaCache<V> {
    map: HashMap<u64, Entry<V>>,
    bytes: usize,
    budget_bytes: usize,
    tick: u64,
    enabled: bool,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl<V: Clone> VecArenaCache<V> {
    pub(crate) fn new(budget_bytes: usize) -> Self {
        Self {
            map: HashMap::new(),
            bytes: 0,
            budget_bytes,
            tick: 0,
            enabled: true,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable/disable the cache (kill switch). Disabling drops all entries so a
    /// re-enable starts clean — mirrors `PlanCaches::set_incremental_enabled`.
    /// Cumulative hit/miss/eviction counters survive a toggle (telemetry).
    pub(crate) fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
        if !on {
            self.map.clear();
            self.bytes = 0;
        }
    }

    /// Drop all cached buffers (releasing their `Arc<wgpu::Buffer>` GPU residency)
    /// but keep the cache ENABLED and its cumulative counters. Used by the engine's
    /// retained-GPU teardown (`clear_retained_gpu`: memory-guard ALARM /
    /// `set_retained(false)`) so the recovery path actually frees this cache's
    /// residency; it repopulates lazily on the next `prepare` (a miss just
    /// re-uploads byte-identical data).
    pub(crate) fn clear(&mut self) {
        self.map.clear();
        self.bytes = 0;
    }

    /// Advance the LRU clock — call once per `prepare` (the cache's render sweep).
    pub(crate) fn begin_frame(&mut self) {
        self.tick = self.tick.wrapping_add(1);
    }

    /// Look up a content-addressed buffer. A hit requires the key AND the source
    /// length to match; it promotes recency and returns a clone (an `Arc`
    /// pointer-bump for the production payload). Always a miss when disabled.
    pub(crate) fn get(&mut self, key: u64, content_len: usize) -> Option<V> {
        if !self.enabled {
            return None;
        }
        match self.map.get_mut(&key) {
            Some(e) if e.content_len == content_len => {
                e.tick = self.tick;
                self.hits += 1;
                Some(e.value.clone())
            }
            _ => {
                self.misses += 1;
                None
            }
        }
    }

    /// Insert (or replace) a content-addressed buffer, then evict LRU entries
    /// until within budget. No-op when disabled.
    pub(crate) fn put(&mut self, key: u64, value: V, content_len: usize, bytes: usize) {
        if !self.enabled {
            return;
        }
        if let Some(old) = self.map.insert(
            key,
            Entry {
                value,
                content_len,
                bytes,
                tick: self.tick,
            },
        ) {
            self.bytes = self.bytes.saturating_sub(old.bytes);
        }
        self.bytes += bytes;
        if self.bytes > self.budget_bytes {
            self.evict_oldest();
        }
    }

    /// Evict least-recently-used entries until at/under budget. Never empties
    /// below one entry — the just-inserted buffer must remain servable even if it
    /// alone exceeds the budget. The just-inserted entry carries the current
    /// (highest) tick, so it is the last possible victim, guarded by that floor.
    ///
    /// Iterative min-by-tick — no allocation or sort: only a handful of entries
    /// are ever over budget at once, and eviction is off the hot path during a
    /// transform drag (the arena is stable → all hits, no `put`s). It runs at all
    /// only when content actually changes the working set past the budget.
    fn evict_oldest(&mut self) {
        while self.bytes > self.budget_bytes && self.map.len() > 1 {
            let Some(oldest) = self.map.iter().min_by_key(|(_, e)| e.tick).map(|(&k, _)| k) else {
                break;
            };
            if let Some(e) = self.map.remove(&oldest) {
                self.bytes = self.bytes.saturating_sub(e.bytes);
                self.evictions += 1;
            }
        }
    }

    pub(crate) fn stats(&self) -> VecArenaStats {
        VecArenaStats {
            entries: self.map.len() as u64,
            bytes: self.bytes as u64,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
        }
    }
}

/// 64-bit content key over `data`, with a `role` discriminator and the source
/// length folded in so the same bytes used as VERTEX vs INDEX vs STORAGE never
/// collide to one entry (the buffers carry different `wgpu` usage flags). Combined
/// with the `content_len` guard in [`VecArenaCache::get`], a false hit needs a
/// full 64-bit collision at an identical length — negligible for a per-session
/// cache of at most a few thousand entries (birthday bound ~1e-12), and the GPU
/// `mutation_replay` parity gate byte-compares the result as the final backstop.
///
/// `FxHasher` is the engine's standard non-crypto hash (as in the tess/page
/// caches); a single pass keeps the per-frame hash of the ~1.5 MB arenas cheap —
/// it replaced a two-pass 128-bit key whose lanes (linear/multiplicative FxHash
/// with different prefixes) were not actually independent, so it cost 2× the
/// hashing for no real added entropy.
pub(crate) fn content_key(role: u8, data: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = rustc_hash::FxHasher::default();
    h.write_u8(role);
    h.write_u64(data.len() as u64);
    h.write(data);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_key_distinguishes_role_and_content() {
        let data = [1u8, 2, 3, 4];
        // Same bytes, different role → different key (a VERTEX buffer can never
        // be served for an INDEX/STORAGE slot).
        assert_ne!(content_key(0, &data), content_key(1, &data));
        // Different bytes, same role → different key.
        assert_ne!(content_key(0, &data), content_key(0, &[1u8, 2, 3, 5]));
        // Stable for identical (role, bytes) — the whole point of the cache.
        assert_eq!(content_key(2, &data), content_key(2, &data));
    }

    #[test]
    fn hit_requires_key_and_length_match() {
        let mut c: VecArenaCache<u32> = VecArenaCache::new(1024);
        let k = content_key(0, &[0u8; 8]);
        c.put(k, 7, 8, 64);
        // Key + length match → hit, returns the stored value, recorded.
        assert_eq!(c.get(k, 8), Some(7));
        // Same key, different source length → MISS (the length guard rejects a
        // would-be key collision at a different length).
        assert_eq!(c.get(k, 9), None);
        // Unknown key → miss.
        assert_eq!(c.get(content_key(0, &[1u8; 8]), 8), None);
        let s = c.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 2);
        assert_eq!(s.entries, 1);
    }

    #[test]
    fn byte_accounting_exact_across_insert_and_replace() {
        let mut c: VecArenaCache<u32> = VecArenaCache::new(1_000_000);
        let k1 = content_key(0, b"a");
        let k2 = content_key(0, b"b");
        c.put(k1, 1, 1, 100);
        c.put(k2, 2, 1, 250);
        assert_eq!(c.stats().bytes, 350);
        assert_eq!(c.stats().entries, 2);
        // Replacing a key subtracts the OLD size and adds the NEW one exactly,
        // both directions.
        c.put(k1, 1, 1, 40); // 100 → 40
        assert_eq!(c.stats().bytes, 290);
        c.put(k2, 2, 1, 1000); // 250 → 1000
        assert_eq!(c.stats().bytes, 1040);
        assert_eq!(c.stats().entries, 2, "replacement keeps entry count");
        assert_eq!(c.stats().evictions, 0, "within budget never evicts");
    }

    #[test]
    fn lru_evicts_oldest_over_budget() {
        // Budget holds two 100-byte entries; a third evicts the LRU and keeps
        // total bytes within budget. Recency is a per-`prepare` frame tick (like
        // residency), so each operation advances `begin_frame` to get distinct
        // ticks — within one frame all entries tie (acceptable: eviction is never
        // correctness-bearing).
        let mut c: VecArenaCache<u32> = VecArenaCache::new(200);
        let ka = content_key(0, b"a");
        let kb = content_key(0, b"b");
        let kc = content_key(0, b"c");
        c.begin_frame();
        c.put(ka, 1, 1, 100);
        c.begin_frame();
        c.put(kb, 2, 1, 100);
        // A later frame touches `a`, so `b` (untouched) becomes the LRU victim.
        c.begin_frame();
        assert_eq!(c.get(ka, 1), Some(1));
        c.begin_frame();
        c.put(kc, 3, 1, 100);
        assert!(c.stats().bytes <= 200, "stays within budget after eviction");
        assert!(c.stats().evictions >= 1, "over-budget put must evict");
        assert_eq!(c.get(kb, 1), None, "the LRU entry was evicted");
        assert_eq!(c.get(ka, 1), Some(1), "the touched entry survived");
        assert_eq!(c.get(kc, 1), Some(3), "the just-inserted entry survived");
    }

    #[test]
    fn keeps_one_entry_larger_than_budget() {
        // A single entry over budget must remain servable (eviction never empties
        // below one entry) — a miss would otherwise be unable to serve the buffer
        // it just built for this frame.
        let mut c: VecArenaCache<u32> = VecArenaCache::new(10);
        c.begin_frame();
        c.put(content_key(0, b"huge"), 1, 4, 1000);
        assert_eq!(c.stats().entries, 1);
        assert_eq!(c.get(content_key(0, b"huge"), 4), Some(1));
        // A second (newer-frame) over-budget put evicts down to just the newest.
        c.begin_frame();
        c.put(content_key(0, b"also"), 2, 4, 1000);
        assert_eq!(c.stats().entries, 1);
        assert_eq!(c.get(content_key(0, b"also"), 4), Some(2));
    }

    #[test]
    fn disabled_cache_never_stores_or_hits() {
        let mut c: VecArenaCache<u32> = VecArenaCache::new(1_000_000);
        c.set_enabled(false);
        let k = content_key(0, b"x");
        c.put(k, 9, 1, 100);
        assert_eq!(c.stats().entries, 0, "disabled put is a no-op");
        assert_eq!(c.stats().bytes, 0);
        assert_eq!(c.get(k, 1), None, "disabled get is always a miss");
    }

    #[test]
    fn re_enable_starts_clean_but_keeps_counters() {
        let mut c: VecArenaCache<u32> = VecArenaCache::new(1_000_000);
        let k = content_key(0, b"x");
        c.put(k, 9, 1, 100);
        assert_eq!(c.get(k, 1), Some(9)); // hits += 1
        c.set_enabled(false); // drops entries + bytes
        assert_eq!(c.stats().entries, 0);
        assert_eq!(c.stats().bytes, 0);
        c.set_enabled(true);
        assert_eq!(c.get(k, 1), None, "entries were dropped on disable");
        // Cumulative counters survive the toggle.
        assert_eq!(c.stats().hits, 1);
        assert_eq!(c.stats().misses, 1);
    }
}

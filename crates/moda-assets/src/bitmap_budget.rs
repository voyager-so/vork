//! Retained external-bitmap budget ledger (the WEB-NEXTJS wasm-OOM family).
//!
//! The wasm editor lane parks every browser-decoded raster `ImageBitmap` in a
//! module registry so an engine swap or a pattern-texture-cache eviction can
//! re-upload without re-decoding. Before this module that registry grew
//! monotonically with distinct sources — the one unbounded memory consumer in
//! the wasm renderer (the wasm heap has the watermark/reinit ladder, the GPU
//! pattern cache has a byte-budget LRU, direct-present surfaces are capped).
//! On an image-heavy multi-page canvas the retained bitmaps alone can reach
//! GBs of browser memory, and the first symptom of the resulting tab-global
//! exhaustion is an allocator OOM abort inside an unrelated tiny wasm
//! allocation — an `unreachable` trap that poisons the instance.
//!
//! This module holds the ACCOUNTING and the eviction POLICY so both can be
//! unit-tested natively (moda-wasm compiles to an empty crate off wasm32). The
//! wasm side owns only the `ImageBitmap` handles and the actual `close()`
//! calls; it mirrors every install/remove/clear into this ledger and asks it
//! which keys to evict.
//!
//! ## Policy: working-set protection, not upload-state protection
//!
//! An entry is TOUCHED at install and on every GPU upload served from the
//! registry. An entry is PROTECTED from eviction while its last touch is at or
//! after the ledger's `protected_floor`; everything colder is evictable,
//! least-recently-touched first, until the total is at or under budget.
//!
//! The floor advances only on `plan_sweep_evictions` — which the host calls
//! ONLY after a presented paint sweep — and it advances with a one-sweep lag:
//! after sweep N the floor sits at sweep N-1's clock, so "protected" means
//! "touched since the sweep before last". That definition makes eviction safe
//! in BOTH GPU-cache modes:
//!
//! - **Non-retained mode** (the 1 GiB memory-alarm state, `?wasmRetained=0`):
//!   the pattern-texture cache is render-local, so every VISIBLE bitmap serves
//!   a registry upload on every painted frame — the visible working set is
//!   always inside the floor and can never be evicted. (Evicting it would
//!   blank the image on the very next frame and heal-churn forever; before
//!   this ledger the planner could do exactly that under the memory alarm.)
//!   If the visible working set alone exceeds the budget, the ledger stays
//!   over budget rather than flicker — the budget is soft by design.
//! - **Retained mode**: a resident GPU texture serves draws without touching
//!   the registry, so a stably-drawn entry goes cold here and IS evictable —
//!   which is harmless: its texture keeps rendering until the GPU LRU drops
//!   it, and that later miss heals through the wanted-key drain.
//!
//! Never-uploaded entries (decoded for pages the camera renderer culled, so
//! they never serve an upload) have a second, FASTER rule — the one-paint
//! cold rule. The host marks the start of every painted sweep
//! (`begin_sweep`); an entry that existed before that mark and still has not
//! uploaded after the paint is provably cold (a fresh install has no resident
//! GPU texture, so a visible one must upload on its first bind) and becomes
//! evictable at that very sweep's enforcement. This is what bounds a STATIC
//! initial load: every feeder install batch ends in a full-repaint
//! `markDirty`, so one painted sweep after the last install is guaranteed by
//! the host — and one is all the cold rule needs. Without it, an idle 40-page
//! deck would sit over budget forever waiting for aging sweeps the host never
//! schedules. (Before this ledger, never-uploaded entries were protected for
//! the canvas LIFETIME, so the cap could not bound the common camera-cull
//! shape at all.)
//!
//! `plan_install_evictions` (the install-time growth check) plans with the
//! SAME floors but never advances them, so a burst of installs between paints
//! cannot erode the last painted working set's protection.
//!
//! ## Engagement: dormant until actual memory pressure
//!
//! The budget must not change behaviour on healthy sessions (owner decision,
//! PR #11617): an image-heavy deck on a machine with plenty of memory gains
//! nothing from evicting and re-decoding. The ledger therefore starts
//! DISENGAGED — it keeps accounting (installs, touches, removals, sweep
//! marks) but both planners return no evictions and the aging floor never
//! advances, so an unconstrained session's registry is byte-for-byte what it
//! was before the budget existed. The host engages it (`set_engaged`) only at
//! the wasm memory ALARM tier — the watermark ladder's established
//! intervention boundary, where this subsystem already sacrifices retained
//! caches — never on capacity/size hints (the warn watermark,
//! `navigator.deviceMemory`) or on recovered panics/traps, whose signatures
//! cannot prove memory pressure (see `wasmBitmapBudget.ts`,
//! `BitmapBudgetEngagement`). Engagement is one-way per session on the host
//! side; the ledger accepts `set_engaged(false)` only for teardown symmetry.

use std::collections::HashMap;

/// One entry's accounting as the ledger sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BitmapUsage {
    /// Decoded RGBA footprint in bytes (width × height × 4).
    bytes: u64,
    /// Monotonic touch stamp — bumped at install and on every GPU upload
    /// served from the registry.
    last_touch: u64,
    /// True once the entry has served a GPU upload. A never-uploaded entry
    /// that existed before a painted sweep began and still has not uploaded
    /// after it is PROVABLY cold: a fresh install has no resident texture, so
    /// a visible one must upload on its first bind — surviving a whole paint
    /// without an upload means the camera culled it (see the cold rule in
    /// `plan_evictions`).
    uploaded: bool,
}

/// Accounting + eviction policy for the retained-`ImageBitmap` registry. The
/// wasm side mirrors every registry mutation into this ledger and closes the
/// bitmaps the planners return.
#[derive(Debug, Default)]
pub struct BitmapBudgetLedger {
    entries: HashMap<String, BitmapUsage>,
    /// Sum of `entries[*].bytes`, maintained incrementally (O(1) under-budget
    /// early-out on the per-sweep call).
    total_bytes: u64,
    /// Monotonic touch clock; 0 is reserved so a floor of 0 protects
    /// everything (the pre-first-sweep state).
    tick: u64,
    /// Entries with `last_touch >= protected_floor` are the working set and
    /// are never evicted. Sits one sweep behind `last_sweep_tick` (see the
    /// module doc for why the lag matters).
    protected_floor: u64,
    /// The clock at the end of the previous `plan_sweep_evictions`; becomes
    /// the floor on the next one.
    last_sweep_tick: u64,
    /// The clock at the START of the current/most recent painted sweep (the
    /// host's `begin_sweep`, called just before it paints). Entries installed
    /// before this that never uploaded were culled by that very paint — the
    /// one-paint cold rule that lets a STATIC initial load be bounded on its
    /// single guaranteed post-install paint. 0 until the first painted sweep,
    /// which keeps the pre-first-paint batch fully protected.
    sweep_start_tick: u64,
    /// False until the host reports actual memory pressure. While disengaged
    /// the planners evict nothing and the floor stays frozen — accounting
    /// only, zero behaviour change for healthy sessions.
    engaged: bool,
}

impl BitmapBudgetLedger {
    pub fn new() -> Self {
        Self::default()
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Record an install (or re-install) of `key` at `bytes`. Touches the
    /// entry, so a fresh install is protected until a painted sweep proves it
    /// cold (never uploads across a whole paint) or it ages past the floor.
    pub fn record_install(&mut self, key: &str, bytes: u64) {
        let touch = self.next_tick();
        let usage = BitmapUsage {
            bytes,
            last_touch: touch,
            uploaded: false,
        };
        if let Some(old) = self.entries.insert(key.to_string(), usage) {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
        }
        self.total_bytes = self.total_bytes.saturating_add(bytes);
    }

    /// Record a GPU upload served from the registry for `key` (a no-op for a
    /// key the ledger does not hold — e.g. a mid-eviction race).
    pub fn record_upload(&mut self, key: &str) {
        let touch = self.next_tick();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_touch = touch;
            entry.uploaded = true;
        }
    }

    /// Mark the START of a painted sweep — the host calls this immediately
    /// before a sweep it is about to paint. The recorded tick is what makes
    /// the one-paint cold rule sound: only entries that already existed here
    /// can be judged "survived this paint without uploading".
    pub fn begin_sweep(&mut self) {
        self.sweep_start_tick = self.next_tick();
    }

    /// Drop `key`'s accounting (released key, failed re-install, eviction
    /// applied by the caller outside the planners).
    pub fn remove(&mut self, key: &str) {
        if let Some(old) = self.entries.remove(key) {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
        }
    }

    /// Reset all accounting (registry teardown). The floors AND engagement
    /// reset too: the next canvas starts like a fresh session — dormant until
    /// its own renderer observes pressure, everything protected until a
    /// painted sweep proves otherwise.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_bytes = 0;
        self.protected_floor = 0;
        self.last_sweep_tick = 0;
        self.sweep_start_tick = 0;
        self.engaged = false;
    }

    /// Retained total in bytes.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Post-paint-sweep eviction plan: keys to evict (least-recently-touched
    /// first, only entries colder than the protected floor) until the total is
    /// at or under `budget_bytes` — then advance the floor (with the one-sweep
    /// lag). The planned keys are REMOVED from the ledger; the caller must
    /// drop and close exactly these handles. Call this only after a sweep
    /// that actually painted, or idle frames would age the working set out
    /// from under the next paint.
    pub fn plan_sweep_evictions(&mut self, budget_bytes: u64) -> Vec<String> {
        if !self.engaged {
            return Vec::new();
        }
        let evicted = self.plan_evictions(budget_bytes);
        // One-sweep lag: protect everything touched since the PREVIOUS sweep,
        // so the paint that just completed — which ran before this call — is
        // always inside the floor (its touches are >= last_sweep_tick). The
        // boundary consumes a tick of its own so an entry touched at the last
        // pre-boundary tick sits strictly BELOW the boundary and ages normally.
        self.protected_floor = self.last_sweep_tick;
        self.last_sweep_tick = self.next_tick();
        evicted
    }

    /// Install-time growth check: same plan, same floor, but the floor does
    /// NOT advance — installs between paints must not erode the last painted
    /// working set's protection. The planned keys are removed from the ledger.
    pub fn plan_install_evictions(&mut self, budget_bytes: u64) -> Vec<String> {
        if !self.engaged {
            return Vec::new();
        }
        self.plan_evictions(budget_bytes)
    }

    /// Report actual memory pressure: the budget begins enforcing (with the
    /// usual two-sweep grace — the floor is wherever it was frozen).
    /// Disengaging again is supported for symmetry but the host latches
    /// one-way: pressure observed once holds for the session.
    pub fn set_engaged(&mut self, engaged: bool) {
        self.engaged = engaged;
    }

    fn plan_evictions(&mut self, budget_bytes: u64) -> Vec<String> {
        if self.total_bytes <= budget_bytes {
            return Vec::new();
        }
        // Two evictability rules, either suffices:
        //  - AGED: touched before the protected floor — not in the working set
        //    of the last two painted sweeps (uploaded entries age this way).
        //  - PROVABLY COLD: never uploaded AND installed before the most
        //    recent painted sweep began. A fresh install has no resident GPU
        //    texture, so a visible one must upload on its first bind; one that
        //    survived a whole paint without uploading was camera-culled. This
        //    is the rule that bounds a STATIC initial load on its single
        //    guaranteed post-install paint (every feeder install ends in a
        //    full-repaint markDirty), where the aged rule alone would need
        //    paints the host never schedules.
        let mut candidates: Vec<(&String, &BitmapUsage)> = self
            .entries
            .iter()
            .filter(|(_, e)| {
                e.last_touch < self.protected_floor
                    || (!e.uploaded && e.last_touch < self.sweep_start_tick)
            })
            .collect();
        candidates.sort_by_key(|(_, e)| e.last_touch);
        let mut running = self.total_bytes;
        let mut evict: Vec<String> = Vec::new();
        for (key, usage) in candidates {
            if running <= budget_bytes {
                break;
            }
            running = running.saturating_sub(usage.bytes);
            evict.push(key.clone());
        }
        for key in &evict {
            if let Some(old) = self.entries.remove(key) {
                self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
            }
        }
        evict
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1024 * 1024;
    const BUDGET: u64 = 384 * MB;

    /// An ENGAGED ledger — the state every eviction-behaviour test wants
    /// (engagement itself is decided host-side by `BitmapBudgetEngagement`
    /// and pinned by the vitest suite; dormancy is proven below).
    fn engaged_ledger() -> BitmapBudgetLedger {
        let mut ledger = BitmapBudgetLedger::new();
        ledger.set_engaged(true);
        ledger
    }

    /// Install `n` keys named `prefix0..n` at `bytes` each.
    fn install_batch(
        ledger: &mut BitmapBudgetLedger,
        prefix: &str,
        n: usize,
        bytes: u64,
    ) -> Vec<String> {
        (0..n)
            .map(|i| {
                let key = format!("{prefix}{i}");
                ledger.record_install(&key, bytes);
                key
            })
            .collect()
    }

    /// EXACTLY one painted sweep as the host drives it — the renderer/host
    /// protocol, in call order, with nothing fabricated:
    ///   1. `begin_sweep`            (render(): beginExternalBitmapSweep,
    ///                                just before repaintAllPages)
    ///   2. the paint's registry uploads (ImageBitmapUploader::upload for
    ///                                every bound-and-copied visible raster)
    ///   3. `plan_sweep_evictions`   (render(): post-paint enforcement,
    ///                                gated on sweepRenderedAny)
    /// Panics if any uploaded (visible) key is evicted by its own sweep —
    /// the blanked-image defect the working-set rules exist to prevent.
    fn painted_frame(
        ledger: &mut BitmapBudgetLedger,
        uploads: &[String],
        budget: u64,
    ) -> Vec<String> {
        ledger.begin_sweep();
        for key in uploads {
            ledger.record_upload(key);
        }
        let evicted = ledger.plan_sweep_evictions(budget);
        for key in uploads {
            assert!(
                !evicted.contains(key),
                "visible bitmap '{key}' evicted by the sweep that painted it"
            );
        }
        evicted
    }

    #[test]
    fn under_budget_evicts_nothing() {
        let mut ledger = engaged_ledger();
        let keys = install_batch(&mut ledger, "k", 2, 10 * MB);
        assert!(painted_frame(&mut ledger, &[], 64 * MB).is_empty());
        // Exactly at budget is under budget (and cold entries stay resident).
        assert!(painted_frame(&mut ledger, &[], 20 * MB).is_empty());
        assert_eq!(ledger.total_bytes(), 20 * MB);
        let _ = keys;
    }

    /// The second review's blocking scenario: a STATIC camera-culled initial
    /// load must be bounded WITHOUT extra sweeps the host never schedules.
    /// The host guarantees exactly one painted sweep after the last install
    /// (feeder `onInstalled` → `markDirty()` → `pendingFullRepaint`), and the
    /// one-paint cold rule makes that single sweep sufficient: everything
    /// installed before the paint began that did not upload during it was
    /// camera-culled and is reclaimable immediately.
    #[test]
    fn static_camera_cull_load_is_bounded_by_its_single_guaranteed_paint() {
        // 40-page deck, 736 MB installed: 3 visible images (60 MB) + 33
        // offscreen (660 MB) + one 16 MB straggler page.
        let mut ledger = engaged_ledger();
        let visible = install_batch(&mut ledger, "page1-", 3, 20 * MB);
        let offscreen = install_batch(&mut ledger, "offscreen", 33, 20 * MB);
        install_batch(&mut ledger, "straggler", 1, 16 * MB);

        // The single post-install paint: camera culls to page 1, so only the
        // visible three upload. Then the canvas goes idle forever.
        let evicted = painted_frame(&mut ledger, &visible, BUDGET);

        assert!(
            ledger.total_bytes() <= BUDGET,
            "static load still {} bytes over the cap after its one guaranteed paint",
            ledger.total_bytes() - BUDGET
        );
        // Only camera-culled entries were reclaimed, oldest first.
        assert!(!evicted.is_empty());
        assert!(evicted
            .iter()
            .all(|k| offscreen.contains(k) || k.starts_with("straggler")));
        for key in &visible {
            assert!(
                ledger.entries.contains_key(key),
                "visible '{key}' must stay resident"
            );
        }

        // Same shape, one oversized offscreen panorama: a single 1 GiB
        // never-uploaded bitmap must not survive the guaranteed paint either.
        let mut ledger = engaged_ledger();
        let visible = install_batch(&mut ledger, "vis", 1, 20 * MB);
        ledger.record_install("pano", 1024 * MB);
        let evicted = painted_frame(&mut ledger, &visible, BUDGET);
        assert_eq!(evicted, vec!["pano".to_string()]);
        assert_eq!(ledger.total_bytes(), 20 * MB);
    }

    /// Entries installed DURING a paint (after its begin mark) have not had a
    /// chance to upload — they are protected through that sweep and judged by
    /// the next one, which their own install's markDirty guarantees.
    #[test]
    fn a_mid_paint_install_is_judged_by_its_own_followup_paint() {
        let mut ledger = engaged_ledger();
        ledger.begin_sweep();
        let batch = install_batch(&mut ledger, "late", 3, 200 * MB);
        assert!(
            ledger.plan_sweep_evictions(BUDGET).is_empty(),
            "mid-paint installs evicted before any chance to render"
        );
        // Their install marked dirty, so the host paints again; page 1 of the
        // batch is visible and uploads, the rest were culled → reclaimed.
        let evicted = painted_frame(&mut ledger, &batch[..1].to_vec(), BUDGET);
        assert_eq!(evicted.len(), 2);
        assert!(evicted.iter().all(|k| k != &batch[0]));
    }

    /// Cross-layer regression for round-1 blocking finding 1: retained mode
    /// OFF makes the GPU pattern cache render-local, so every visible bitmap
    /// re-uploads from the registry on every painted frame. Sustained
    /// over-budget rendering must evict only cold entries, keep every visible
    /// raster painted, and settle without evict→heal→re-decode churn.
    #[test]
    fn non_retained_over_budget_keeps_visible_rasters_painted() {
        let mut ledger = engaged_ledger();
        // 20 visible (200 MB) + 30 cold offscreen (600 MB): far over budget.
        let visible = install_batch(&mut ledger, "vis", 20, 10 * MB);
        install_batch(&mut ledger, "cold", 30, 20 * MB);

        let mut total_evicted = 0usize;
        for _ in 0..50 {
            total_evicted += painted_frame(&mut ledger, &visible, BUDGET).len();
        }
        assert!(ledger.total_bytes() <= BUDGET);
        // Each cold bitmap was evicted at most once (no churn), and the
        // visible set — asserted painted every frame by `painted_frame` —
        // never needed a re-decode.
        assert!(
            total_evicted <= 30,
            "eviction churned: {total_evicted} evictions"
        );
        assert!(
            ledger.total_bytes() >= 200 * MB,
            "the visible 200 MB working set must stay fully resident"
        );
    }

    /// Round-1 finding 1's worst case: the visible working set ALONE exceeds
    /// the budget with retained mode off. The old planner ping-ponged visible
    /// bitmaps (evict A → transparent → heal A → evict B → …); the working-set
    /// rules must instead hold everything resident — over budget, soft cap —
    /// with zero evictions frame after frame.
    #[test]
    fn non_retained_visible_set_over_budget_never_flickers() {
        let mut ledger = engaged_ledger();
        let visible = install_batch(&mut ledger, "vis", 5, 100 * MB); // 500 MB visible
        for _ in 0..50 {
            let evicted = painted_frame(&mut ledger, &visible, BUDGET);
            assert!(
                evicted.is_empty(),
                "evicted {evicted:?} out of an all-visible working set"
            );
        }
        assert_eq!(ledger.total_bytes(), 500 * MB);
    }

    /// Uploaded-but-stale entries (visited pages in retained mode, where a
    /// resident texture serves draws without registry touches) age out via
    /// the two-sweep floor, least-recently-touched first, stopping at budget.
    #[test]
    fn stale_uploaded_entries_age_out_in_lru_order_and_eviction_stops_at_budget() {
        let mut ledger = engaged_ledger();
        let keys = install_batch(&mut ledger, "k", 3, 30 * MB);
        // Each key uploads once (its page was visited), k2 most recently.
        painted_frame(&mut ledger, &keys[0..1].to_vec(), u64::MAX);
        painted_frame(&mut ledger, &keys[1..2].to_vec(), u64::MAX);
        painted_frame(&mut ledger, &keys[2..3].to_vec(), u64::MAX);
        // Two more painted sweeps age k0/k1 past the floor; k2 stays inside.
        painted_frame(&mut ledger, &keys[2..3].to_vec(), u64::MAX);
        // 90 MB total, 40 MB budget: k0 (oldest) then k1; k2 survives.
        let evicted = painted_frame(&mut ledger, &keys[2..3].to_vec(), 40 * MB);
        assert_eq!(evicted, vec![keys[0].clone(), keys[1].clone()]);
        assert_eq!(ledger.total_bytes(), 30 * MB);

        // Stop as soon as under budget: one big stale entry alone suffices.
        let mut ledger = engaged_ledger();
        ledger.record_install("big", 50 * MB);
        ledger.record_install("small", 10 * MB);
        painted_frame(&mut ledger, &["big".into(), "small".into()], u64::MAX);
        painted_frame(&mut ledger, &[], u64::MAX);
        painted_frame(&mut ledger, &[], u64::MAX);
        assert_eq!(
            painted_frame(&mut ledger, &[], 15 * MB),
            vec!["big".to_string()]
        );
        assert_eq!(ledger.total_bytes(), 10 * MB);
    }

    /// Install-time enforcement (a heal re-install, or any batch between
    /// paints) advances no floor and spares both the painted working set and
    /// the fresh install — round-1's evict-A → heal-A → evict-B flicker loop
    /// is structurally impossible.
    #[test]
    fn install_evictions_between_paints_spare_the_painted_working_set() {
        let mut ledger = engaged_ledger();
        let visible = install_batch(&mut ledger, "vis", 2, 150 * MB);
        let cold = install_batch(&mut ledger, "cold", 4, 100 * MB);
        painted_frame(&mut ledger, &visible, u64::MAX);

        // Between paints: a 300 MB install lands (e.g. a healed key's batch).
        ledger.record_install("healed", 300 * MB);
        let evicted = ledger.plan_install_evictions(BUDGET);
        for key in &visible {
            assert!(
                !evicted.contains(key),
                "install-time eviction took painted '{key}'"
            );
        }
        assert!(
            !evicted.contains(&"healed".to_string()),
            "install-time eviction took the fresh install"
        );
        assert!(evicted.iter().all(|k| cold.contains(k)));
        // The visible set still re-uploads fine on the next non-retained frame.
        painted_frame(&mut ledger, &visible, BUDGET);
    }

    /// Owner requirement (PR #11617): the budget must NOT change behaviour on
    /// healthy sessions. A default (disengaged) ledger driven through the
    /// exact host protocol — sweep marks included — over an over-budget
    /// image-heavy deck must evict NOTHING in either GPU-cache shape, so no
    /// key can ever miss and no re-decode can ever be triggered. (What makes
    /// a session engage at all is decided host-side by
    /// `BitmapBudgetEngagement`, tested through that production seam in
    /// tests/vitest/renderers/wasmBitmapBudget.test.ts.)
    #[test]
    fn healthy_session_over_budget_deck_is_completely_untouched() {
        let mut ledger = BitmapBudgetLedger::new(); // deliberately NOT engaged
        let visible = install_batch(&mut ledger, "vis", 5, 50 * MB);
        install_batch(&mut ledger, "offscreen", 40, 20 * MB); // 1050 MB total
        for frame in 0..50 {
            // Alternate cache shapes: non-retained frames re-upload every
            // visible key; retained frames serve draws from resident textures
            // (no registry touch). Neither may evict on a healthy session.
            let uploads: &[String] = if frame % 2 == 0 { &visible } else { &[] };
            let evicted = painted_frame(&mut ledger, uploads, BUDGET);
            assert!(
                evicted.is_empty(),
                "healthy session evicted at frame {frame}"
            );
        }
        assert!(ledger.plan_install_evictions(BUDGET).is_empty());
        assert_eq!(
            ledger.total_bytes(),
            1050 * MB,
            "healthy-session accounting drifted"
        );
    }

    /// Pressure arriving mid-session engages the budget: the very next
    /// painted sweep proves the never-uploaded cold set and bounds it, while
    /// the painted working set stays untouched.
    #[test]
    fn mid_session_engagement_bounds_the_cold_set_on_the_next_paint() {
        let mut ledger = BitmapBudgetLedger::new();
        let visible = install_batch(&mut ledger, "vis", 5, 10 * MB);
        let cold = install_batch(&mut ledger, "cold", 30, 20 * MB); // 650 MB total
        for _ in 0..10 {
            assert!(painted_frame(&mut ledger, &visible, BUDGET).is_empty());
        }
        ledger.set_engaged(true); // the ALARM watermark fired
        let evicted = painted_frame(&mut ledger, &visible, BUDGET);
        assert!(!evicted.is_empty());
        assert!(evicted.iter().all(|k| cold.contains(k)));
        assert!(ledger.total_bytes() <= BUDGET);
    }

    #[test]
    fn remove_and_clear_keep_the_byte_total_honest() {
        let mut ledger = engaged_ledger();
        let keys = install_batch(&mut ledger, "k", 3, 10 * MB);
        ledger.remove(&keys[1]);
        ledger.remove("never-installed");
        assert_eq!(ledger.total_bytes(), 20 * MB);
        // Re-install replaces, not doubles.
        ledger.record_install(&keys[0], 25 * MB);
        assert_eq!(ledger.total_bytes(), 35 * MB);
        ledger.clear();
        assert_eq!(ledger.total_bytes(), 0);
        // Clear resets engagement AND the sweep marks: the next canvas's
        // over-budget batch is fully protected again until ITS first paint.
        ledger.set_engaged(true);
        install_batch(&mut ledger, "next", 3, 200 * MB);
        assert!(ledger.plan_install_evictions(BUDGET).is_empty());
    }
}

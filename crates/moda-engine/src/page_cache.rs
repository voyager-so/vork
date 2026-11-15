//! page_cache.rs — the resident scenes of recently shown Design pages (docs/roadmap/design-pages.md §4).
//!
//! The editor feeds the engine only the page on screen. Switching pages used to replace the resident scene
//! with the next page's (a full parse and hydrate) and rebuild its camera plan (tessellation and text through
//! the shared caches, then the whole plan): seconds at 20k nodes, and the same again on the way back. This
//! keeps the last few pages' resident scenes, each with its retained camera plan and scene memos
//! ([`ScenePlanState`]), parked under the page's key within a page count and a byte budget. Showing a parked
//! page swaps it back in: the host then patches whatever changed on it while it was away (its scene-patch lane
//! against the snapshot it parked with), and the first frame reuses the plan.
//!
//! Least recently parked goes first when the budget is exceeded. A page whose own footprint is over the byte
//! budget is not kept. Correctness never depends on a hit: a miss is the full feed it always was.

use crate::residency::{PlanCaches, ResidentScene, ScenePlanState};

/// A page that is not on screen: its resident scene and the plan state that goes with it.
struct ParkedScene {
    key: String,
    resident: ResidentScene,
    plan: ScenePlanState,
    /// The resident scene's footprint (the host measures it at hydrate).
    resident_bytes: usize,
    /// Resident scene + retained plan.
    bytes: usize,
}

/// Parked pages, least recently parked first.
pub struct PageCache {
    entries: Vec<ParkedScene>,
    max_pages: usize,
    max_bytes: usize,
    pub parks: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Parks refused because the page alone was over the byte budget.
    pub refused: u64,
}

/// Default budget: three pages, 384 MiB (the host sets its own from the memory policy).
pub const DEFAULT_MAX_PAGES: usize = 3;
pub const DEFAULT_MAX_BYTES: usize = 384 << 20;

impl Default for PageCache {
    fn default() -> Self {
        PageCache {
            entries: Vec::new(),
            max_pages: DEFAULT_MAX_PAGES,
            max_bytes: DEFAULT_MAX_BYTES,
            parks: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            refused: 0,
        }
    }
}

impl PageCache {
    /// Keep at most `max_pages` pages and `max_bytes` bytes parked (evicting now if over). `0` pages turns
    /// the cache off.
    pub fn set_budget(&mut self, max_pages: usize, max_bytes: usize) {
        self.max_pages = max_pages;
        self.max_bytes = max_bytes;
        self.evict_to_budget();
    }

    /// Park the resident scene under `key` with its plan state, leaving an empty resident scene and plan
    /// state behind. `resident_bytes` is the scene's footprint (the host measures it at hydrate). Returns
    /// whether it was kept (false: no scene, or over budget on its own — it is dropped either way).
    pub fn park(
        &mut self,
        key: &str,
        resident: &mut ResidentScene,
        caches: &mut PlanCaches,
        resident_bytes: usize,
    ) -> bool {
        self.remove(key);
        let scene = resident.swap_scene(ResidentScene::empty());
        let plan = caches.take_scene_state();
        if scene.scene().is_none() || self.max_pages == 0 {
            return false;
        }
        let bytes = resident_bytes + plan.plan_bytes();
        if bytes > self.max_bytes {
            self.refused += 1;
            return false;
        }
        self.parks += 1;
        self.entries.push(ParkedScene {
            key: key.to_string(),
            resident: scene,
            plan,
            resident_bytes,
            bytes,
        });
        self.evict_to_budget();
        self.entries.iter().any(|e| e.key == key)
    }

    /// Bring back the page parked under `key`: it becomes the resident scene with its plan state (whatever
    /// was resident is dropped; the host parks it first when it wants it). False when nothing is parked
    /// under `key` (the host full-feeds).
    pub fn unpark(&mut self, key: &str, resident: &mut ResidentScene, caches: &mut PlanCaches) -> bool {
        match self.take(key) {
            Some(parked) => {
                Self::install(parked, resident, caches);
                true
            }
            None => false,
        }
    }

    /// A page switch in one step: park the resident scene under `park_key` (when given) and make the page
    /// parked under `show_key` resident (when given and parked). The page being shown is taken out before
    /// the outgoing one is parked, so the budget never evicts it. Returns (parked, shown, the shown scene's
    /// resident bytes). With nothing shown, nothing is resident afterwards (the host full-feeds).
    pub fn switch(
        &mut self,
        park_key: Option<&str>,
        show_key: Option<&str>,
        resident: &mut ResidentScene,
        caches: &mut PlanCaches,
        resident_bytes: usize,
    ) -> (bool, bool, usize) {
        let incoming = show_key.and_then(|k| self.take(k));
        let parked = match park_key {
            Some(k) => self.park(k, resident, caches, resident_bytes),
            None => {
                drop(resident.swap_scene(ResidentScene::empty()));
                drop(caches.take_scene_state());
                false
            }
        };
        match incoming {
            Some(p) => {
                let bytes = p.resident_bytes;
                Self::install(p, resident, caches);
                (parked, true, bytes)
            }
            None => (parked, false, 0),
        }
    }

    fn take(&mut self, key: &str) -> Option<ParkedScene> {
        match self.entries.iter().position(|e| e.key == key) {
            Some(i) => {
                self.hits += 1;
                Some(self.entries.remove(i))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    fn install(parked: ParkedScene, resident: &mut ResidentScene, caches: &mut PlanCaches) {
        drop(resident.swap_scene(parked.resident));
        drop(caches.take_scene_state());
        caches.restore_scene_state(parked.plan);
    }

    /// Drop the page parked under `key`, if any.
    pub fn remove(&mut self, key: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.key != key);
        before != self.entries.len()
    }

    /// Drop every parked page (memory pressure, a cache reset, a render-option change).
    pub fn clear(&mut self) {
        self.evictions += self.entries.len() as u64;
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes parked now.
    pub fn bytes(&self) -> usize {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    /// Parked keys, least recently parked first.
    pub fn keys(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.key.clone()).collect()
    }

    fn evict_to_budget(&mut self) {
        while !self.entries.is_empty() && (self.entries.len() > self.max_pages || self.bytes() > self.max_bytes) {
            self.entries.remove(0);
            self.evictions += 1;
        }
    }

    pub fn stats_json(&self) -> String {
        moda_scene::value::json!({
            "pages": self.len(),
            "bytes": self.bytes(),
            "keys": self.keys(),
            "max_pages": self.max_pages,
            "max_bytes": self.max_bytes,
            "parks": self.parks,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "refused": self.refused,
        })
        .to_string()
    }
}

#[cfg(test)]
#[path = "page_cache_tests.rs"]
mod tests;

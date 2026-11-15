//! One export job's plan caches (Moda's export study, idea S1).
//!
//! A multi-page export (a deck's PDF or PNGs, a carousel, one ad in many sizes, a print PDF) renders its pages in
//! one process (`moda-render batch`, the hosts' `moda-render slices` server). Without caches every page shaped and
//! laid out its text from scratch, although the pages of one job repeat much of it: the same bullet style, the
//! same running text, the same footer. [`ExportJobCaches`] is one [`PlanCaches`] for the job:
//!
//! - **Keyed by content, not by node.** The node hashes it keys on leave out the node's `id` and `name`
//!   ([`residency::content_hashes_for_scene`]), and a text batch is looked up by its content key alone, so a text
//!   layer on page 7 that lays out exactly like one on page 2 (same words, style, box and scale) reuses page 2's
//!   layout. A layer whose layout differs in anything (another font size, width, colour, font) has another key
//!   and is laid out afresh: there is no invalidation to get wrong.
//! - **Two generations.** After each page, entries that page didn't use are dropped (two-generation
//!   eviction), so the caches hold one page's working set however many pages the job has.
//! - **One job.** The caches live as long as the job's process and see one document (or its per-request
//!   overrides, each with its own hashes): nothing carries over to another export.
//! - **Same pixels, same report.** A plan built through these caches is the plan an uncached build makes; a text
//!   batch served from them replays the resource events (font substitutions, fallbacks, the vector lane's served
//!   count) its build recorded, under the layer that drew it. The page plan cache (R5) is not used: a page plan
//!   holds no events to replay, and an export job renders each page once.
//! - **Plain shapes** tessellate through the process-wide command-keyed cache as without caches
//!   (`tess::tessellate_combined_cached`), which already shares geometry across the pages of a process.

use crate::frame::{PageRegionPx, PlanWarnings, RenderOptions};
use crate::residency::{self, HashMap, NodeHashes, PlanCaches};
use crate::seams::CoveragePlan;
use anyhow::Result;

/// The per-node hashes an [`ExportJobCaches`] keys on, for one scene.
pub type JobHashes = HashMap<String, NodeHashes>;

/// One export job's plan caches. See the module docs.
pub struct ExportJobCaches {
    caches: PlanCaches,
    pages: u64,
    peak_bytes: usize,
}

impl Default for ExportJobCaches {
    fn default() -> Self {
        Self::new()
    }
}

/// What the job's caches did (the CLI's `--timings`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JobCacheStats {
    pub pages: u64,
    pub text_hits: u64,
    pub text_misses: u64,
    pub tess_hits: u64,
    pub tess_misses: u64,
    /// Bytes held after the last page (one page's working set).
    pub bytes: usize,
    /// The most held after any page.
    pub peak_bytes: usize,
}

impl ExportJobCaches {
    pub fn new() -> Self {
        let mut caches = PlanCaches::default();
        caches.enable_job_sharing();
        ExportJobCaches {
            caches,
            pages: 0,
            peak_bytes: 0,
        }
    }

    /// The hashes to plan `scene`'s pages with (compute once per scene and pass them to every page of it).
    pub fn hashes(scene: &moda_scene::Scene) -> JobHashes {
        residency::content_hashes_for_scene(scene)
    }

    /// Plan one page (or, with `region`, the region of its raster a tiled or cropped render draws), then drop
    /// the entries the page didn't use.
    pub fn plan_page(
        &mut self,
        scene: &moda_scene::Scene,
        hashes: &JobHashes,
        page_index: usize,
        opts: &RenderOptions,
        region: Option<PageRegionPx>,
    ) -> Result<(CoveragePlan, PlanWarnings)> {
        self.caches.begin_frame();
        let planned = match region {
            None => crate::frame::build_coverage_plan_cached(
                scene,
                page_index,
                opts,
                Some((&mut self.caches, hashes)),
            ),
            Some(r) => crate::frame::build_coverage_plan_region_cached(
                scene,
                page_index,
                opts,
                r,
                Some((&mut self.caches, hashes)),
            ),
        };
        self.caches.end_job_page();
        self.pages += 1;
        self.peak_bytes = self.peak_bytes.max(self.caches.job_bytes());
        planned
    }

    /// [`crate::render_page`] through the job's caches.
    pub fn render_page(
        &mut self,
        engine: &crate::Engine,
        scene: &moda_scene::Scene,
        hashes: &JobHashes,
        page_index: usize,
        opts: &RenderOptions,
    ) -> Result<crate::RenderedPage> {
        let (plan, warnings) = self.plan_page(scene, hashes, page_index, opts, None)?;
        rendered(engine, page_index, &plan, warnings)
    }

    /// [`crate::render_page_region`] through the job's caches.
    pub fn render_page_region(
        &mut self,
        engine: &crate::Engine,
        scene: &moda_scene::Scene,
        hashes: &JobHashes,
        page_index: usize,
        opts: &RenderOptions,
        region: PageRegionPx,
    ) -> Result<crate::RenderedPage> {
        let (plan, warnings) = self.plan_page(scene, hashes, page_index, opts, Some(region))?;
        rendered(engine, page_index, &plan, warnings)
    }

    pub fn stats(&self) -> JobCacheStats {
        JobCacheStats {
            pages: self.pages,
            text_hits: self.caches.text_hits,
            text_misses: self.caches.text_misses,
            tess_hits: self.caches.tess_hits,
            tess_misses: self.caches.tess_misses,
            bytes: self.caches.job_bytes(),
            peak_bytes: self.peak_bytes,
        }
    }

    pub fn stats_json(&self) -> moda_scene::value::Value {
        let s = self.stats();
        moda_scene::value::json!({
            "pages": s.pages,
            "text_hits": s.text_hits,
            "text_misses": s.text_misses,
            "tess_hits": s.tess_hits,
            "tess_misses": s.tess_misses,
            "bytes": s.bytes,
            "peak_bytes": s.peak_bytes,
        })
    }
}

fn rendered(
    engine: &crate::Engine,
    page_index: usize,
    plan: &CoveragePlan,
    warnings: PlanWarnings,
) -> Result<crate::RenderedPage> {
    let rgba = engine.render_plan(plan)?;
    Ok(crate::RenderedPage {
        page_index,
        page_id: plan.page_id.clone(),
        width: plan.width_px,
        height: plan.height_px,
        rgba,
        skipped: warnings.skipped,
        dropped: warnings.dropped,
    })
}

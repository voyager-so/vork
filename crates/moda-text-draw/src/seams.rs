//! seams.rs — the text half of moda-engine's cross-track interface contract (`TextQuadBatch`, `DegradedQuad`),
//! excerpted; see PROVENANCE.md.

pub use moda_text::ops::{TextOp, TextOpPipeline};

/// The draws of one richtext/shapeText/connectorText item: the vector glyph ops (`vector`), or the degraded
/// raster (`degraded`), plus the decoration and emoji ops (`ops`). `--dump-text-quads` serializes them as JSON;
/// `moda_engine::text_draw` renders them.
#[derive(Clone)]
pub struct TextQuadBatch {
    pub node_id: String,
    /// Render z (fractional: node z, or shape z + SHAPE_TEXT_Z_OFFSET).
    pub z: f64,
    /// Column-major model matrix (same construction as shapes).
    pub model_matrix: [f32; 16],
    pub page_clip_bounds: [f32; 4],
    /// The ordered non-glyph op stream (emoji quads, decorations). Draw order == blend order; every op shares
    /// the batch depth.
    pub ops: Vec<TextOp>,
    /// Emoji cluster strings (indexed by `TextOpPipeline::Emoji.emoji`).
    pub emojis: Vec<String>,
    /// M4d (additive, interface-change note in the SEAMS.md change log):
    /// Degraded-tier raster quad for fallback-class nodes rendered behind
    /// `--best-effort-text`. When set, `ops` is empty and the
    /// executor draws ONE textured quad via the extracted `text-emoji`
    /// image module (the production Canvas2D display pipeline,
    /// `richtext/shaders/richtext.wgsl.ts`, is not part of the extracted
    /// export surface — its edge-alpha-gamma compensation is a recorded
    /// Degraded-tier divergence inside the 0.90–0.97 band).
    pub degraded: Option<DegradedQuad>,
    /// The vector text engine ops. When set, the executor draws the ordered
    /// `VectorNodeOps` stream through the extracted vector pipelines
    /// (`vector-text-tess` / `vector-text-analytic` /
    /// `vector-text-analytic-additive` (glow, task #142) /
    /// `vector-text-analytic-gradient` / `<id>.vector-text-fill`).
    /// `fill_shader` doubles as the vector shader-fill instance (module id
    /// formatted `<id>.vector-text-fill` at draw time); `VectorNodeOps::gradient`
    /// carries the gradient uniforms (contiguous 176-byte binding 3).
    pub vector: Option<moda_text::vector::VectorNodeOps>,
    /// Additive: the node's `distort` warp as the local-px 3×3
    /// homography rows packed for the vector `DrawUniforms` distort rows —
    /// row-major `[h11, h12, h13, h21, h22, h23, h31, h32, h33]`. None ⇒
    /// identity rows with the enabled flag 0. Vector-engine draws only (the
    /// Degraded tier stays unwarped — the accepted fallback gap).
    pub distort: Option<[f32; 9]>,
    /// vork: what this batch draws, as one number: the content key its plan cache keeps it under (text hash,
    /// opacity, render scale, model scale band, font host generation; moda-engine `text_draw::text_cache_key`).
    /// Two batches with the same key and the same placement (z, model, clip) draw the same pixels, whichever
    /// allocation holds them, so a consumer that must know "the same text as before" names a batch by this and
    /// its placement, not by its address (a layer capture's content key, moda-engine
    /// `residency::fold_item_geometry`). `None`: built outside a plan cache; such a batch has only its address.
    ///
    /// It is a name for the fields above, not one more of them: two batches are equal, and print alike, whether
    /// or not either carries it (a plan built through a cache equals the same plan built without one).
    pub content_key: Option<u64>,
}

impl PartialEq for TextQuadBatch {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
            && self.z == other.z
            && self.model_matrix == other.model_matrix
            && self.page_clip_bounds == other.page_clip_bounds
            && self.ops == other.ops
            && self.emojis == other.emojis
            && self.degraded == other.degraded
            && self.vector == other.vector
            && self.distort == other.distort
    }
}

impl std::fmt::Debug for TextQuadBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextQuadBatch")
            .field("node_id", &self.node_id)
            .field("z", &self.z)
            .field("model_matrix", &self.model_matrix)
            .field("page_clip_bounds", &self.page_clip_bounds)
            .field("ops", &self.ops)
            .field("emojis", &self.emojis)
            .field("degraded", &self.degraded)
            .field("vector", &self.vector)
            .field("distort", &self.distort)
            .finish()
    }
}

/// Placement + raster for one Degraded-tier text quad (M4d). Mirrors the
/// Canvas2D display-quad math (`richtext/transforms.ts:77-117`): quad size
/// is the texture's logical size; `offset` is the alignment offset
/// ((node − texture) × {0,½,1} per detected horizontal align /
/// `verticalAlign`), node-local; `opacity` = node opacity × the raster's
/// `fill_alpha` (uniform-alpha quirk).
#[derive(Debug, Clone, PartialEq)]
pub struct DegradedQuad {
    pub raster: std::sync::Arc<moda_text::degraded::DegradedRaster>,
    pub offset: [f32; 2],
    pub opacity: f32,
}

impl TextQuadBatch {
    /// The local-space quads this batch draws, including glyph transforms, strokes and raster padding.
    pub fn local_bounds(&self) -> Option<[f64; 4]> {
        let mut bounds: Option<[f64; 4]> = None;
        let mut point = |x: f64, y: f64| {
            bounds = Some(match bounds {
                None => [x, y, x, y],
                Some(b) => [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)],
            });
        };
        if let Some(vector) = &self.vector {
            for op in &vector.ops {
                for xy in op.tris.chunks_exact(2) { point(xy[0].into(), xy[1].into()); }
                for glyph in &op.records {
                    if glyph.glyph_opacity <= 0. { continue; }
                    let [x0, y0, x1, y1] = glyph.em_box.map(f64::from);
                    let size = f64::from(glyph.font_size_px);
                    let cx = f64::from(glyph.pen_x) + (x0 + x1) * size * 0.5;
                    let cy = f64::from(glyph.baseline_y) + (y0 + y1) * size * 0.5;
                    let [a, c, b, d] = glyph.glyph_lin.map(f64::from);
                    for (x, y) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] {
                        let dx = (x - (x0 + x1) * 0.5) * size;
                        let dy = (y - (y0 + y1) * 0.5) * size;
                        point(cx + a * dx + c * dy + f64::from(glyph.glyph_off[0]),
                            cy + b * dx + d * dy + f64::from(glyph.glyph_off[1]));
                    }
                }
            }
            for emoji in &vector.emojis {
                for xy in emoji.corners.chunks_exact(2) { point(xy[0].into(), xy[1].into()); }
            }
        }
        for op in &self.ops {
            for xyuv in op.verts.chunks_exact(4) { point(xyuv[0].into(), xyuv[1].into()); }
        }
        if let Some(quad) = &self.degraded {
            let [x, y] = quad.offset.map(f64::from);
            point(x, y);
            point(x + quad.raster.logical_w, y + quad.raster.logical_h);
        }
        bounds
    }
}

//! tiled.rs — tiled page export: render a page whose full raster exceeds the
//! single-target envelope as a grid of within-envelope tiles, streaming the
//! stitched rows into a PNG encoder so the full uncompressed raster is never
//! materialized host-side.
//!
//! This is the native lane's answer to the retired browser exporter's
//! `utils/export/tiledRenderer.ts`: same grid walk (row-major, sequential —
//! correctness first; the pool's concurrency budget stays owned by the
//! caller), but the stitch is a row-band buffer feeding `png`'s stream writer
//! instead of a Canvas2D the size of the output (the browser path's OOM
//! source).
//!
//! Each tile is a [`frame::build_coverage_plan_region`] render: identical draw
//! lists to the full-page plan, drawn with the view-projection of the tile's
//! raster band through a viewport shifted to the tile origin
//! (`seams::RegionViewport`; a projection change, not a re-layout — see
//! `frame.rs`), so the stitched grid reproduces the single-target render.
//! The tiled-seam golden (`tests/tiled_seam.rs`) pins that equivalence. Pages with active backdrop /
//! layer-blur content refuse (their pixels sample the TARGET, so tile edges
//! would seam) — the region plan builder bails with
//! [`frame::REGION_BACKDROP_ERROR`] / [`frame::REGION_LAYER_BLUR_ERROR`].
//!
//! Memory: peak host allocation is ONE band — `tile_rows x out_w x 4` bytes,
//! where `tile_rows` adapts downward on very wide outputs so the band stays
//! under [`BAND_BYTES_BUDGET`] — plus the encoder's compression state. VRAM
//! peaks at one tile's attachments (tiles are within the msaa4 envelope by
//! construction).

use anyhow::{bail, Context, Result};
use moda_scene::Scene;

use crate::frame::{self, Engine, PageRegionPx, RenderOptions};

/// Default tile edge, in output pixels. 4096 puts a tile at 16.7 Mpx —
/// comfortably inside the msaa4 single-target envelope (~0.5 GiB transient)
/// while keeping the grid small (a 16384-wide row is 4 tiles).
pub const DEFAULT_TILE_PX: u32 = 4_096;

/// Ceiling for the stitch band (`tile_rows x out_w x 4`). 256 MiB caps host
/// memory on pathologically wide outputs by shrinking the band's row count
/// (never below [`MIN_BAND_ROWS`]); at the policy's tiled axis cap the band
/// stays well under it at the default tile size.
pub const BAND_BYTES_BUDGET: u64 = 256 * 1024 * 1024;

/// Floor for the adaptive band height — bands this shallow keep the tile count
/// sane even when [`BAND_BYTES_BUDGET`] binds on an extreme aspect ratio.
pub const MIN_BAND_ROWS: u32 = 256;

/// What a tiled render produced (the PNG bytes went to the caller's writer).
#[derive(Debug, Clone)]
pub struct TiledPageRender {
    pub page_index: usize,
    pub page_id: String,
    /// Full output raster size — identical to what a single-target render of
    /// the same `(page, scale)` would produce (`ceil(page_dim x scale)`).
    pub width: u32,
    pub height: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    /// Plan warnings, deduplicated across tiles (every tile plans the same
    /// display list, so warnings repeat per tile).
    pub skipped: Vec<String>,
    /// Fail-closed refusals, deduplicated the same way. See
    /// [`crate::RenderedPage::dropped`].
    pub dropped: Vec<String>,
}

/// Render one page as a tile grid, streaming stitched rows into `out` as PNG.
///
/// `opts.scale` drives the output size exactly like [`crate::render_page`];
/// `tile_px` is the tile edge (0 → [`DEFAULT_TILE_PX`]). The caller owns
/// bounding the total output size (the stills policy's tiled sanity cap) —
/// this function only bounds per-tile and per-band allocations.
pub fn render_page_tiled_to<W: std::io::Write>(
    engine: &Engine,
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    tile_px: u32,
    out: W,
) -> Result<TiledPageRender> {
    let page = scene.doc().pages.get(page_index).ok_or_else(|| {
        anyhow::anyhow!(
            "page index {page_index} out of range ({} pages)",
            scene.doc().pages.len()
        )
    })?;
    let page_id = page.id().to_string();
    let (out_w, out_h) = crate::export_scale::output_dims(page.width(), page.height(), opts.scale);

    let tile_px = if tile_px == 0 {
        DEFAULT_TILE_PX
    } else {
        tile_px
    };
    let max_axis = engine.gpu.device.limits().max_texture_dimension_2d;
    if tile_px > max_axis {
        bail!("tile edge {tile_px} exceeds the device texture limit {max_axis}");
    }
    // Adaptive band height: a band spans the FULL output width, so very wide
    // outputs shrink the band's rows to stay under the byte budget.
    let budget_rows = (BAND_BYTES_BUDGET / (out_w as u64 * 4)).min(tile_px as u64) as u32;
    let band_rows = budget_rows.max(MIN_BAND_ROWS).min(tile_px);
    let tiles_x = out_w.div_ceil(tile_px);
    let tiles_y = out_h.div_ceil(band_rows);

    let mut encoder = png::Encoder::new(out, out_w, out_h);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(crate::encode::png_compression());
    let mut writer = encoder.write_header().context("png header")?;
    let mut stream = writer.stream_writer().context("png stream writer")?;

    let mut band: Vec<u8> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    for ty in 0..tiles_y {
        let y = ty * band_rows;
        let h = band_rows.min(out_h - y);
        let band_len = (out_w as usize) * (h as usize) * 4;
        band.clear();
        band.resize(band_len, 0);
        for tx in 0..tiles_x {
            let x = tx * tile_px;
            let w = tile_px.min(out_w - x);
            let (band_x, band_w) = crate::seams::region_band(x, tile_px, out_w);
            let (band_y, band_h) = crate::seams::region_band(y, band_rows, out_h);
            let region = PageRegionPx {
                x,
                y,
                w,
                h,
                band_x,
                band_y,
                band_w,
                band_h,
            };
            let (plan, warnings) =
                frame::build_coverage_plan_region(scene, page_index, opts, region)
                    .with_context(|| format!("tile ({tx},{ty}) plan"))?;
            let rgba = engine
                .render_plan(&plan)
                .with_context(|| format!("tile ({tx},{ty}) render ({w}x{h} at {x},{y})"))?;
            debug_assert_eq!(rgba.len(), (w as usize) * (h as usize) * 4);
            for warning in warnings.skipped {
                if !skipped.contains(&warning) {
                    skipped.push(warning);
                }
            }
            for refusal in warnings.dropped {
                if !dropped.contains(&refusal) {
                    dropped.push(refusal);
                }
            }
            // Blit the tile's rows into the band at its x offset.
            let tile_row_bytes = (w as usize) * 4;
            let band_row_bytes = (out_w as usize) * 4;
            let x_bytes = (x as usize) * 4;
            for row in 0..h as usize {
                let src = &rgba[row * tile_row_bytes..(row + 1) * tile_row_bytes];
                let dst_start = row * band_row_bytes + x_bytes;
                band[dst_start..dst_start + tile_row_bytes].copy_from_slice(src);
            }
        }
        use std::io::Write as _;
        stream
            .write_all(&band)
            .with_context(|| format!("png stream write (band {ty})"))?;
    }
    stream.finish().context("png stream finish")?;

    Ok(TiledPageRender {
        page_index,
        page_id,
        width: out_w,
        height: out_h,
        tiles_x,
        tiles_y,
        skipped,
        dropped,
    })
}

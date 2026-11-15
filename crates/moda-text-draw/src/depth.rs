//! depth.rs — how render-z becomes depth.
//!
//! Larger z = closer to the camera = SMALLER depth; compare function `less`,
//! depth cleared to 1.0. Pages sit at 0.99999 (fill) / 0.99998 (stroke),
//! hardcoded in the page WGSL. Strokes add `STROKE_DEPTH_OFFSET` in the
//! vertex shader (the constant in every path module's WGSL).
//!
//! The 2D pass needs depth for order: opaque bodies write it and draw batched
//! by paint kind, not in z order, and every transparent draw tests against it.
//! So each render-z slot needs depths of its own, and the count of slots a
//! frame can order is what the depth attachment can tell apart.
//!
//! The attachment is `Depth24Plus`: 24-bit unorm or 32-bit float. Both keep
//! two depths in [0, 1) apart when they differ by two or more [`DEPTH_UNIT`]s
//! (2^-24), and an `f32` holds every multiple of it there exactly. One slot is
//! [`DEPTH_UNITS_PER_Z`] = 40 units. The nearest depths the engine sets apart
//! on purpose are a twentieth of a slot from each other (brush artwork starts
//! +0.05 above its body's fill): 2 units. The others are eighths (a stroke at
//! +5/8 against text at +1/2 and a capture's inner shadow at +3/4; an
//! arrowhead's stroke at +7/8 against the next slot): 5 units, each depth an
//! exact multiple of the unit from the CPU through the vertex shader's add.
//! Anything finer inside a slot (a paint stack's layers, a Transform effect's
//! copies, a brush's placed shapes) draws in painter's order in the
//! transparent phase and writes no depth.
//!
//! That gives [`MAX_RENDER_Z`] slots between [`SHAPE_DEPTH_MAX`] and
//! [`SHAPE_DEPTH_MIN`]. A plan that draws an item past them is refused ([`ensure_render_z`]);
//! nothing past the range is drawn at a clamped or clipped depth.
//!
//! Moda's WebGPU model (`WebGPUZIndexUtils.linearDepth`) spreads 50,000 slots
//! over the same band, 268 units a slot, and puts slot 56,250 and later behind
//! the near plane. The order of any two depths is the
//! same in both models; only the spacing differs.

/// The depth step every `Depth24Plus` attachment resolves: 2^-24.
pub const DEPTH_UNIT: f64 = 1.0 / 16_777_216.0;
/// Depth units one render-z slot takes.
pub const DEPTH_UNITS_PER_Z: f64 = 40.0;
/// Depth one render-z slot takes.
pub const DEPTH_PER_Z: f64 = DEPTH_UNITS_PER_Z * DEPTH_UNIT;

/// The nearest depth a shape takes.
pub const SHAPE_DEPTH_MIN: f64 = 0.1;
/// The depth of render-z 0: 0.9 as an `f32` holds it, a whole number of units.
pub const SHAPE_DEPTH_MAX: f64 = 0.9_f32 as f64;

/// The render-z slots depth can order: a plan's drawn items may reach at most
/// this slot (335,544).
pub const MAX_RENDER_Z: u64 = ((SHAPE_DEPTH_MAX - SHAPE_DEPTH_MIN) / DEPTH_PER_Z) as u64;

/// Depth the page-stroke quad writes (exports pass strokeWidth 0, unused).
pub const PAGE_STROKE_DEPTH: f64 = 0.99998;

/// The depth of render-z `z`: [`DEPTH_PER_Z`] nearer per slot from
/// [`SHAPE_DEPTH_MAX`]; a negative z sits at z 0. z may be fractional (text
/// +0.5, arrowheads +0.25, inner shadows +0.5, a layer's or a copy's share of
/// its node's slot).
pub fn linear_depth(z_index: f64) -> f64 {
    SHAPE_DEPTH_MAX - z_index.max(0.0) * DEPTH_PER_Z
}

/// How much nearer a stroke's vertices sit than their body's fill: 5/8 of a
/// slot, the ratio of Moda's `WebGPUZIndexUtils.getStrokeOffset` to its slot.
/// The path modules' WGSL carries the same value as `STROKE_DEPTH_OFFSET`
/// (`moda_shaders::manifest().pinned_constants`).
pub fn stroke_depth_offset() -> f64 {
    -0.625 * DEPTH_PER_Z
}

/// Refuses a plan whose drawn items reach render-z slot `drawn_end` (one past
/// the highest slot any of them sits in) when depth cannot order that many.
/// Slots the cull window skipped draw nothing and do not count.
pub fn ensure_render_z(drawn_end: u64) -> anyhow::Result<()> {
    anyhow::ensure!(
        drawn_end <= MAX_RENDER_Z,
        "this frame's 2D items reach draw-order slot {drawn_end} (one slot per ordered item) and the 2D depth \
         buffer orders at most {MAX_RENDER_Z} in one frame; draw fewer items in it"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_range_is_stated() {
        assert_eq!(MAX_RENDER_Z, 335_544);
        assert!(ensure_render_z(MAX_RENDER_Z).is_ok());
        let refusal = ensure_render_z(MAX_RENDER_Z + 1).unwrap_err().to_string();
        assert!(refusal.contains("335545") && refusal.contains("335544"), "{refusal}");
        // The last slot's nearest depth (an arrowhead's stroke, +7/8) is still a shape depth in front of the
        // near plane.
        let nearest = linear_depth((MAX_RENDER_Z - 1) as f64 + 0.25) + stroke_depth_offset();
        assert!(nearest >= SHAPE_DEPTH_MIN && nearest > 0.0, "{nearest}");
    }

    /// Every eighth of every slot has an `f32` depth of its own, a whole number of units, and the stroke add
    /// in `f32` (what the vertex shader does) is exact.
    #[test]
    fn eighths_of_a_slot_are_exact_and_apart() {
        let stroke = stroke_depth_offset() as f32;
        assert_eq!(stroke as f64, stroke_depth_offset());
        let whole_units = |d: f32| (d as f64 / DEPTH_UNIT).fract() == 0.0;
        for slot in [0, 1, 2, 56_249, 56_250, 99_999, 100_000, MAX_RENDER_Z - 2, MAX_RENDER_Z - 1] {
            let mut previous = f32::INFINITY;
            for eighth in 0..16 {
                let z = slot as f64 + eighth as f64 / 8.0;
                let depth = linear_depth(z) as f32;
                assert_eq!(depth as f64, linear_depth(z), "z {z}");
                assert!(whole_units(depth), "z {z}");
                assert!(
                    (previous as f64 - depth as f64) >= 5.0 * DEPTH_UNIT,
                    "z {z}: {previous} then {depth}"
                );
                // A stroke at z draws at the depth of z + 5/8.
                assert_eq!(depth + stroke, linear_depth(z + 0.625) as f32, "z {z}");
                previous = depth;
            }
        }
    }

    #[test]
    fn nearer_z_is_smaller_depth_and_negative_z_sits_at_zero() {
        assert_eq!(linear_depth(0.0), SHAPE_DEPTH_MAX);
        assert_eq!(linear_depth(-3.0), SHAPE_DEPTH_MAX);
        assert!(linear_depth(100_000.0) < linear_depth(99_999.875));
        assert!(linear_depth(100_000.0) > 0.0);
    }

    /// Brush artwork's first shape (+0.05) is two units or more nearer than its body's fill, in `f32`.
    #[test]
    fn a_twentieth_of_a_slot_is_two_units() {
        for slot in [0u64, 1, 56_250, 100_000, MAX_RENDER_Z - 1] {
            let fill = linear_depth(slot as f64) as f32;
            let artwork = linear_depth(slot as f64 + 0.05) as f32;
            assert!(fill as f64 - artwork as f64 >= 2.0 * DEPTH_UNIT, "slot {slot}: {fill} then {artwork}");
        }
    }

    /// The WGSL's stroke offset is this module's.
    #[test]
    fn the_shaders_stroke_offset_is_this_one() {
        assert_eq!(
            moda_shaders::manifest().pinned_constants.stroke_depth_offset as f32,
            stroke_depth_offset() as f32
        );
    }
}

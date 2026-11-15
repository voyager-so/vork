//! Gradient mesh fills (`moda_scene::mesh_gradient`, Illustrator's gradient mesh) in the engine.
//!
//! **Baked, then drawn as an image.** The mesh is rasterized on the CPU ([`moda_scene::mesh_gradient::bake`]: each
//! Coons patch diced to [`DICE_TEXELS`](moda_scene::mesh_gradient::DICE_TEXELS) and Gouraud-shaded) into a
//! premultiplied image over its control bounds, at [`TEXELS_PER_PX`] texels per node px (at most [`MAX_DIM`] per
//! side), and drawn by the image-fill lane: the node's mesh layer becomes an image layer placed `crop` over the
//! bounds, so the shape clips it, anti-aliases its edge and takes its opacity, blend and effects like any image
//! fill. Outside the patches the image is transparent. A mesh's colour is smooth, so two texels per px sampled
//! bilinearly stays within an 8-bit step of the patch colour at any zoom; the plan does not depend on the zoom.
//!
//! Bakes are cached by the mesh, its paint box and resolution ([`CACHE_ENTRIES`] per thread), so camera motion
//! and replans reuse them; the texture cache keys the upload by the same hash.

use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use moda_scene::mesh_gradient::{bake, MeshGradientPaint};
use moda_scene::paint::{Fill, ImagePaint, ImagePlacement, Paint, PaintPoint, PaintSize};
use moda_scene::types::RawNode;
use moda_scene::value::Value;

use crate::color::color_to_normalized_rgba;
use crate::paint::{DecodedImage, PatternTexture, PatternTextureState};

/// Texels per node-local px.
pub const TEXELS_PER_PX: f64 = 2.0;
/// Most texels along either side of a baked mesh.
pub const MAX_DIM: u32 = 2048;
/// Bakes kept per thread.
const CACHE_ENTRIES: usize = 32;

thread_local! {
    static BAKES: RefCell<HashMap<u64, (PatternTexture, [f64; 4])>> = RefCell::new(HashMap::new());
}

fn parse(color: &Value) -> Option<[f64; 4]> {
    let c = color.as_str().and_then(color_to_normalized_rgba)?;
    Some([c.r, c.g, c.b, c.a])
}

/// The baked texture for `mesh` over `paint_box` and the node-local rect `[x, y, w, h]` it covers; `None` when the
/// mesh is invalid or has no area (the layer then draws nothing).
pub fn baked(mesh: &MeshGradientPaint, paint_box: [f64; 4]) -> Option<(PatternTexture, [f64; 4])> {
    let mut h = DefaultHasher::new();
    moda_scene::value::to_string(mesh).ok()?.hash(&mut h);
    paint_box.map(f64::to_bits).hash(&mut h);
    TEXELS_PER_PX.to_bits().hash(&mut h);
    let key = h.finish();
    if let Some(hit) = BAKES.with(|b| b.borrow().get(&key).cloned()) {
        return Some(hit);
    }
    let placed = mesh.place(paint_box, parse)?;
    let baked = bake(&placed, TEXELS_PER_PX, MAX_DIM)?;
    let texture = PatternTexture {
        state: PatternTextureState::Loaded,
        key: format!("mesh:{key:016x}"),
        image: DecodedImage {
            width: baked.width,
            height: baked.height,
            rgba_premultiplied: Arc::new(baked.rgba_premultiplied),
            straight: false,
            external: None,
        },
        scale_factor: (1.0, 1.0),
    };
    let rect = [baked.origin[0], baked.origin[1], baked.size[0], baked.size[1]];
    BAKES.with(|b| {
        let mut b = b.borrow_mut();
        if b.len() >= CACHE_ENTRIES {
            b.clear();
        }
        b.insert(key, (texture.clone(), rect));
    });
    Some((texture, rect))
}

/// `node` with its top visible fill's mesh replaced by an image paint that places the bake `crop` over `rect`
/// (the layer's opacity, blend and effects kept).
pub fn as_image_node(node: &RawNode, texture: &PatternTexture, rect: [f64; 4]) -> RawNode {
    let mut fills = moda_scene::paint::node_fills(node);
    let Some(top) = fills.iter().rposition(|f| moda_scene::paint::shown(f.visible, f.opacity)) else {
        return node.clone();
    };
    let (tw, th) = (texture.image.width as f64, texture.image.height as f64);
    let image = Paint::Image(ImagePaint {
        src: Some(texture.key.clone()),
        placement: ImagePlacement {
            fit: Some("crop".into()),
            offset: Some(PaintPoint { x: rect[0], y: rect[1] }),
            scale: Some(PaintPoint { x: rect[2] / tw, y: rect[3] / th }),
            natural_size: Some(PaintSize { width: tw, height: th }),
            ..Default::default()
        },
        adjust: None,
    });
    fills[top] = Fill { paint: image, ..fills[top].clone() };
    let mut m = node.fields().clone();
    moda_scene::paint::set_fills(&mut m, fills);
    node.with_fields(m)
}

/// A 1×1 transparent texture: what an invalid mesh draws.
pub fn transparent() -> PatternTexture {
    PatternTexture {
        state: PatternTextureState::Loaded,
        key: "mesh:invalid".into(),
        image: DecodedImage { width: 1, height: 1, rgba_premultiplied: Arc::new(vec![0; 4]), straight: false, external: None },
        scale_factor: (1.0, 1.0),
    }
}

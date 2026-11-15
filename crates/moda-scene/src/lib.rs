//! moda-scene — scene JSON → normalized, variable-resolved, expanded display
//! list + feature manifest (PLAN.md M0c).
//!
//! Pipeline (mirrors the production TS renderer's CPU side):
//!
//! ```text
//! canvas_data JSON
//!   └─ hydrate (filter invalid types, normalize nodes and pages)        scene.rs / normalize.rs
//!       └─ sanitize page membership + rebuild indexes                   scene.rs
//!           └─ HTML variable fallbacks (before expansion)               variables.rs
//!               └─ expand tables / generated / page backgrounds         expansion.rs
//!                   └─ flattened z-order → display list                 display_list.rs
//!   └─ assess() pure manifest walk (no GPU, no assets)                  assess.rs
//! ```
//!
//! Normative TS sources are cited per item in each module. `types.rs` is
//! FROZEN at M0c exit (PLAN §0 rule 4).

pub mod appearance;
pub mod assess;
pub mod backdrop_kinds;
pub mod blend;
pub mod boolean_node;
pub mod charts;
pub mod corner_smoothing;
pub mod rect_corners;
pub mod defaults;
pub mod display_list;
pub mod effect;
pub mod envelope;
pub mod envelope_raster;
// Shared distort homography math (TS projectiveTransform.ts mirror) — used by
// the distort readers and the v3 animation tilt/skew bakers.
mod distort;
pub mod expansion;
pub mod fields;
pub mod flipbook;
pub mod generator;
pub mod generator_shapes;
pub mod flags;
pub mod live_effects;
pub mod mesh_gradient;
pub mod text_warp;
pub mod normalize;
pub mod page_render;
pub mod paint;
pub mod path_bool;
pub mod path_geom;
pub mod path_length;
pub mod feed_points;
pub mod json_text;
pub mod path_points;
pub mod pattern;
pub mod rectangle_path;
pub mod repeat;
pub mod scene;
pub mod shader_effects;
pub mod shape_outline;
pub mod stamps;
pub mod stroke_brush;
pub mod stroke_model;
pub mod stroke_outline;
pub mod table_spans;
pub mod transform;
pub mod types;
pub mod value;
pub mod variables;
pub mod vector_network;
pub mod worlds;

pub use assess::{assess, manifest_to_json, AssessOptions, ShaderRegistry};
pub use display_list::{
    build_display_list, build_display_list_for_canvas, canvas_root_content_bounds,
    canvas_root_frame_size, CANVAS_ROOT_FRAME_MARGIN,
};
pub use generator_shapes::generator_shapes_json;
pub use live_effects::geometry_effects_json;
pub use path_bool::{
    boolean_paths, boolean_paths_json, pathfinder_paths, pathfinder_paths_json, region_fill_json,
    shape_builder_json,
};
pub use path_geom::fit::fit_points_json;
pub use path_geom::offset_path::offset_path_json;
pub use path_geom::simplify::simplify_path_json;
pub use scene::{extract_canvas_data, extract_canvas_data_owned, load_scene_json, ReexpandOutcome, Scene};
pub use stroke_outline::{stroke_outline_json, stroke_outline_json_with};
pub use types::{
    AxisStatus, DisplayList, Manifest, PaintItem, PaintKind, PaintPhase, NodeField, RawNode, RawPage,
    SceneDocument,
};

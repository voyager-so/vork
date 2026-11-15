//! Known-field tables per node type — the permissive-serde "spillover"
//! detector (PLAN M0c). A field on a node that is not in its type's table is
//! reported as the manifest axis `unknown:node-field` with status `unknown`
//! (PLAN §3: "the schema evolved and a human should look").
//!
//! Tables are transcribed from `core/scene/nodes.ts` via
//! `research/inventory/scene-schema.md` §3–§4. There are no load-time
//! migrations, so old field names are not listed: a document still carrying
//! one reports it as unknown.

/// `BaseNodeProperties` (nodes.ts) + transient render-time fields.
const BASE: &[&str] = &[
    "id",
    "type",
    "locked",
    "hidden",
    "opacity",
    // Node-level normalized pivot in [0,1] (`pivotX`/`pivotY`, BaseNodeProperties)
    // — the rotation/scale origin (nodePivot.ts). Omitting them here would
    // false-decline docs that set a node pivot over a supported field.
    "pivotX",
    "pivotY",
    // The node's shear in degrees (`skewX`, BaseNodeProperties): `T · R · Shear · S`, read by
    // `Scene::accumulated_affine` (transform::from_trsk), so every lane draws it.
    "skewX",
    // Saved export rows (`exportSettings`, BaseNodeProperties) — the Details panel's
    // Export section. Editor-only: what to export, never how the node renders.
    "exportSettings",
    // Component property bindings of a main's layer (`propertyRefs`, design-systems.md §6.2). The editor applies
    // property values into an instance's stored layers, so nothing here reads it.
    "propertyRefs",
    // Figma's "Ignore auto layout" (`layoutPositioning: 'absolute'`, BaseNodeProperties, E2.3).
    // Auto layout is resolved in the editor and baked into each child's x/y before a doc
    // reaches us, so nothing here reads it; an absolute child simply renders at its own x/y.
    "layoutPositioning",
    // Auto-layout sizing (`widthMode` / `heightMode`, BaseNodeProperties since E2.2): a leaf
    // shape can Fill its auto-layout parent. Resolved in the editor and baked into width/height
    // before a doc reaches us, like layoutPositioning, so nothing here reads it.
    "widthMode",
    "heightMode",
    // Figma's min / max width and height (`minWidth` … `maxHeight`, BaseNodeProperties): limits the
    // editor's auto layout clamps sizes to. Baked into width/height before a doc reaches us.
    "minWidth",
    "maxWidth",
    "minHeight",
    "maxHeight",
    // A child's cell, spans and cell alignment in a Grid auto-layout frame (`gridColumn` … `gridAlignY`,
    // BaseNodeProperties). The editor's grid layout bakes them into x/y and width/height.
    "gridColumn",
    "gridRow",
    "gridColumnSpan",
    "gridRowSpan",
    "gridAlignX",
    "gridAlignY",
    // Figma's constraints (`constraints: { horizontal, vertical }`, BaseNodeProperties, E2.7): how
    // a frame's child follows the frame when it is resized. Applied in the editor at resize time
    // and baked into x/y/size, so nothing here reads it.
    "constraints",
    // Clip to the layer below (`clipToBelow`, BaseNodeProperties; Photoshop's clipping mask): the engine draws the
    // layer only where its base (`Scene::clip_below_base`) has alpha, the base still visible.
    "clipToBelow",
    // The overprint mode (`overprintMode`, BaseNodeProperties; PDF OPM) of the layer's overprinting fills and
    // strokes: written by the CMYK PDF export, nothing here draws it.
    "overprintMode",
    // Figma's mask types (E7.18): how a group's clip-mask node masks (alpha /
    // vector / luminance). Read by the engine's mask coverage (effects/mask.rs).
    "maskType",
    "maskInvert",
    "maskFeather",
    // Variables v2: field → variable bindings and explicit modes per collection.
    // The bound fields keep their literals, which are what renders here.
    "bindings",
    "variableModes",
    // Styles (design-systems.md §5): slot → style id. The styled fields keep the
    // style's values, which are what renders here.
    "styleRefs",
    "metadata",
    // Transient: marks a node a flipbook's onion skin added when the scene was made (`crate::flipbook::ONION_KEY`).
    "_onion",
    // Canvas/layer display label. Not a declared schema field (the canonical node
    // name lives at `metadata.name`), but legacy/agent write paths persist a
    // top-level `name` on nodes AND the envelope, and the collab projection
    // round-trips any undeclared top-level key through its `meta` catch-all
    // (projection.py `apply_canvas_data_to_doc`). It is a pure label with no
    // render effect, so recognizing it avoids a false decline.
    "name",
    // Duplicated/pasted node's reference to its source node id. Not a declared
    // schema field — the duplication/paste bookkeeping writes it as an untyped
    // `(node as any).originalId` (DuplicationService.ts `offsetTopLevelNodes`,
    // PastePlacementService.ts) purely to map a copy back to its original for
    // offsetting; the renderer never reads it. Measured on live prod as a
    // residual false-decline driver, so recognize it on every node kind.
    "originalId",
    "_pageClipBoundsOverride",
    "_pageOrigin",
    "_maskContext",
    "_transformSourceId",
    // Natural (intrinsic) dimensions of a pattern-image fill's source image,
    // stashed for pattern-editing UX. Not a declared schema field — the image
    // placement path writes it as an untyped `(rectNode as any)._patternImageInfo`
    // (ImageService.ts) purely so the editor can reason about the source image;
    // the renderer never reads it (the render-affecting fields are
    // image paint's `src` + `placement.naturalSize`, in `fills`,
    // which carries the same dims). Measured on live prod (2026-07-03) as a
    // residual false-decline driver on thumbnail + vision renders, same class as
    // `name` / `originalId`, so recognize it on every node kind.
    //
    // NOTE (why not a blanket "ignore any `_`-prefixed field" rule): the four
    // underscore fields above are DECLARED `BaseNodeProperties` schema
    // (nodes.ts) AND render-affecting — the engine reads them (e.g.
    // `_transformSourceId` in scene.rs `paint_transform_id`; `_pageOrigin` /
    // `_pageClipBoundsOverride` in frame.rs). So the `_` prefix is NOT a reliable
    // "internal / non-render-affecting" signal here; a blanket ignore would
    // silence exactly the schema-drift the spillover detector exists to catch (a
    // future render-affecting `_`-field would silently mis-render natively instead
    // of safely declining). Recognize benign internal fields one at a time.
    "_patternImageInfo",
    // Illustrator's Text Wrap (E8.14): area and box text below this layer flow around its outline
    // (`moda_engine::text_area`).
    "wrapsText",
    // A layer's 2.5D transform (`transform3d`, motion-model.md §12.6, 3D phase 1): moda-render's graph lane projects
    // it (moda-cli `graph_space.rs`); a plain render draws the layer in 2D.
    "transform3d",
    // A 3D layer (`model3d`, motion-model.md §12.6 phase 2): moda-render's graph lane draws it as a lit solid
    // (moda-cli `graph_solid.rs`); a plain render draws the layer as it is.
    "model3d",
    // A reference layer (`reference`, raster-layers.md §12.1): the editor's bucket, magic wand and Select ▸ Painted
    // pixels read the composite of these layers. Editor-only: it never changes what renders.
    "reference",
    // A reference image (`exportExcluded`): the export hosts drop the layer before a job reaches the renderer
    // (export-hosts `exportExcluded.mjs`), so a render here draws it like any layer (screenshots show it).
    "exportExcluded",
    // A generated node's identity (format 29, generated-content RFC §10): its key path under the generator and its
    // content hash. Editor-only (reconciliation reads them); the node renders as itself.
    "genKey",
    "genHash",
];

/// The fill stack (`fills`, E3.1 paint stacks; `paint.rs`).
const FILL: &[&str] = &["fills"];

/// The stroke stack (`strokes`): each stroke carries its own width, caps,
/// joins, dashes, width profile / brush objects and arrowheads.
const STROKE: &[&str] = &["strokes"];

/// Illustrator's Appearance order (E8.2, `crate::appearance`): the fill and stroke ids (and a group's
/// `contents`) bottom to top.
const APPEARANCE: &[&str] = &["appearanceOrder"];

/// `TrimPathProperties` (nodes.ts) — After-Effects-style stroke trim. Scalar
/// fractions of arc length; absent = identity. Carried by the strokeable
/// geometric shapes and lines (not container/table/richtext).
const TRIM_PATH: &[&str] = &[
    "trimPathStart",
    "trimPathEnd",
    "trimPathOffset",
    "trimPathMode",
    "trimPathBy",
];

/// `TextFrameProperties` (nodes.ts:631-667).
const TEXT_FRAME: &[&str] = &[
    "textContent",
    "textPadding",
    "textHorizontalAlign",
    "textVerticalAlign",
    "textResizeMode",
];

/// AABB geometry shared by most shapes (scene-schema.md §2).
const GEOM: &[&str] = &[
    "x",
    "y",
    "width",
    "height",
    "rotation",
    "scale",
    "aspectRatio",
];

/// `EffectsProperties` (effect.ts): the one effect list (drop / inner
/// shadows, blurs, backdrops, shaders, text effects; `moda_scene::effect`).
const EFFECTS: &[&str] = &["effects"];

// `cornerSmoothing`: Figma's squircle corners (0–1), drawn through
// `corner_smoothing::smoothed_rect_path` (E7.11).
// `cornerStyle`: live corners (round / inverted / chamfer), drawn through the outline
// (`round_path_corners_with`) instead of the SDF.
const RECTANGLE_ONLY: &[&str] = &["cornerRadius", "cornerSmoothing", "cornerStyle", "blendMode"];

// Canonical planar distortion on every distort-capable box type
// (`DISTORT_CAPABLE_NODE_TYPES`, core/scene/distortGeometry.ts). KNOWN as of
// node-distort PR 2: the native engine renders the warp (moda-engine
// `paint::node_distort` → geometry pre-warp / pattern UV inverse map / vector
// text DrawUniforms rows), so a distort-carrying doc no longer declines to the
// fallback renderer.
const DISTORT: &[&str] = &["distort"];

const CONTAINER_ONLY: &[&str] = &[
    // The layer blend mode (Figma's Layer row): the layer's drawn result blends as one layer.
    "blendMode",
    "children",
    "clipContent",
    "cornerRadius",
    "cornerSmoothing",
    "flexLayout",
    // Figma's layout grids (`layoutGrids`, ContainerNode): an editor overlay and snapping / constraint aid,
    // never drawn by the renderer (grids don't export).
    "layoutGrids",
    // Ruler guides of a top-level frame (`guides`, ContainerNode; core/scene/guides.ts): an editor overlay and
    // snapping aid, never drawn by the renderer.
    "guides",
    // A top-level frame's print bleed (`bleed`, ContainerNode; core/scene/print.ts): drawn as an editor guide; a print
    // PDF export grows the page by it before rendering, so the renderer never reads it.
    "bleed",
    // Components (design-systems.md §6): a main component's `component` and an instance's `instance` (source and
    // overrides). The editor stores an instance's layers as ordinary child nodes (a derived cache), so nothing here
    // reads either: an instance renders as the frame it is.
    "component",
    "instance",
    // A component set (variants, design-systems.md §6.2): the frame holding the variants and their shared
    // properties. Editor-only; the set renders as the frame it is.
    "componentSet",
    // The frame's motion graph (`graph`, format 26, motion-model.md §2.1; moda-motion `FrameGraph`). The static design
    // is the tree; nothing here evaluates the graph yet.
    "graph",
    // A main component's 2D rig (`rig`, character-assets.md §3.1; moda-motion `rig`): bones, parts, slots, poses.
    // The tree is the rest pose; the motion overlay writes posed parts onto their own fields, so nothing here reads it.
    "rig",
];

/// `SectionNode` (nodes.ts, design-editor E7.10): Figma's section. Its box, children, fill and stroke; no
/// rotation, scale or effects. `contentsHidden` (Hide contents) skips its children in the z-order.
const SECTION_ONLY: &[&str] = &[
    "x",
    "y",
    "width",
    "height",
    "children",
    "contentsHidden",
];

const ELLIPSE_ONLY: &[&str] = &[
    "startAngle",
    "sweepAngle",
    "innerRadiusRatio",
    "blendMode",
];

const LINE_ONLY: &[&str] = &[
    // The layer blend mode (Figma's Layer row): the layer's drawn result blends as one layer.
    "blendMode",
    "points",
    "routing",
    "waypoints",
    "cornerRadius",
    "autoRoute",
    "pointerLength",
    "pointerWidth",
    // Arrowheads live on the stroke (`strokes[].startArrow` / `endArrow`, E3.1).
    "startAnchor",
    "endAnchor",
    // ConnectorTextProperties (nodes.ts:675-729)
    "textContent",
    "textPosition",
    "textOffsetX",
    "textOffsetY",
    "textPadding",
    "textAlign",
    "textRotation",
    "textBackground",
    "textBackgroundColor",
];

const PATH_ONLY: &[&str] = &[
    "points",
    "fillRule",
    "closed",
    "blendMode",
    // Editable uniform corner rounding, applied to `points` at the
    // single derive point (`moda_engine::tess::path_cmds_for_node`).
    "cornerRadius",
    // Live corners: the path-wide style (round / inverted / chamfer); a point's own
    // `cornerStyle` / `cornerRadius` override it (`round_path_corners_with`).
    "cornerStyle",
    // Pen samples recorded while drawing (brush plan §4.5); path-level, not
    // stroke-scoped. Read by brushes set to vary with pressure/tilt/rotation.
    "inkChannels",
    // Figma's vector network (E7.9): authoritative geometry when present;
    // `points` / `closed` / `fillRule` are derived from it. Drawn natively by
    // `moda_engine::tess::network` (fills per region, strokes per segment).
    "vectorNetwork",
];

const STAR_ONLY: &[&str] = &["numPoints", "innerRadiusRatio", "cornerRadius", "blendMode"];

const POLYGON_ONLY: &[&str] = &["sides", "cornerRadius", "blendMode"];

const GROUP_ONLY: &[&str] = &[
    "x",
    "y",
    "rotation",
    "scaleX",
    "scaleY",
    "children",
    "clipMaskNodeId",
    "clipMode",
    "clipInherit",
    // The group's blend mode (#855): its subtree flattens, then blends over the backdrop.
    "blendMode",
    // Layer filters on the group (#1177): a layer blur blurs its whole content as one layer.
    "effects",
    // Envelope distort (`envelope`, E8.17): a warp preset, mesh or top-object outline that bends the group's
    // content, resolved by `envelope.rs` into transients drawn in place of the children.
    "envelope",
    // Repeat objects (`repeat`, Object ▸ Repeat): radial, grid or mirror copies of the group's children, resolved
    // by `repeat.rs` into transients drawn in place of the children.
    "repeat",
    // A live Image Trace object's state (`imageTrace`, E8.15): options, stats and the id of the
    // hidden source child. Editor-only: the group renders as its children (paths shown, the
    // source image hidden), so nothing here reads it.
    "imageTrace",
    // A flipbook drawing's exposure in timeline frames (`hold`, motion-model.md §12.10), read through its flipbook.
    "hold",
    // A flipbook (format 28, motion-model.md §12.10): the group draws one child, `flipbook.current`
    // (`crate::flipbook`), which the timeline sets frame by frame.
    "flipbook",
    // The onion skin preview on a flipbook (editor state, never saved or exported; `crate::flipbook::expand_onion`).
    "onionSkin",
];

/// `BooleanNode` (nodes.ts, E7.1): a group-like transform, its children, the operation, and
/// the node's own paint and effects (the EFFECTS / FILL / STROKE tables).
const BOOLEAN_ONLY: &[&str] = &[
    "x",
    "y",
    "rotation",
    "scaleX",
    "scaleY",
    "children",
    "booleanOperation",
    "blendMode",
];

const RICHTEXT_ONLY: &[&str] = &[
    // The layer blend mode (Figma's Layer row): the layer's drawn result blends as one layer.
    "blendMode",
    // Per-glyph kinetic-typography deltas (KineticText). Meant to be
    // TRANSIENT (per-frame animation output, not persisted), but it IS a
    // RichTextNode schema field and the native vector text lane renders it
    // (`build_richtext_item` routes non-empty `unitTransforms` to the vector arm),
    // so a doc that carries it must not spill as `unknown:node-field`.
    "unitTransforms",
    "htmlContent",
    // The glyph colour override (E3.1 paint stacks; absent = the run colours).
    "textFills",
    "paragraphSpacing",
    "maxLines",
    // E6.3 "Truncate text" ('ending'); with maxLines and the box height it
    // hides overflowing lines behind an ellipsis (moda-text `truncation`).
    "textTruncation",
    // F7.2 type on a path: the text runs along this path (moda-text `text_path`).
    "textPath",
    // E8.14 area type: the outline and Area Type Options the text fits (moda-text `area`), and the
    // thread frame this one continues (`moda_engine::text_area`).
    "area",
    "threadPrev",
    "verticalAlign",
    "resizeMode",
    // Opt-in AABB clip of the text render to the node's own box, set by
    // table expansion on cell content nodes so overflowing cell text can't paint
    // over its neighbours. KNOWN schema, HONOURED natively — the frame.rs richtext
    // arm intersects the effective clip rect with the node AABB — so a doc that
    // carries it must not spill as `unknown:node-field` (which would leave the
    // native engine silently ignoring the clip and diverging from the browser).
    "clipTextToBounds",
];

/// A raster paint layer's own fields (spike). `pixelSelection` is editor state (the marching ants' key, kept on the
/// node so selecting is an undo step): never saved and never rendered, so both renderers ignore it alike.
// `hold`: a flipbook drawing's exposure in timeline frames (motion-model.md §12.10), read through its flipbook.
// `assistants`: drawing assistants (perspective / ellipse rulers, docs/design/drawing-assistants.md), an editor
// painting aid that never renders, so both renderers ignore it alike.
const PAINT_ONLY: &[&str] = &[
    "surface",
    "blendMode",
    "lockTransparency",
    "pixelSelection",
    "_paintLayer",
    "hold",
    "assistants",
];

const BLEND_ONLY: &[&str] = &[
    "sourceIds",
    "spineId",
    "spacing",
    "steps",
    "distance",
    "orientation",
    "easeIn",
    "easeOut",
    "reverseStack",
    // Applied to every step (`blend::expand_blend_node`).
    "blendMode",
];

const TABLE_ONLY: &[&str] = &[
    // The layer blend mode (Figma's Layer row): the layer's drawn result blends as one layer.
    "blendMode",
    "rows",
    "columns",
    "cells",
    "backgroundColors",
    "headerBackgroundColor",
    "columnStrokeConfig",
    "rowStrokeConfig",
    "cornerRadius",
    // Whether row heights are re-measured from cell content ('auto') or kept as authored
    // (absent/'fixed'). Nothing in the engine READS it: row heights are already
    // resolved into `rows[].size` by the time a doc reaches us, so measurement is a pure
    // frontend concern. It is mirrored purely so a doc carrying it doesn't spill as
    // `unknown:node-field` and send the whole page to the Chromium fallback.
    "rowSizing",
];

const GENERATED_ONLY: &[&str] = &[
    // The layer blend mode (Figma's Layer row): the layer's drawn result blends as one layer.
    "blendMode",
    "generatedKind",
    "generatedSource",
    "renderMode",
    "cachedAssetUrl",
    "cachedVectorPath",
    "cachedNaturalDimensions",
    // A chart's part values for one motion frame (charts/motion_parts.rs): the render feed only, never saved.
    "_motionParts",
];

// `_generatorOutput` / `_generatorStatus`: the editor feed only (its worker's output, never saved).
// Format 29 (generated-content RFC §4.1): `children` is a nodes-mode generator's derived cache (drawn as its
// children), `sources` its source layers; `outputKey`, `overrides` and `host` are the editor's.
// `_compositionTime`: the frame's time inside a composition, written per frame by moda-render's graph lane.
const GENERATOR_ONLY: &[&str] = &[
    "blendMode",
    "code",
    "params",
    "seed",
    "seedIdentity",
    "sourceVisible",
    "children",
    "sources",
    "outputKey",
    "overrides",
    "host",
    "_generatorOutput",
    "_generatorStatus",
    "_compositionTime",
];

fn contains(groups: &[&[&str]], key: &str) -> bool {
    groups.iter().any(|g| g.contains(&key))
}

/// True when `key` is a known field for node type `node_type`.
/// Unknown node types accept nothing (the type itself is already the signal).
pub fn is_known_field(node_type: &str, key: &str) -> bool {
    let groups: &[&[&str]] = match node_type {
        "rectangle" => &[
            BASE,
            APPEARANCE,
            GEOM,
            EFFECTS,
            FILL,
            STROKE,
            TRIM_PATH,
            TEXT_FRAME,
            DISTORT,
            RECTANGLE_ONLY,
        ],
        "container" => &[
            BASE,
            APPEARANCE,
            GEOM,
            EFFECTS,
            FILL,
            STROKE,
            CONTAINER_ONLY,
        ],
        "section" => &[BASE, FILL, STROKE, SECTION_ONLY],
        "ellipse" => &[
            BASE,
            APPEARANCE,
            GEOM,
            EFFECTS,
            FILL,
            STROKE,
            TRIM_PATH,
            TEXT_FRAME,
            DISTORT,
            ELLIPSE_ONLY,
        ],
        "line" => &[
            BASE,
            APPEARANCE,
            EFFECTS,
            STROKE,
            TRIM_PATH,
            LINE_ONLY,
        ],
        "path" => &[
            BASE,
            APPEARANCE,
            GEOM,
            EFFECTS,
            FILL,
            STROKE,
            TRIM_PATH,
            TEXT_FRAME,
            PATH_ONLY,
        ],
        "star" => &[
            BASE, GEOM, EFFECTS, FILL, STROKE, TRIM_PATH, TEXT_FRAME,
            DISTORT, STAR_ONLY, APPEARANCE,
        ],
        "polygon" => &[
            BASE,
            APPEARANCE,
            GEOM,
            EFFECTS,
            FILL,
            STROKE,
            TRIM_PATH,
            TEXT_FRAME,
            DISTORT,
            POLYGON_ONLY,
        ],
        // Appearance on groups (E8.2): fills and strokes painted on the merged outline of the contents.
        "group" => &[BASE, GROUP_ONLY, FILL, STROKE, APPEARANCE],
        // A boolean draws as its result path, whose paint stack follows the order like any path's.
        "boolean" => &[
            BASE,
            APPEARANCE,
            EFFECTS,
            FILL,
            STROKE,
            BOOLEAN_ONLY,
        ],
        "richtext" => &[BASE, GEOM, EFFECTS, STROKE, FILL, DISTORT, RICHTEXT_ONLY],
        "blend" => &[BASE, BLEND_ONLY],
        "table" => &[BASE, GEOM, STROKE, TABLE_ONLY],
        "generated" => &[BASE, GEOM, STROKE, GENERATED_ONLY],
        "generator" => &[BASE, GEOM, EFFECTS, GENERATOR_ONLY],
        // Raster paint layer (spike): hydrates as a rectangle (`normalize::normalize_paint_in_place`).
        "paint" => &[BASE, GEOM, EFFECTS, PAINT_ONLY],
        _ => return false,
    };
    contains(groups, key)
}

/// Unknown (spillover) fields on a node — feeds `unknown:node-field`.
pub fn unknown_fields(node: &crate::types::RawNode) -> Vec<String> {
    let t = node.node_type();
    node.fields()
        .iter()
        // A `null` value means the field is absent/unset, not schema drift. Pydantic
        // `model_dump()` without `exclude_none` emits EVERY optional field as null, so a plain
        // node round-tripped through a backend save carries e.g. `distort: null` (and dozens of
        // others). A null carries no renderable data, so it must not spill as
        // `unknown:node-field` — otherwise a not-yet-recognized optional field (like `distort`
        // before its follow-up renderer) would false-decline every saved doc to the fallback.
        // Real (non-null) unknown values still spill; static TS↔Rust schema drift is still caught
        // by the `scene_schema_parity` gate, which checks the declared surface, not runtime values.
        .filter(|(k, v)| !v.is_null() && !is_known_field(t, k))
        .map(|(k, _)| k.to_string())
        .collect()
}

/// Known top-level keys of the `canvas_data` envelope.
///
/// The authoritative producer is the collab projection
/// (`backend/app/collab/projection.py` `project_doc_to_canvas_data`), which
/// spreads every `meta` Y.Map key to the top level: the enumerated `META_KEYS`
/// (`kind` / `documentType` / `category` / `authoringModel` / `schemaVersion` / `layoutMode` /
/// `gridColumns` / `activeTemplatePackageIds` / `width` / `height`) PLUS any
/// undeclared top-level key the round-trip parks in `meta` (e.g. a legacy/agent
/// `name`). These are benign document metadata — classification, version, legacy
/// single-page dims, display label — that the assess walker never reads. Before
/// they were listed here every collab-projected doc carrying them tripped
/// `unknown:envelope-field` and false-declined to Chromium. The
/// `tests/scene_schema_parity.rs` guard keeps this in lockstep with BOTH canonical
/// top-level producers — the projection `META_KEYS` and the frontend serializer's
/// `getSceneState()` return-type key set — which is the parity proof that lets
/// `unknown:envelope-field` be non-gating (assess.rs `axis_elevates_verdict`).
pub const ENVELOPE_KEYS: &[&str] = &[
    "nodes",
    "pages",
    "category",
    "authoringModel",
    // Variables v2 (design-systems.md §4.1): collections, variables and the
    // document's explicit modes. Never read here: the engine renders the cached
    // literals the editor keeps next to every binding (variables.rs).
    "variableCollections",
    "variables",
    "variableModes",
    // Styles (design-systems.md §5.1). Never read here: layers keep the style's
    // values as literals.
    "styles",
    // Document brush table (brushes.md §4.4; `Scene.getSceneState().brushes`,
    // emitted by the save serializer only when non-empty). Brush artwork is
    // never on a page; the outline/brush renderers (plan P8, P16–P19) read it.
    "brushes",
    // Document pattern table (E8.5, `crate::pattern`; emitted only when non-empty). Pattern artwork is never
    // on a page; the engine's pattern paint reads it.
    "patterns",
    // Document code table (live-nodes.md §3.2): generator nodes' `CodeDef`s by content hash.
    "code",
    "animations",
    "activeTemplatePackageIds",
    "config",
    "canvasRootNodeIds",
    // A Design document's pages (docs/roadmap/design-pages.md): the scene draws
    // one of them (`activeCanvasId`, else the first) as its canvas root.
    "canvases",
    "activeCanvasId",
    "layoutMode",
    "gridColumns",
    "stateVersion",
    // Benign document metadata from the collab projection (see doc above):
    // `META_KEYS` classification/version/legacy-dims + the round-tripped `name`.
    "kind",
    "documentType",
    "schemaVersion",
    "width",
    "height",
    "name",
    // Arbitrary document-metadata blob. The PPTX import writes it at the top
    // level (`canvas_data["metadata"] = canvas_doc.metadata`,
    // pptx_import_service.py) and the collab projection round-trips it through
    // its `meta` catchall (it is not one of the enumerated `META_KEYS`). The
    // assess walker never reads it, so it is benign; recognizing it stops a
    // residual false-decline measured on live prod traffic.
    "metadata",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RawNode;
    use crate::value::json;

    #[test]
    fn known_fields_accept_documented_schema() {
        // nodes.ts:741-762 (rectangle), 1275-1375 (richtext)
        assert!(is_known_field("rectangle", "cornerRadius"));
        assert!(is_known_field("rectangle", "cornerSmoothing"));
        assert!(is_known_field("container", "cornerSmoothing"));
        assert!(is_known_field("container", "component"));
        assert!(is_known_field("container", "instance"));
        assert!(is_known_field("container", "componentSet"));
        assert!(is_known_field("rectangle", "propertyRefs"));
        // `path` carries its own uniform cornerRadius now.
        assert!(is_known_field("path", "cornerRadius"));
        assert!(!is_known_field("rectangle", "perspective")); // the old corner-pin field is gone
                                                              // Canonical planar distortion: KNOWN on the five distort-capable
                                                              // box types as of node-distort PR 2 (the native engine renders the warp).
                                                              // Any other kind carrying `distort` is out-of-schema and still spills.
        for t in ["rectangle", "ellipse", "star", "polygon", "richtext"] {
            assert!(
                is_known_field(t, "distort"),
                "distort must be native-known on the capable types ({t})"
            );
        }
        for t in ["line", "group", "container", "path"] {
            assert!(
                !is_known_field(t, "distort"),
                "distort must stay unknown on non-capable types ({t})"
            );
        }
        assert!(is_known_field("richtext", "htmlContent"));
        // E-M.7: one `effects` list on every effect-capable kind; the flat
        // effect keys it replaced spill.
        for t in ["rectangle", "container", "ellipse", "path", "star", "polygon", "line", "richtext"] {
            assert!(is_known_field(t, "effects"), "effects must be known on {t}");
            for flat in ["shadowColor", "shadowBlur", "innerShadowColor", "backgroundBlur", "layerBlur", "shaderEffects"] {
                assert!(!is_known_field(t, flat), "flat effect key {flat} must spill on {t}");
            }
        }
        assert!(is_known_field("group", "effects")); // group layer filters (#1177)
        assert!(is_known_field("group", "envelope")); // envelope distort (E8.17)
        assert!(is_known_field("group", "repeat")); // repeat objects
        assert!(!is_known_field("rectangle", "envelope"));
        assert!(!is_known_field("line", "pointerAtBeginning")); // stored arrowheads are strokes[].startArrow/endArrow
                                                                // E3.1 paint stacks: a line's arrowheads moved onto its stroke.
        assert!(!is_known_field("line", "startArrowShape"));
        assert!(!is_known_field("line", "endArrowShape"));
        assert!(!is_known_field("ellipse", "radiusX")); // the old center+radius form is gone
                                                        // Paint stacks (E3.1): every fillable node stores `fills` (a video is a
                                                        // fill's paint), every strokeable node `strokes`; the flat keys are gone.
        for t in [
            "rectangle",
            "container",
            "ellipse",
            "path",
            "star",
            "polygon",
            "richtext",
        ] {
            assert!(is_known_field(t, "fills"), "fills must be known on {t}");
        }
        for t in [
            "rectangle",
            "container",
            "ellipse",
            "path",
            "star",
            "polygon",
            "richtext",
            "line",
            "table",
            "generated",
        ] {
            assert!(is_known_field(t, "strokes"), "strokes must be known on {t}");
        }
        for t in ["rectangle", "ellipse", "star", "line", "richtext"] {
            for flat in [
                "fill",
                "fillPriority",
                "fillVideoSrc",
                "fillVideoPosterSrc",
                "fillPatternImageSrc",
                "fillShaderId",
                "stroke",
                "strokeWidth",
                "lineCap",
                "dash",
            ] {
                assert!(
                    !is_known_field(t, flat),
                    "flat key {flat} must spill on {t}"
                );
            }
        }
        assert!(!is_known_field("line", "fills")); // lines carry no fill
        // Appearance on groups (E8.2): a group's fills and strokes paint its merged outline.
        assert!(is_known_field("group", "fills"));
        assert!(is_known_field("group", "strokes"));
        assert!(is_known_field("group", "appearanceOrder"));
        assert!(is_known_field("rectangle", "appearanceOrder"));
        // Rich text's glyph-colour override (its `fills` is the box background).
        assert!(is_known_field("richtext", "textFills"));
        assert!(!is_known_field("rectangle", "textFills"));
        assert!(!is_known_field("rectangle", "futureProp"));

        // Node-level pivot (BaseNodeProperties) — the native v3 evaluator reads
        // these, so every node kind must recognize them (no false decline).
        assert!(is_known_field("rectangle", "pivotX"));
        assert!(is_known_field("richtext", "pivotY"));
        assert!(is_known_field("group", "pivotX"));
        // Benign top-level label bled onto nodes by collab/agent write paths.
        assert!(is_known_field("rectangle", "name"));
        // Duplicated/pasted node's source-node reference — bookkeeping only, no
        // render effect; recognized on every kind (measured live-prod decline).
        assert!(is_known_field("rectangle", "originalId"));
        assert!(is_known_field("group", "originalId"));
        assert!(is_known_field("richtext", "originalId"));
        // Internal pattern-image annotation (ImageService.ts) — non-render-affecting
        // UX metadata; recognized on every kind (measured live-prod decline).
        assert!(is_known_field("rectangle", "_patternImageInfo"));
        assert!(is_known_field("ellipse", "_patternImageInfo"));
        // NARROW fix, NOT a blanket underscore rule: an arbitrary unrecognized
        // `_`-prefixed field STILL spills (the `_` prefix is a render-affecting
        // schema convention here — see BASE doc), so drift stays caught.
        assert!(!is_known_field("rectangle", "_futureInternalProp"));
        // Per-glyph kinetic-typography deltas render on the native vector lane.
        assert!(is_known_field("richtext", "unitTransforms"));
        assert!(!is_known_field("rectangle", "unitTransforms")); // richtext-only

        // Brushes / variable-width strokes (brush plan §4): the width profile,
        // taper, wave, dynamics and brush live on each `strokes[]` entry now
        // (E3.1), so the flat keys spill everywhere; pen samples stay path-level.
        for f in [
            "strokeProfile",
            "strokeTaper",
            "strokeWave",
            "strokeDynamic",
            "strokeBrush",
        ] {
            for t in ["path", "line", "rectangle", "ellipse", "star", "polygon"] {
                assert!(!is_known_field(t, f), "{f} must be unknown on {t}");
            }
        }
        assert!(is_known_field("path", "inkChannels"));
        for t in ["line", "rectangle", "ellipse", "star", "polygon"] {
            assert!(
                !is_known_field(t, "inkChannels"),
                "inkChannels must stay unknown on {t}"
            );
        }
    }

    #[test]
    fn envelope_keys_include_collab_projected_metadata() {
        // The collab projection spreads the `meta` Y.Map to the top level; these
        // benign metadata keys must not spill as `unknown:envelope-field`.
        for k in [
            "kind",
            "documentType",
            "schemaVersion",
            "width",
            "height",
            "name",
            // Arbitrary document-metadata blob round-tripped through the
            // projection's `meta` catchall (PPTX import writes it).
            "metadata",
        ] {
            assert!(
                ENVELOPE_KEYS.contains(&k),
                "envelope key `{k}` must be recognized"
            );
        }
    }

    #[test]
    fn layout_positioning_is_known_on_every_node_type() {
        // Any node can be a child of an auto-layout container, so the E2.3 absolute-position
        // flag must never send a page to the fallback renderer.
        for t in [
            "rectangle", "container", "section", "ellipse", "line", "path", "star", "polygon", "group",
            "richtext", "blend", "table", "generated",
        ] {
            assert!(
                is_known_field(t, "layoutPositioning"),
                "layoutPositioning must be known on {t}"
            );
            // E2.2: any auto-layout child can carry a sizing mode (Fill), not just containers.
            assert!(
                is_known_field(t, "widthMode"),
                "widthMode must be known on {t}"
            );
            assert!(
                is_known_field(t, "heightMode"),
                "heightMode must be known on {t}"
            );
            // E2.7: any frame child can carry constraints.
            assert!(
                is_known_field(t, "constraints"),
                "constraints must be known on {t}"
            );
            if t == "container" {
                assert!(
                    is_known_field(t, "layoutGrids"),
                    "layoutGrids must be known on containers"
                );
                assert!(is_known_field(t, "guides"), "ruler guides must be known on containers");
            }
            for f in [
                "minWidth", "maxWidth", "minHeight", "maxHeight", "gridColumn", "gridRow",
                "gridColumnSpan", "gridRowSpan", "gridAlignX", "gridAlignY",
            ] {
                assert!(is_known_field(t, f), "{f} must be known on {t}");
            }
        }
        let node = RawNode::from_value(json!({
            "id": "r1", "type": "rectangle", "x": 4, "y": 6,
            "width": 10, "height": 10, "rotation": 0, "scale": 1,
            "layoutPositioning": "absolute",
            "constraints": { "horizontal": "left-right", "vertical": "bottom" },
            "minWidth": 8, "maxHeight": 40,
            "gridColumn": 2, "gridRow": 1, "gridColumnSpan": 2, "gridAlignX": "center"
        }))
        .unwrap();
        assert!(unknown_fields(&node).is_empty());
    }

    #[test]
    fn unknown_field_spillover_detected() {
        let node = RawNode::from_value(json!({
            "id": "r1", "type": "rectangle", "x": 0, "y": 0,
            "width": 10, "height": 10, "rotation": 0, "scale": 1,
            "futureProp": {"v": 2}
        }))
        .unwrap();
        assert_eq!(unknown_fields(&node), vec!["futureProp".to_string()]);
    }

    #[test]
    fn null_valued_unknown_field_does_not_spill() {
        // A plain node round-tripped through a backend `model_dump()` (no exclude_none) carries
        // `distort: null` and other unset optionals. Null = absent, so it must NOT spill as
        // unknown:node-field — else every saved doc would false-decline over an inert null.
        let with_nulls = RawNode::from_value(json!({
            "id": "r1", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10,
            "rotation": 0, "scale": 1, "distort": null, "futureProp": null
        }))
        .unwrap();
        assert!(unknown_fields(&with_nulls).is_empty());

        // A REAL (non-null) distort is KNOWN schema as of node-distort PR 2 —
        // the native engine renders the warp, so the doc must NOT decline.
        let with_real = RawNode::from_value(json!({
            "id": "r1", "type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10,
            "rotation": 0, "scale": 1,
            "distort": {"type": "matrix", "matrix": [1, 0, 0, 0, 1, 0, 0.3, 0.1, 1]}
        }))
        .unwrap();
        assert!(unknown_fields(&with_real).is_empty());

        // ...but on a non-capable type it is out-of-schema and still spills
        // (declines to the fallback rather than rendering silently unwarped).
        let on_line = RawNode::from_value(json!({
            "id": "l1", "type": "line", "points": [0, 0, 10, 10],
            "distort": {"type": "matrix", "matrix": [1, 0, 0, 0, 1, 0, 0.3, 0.1, 1]}
        }))
        .unwrap();
        assert_eq!(unknown_fields(&on_line), vec!["distort".to_string()]);
    }
}

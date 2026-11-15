/** Generated from vork-scene Rust input types. Run npm run generate; do not edit. */

export type Layer = TrackLayer | ClipLayer;
export type Blend = "replace" | "combine";
export type Interpolation = "STEP" | "LINEAR" | "CUBICSPLINE";
/**
 * Keyframes inline as `[time, value]` pairs, or two buffers for bulk data.
 */
export type Keys =
  | unknown[][]
  | {
      times: NameOr3;
      values: NameOr3;
    };
/**
 * A resource name or inline definition.
 */
export type NameOr3 = string | Buffer;
export type Buffer = InlineBuffer | FileBuffer | DenseBuffer | DenseFileBuffer;
/**
 * Inline numeric elements. Matrices contain 16 column-major components.
 */
export type InlineDataSchema = number[] | number[][];
export type BufferType = "f32" | "i32" | "u32" | "vec2" | "vec3" | "vec4" | "uvec4" | "mat4";
/**
 * A track path segment: a name, or a zero-based index (`["nodes", "title", "glyphs", 3, "opacity"]`).
 */
export type PathSegment = string | number;
/**
 * An ordered image-graph step. Resource names remain shorthand for an unnamed step.
 */
export type CompositeEntry = NameOr | CompositeNode;
/**
 * A resource name or inline definition.
 */
export type NameOr = string | Material;
/**
 * A material input value. Resources are explicit tagged objects; a plain string is
 * only ever an enum token of the program's reflected input type.
 */
export type Input =
  | number
  | number[]
  | boolean
  | GeometryBinding
  | TextureBinding
  | AttributeBinding
  | GridBinding
  | AovBinding
  | ChannelBinding
  | StepBinding
  | string;
/**
 * A resource name or inline definition.
 */
export type NameOr4 = string | Image;
/**
 * Blender colour space names, verbatim.
 */
export type ColorSpace = "sRGB" | "Linear Rec.709" | "Linear Rec.2020" | "ACEScg" | "Display P3" | "Non-Color";
export type MagFilter = "nearest" | "linear";
export type MinFilter =
  | "nearest"
  | "linear"
  | "nearest_mipmap_nearest"
  | "linear_mipmap_nearest"
  | "nearest_mipmap_linear"
  | "linear_mipmap_linear";
export type Background = [number, number, number] | BackgroundMode | string;
export type BackgroundMode = "environment" | "transparent";
export type Clock = number | string;
export type Wrap = "repeat" | "clamp_to_edge" | "mirrored_repeat";
export type Channel = "r" | "g" | "b" | "a" | "rgb";
export type Factor = number | number[];
export type KernelChannel = "color" | "depth" | "view_normal" | "object_index" | "motion" | "shadow_catcher";
export type Opacity = "coverage" | "transmittance";
/**
 * A resource name or inline definition.
 */
export type NameOr2 = string | Program;
export type Program = FileProgram | SourceProgram;
/**
 * Number of threads, independent of the internal workgroup size.
 */
export type ComputeDispatch = number | [number, number, number];
export type ComputeSchedule = ("frame" | "install") | "simulation";
export type CompositeResolution = [number, number] | number;
/**
 * `2d`: a canvas. `3d`: a closed world (on a node in a canvas).
 */
export type Space = "2d" | "3d";
export type Geometry = BufferGeometry | StdGeometry;
export type SubdivisionType = "catmull_clark" | "simple";
export type Topology = ("triangles" | "points" | "curves") | "implicit";
/**
 * Native deformation (including subdivision), geometry compute, and displacement.
 */
export type GeometryStage = "deform" | "compute" | "displace";
/**
 * Translation, rotation and scale (glTF node TRS), flat on the node. Rotation is
 * stored as a quaternion; an authored `{"euler": [x,y,z]}` converts losslessly.
 */
export type Node = TransformNode | MeshNode | AssetNode | CameraNode | LightNode | TextNode | PathNode;
/**
 * An animation layer owned by a node: tracks on this node's clock, paths starting at the node.
 * Like a scene layer it keeps its own clock: source time = offset + node time * speed.
 */
export type NodeLayer = NodeTrackLayer | NodeClipLayer;
/**
 * An authored rotation: a glTF quaternion or a tagged Euler triple.
 */
export type Rotation =
  | [number, number, number, number]
  | {
      /**
       * Radians about X, then Y, then Z (fixed parent axes; Blender XYZ).
       *
       * @minItems 3
       * @maxItems 3
       */
      euler: [number, number, number];
    };
/**
 * Aiming references a node origin or a fixed point in world coordinates.
 */
export type AimTarget = string | [number, number, number];
/**
 * How sound follows this node's clock speed.
 */
export type Pitch = "follow" | "keep";
/**
 * A resource name or inline definition.
 */
export type NameOr5 = string | Geometry;
export type LightShape = "sun" | "point" | "spot" | "rectangle" | "ellipse" | "environment";
/**
 * `content`: a string, or styled runs laid out as one text.
 */
export type Content = string | Run[];
/**
 * `fill` and `stroke` (§7): `"none"`, a CSS colour, a CSS gradient function, or an image.
 */
export type Paint = NoPaint | string | ImagePaint;
/**
 * The string `"none"` (SVG `fill="none"`): nothing is painted.
 */
export type NoPaint = "none";
/**
 * CSS `object-fit`.
 */
export type Fit = "contain" | "cover" | "fill" | "none";
/**
 * `font`: a font file (.ttf/.otf), or a list of files.
 */
export type FontFiles = string | string[];
/**
 * CSS `font-style`.
 */
export type FontStyle = "normal" | "italic" | "oblique";
/**
 * CSS `font-weight`: a number from 1 to 1000, `normal` (400) or `bold` (700).
 */
export type FontWeight = FontWeightKeyword | number;
export type FontWeightKeyword = "normal" | "bold";
/**
 * CSS `direction`.
 */
export type Direction = "ltr" | "rtl";
/**
 * CSS `line-break`.
 */
export type LineBreak = "auto" | "loose" | "normal" | "strict" | "anywhere";
/**
 * CSS `line-height`: `normal` (the font's own line metrics) or a multiple of the font size.
 */
export type LineHeight = LineHeightNormal | number;
export type LineHeightNormal = "normal";
/**
 * CSS `text-align`.
 */
export type TextAlign = "start" | "end" | "left" | "right" | "center" | "justify";
/**
 * CSS `white-space`.
 */
export type WhiteSpace = "normal" | "nowrap" | "pre" | "pre-wrap" | "pre-line" | "break-spaces";
/**
 * CSS `word-break`.
 */
export type WordBreak = "normal" | "break-all" | "keep-all" | "break-word";
/**
 * SVG `fill-rule`.
 */
export type FillRule = "nonzero" | "evenodd";
/**
 * SVG `stroke-dasharray`: `"none"` (solid) or the dash and gap lengths in px; an odd count repeats.
 */
export type DashArray = NoDash | number[];
/**
 * The string `"none"`: a solid stroke.
 */
export type NoDash = "none";
/**
 * SVG `stroke-linecap`.
 */
export type LineCap = "butt" | "round" | "square";
/**
 * SVG `stroke-linejoin`; SVG 2's `miter-clip` and `arcs` are refused as not built.
 */
export type LineJoin = "miter" | "round" | "bevel";
export type AovType = "value" | "color";
/**
 * Output rate: decimal number or reduced rational string (frames per second).
 */
export type Time = number | string;
export type MotionBlurPosition = "start" | "center" | "end";

export interface Scene {
  /**
   * Editor JSON Schema address; metadata only, never fetched by the renderer.
   */
  $schema?: string | null;
  animation?: Layer[];
  buffers?: {
    [k: string]: Buffer;
  };
  /**
   * `{space: "2d"}`: the document is a 2D canvas. Absent: a 3D world.
   */
  composition?: Composition | null;
  geometry?: {
    [k: string]: Geometry;
  };
  /**
   * Ordered geometry lowerings; compute contains ordinary geometry programs.
   */
  geometry_stages?: GeometryStage[];
  images?: {
    [k: string]: Image;
  };
  materials?: {
    [k: string]: Material;
  };
  nodes: Node[];
  programs?: {
    [k: string]: Program;
  };
  /**
   * Authored straight segments inside a world-space box.
   */
  propagation?: Propagation | null;
  render?: Render;
  /**
   * Fixed-clock retained compute state, independent of output FPS.
   */
  simulation?: Simulation | null;
  version: 2;
}
export interface TrackLayer {
  blend?: Blend;
  clip?: null;
  loop?: boolean;
  name?: string | null;
  offset?: number;
  speed?: number;
  tracks: Track[];
  weight?: number;
}
/**
 * A keyed property.
 */
export interface Track {
  interpolation?: Interpolation | null;
  keys: Keys;
  /**
   * Segments naming the keyed property, e.g. ["nodes","Bone.001","rotation"].
   */
  path: PathSegment[];
}
export interface InlineBuffer {
  count?: number | null;
  data: InlineDataSchema;
  dims?: null;
  file?: null;
  /**
   * OpenVDB/NanoVDB field name; buffer.type describes voxel values.
   */
  grid?: string | null;
  offset?: number;
  type: BufferType;
}
export interface FileBuffer {
  count?: number | null;
  data?: null;
  dims?: null;
  file: string;
  /**
   * OpenVDB/NanoVDB field name; buffer.type describes voxel values.
   */
  grid?: string | null;
  offset?: number;
  type: BufferType;
}
export interface DenseBuffer {
  count?: null;
  data?: InlineDataSchema | null;
  /**
   * Dense typed array, x contiguous. Initially zero unless data or a binary file is supplied.
   * Field consumers interpret cell centres in local [-1,1]³.
   *
   * @minItems 3
   * @maxItems 3
   */
  dims: [number, number, number];
  file?: null;
  grid?: null;
  offset?: number;
  type: BufferType;
}
export interface DenseFileBuffer {
  count?: number | null;
  data?: null;
  /**
   * Dense typed array, x contiguous. Initially zero unless data or a binary file is supplied.
   * Field consumers interpret cell centres in local [-1,1]³.
   *
   * @minItems 3
   * @maxItems 3
   */
  dims: [number, number, number];
  file: string;
  grid?: null;
  offset?: number;
  type: BufferType;
}
export interface ClipLayer {
  blend?: Blend;
  clip: Clip;
  loop?: boolean;
  name?: string | null;
  offset?: number;
  speed?: number;
  tracks?: null;
  weight?: number;
}
export interface Clip {
  file?: string | null;
  index?: number | null;
  name?: string | null;
  target: string;
}
/**
 * `composition`: on the scene, `{space: "2d"}` makes the document a canvas; on a node, a canvas or a world (compose.rs).
 */
export interface Composition {
  /**
   * Working RGB encoding; omitted: Linear Rec.709 for worlds, sRGB for canvases.
   */
  color_space?: "Linear Rec.709" | "Linear Rec.2020" | "ACEScg" | "sRGB" | "Display P3";
  /**
   * This composition's image graph, run in its working space before its view.
   */
  compositor?: CompositeEntry[];
  /**
   * SVG 2 `preserveAspectRatio`; omitted means `xMidYMid meet`; unsupported on a canvas inside a canvas.
   */
  preserveAspectRatio?: string | null;
  space: Space;
  /**
   * A resource name or inline definition.
   */
  view?: string | Material;
  /**
   * SVG 2 `viewBox`: min-x, min-y, width, height of the canvas's own coordinates; unsupported on a canvas inside a canvas.
   *
   * @minItems 4
   * @maxItems 4
   */
  viewBox?: [number, number, number, number] | null;
}
/**
 * A program instance. Its surface entry shades the assigned geometry.
 */
export interface Material {
  double_sided?: boolean;
  inputs?: {
    [k: string]: Input;
  };
  /**
   * Meaning of geometry_opacity; the renderer owns its integration method.
   */
  opacity?: Opacity | null;
  /**
   * Highest priority owns an overlapping dielectric region. Equal priorities
   * use the most recently entered solid; simultaneous entries use solid order.
   */
  priority?: number;
  program?: NameOr2;
}
/**
 * Read-only evaluated geometry of a named mesh object, sampled by the compute clock.
 */
export interface GeometryBinding {
  geometry: string;
}
/**
 * An image handle. Sampling and transformations belong to the receiving program.
 */
export interface TextureBinding {
  image: NameOr4;
}
export interface Image {
  /**
   * Explicit input encoding; omitted uses embedded colour metadata, then the format fallback.
   */
  color_space?: ColorSpace | null;
  /**
   * A rendered image's composition: its image graph, run on the capture before it is published.
   */
  composition?: ImageComposition | null;
  file?: string | null;
  mag_filter?: MagFilter;
  min_filter?: MinFilter;
  render?: ImageRender | null;
  /**
   * Seconds into media; omitted follows scene time. EXR sequences honor
   * framesPerSecond metadata; other sequences use render.fps.
   */
  time?: Clock | null;
  wrap_s?: Wrap;
  wrap_t?: Wrap;
}
/**
 * `images.<name>.composition`: the image graph of a rendered image.
 */
export interface ImageComposition {
  compositor?: CompositeEntry[];
}
/**
 * Settings that a nested image may author. The parent owns transport and time;
 * a capture has a closed shutter and runs the automatic per-step compositor.
 */
export interface ImageRender {
  background?: Background;
  camera?: string | null;
  /**
   * @minItems 2
   * @maxItems 2
   */
  resolution?: [number, number];
  root?: string | null;
}
/**
 * Install-resolved named float vertex data. Custom programs declare Attribute<float>
 * through Attribute<float4>; read_attribute returns value, dx and dy. Missing or
 * incompatible streams on any bound geometry fail install.
 */
export interface AttributeBinding {
  attribute: string;
  channel?: Channel | null;
  factor?: Factor | null;
}
/**
 * A volume grid buffer for a FloatGrid or VectorGrid program input.
 */
export interface GridBinding {
  grid: NameOr3;
}
/**
 * A render.aovs entry: an Aov handle for surface programs, or the AOV image in compositor programs.
 */
export interface AovBinding {
  aov: string;
}
/**
 * A kernel image channel read by a compositor program.
 */
export interface ChannelBinding {
  channel: KernelChannel;
}
/**
 * An earlier named compositor entry, local to this graph.
 */
export interface StepBinding {
  step: string;
}
export interface FileProgram {
  bindings?: {
    [k: string]: NameOr3;
  };
  dispatch?: ComputeDispatch | null;
  file: string;
  schedule?: ComputeSchedule | null;
  source?: null;
  source_origin?: null;
}
export interface SourceProgram {
  bindings?: {
    [k: string]: NameOr3;
  };
  dispatch?: ComputeDispatch | null;
  file?: null;
  schedule?: ComputeSchedule | null;
  source: string;
  source_origin?: string | null;
}
export interface CompositeNode {
  material: NameOr;
  name?: string | null;
  resolution?: CompositeResolution | null;
}
export interface BufferGeometry {
  attributes: {
    [k: string]: NameOr3;
  };
  /**
   * Live point prefix within the installed POSITION/width capacity.
   */
  count?: NameOr3 | null;
  /**
   * Endpoint vertex indices and OpenSubdiv/USD edge sharpness. Used by std subdivision.
   */
  crease_edge?: EdgeCrease[];
  /**
   * USD-style point counts, one u32 per polyline curve.
   */
  curve_vertex_counts?: NameOr3 | null;
  /**
   * Corners per polygon in the index buffer (USD/OpenSubdiv cage); absent means triangles.
   */
  face_vertex_counts?: number[] | null;
  /**
   * Temporal point identities; a changed identity is a birth, even when a row is reused.
   */
  ids?: NameOr3 | null;
  indices?: NameOr3 | null;
  material_ranges?: MaterialRange[];
  morphs?: {
    [k: string]: Morph;
  };
  /**
   * Solver-owned sticky failure receipt, collected with evaluated geometry.
   */
  status?: GeometryStatus | null;
  /**
   * Blender Subdivision Surface, evaluated after deformation and before displacement.
   */
  subdivision?: SubdivisionSurface | null;
  topology?: Topology;
  /**
   * USD-style diameter in metres, one f32 per control point.
   */
  widths?: NameOr3 | null;
}
/**
 * An undirected cage edge. Crease is OpenSubdiv sharpness: 0 is smooth; 10 or greater is infinitely sharp.
 */
export interface EdgeCrease {
  crease: number;
  /**
   * @minItems 2
   * @maxItems 2
   */
  vertices: [number, number];
}
export interface MaterialRange {
  count: number;
  material_index: number;
  start: number;
}
export interface Morph {
  NORMAL?: NameOr3 | null;
  POSITION: NameOr3;
}
export interface GeometryStatus {
  buffer: NameOr3;
  message: string;
}
export interface SubdivisionSurface {
  render_levels?: number;
  subdivision_type?: SubdivisionType;
  use_limit_surface?: boolean;
}
/**
 * Authoring shorthand expanded by vork-std before kernel scene decoding.
 * Builder-specific options are validated by the selected standard-library builder.
 */
export interface StdGeometry {
  /**
   * Text font path, resolved relative to the scene.
   */
  font?: string | null;
  std: string;
  [k: string]: unknown;
}
export interface TransformNode {
  /**
   * When the node exists, in parent seconds. Off: no picture, no sound, no evaluation, children included.
   */
  active?: boolean | [Clock, Clock];
  /**
   * Layers evaluated on this node's clock; paths start at this node.
   */
  animation?: NodeLayer[];
  /**
   * Sound source and gain over this node's complete subtree mix.
   */
  audio?: Audio | null;
  children?: Node[];
  /**
   * 2D canvas only: the named node's shape clips this node, drawn in this node's coordinates (SVG
   * clip-path, userSpaceOnUse).
   */
  "clip-path"?: string | null;
  /**
   * `{space: "2d"}`: a canvas whose children paint to one picture; `{space: "3d"}`, in a canvas: a closed
   * 3D world rendered to one picture and placed in the canvas (END-STATE-API §7).
   */
  composition?: Composition | null;
  /**
   * 2D canvas only: CSS filter functions over the node's picture, in order.
   * Layer blur refuses sigma above 100 target pixels after accumulated transforms.
   */
  filter?: string | null;
  /**
   * Cycles holdout: transparent black on transparent film; opaque black otherwise.
   */
  is_holdout?: boolean;
  /**
   * Cycles shadow catcher. Combined treats this surface as a holdout.
   */
  is_shadow_catcher?: boolean;
  light_link_tags?: string[];
  /**
   * 2D canvas only: the named node's picture masks this node by its luminance, drawn in this node's
   * coordinates (SVG mask, userSpaceOnUse).
   */
  mask?: string | null;
  name?: string;
  /**
   * 2D canvas only: the node's whole picture, children included, placed once (CSS opacity).
   */
  opacity?: number;
  pass_index?: number | null;
  rotation?: Rotation;
  /**
   * @minItems 3
   * @maxItems 3
   */
  scale?: [number, number, number];
  target?: AimTarget | null;
  /**
   * The clock for node-owned animation and media: local = offset + (parent - start) * speed, wrapped or held in the trimmed section.
   */
  time?: NodeClock | null;
  /**
   * @minItems 3
   * @maxItems 3
   */
  translation?: [number, number, number];
  type?: "node";
  visible?: boolean;
  visible_camera?: boolean;
  visible_diffuse?: boolean;
  visible_glossy?: boolean;
  visible_shadow?: boolean;
  visible_transmission?: boolean;
}
export interface NodeTrackLayer {
  blend?: Blend;
  clip?: null;
  loop?: boolean;
  name?: string | null;
  offset?: number;
  speed?: number;
  tracks: Track[];
  weight?: number;
}
export interface NodeClipLayer {
  blend?: Blend;
  clip: Clip;
  loop?: boolean;
  name?: string | null;
  offset?: number;
  speed?: number;
  tracks?: null;
  weight?: number;
}
/**
 * Sound attached to any node. Omit file for a group gain.
 */
export interface Audio {
  file?: string | null;
  gain?: number;
}
/**
 * The node clock: where in the content to begin, how much to use, how fast,
 * whether to repeat, and when on the parent's clock it starts.
 */
export interface NodeClock {
  /**
   * Content seconds used; omitted: the rest of the one source after `offset`.
   */
  duration?: Clock | null;
  loop?: boolean;
  offset?: Clock;
  pitch?: Pitch;
  speed?: Clock;
  /**
   * Parent seconds at which local time is `offset`; omitted: the window's start, else 0.
   */
  start?: Clock | null;
}
export interface MeshNode {
  /**
   * When the node exists, in parent seconds. Off: no picture, no sound, no evaluation, children included.
   */
  active?: boolean | [Clock, Clock];
  /**
   * Layers evaluated on this node's clock; paths start at this node.
   */
  animation?: NodeLayer[];
  /**
   * Sound source and gain over this node's complete subtree mix.
   */
  audio?: Audio | null;
  children?: Node[];
  /**
   * 2D canvas only: the named node's shape clips this node, drawn in this node's coordinates (SVG
   * clip-path, userSpaceOnUse).
   */
  "clip-path"?: string | null;
  /**
   * `{space: "2d"}`: a canvas whose children paint to one picture; `{space: "3d"}`, in a canvas: a closed
   * 3D world rendered to one picture and placed in the canvas (END-STATE-API §7).
   */
  composition?: Composition | null;
  /**
   * 2D canvas only: CSS filter functions over the node's picture, in order.
   * Layer blur refuses sigma above 100 target pixels after accumulated transforms.
   */
  filter?: string | null;
  geometry: NameOr5;
  instances?: Instances | null;
  /**
   * Cycles holdout: transparent black on transparent film; opaque black otherwise.
   */
  is_holdout?: boolean;
  /**
   * Cycles shadow catcher. Combined treats this surface as a holdout.
   */
  is_shadow_catcher?: boolean;
  light_link_tags?: string[];
  /**
   * 2D canvas only: the named node's picture masks this node by its luminance, drawn in this node's
   * coordinates (SVG mask, userSpaceOnUse).
   */
  mask?: string | null;
  materials?: NameOr[];
  morph_weights?: {
    [k: string]: number;
  };
  name?: string;
  /**
   * 2D canvas only: the node's whole picture, children included, placed once (CSS opacity).
   */
  opacity?: number;
  pass_index?: number | null;
  rotation?: Rotation;
  /**
   * @minItems 3
   * @maxItems 3
   */
  scale?: [number, number, number];
  skin?: Skin | null;
  target?: AimTarget | null;
  /**
   * The clock for node-owned animation and media: local = offset + (parent - start) * speed, wrapped or held in the trimmed section.
   */
  time?: NodeClock | null;
  /**
   * @minItems 3
   * @maxItems 3
   */
  translation?: [number, number, number];
  type: "mesh";
  visible?: boolean;
  visible_camera?: boolean;
  visible_diffuse?: boolean;
  visible_glossy?: boolean;
  visible_shadow?: boolean;
  visible_transmission?: boolean;
}
export interface Instances {
  attributes: {
    [k: string]: NameOr3;
  };
  count?: NameOr3 | null;
}
export interface Skin {
  inverse_bind_matrices: NameOr3;
  joints: string[];
}
export interface AssetNode {
  /**
   * When the node exists, in parent seconds. Off: no picture, no sound, no evaluation, children included.
   */
  active?: boolean | [Clock, Clock];
  /**
   * Layers evaluated on this node's clock; paths start at this node.
   */
  animation?: NodeLayer[];
  /**
   * Sound source and gain over this node's complete subtree mix.
   */
  audio?: Audio | null;
  children?: Node[];
  /**
   * 2D canvas only: the named node's shape clips this node, drawn in this node's coordinates (SVG
   * clip-path, userSpaceOnUse).
   */
  "clip-path"?: string | null;
  /**
   * `{space: "2d"}`: a canvas whose children paint to one picture; `{space: "3d"}`, in a canvas: a closed
   * 3D world rendered to one picture and placed in the canvas (END-STATE-API §7).
   */
  composition?: Composition | null;
  file: string;
  /**
   * 2D canvas only: CSS filter functions over the node's picture, in order.
   * Layer blur refuses sigma above 100 target pixels after accumulated transforms.
   */
  filter?: string | null;
  /**
   * Cycles holdout: transparent black on transparent film; opaque black otherwise.
   */
  is_holdout?: boolean;
  /**
   * Cycles shadow catcher. Combined treats this surface as a holdout.
   */
  is_shadow_catcher?: boolean;
  light_link_tags?: string[];
  /**
   * 2D canvas only: the named node's picture masks this node by its luminance, drawn in this node's
   * coordinates (SVG mask, userSpaceOnUse).
   */
  mask?: string | null;
  materials?: {
    [k: string]: NameOr;
  };
  name?: string;
  nodes?: {
    [k: string]: AssetNodeOverrides;
  };
  /**
   * 2D canvas only: the node's whole picture, children included, placed once (CSS opacity).
   */
  opacity?: number;
  pass_index?: number | null;
  rotation?: Rotation;
  /**
   * @minItems 3
   * @maxItems 3
   */
  scale?: [number, number, number];
  target?: AimTarget | null;
  /**
   * The clock for node-owned animation and media: local = offset + (parent - start) * speed, wrapped or held in the trimmed section.
   */
  time?: NodeClock | null;
  /**
   * @minItems 3
   * @maxItems 3
   */
  translation?: [number, number, number];
  type: "asset";
  visible?: boolean;
  visible_camera?: boolean;
  visible_diffuse?: boolean;
  visible_glossy?: boolean;
  visible_shadow?: boolean;
  visible_transmission?: boolean;
}
/**
 * Visibility is ANDed with ancestors. Reflections and diffuse transport query
 * scene geometry independently of camera visibility. Diffuse visibility controls AO
 * and indirect occlusion; specular visibility controls reflected appearance.
 */
export interface AssetNodeOverrides {
  light_link_tags?: string[] | null;
  pass_index?: number | null;
  visible?: boolean;
  visible_camera?: boolean;
  visible_diffuse?: boolean;
  visible_glossy?: boolean;
  visible_shadow?: boolean;
  visible_transmission?: boolean;
}
export interface CameraNode {
  /**
   * When the node exists, in parent seconds. Off: no picture, no sound, no evaluation, children included.
   */
  active?: boolean | [Clock, Clock];
  /**
   * Layers evaluated on this node's clock; paths start at this node.
   */
  animation?: NodeLayer[];
  /**
   * Sound source and gain over this node's complete subtree mix.
   */
  audio?: Audio | null;
  children?: Node[];
  /**
   * 2D canvas only: the named node's shape clips this node, drawn in this node's coordinates (SVG
   * clip-path, userSpaceOnUse).
   */
  "clip-path"?: string | null;
  /**
   * `{space: "2d"}`: a canvas whose children paint to one picture; `{space: "3d"}`, in a canvas: a closed
   * 3D world rendered to one picture and placed in the canvas (END-STATE-API §7).
   */
  composition?: Composition | null;
  /**
   * 2D canvas only: CSS filter functions over the node's picture, in order.
   * Layer blur refuses sigma above 100 target pixels after accumulated transforms.
   */
  filter?: string | null;
  focus_object?: string | null;
  /**
   * Cycles holdout: transparent black on transparent film; opaque black otherwise.
   */
  is_holdout?: boolean;
  /**
   * Cycles shadow catcher. Combined treats this surface as a holdout.
   */
  is_shadow_catcher?: boolean;
  light_link_tags?: string[];
  /**
   * 2D canvas only: the named node's picture masks this node by its luminance, drawn in this node's
   * coordinates (SVG mask, userSpaceOnUse).
   */
  mask?: string | null;
  material?: NameOr | null;
  name?: string;
  /**
   * 2D canvas only: the node's whole picture, children included, placed once (CSS opacity).
   */
  opacity?: number;
  pass_index?: number | null;
  rotation?: Rotation;
  /**
   * @minItems 3
   * @maxItems 3
   */
  scale?: [number, number, number];
  target?: AimTarget | null;
  /**
   * The clock for node-owned animation and media: local = offset + (parent - start) * speed, wrapped or held in the trimmed section.
   */
  time?: NodeClock | null;
  /**
   * @minItems 3
   * @maxItems 3
   */
  translation?: [number, number, number];
  type: "camera";
  visible?: boolean;
  visible_camera?: boolean;
  visible_diffuse?: boolean;
  visible_glossy?: boolean;
  visible_shadow?: boolean;
  visible_transmission?: boolean;
}
export interface LightNode {
  /**
   * When the node exists, in parent seconds. Off: no picture, no sound, no evaluation, children included.
   */
  active?: boolean | [Clock, Clock];
  angle?: number;
  /**
   * Layers evaluated on this node's clock; paths start at this node.
   */
  animation?: NodeLayer[];
  /**
   * Sound source and gain over this node's complete subtree mix.
   */
  audio?: Audio | null;
  children?: Node[];
  /**
   * 2D canvas only: the named node's shape clips this node, drawn in this node's coordinates (SVG
   * clip-path, userSpaceOnUse).
   */
  "clip-path"?: string | null;
  /**
   * None: LightShape::default_color (white, or Blender's world colour for an
   * environment without an image).
   *
   * @minItems 3
   * @maxItems 3
   */
  color?: [number, number, number] | null;
  /**
   * `{space: "2d"}`: a canvas whose children paint to one picture; `{space: "3d"}`, in a canvas: a closed
   * 3D world rendered to one picture and placed in the canvas (END-STATE-API §7).
   */
  composition?: Composition | null;
  energy?: number | null;
  /**
   * 2D canvas only: CSS filter functions over the node's picture, in order.
   * Layer blur refuses sigma above 100 target pixels after accumulated transforms.
   */
  filter?: string | null;
  illuminates?: string[] | null;
  image?: NameOr4 | null;
  /**
   * Cycles holdout: transparent black on transparent film; opaque black otherwise.
   */
  is_holdout?: boolean;
  /**
   * Cycles shadow catcher. Combined treats this surface as a holdout.
   */
  is_shadow_catcher?: boolean;
  light_link_tags?: string[];
  /**
   * 2D canvas only: the named node's picture masks this node by its luminance, drawn in this node's
   * coordinates (SVG mask, userSpaceOnUse).
   */
  mask?: string | null;
  /**
   * Environment radiance material instance.
   */
  material?: NameOr | null;
  /**
   * Physical world fog material on an environment light.
   */
  medium?: NameOr | null;
  name?: string;
  /**
   * 2D canvas only: the node's whole picture, children included, placed once (CSS opacity).
   */
  opacity?: number;
  pass_index?: number | null;
  radius?: number;
  rotation?: Rotation;
  /**
   * @minItems 3
   * @maxItems 3
   */
  scale?: [number, number, number];
  shadowed_by?: string[] | null;
  shape: LightShape;
  size?: number;
  size_y?: number | null;
  spot_blend?: number;
  spot_size?: number;
  target?: AimTarget | null;
  /**
   * None: `color` alone. Some(kelvin): `color` × std.color blackbody(kelvin).
   */
  temperature?: number | null;
  /**
   * The clock for node-owned animation and media: local = offset + (parent - start) * speed, wrapped or held in the trimmed section.
   */
  time?: NodeClock | null;
  /**
   * @minItems 3
   * @maxItems 3
   */
  translation?: [number, number, number];
  type: "light";
  visible?: boolean;
  visible_camera?: boolean;
  visible_diffuse?: boolean;
  visible_glossy?: boolean;
  visible_shadow?: boolean;
  visible_transmission?: boolean;
}
/**
 * 2D canvas only (END-STATE-API §7, text): CSS's names for the properties of text; the origin
 * is the top-left of the text box; the glyph outlines take SVG's fill, stroke and stroke-width.
 */
export interface TextNode {
  /**
   * When the node exists, in parent seconds. Off: no picture, no sound, no evaluation, children included.
   */
  active?: boolean | [Clock, Clock];
  /**
   * Layers evaluated on this node's clock; paths start at this node.
   */
  animation?: NodeLayer[];
  /**
   * Sound source and gain over this node's complete subtree mix.
   */
  audio?: Audio | null;
  children?: Node[];
  /**
   * 2D canvas only: the named node's shape clips this node, drawn in this node's coordinates (SVG
   * clip-path, userSpaceOnUse).
   */
  "clip-path"?: string | null;
  /**
   * `{space: "2d"}`: a canvas whose children paint to one picture; `{space: "3d"}`, in a canvas: a closed
   * 3D world rendered to one picture and placed in the canvas (END-STATE-API §7).
   */
  composition?: Composition | null;
  content: Content;
  direction?: Direction;
  fill?: Paint;
  /**
   * 2D canvas only: CSS filter functions over the node's picture, in order.
   * Layer blur refuses sigma above 100 target pixels after accumulated transforms.
   */
  filter?: string | null;
  font: FontFiles;
  "font-feature-settings"?: string;
  "font-size"?: number;
  "font-style"?: FontStyle;
  "font-variation-settings"?: string;
  "font-weight"?: FontWeight;
  /**
   * Cycles holdout: transparent black on transparent film; opaque black otherwise.
   */
  is_holdout?: boolean;
  /**
   * Cycles shadow catcher. Combined treats this surface as a holdout.
   */
  is_shadow_catcher?: boolean;
  "letter-spacing"?: number;
  light_link_tags?: string[];
  "line-break"?: LineBreak;
  "line-height"?: LineHeight;
  /**
   * 2D canvas only: the named node's picture masks this node by its luminance, drawn in this node's
   * coordinates (SVG mask, userSpaceOnUse).
   */
  mask?: string | null;
  name?: string;
  /**
   * 2D canvas only: the node's whole picture, children included, placed once (CSS opacity).
   */
  opacity?: number;
  pass_index?: number | null;
  rotation?: Rotation;
  /**
   * @minItems 3
   * @maxItems 3
   */
  scale?: [number, number, number];
  stroke?: Paint;
  "stroke-width"?: number;
  target?: AimTarget | null;
  "text-align"?: TextAlign;
  /**
   * The clock for node-owned animation and media: local = offset + (parent - start) * speed, wrapped or held in the trimmed section.
   */
  time?: NodeClock | null;
  /**
   * @minItems 3
   * @maxItems 3
   */
  translation?: [number, number, number];
  type: "text";
  visible?: boolean;
  visible_camera?: boolean;
  visible_diffuse?: boolean;
  visible_glossy?: boolean;
  visible_shadow?: boolean;
  visible_transmission?: boolean;
  "white-space"?: WhiteSpace;
  width?: number | null;
  "word-break"?: WordBreak;
  "word-spacing"?: number;
}
/**
 * One styled run of `content`: its text and the CSS and SVG properties it sets; where it sets none, the
 * node's apply.
 */
export interface Run {
  content: string;
  fill?: Paint | null;
  font?: FontFiles | null;
  "font-feature-settings"?: string | null;
  "font-size"?: number | null;
  "font-style"?: FontStyle | null;
  "font-variation-settings"?: string | null;
  "font-weight"?: FontWeight | null;
  "letter-spacing"?: number | null;
  stroke?: Paint | null;
}
/**
 * An image as a paint: bound as every image is (`{image: name}`), drawn in the shape's bounding box
 * and clipped by the shape.
 */
export interface ImagePaint {
  fit?: Fit | null;
  image: string;
}
/**
 * 2D canvas only (END-STATE-API §7, path): SVG 2's path data and paint properties, with SVG's
 * names and meanings; colours are CSS colours.
 */
export interface PathNode {
  /**
   * When the node exists, in parent seconds. Off: no picture, no sound, no evaluation, children included.
   */
  active?: boolean | [Clock, Clock];
  /**
   * Layers evaluated on this node's clock; paths start at this node.
   */
  animation?: NodeLayer[];
  /**
   * Sound source and gain over this node's complete subtree mix.
   */
  audio?: Audio | null;
  children?: Node[];
  /**
   * 2D canvas only: the named node's shape clips this node, drawn in this node's coordinates (SVG
   * clip-path, userSpaceOnUse).
   */
  "clip-path"?: string | null;
  /**
   * `{space: "2d"}`: a canvas whose children paint to one picture; `{space: "3d"}`, in a canvas: a closed
   * 3D world rendered to one picture and placed in the canvas (END-STATE-API §7).
   */
  composition?: Composition | null;
  d?: string;
  fill?: Paint;
  "fill-opacity"?: number;
  "fill-rule"?: FillRule;
  /**
   * 2D canvas only: CSS filter functions over the node's picture, in order.
   * Layer blur refuses sigma above 100 target pixels after accumulated transforms.
   */
  filter?: string | null;
  /**
   * Cycles holdout: transparent black on transparent film; opaque black otherwise.
   */
  is_holdout?: boolean;
  /**
   * Cycles shadow catcher. Combined treats this surface as a holdout.
   */
  is_shadow_catcher?: boolean;
  light_link_tags?: string[];
  /**
   * 2D canvas only: the named node's picture masks this node by its luminance, drawn in this node's
   * coordinates (SVG mask, userSpaceOnUse).
   */
  mask?: string | null;
  name?: string;
  /**
   * 2D canvas only: the node's whole picture, children included, placed once (CSS opacity).
   */
  opacity?: number;
  "paint-order"?: string;
  pass_index?: number | null;
  pathLength?: number | null;
  rotation?: Rotation;
  /**
   * @minItems 3
   * @maxItems 3
   */
  scale?: [number, number, number];
  stroke?: Paint;
  "stroke-dasharray"?: DashArray;
  "stroke-dashoffset"?: number;
  "stroke-linecap"?: LineCap;
  "stroke-linejoin"?: LineJoin;
  "stroke-miterlimit"?: number;
  "stroke-opacity"?: number;
  "stroke-width"?: number;
  target?: AimTarget | null;
  /**
   * The clock for node-owned animation and media: local = offset + (parent - start) * speed, wrapped or held in the trimmed section.
   */
  time?: NodeClock | null;
  /**
   * @minItems 3
   * @maxItems 3
   */
  translation?: [number, number, number];
  type: "path";
  visible?: boolean;
  visible_camera?: boolean;
  visible_diffuse?: boolean;
  visible_glossy?: boolean;
  visible_shadow?: boolean;
  visible_transmission?: boolean;
}
export interface Propagation {
  /**
   * World-space minimum and maximum in metres; rays leaving continue straight.
   *
   * @minItems 2
   * @maxItems 2
   */
  bounds: [[number, number, number], [number, number, number]];
  /**
   * Program exporting PropagationStep propagation(PropagationContext, Inputs).
   */
  program: string;
}
export interface Render {
  /**
   * Shader AOV declarations; color outputs contain scene-linear RGB.
   */
  aovs?: Aov[];
  background?: Background;
  camera?: string | null;
  duration?: Clock;
  fps?: Time;
  motion_blur?: MotionBlur | null;
  /**
   * @minItems 2
   * @maxItems 2
   */
  resolution?: [number, number];
  root?: string | null;
}
export interface Aov {
  name: string;
  type: AovType;
}
export interface MotionBlur {
  position?: MotionBlurPosition;
  shutter?: number;
}
/**
 * A deterministic fixed-clock schedule over ordinary typed named buffers.
 */
export interface Simulation {
  /**
   * Positive fixed tick duration, in seconds; independent of render FPS.
   */
  dt: number | string;
  /**
   * Ordered compute program names establishing S0.
   */
  init: string[];
  /**
   * Simulation randomness is independent of image sampling randomness.
   */
  seed: number;
  /**
   * Complete persistent state and scratch buffers used by these passes.
   */
  state: string[];
  /**
   * Ordered compute program names advancing Sn to Sn+1.
   */
  steps: string[];
  /**
   * Time of the initialized state S0, in seconds.
   */
  t0: number | string;
}

/**
 * Profile file contents.
 */
export interface ProfileFile {
  description?: string;
  /**
   * Quality/cost level, independent of the profile's name: 0 draft, 1 final, 2 reference, 3 previs (a separate blocking look).
   */
  level: number;
  settings?: Settings;
  /**
   * The profile's own revision, recorded in output metadata.
   */
  version: number;
}
/**
 * Sampling seed, sample budget and denoising intent.
 */
export interface Settings {
  /**
   * Enable optional scene-linear OIDN. Off retains profile-owned reconstruction.
   */
  denoise?: boolean;
  /**
   * Override the profile sample budget; more samples means cleaner and slower.
   */
  samples?: number;
  /**
   * Deterministic sampling seed, independent of the selected technique.
   */
  seed?: number;
}

/**
 * RGB encodings with pinned primaries and white points.
 */
export type WorkingColorSpace = "Linear Rec.709" | "Linear Rec.2020" | "ACEScg" | "sRGB" | "Display P3";
/**
 * Display encoding owned by a destination, resolved together with composition.view.
 */
export type OutputDisplay = "sRGB" | "Display P3" | "Rec.2100-PQ" | "Rec.2100-HLG";

/**
 * A render-call destination. Raw destinations bypass the view and have no default display.
 */
export interface OutputDestination {
  color_space?: WorkingColorSpace;
  display?: OutputDisplay;
  file: string;
  peak_luminance?: number;
  reference_white?: number;
}

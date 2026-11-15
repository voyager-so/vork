//! Drawing solids: one depth-tested, 4× multisampled pass into a pool texture (`Rgba16Float`, premultiplied,
//! sRGB-encoded like the renderer's captures), lit in linear light. Every solid of a [`SolidScene`] shares the pass's
//! depth buffer, so solids that cross cut each other per pixel; opaque solids draw first, then translucent ones far to
//! near, each blended over (premultiplied) at its nearest surface only (a depth-only pass of it first).
//!
//! Lighting, a metallic–roughness material (glTF's): base colour (times a texture), `metalness`, `roughness`,
//! `emissive` and `opacity`. Ambient lights add their colour, hemisphere lights the sky's from above and the ground's
//! from below; directional, point and spot lights add Cook–Torrance: Lambert diffuse (none on a metal) and the GGX
//! (Trowbridge–Reitz) highlight with Smith's height-correlated visibility and Schlick's Fresnel, whose size follows
//! `roughness` and whose colour follows `metalness`. Intensity 1 lights a white surface facing it to white (the
//! radiance is scaled by π, three.js's legacy convention). A spot light lights a cone that fades out over its
//! penumbra; a point or spot light with a range fades out at it (`(1 − (d / range)⁴)²`, Unreal's window), else doesn't
//! fall off. Emission adds its colour unlit. `Shading::Flat` uses each triangle's own normal; `Shading::Toon` bands
//! the diffuse into three steps and the highlight into one.
//!
//! Around the solids: with an [`Environment`](super::env::Environment), every surface reflects it (the split-sum
//! approximation: its level for the roughness along the reflected ray, times Karis's analytic environment BRDF) and is
//! lit by its irradiance (nine SH coefficients). Without one, a sky-over-ground made from the ambient light, by
//! Schlick's Fresnel (whole on metals, only its grazing rim on the rest).
//!
//! Shadows: each directional or spot light with `shadow` renders the scene's casters into a 2048² depth map
//! (orthographic over the casters' box, or the spot's perspective), and a receiving surface reads it with 16
//! percentage-closer taps over a disc `softness` px wide (a fixed Poisson pattern turned per pixel by interleaved
//! gradient noise: the same on every run). Reflection: a solid with `reflect` (a plane) mirrors the scene's other solids
//! drawn in its plane (a second pass, the camera mirrored, lit as they are), blurred by its roughness.

use std::cell::RefCell;
use std::sync::{Arc, Weak};

use crate::pixel::image::Storage;
use crate::pixel::pool::{PoolTexture, TexturePool};
use crate::pixel::{Domain, PixelError};

use super::env::Environment;
use super::Mesh;

/// At most this many lights shade a solid.
pub const MAX_LIGHTS: usize = 8;
/// At most this many lights cast shadows.
pub const MAX_SHADOWS: usize = 4;
/// A shadow map's side.
pub const SHADOW_SIZE: u32 = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
    Ambient,
    Hemisphere,
    Directional,
    Point,
    Spot,
}

/// A light in the frame's world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    pub kind: LightKind,
    /// Linear RGB times intensity (a hemisphere's from above).
    pub color: [f64; 3],
    /// Point and spot: where it is. Directional: the direction toward it (from the scene).
    pub vector: [f64; 3],
    /// Spot: the unit direction from its target toward it.
    pub axis: [f64; 3],
    /// Hemisphere: the colour from below (linear RGB times intensity).
    pub ground: [f64; 3],
    /// Spot: the cone's full width and the share of its half-angle its edge fades over (0–1).
    pub angle: f64,
    pub penumbra: f64,
    /// Point and spot: px at which it has faded out; None: no falloff.
    pub range: Option<f64>,
    /// Directional and spot: whether it casts shadows, and how wide their edge is (px).
    pub shadow: bool,
    pub softness: f64,
}

impl Light {
    /// A light with no cone, ground, range or shadow.
    pub fn simple(kind: LightKind, color: [f64; 3], vector: [f64; 3]) -> Light {
        Light { kind, color, vector, axis: [0.0, 0.0, -1.0], ground: color, angle: 180.0, penumbra: 0.0, range: None, shadow: false, softness: 0.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shading {
    Smooth,
    Flat,
    Toon,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    pub metalness: f64,
    pub roughness: f64,
    pub shading: Shading,
    pub opacity: f64,
    /// Light the surface gives off, unlit: linear RGB times intensity.
    pub emissive: [f64; 3],
    pub cast_shadow: bool,
    pub receive_shadow: bool,
    /// 0–1: how much of the mirror image of the scene's other solids a plane shows.
    pub reflect: f64,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            metalness: 0.0,
            roughness: 0.5,
            shading: Shading::Smooth,
            opacity: 1.0,
            emissive: [0.0; 3],
            cast_shadow: true,
            receive_shadow: true,
            reflect: 0.0,
        }
    }
}

/// One solid in a scene: its mesh, where it is and what it is made of.
#[derive(Clone, Debug)]
pub struct SolidItem<'a> {
    pub mesh: &'a Mesh,
    /// The solid's coordinates → the frame's world (row-major 4×4).
    pub model: [[f64; 4]; 4],
    pub material: Material,
}

/// The environment around a scene: the image, times `intensity`, turned `rotation` degrees about the vertical axis.
#[derive(Clone, Debug)]
pub struct EnvironmentLight {
    pub environment: Arc<Environment>,
    pub intensity: f64,
    pub rotation: f64,
}

/// Solids drawn together, and how they are seen.
#[derive(Clone, Debug)]
pub struct SolidScene<'a> {
    pub items: Vec<SolidItem<'a>>,
    /// Every solid that can cast a shadow on them (the frame's solids, these included; those with `cast_shadow`).
    pub casters: Vec<SolidItem<'a>>,
    /// World → homogeneous output pixel `(X, Y, W)` (the camera and the output's pixel map; W = depth).
    pub project: [[f64; 4]; 3],
    /// The camera's position in the world.
    pub eye: [f64; 3],
    pub lights: Vec<Light>,
    pub environment: Option<EnvironmentLight>,
}

const SHADER: &str = r#"
// color.w: the kind (0 ambient, 1 directional, 2 point, 3 spot, 4 hemisphere). v.w: a spot's cos of its inner cone.
// axis.w: a spot's cos of its outer cone. ground.w: the range (0: none). shadow: x its map's layer (−1: none),
// y softness px, z 1 for a perspective map, w the map's width in world px (perspective: per px of depth).
struct Light { color: vec4<f32>, v: vec4<f32>, axis: vec4<f32>, ground: vec4<f32>, shadow: vec4<f32> };
struct U {
    clip: mat4x4<f32>,
    mirror: mat4x4<f32>,
    shadow: array<mat4x4<f32>, 4>,
    eye: vec4<f32>,
    // x: lights, y: 1 in the reflection pass (linear output, the plane's far side cut).
    count: vec4<f32>,
    plane: vec4<f32>,
    // x: the environment's intensity, y: its turn (radians), z: its levels, w: 1 with an environment.
    env: vec4<f32>,
    sh: array<vec4<f32>, 9>,
    lights: array<Light, 8>,
};
struct Item {
    model: mat4x4<f32>,
    nrm: mat4x4<f32>,
    mat: vec4<f32>,
    emissive: vec4<f32>,
    // x: 1 receives shadows, y: reflect, z: 1 with a reflection image.
    flags: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var shadow_map: texture_depth_2d_array;
@group(0) @binding(2) var shadow_cmp: sampler_comparison;
@group(0) @binding(3) var env_map: texture_2d<f32>;
@group(0) @binding(4) var env_samp: sampler;
@group(0) @binding(5) var mirror_img: texture_2d<f32>;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;
@group(2) @binding(0) var<uniform> it: Item;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) col: vec4<f32>,
};
struct VOut {
    @invariant @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) col: vec4<f32>,
};

@vertex
fn vs(v: VIn) -> VOut {
    var o: VOut;
    let world = it.model * vec4<f32>(v.pos, 1.0);
    o.clip = u.clip * (u.mirror * world);
    o.world = world.xyz;
    o.nrm = (it.nrm * vec4<f32>(v.nrm, 0.0)).xyz;
    o.uv = v.uv;
    o.col = v.col;
    return o;
}

fn encode(l: f32) -> f32 {
    let c = max(l, 0.0);
    if (c <= 0.0031308) { return c * 12.92; }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

const PI: f32 = 3.14159265;

// How much of a light reaches `p`: a spot's cone and a range's window; and the unit vector toward the light.
fn toward(lt: Light, kind: u32, p: vec3<f32>) -> vec4<f32> {
    if (kind == 1u) { return vec4<f32>(normalize(lt.v.xyz), 1.0); }
    let d = lt.v.xyz - p;
    let dist = max(length(d), 1e-4);
    let l = d / dist;
    var k = 1.0;
    if (lt.ground.w > 0.0) {
        let x = clamp(1.0 - pow(dist / lt.ground.w, 4.0), 0.0, 1.0);
        k = x * x;
    }
    if (kind == 3u) {
        k = k * smoothstep(lt.axis.w, max(lt.v.w, lt.axis.w + 1e-5), dot(l, normalize(lt.axis.xyz)));
    }
    return vec4<f32>(l, k);
}

// GGX (Trowbridge–Reitz) times Smith's height-correlated visibility.
fn specular_dv(nh: f32, nl: f32, nv: f32, a: f32) -> f32 {
    let a2 = a * a;
    let f = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * f * f);
    let gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
    let gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
    return d * 0.5 / max(gv + gl, 1e-5);
}

var<private> POISSON: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216), vec2<f32>(0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870), vec2<f32>(0.34495938, 0.29387760),
    vec2<f32>(-0.91588581, 0.45771432), vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543, 0.27676845), vec2<f32>(0.97484398, 0.75648379),
    vec2<f32>(0.44323325, -0.97511554), vec2<f32>(0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023), vec2<f32>(0.79197514, 0.19090188),
    vec2<f32>(-0.24188840, 0.99706507), vec2<f32>(-0.81409955, 0.91437590),
    vec2<f32>(0.19984126, 0.78641367), vec2<f32>(0.14383161, -0.14100790)
);

// How lit `p` (normal `n`) is by light `k`'s shadow map: 1 in light, 0 in shadow.
fn shadowed(k: u32, p: vec3<f32>, n: vec3<f32>, l: vec3<f32>, frag: vec2<f32>) -> f32 {
    let sh = u.lights[k].shadow;
    if (sh.x < -0.5) { return 1.0; }
    let layer = i32(sh.x + 0.5);
    let m = u.shadow[layer];
    let persp = sh.z > 0.5;
    let depth = select(1.0, max((m * vec4<f32>(p, 1.0)).w, 1e-3), persp);
    let width = sh.w * depth;
    let texel = width / 2048.0;
    // Read from a little off the surface (out along its normal and toward the light), and compare against a depth
    // pulled toward the light by how far the surface's slope carries it across the taps' disc, so a surface doesn't
    // shadow itself.
    let q = p + (n + l) * texel;
    let c = m * vec4<f32>(q, 1.0);
    let ndc = c.xyz / c.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z > 1.0 || ndc.z < 0.0) { return 1.0; }
    let r = max(sh.y / width, 1.5 / 2048.0);
    let c1 = m * vec4<f32>(q + l, 1.0);
    let per_px = abs(ndc.z - c1.z / c1.w);
    let nl = clamp(dot(n, l), 0.05, 1.0);
    let slope = min(sqrt(1.0 - nl * nl) / nl, 10.0);
    let reference = ndc.z - per_px * (r * width * slope + texel);
    // Interleaved gradient noise (Jimenez 2014): a turn per pixel, the same on every run.
    let a = 6.2831853 * fract(52.9829189 * fract(dot(frag, vec2<f32>(0.06711056, 0.00583715))));
    let rot = mat2x2<f32>(cos(a), sin(a), -sin(a), cos(a));
    var lit = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let o = rot * POISSON[i] * r;
        lit = lit + textureSampleCompareLevel(shadow_map, shadow_cmp, uv + o, layer, reference);
    }
    return lit / 16.0;
}

// The environment along `d` (a world direction), at level `lod`.
fn env_at(d: vec3<f32>, lod: f32) -> vec3<f32> {
    let lon = atan2(d.x, d.z) - u.env.y;
    let lat = asin(clamp(-d.y, -1.0, 1.0));
    let uv = vec2<f32>(lon / 6.2831853 + 0.5, 0.5 - lat / PI);
    return textureSampleLevel(env_map, env_samp, uv, lod).rgb * u.env.x;
}

// The irradiance over π (the light a white Lambert surface reflects) facing `n`.
fn irradiance(n: vec3<f32>) -> vec3<f32> {
    var b = array<f32, 9>(
        0.282095, 0.488603 * n.y, 0.488603 * n.z, 0.488603 * n.x,
        1.092548 * n.x * n.y, 1.092548 * n.y * n.z, 0.315392 * (3.0 * n.z * n.z - 1.0),
        1.092548 * n.x * n.z, 0.546274 * (n.x * n.x - n.y * n.y));
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i = i + 1u) { e = e + u.sh[i].rgb * b[i]; }
    return max(e, vec3<f32>(0.0));
}

// Karis's analytic fit of the split sum's environment BRDF ("Physically Based Shading on Mobile", 2014).
fn env_brdf(f0: vec3<f32>, rough: f32, nv: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let reflection = u.count.y > 0.5;
    if (reflection && dot(u.plane.xyz, i.world) + u.plane.w < 0.0) { discard; }
    let texel = textureSample(tex, samp, i.uv);
    let face = normalize(cross(dpdx(i.world), dpdy(i.world)));
    let shading = u32(it.mat.z + 0.5);
    var n = normalize(i.nrm);
    if (shading == 1u) { n = face; }
    let v = normalize(u.eye.xyz - i.world);
    if (dot(n, v) < 0.0) { n = -n; }
    let base = i.col.rgb * texel.rgb;
    let alpha = i.col.a * texel.a * it.mat.w;
    let metal = clamp(it.mat.x, 0.0, 1.0);
    let rough = clamp(it.mat.y, 0.045, 1.0);
    let a = rough * rough;
    let shin = 2.0 / (rough * rough * rough * rough) - 2.0;
    let f0 = mix(vec3<f32>(0.04), base, metal);
    let nv = clamp(dot(n, v), 1e-4, 1.0);
    var sky = vec3<f32>(0.0);
    var ground = vec3<f32>(0.0);
    var diffuse = vec3<f32>(0.0);
    var spec = vec3<f32>(0.0);
    let count = u32(u.count.x + 0.5);
    for (var k = 0u; k < 8u; k = k + 1u) {
        if (k >= count) { break; }
        let lt = u.lights[k];
        let kind = u32(lt.color.w + 0.5);
        if (kind == 0u) { sky = sky + lt.color.rgb; ground = ground + lt.color.rgb; continue; }
        if (kind == 4u) { sky = sky + lt.color.rgb; ground = ground + lt.ground.rgb; continue; }
        let tl = toward(lt, kind, i.world);
        let l = tl.xyz;
        var radiance = lt.color.rgb * tl.w;
        if (it.flags.x > 0.5 && tl.w > 0.0) { radiance = radiance * shadowed(k, i.world, n, l, i.clip.xy); }
        let nl = clamp(dot(n, l), 0.0, 1.0);
        let h = normalize(l + v);
        if (shading == 2u) {
            var d = nl;
            var s = pow(max(dot(n, h), 0.0), shin) * (shin + 8.0) / 25.13274;
            d = select(select(0.2, 0.6, d > 0.15), 1.0, d > 0.6);
            s = select(0.0, 1.0, s > 0.5) * select(0.0, 1.0, nl > 0.0);
            diffuse = diffuse + radiance * d;
            spec = spec + radiance * s * f0;
            continue;
        }
        let vh = clamp(dot(v, h), 0.0, 1.0);
        let f = f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - vh, 0.0, 1.0), 5.0);
        // π × the BRDF × N·L: a light of 1 head on a white Lambert surface gives 1.
        diffuse = diffuse + radiance * nl * (vec3<f32>(1.0) - f);
        spec = spec + radiance * nl * PI * f * specular_dv(clamp(dot(n, h), 0.0, 1.0), nl, nv, a);
    }
    // The ambient light, from above (y is down) and below.
    let up = clamp(0.5 - 0.5 * n.y, 0.0, 1.0);
    let ambient = mix(ground, sky, up);
    let r = reflect(-v, n);
    var lit = vec3<f32>(0.0);
    if (u.env.w > 0.5) {
        // The environment: its irradiance, and its reflection by the split sum.
        let specular_env = env_at(r, rough * u.env.z) * env_brdf(f0, rough, nv);
        let kd = (vec3<f32>(1.0) - env_brdf(f0, rough, nv)) * (1.0 - metal);
        lit = base * (1.0 - metal) * (ambient + diffuse) + base * kd * irradiance(n) * u.env.x + specular_env + spec;
    } else {
        // A sky-over-ground environment made from the ambient light, reflected: what makes a metal read as metal.
        let env_sky = sky * 2.2 + vec3<f32>(0.12);
        let env_ground = ground * 0.35;
        let env = mix(env_ground, env_sky, clamp(0.5 - 0.5 * r.y, 0.0, 1.0)) * (1.0 - 0.6 * rough);
        let fres = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - nv, 5.0) * (1.0 - rough);
        // Metals reflect it whole; other surfaces only at a grazing angle (their head-on 4 % would grey every colour).
        let envw = mix(fres - f0, fres, metal);
        lit = base * (1.0 - metal) * (ambient + diffuse) + envw * env + spec;
    }
    lit = lit + it.emissive.rgb;
    if (it.flags.z > 0.5) {
        // The mirror image (linear, premultiplied), blurred by the roughness, over the surface.
        let size = vec2<f32>(textureDimensions(mirror_img));
        let radius = 40.0 * rough * rough;
        var m = vec4<f32>(0.0);
        for (var t = 0u; t < 16u; t = t + 1u) {
            let q = clamp(i.clip.xy + POISSON[t] * radius, vec2<f32>(0.0), size - vec2<f32>(1.0));
            m = m + textureLoad(mirror_img, vec2<i32>(q), 0);
        }
        m = m / 16.0 * it.flags.y;
        lit = lit * (1.0 - m.a) + m.rgb;
    }
    if (reflection) { return vec4<f32>(lit * alpha, alpha); }
    let c = vec3<f32>(encode(lit.r), encode(lit.g), encode(lit.b));
    return vec4<f32>(c * alpha, alpha);
}
"#;

/// The shadow maps' pass: depth only, through a light's matrix.
const SHADOW_SHADER: &str = r#"
struct S { m: mat4x4<f32> };
struct Item { model: mat4x4<f32> };
@group(0) @binding(0) var<uniform> s: S;
@group(1) @binding(0) var<uniform> it: Item;
@vertex
fn vs(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return s.m * (it.model * vec4<f32>(pos, 1.0));
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightData {
    color: [f32; 4],
    v: [f32; 4],
    axis: [f32; 4],
    ground: [f32; 4],
    shadow: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    clip: [[f32; 4]; 4],
    mirror: [[f32; 4]; 4],
    shadow: [[[f32; 4]; 4]; MAX_SHADOWS],
    eye: [f32; 4],
    count: [f32; 4],
    plane: [f32; 4],
    env: [f32; 4],
    sh: [[f32; 4]; 9],
    lights: [LightData; MAX_LIGHTS],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ItemData {
    model: [[f32; 4]; 4],
    nrm: [[f32; 4]; 4],
    mat: [f32; 4],
    emissive: [f32; 4],
    flags: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    nrm: [f32; 3],
    uv: [f32; 2],
    col: [f32; 4],
}

const SAMPLES: u32 = 4;
/// The nearest depth drawn, frame px in front of the camera. Depth is stored reversed (`NEAR / depth`: 1 at the near
/// plane, toward 0 far away), which keeps a float depth buffer's precision even across the frame's thousands of px.
const NEAR: f64 = 1.0;

type M4 = [[f64; 4]; 4];

const IDENTITY: M4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

fn to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// A column-major `mat4x4<f32>` from a row-major 4×4.
fn cols(m: &M4) -> [[f32; 4]; 4] {
    let mut o = [[0.0f32; 4]; 4];
    for (c, col) in o.iter_mut().enumerate() {
        for (r, v) in col.iter_mut().enumerate() {
            *v = m[r][c] as f32;
        }
    }
    o
}

/// The inverse transpose of `m`'s upper 3×3 (normals), as a 4×4.
fn normal_matrix(m: &M4) -> M4 {
    let a = |r: usize, c: usize| m[r][c];
    let cof = |r: usize, c: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
        a(r1, c1) * a(r2, c2) - a(r1, c2) * a(r2, c1)
    };
    let det = a(0, 0) * cof(0, 0) + a(0, 1) * cof(0, 1) + a(0, 2) * cof(0, 2);
    let s = if det.abs() > 1e-300 { 1.0 / det } else { 1.0 };
    let mut o = [[0.0; 4]; 4];
    for (r, row) in o.iter_mut().enumerate().take(3) {
        for (c, v) in row.iter_mut().enumerate().take(3) {
            // (M⁻¹)ᵀ[r][c] = cofactor(r, c) / det.
            *v = cof(r, c) * s;
        }
    }
    o[3][3] = 1.0;
    o
}

fn apply(m: &M4, p: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(a: [f64; 3]) -> Option<[f64; 3]> {
    let l = dot(a, a).sqrt();
    (l > 1e-12 && l.is_finite()).then(|| [a[0] / l, a[1] / l, a[2] / l])
}

/// A light's frame: forward `f` and two axes across it.
fn basis(f: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let r = unit(cross(f, [0.0, -1.0, 0.0])).or_else(|| unit(cross(f, [0.0, 0.0, 1.0]))).unwrap_or([1.0, 0.0, 0.0]);
    (r, cross(r, f))
}

/// The world corners of every caster's box.
fn caster_corners(casters: &[SolidItem<'_>]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for c in casters.iter().filter(|c| c.material.cast_shadow) {
        let Some((lo, hi)) = c.mesh.bounds() else { continue };
        for i in 0..8 {
            let p = [if i & 1 == 0 { lo[0] } else { hi[0] }, if i & 2 == 0 { lo[1] } else { hi[1] }, if i & 4 == 0 { lo[2] } else { hi[2] }];
            out.push(apply(&c.model, p));
        }
    }
    out
}

/// A shadow-casting light's world → map clip matrix (depth 0–1, the map's top row at clip y = 1) and the map's width
/// in world px (a spot's: per px of depth); None when no caster is in front of it.
pub fn shadow_matrix(light: &Light, corners: &[[f64; 3]]) -> Option<(M4, f64, bool)> {
    if corners.is_empty() {
        return None;
    }
    let soft = light.softness.max(0.0);
    match light.kind {
        LightKind::Directional => {
            let f = unit([-light.vector[0], -light.vector[1], -light.vector[2]])?;
            let (r, up) = basis(f);
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for p in corners {
                let q = [dot(*p, r), dot(*p, up), dot(*p, f)];
                for k in 0..3 {
                    lo[k] = lo[k].min(q[k]);
                    hi[k] = hi[k].max(q[k]);
                }
            }
            let ext = (hi[0] - lo[0]).max(hi[1] - lo[1]);
            let ext = ext * 1.04 + 2.0 * soft + 8.0;
            let (cx, cy) = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0);
            let (z0, z1) = (lo[2] - 2.0, hi[2] + 2.0);
            let k = 2.0 / ext;
            let row = |a: [f64; 3], s: f64, off: f64| [a[0] * s, a[1] * s, a[2] * s, off];
            Some((
                [row(r, k, -cx * k), row(up, k, -cy * k), row(f, 1.0 / (z1 - z0), -z0 / (z1 - z0)), [0.0, 0.0, 0.0, 1.0]],
                ext,
                false,
            ))
        }
        LightKind::Spot => {
            let p = light.vector;
            let f = unit([-light.axis[0], -light.axis[1], -light.axis[2]])?;
            let (r, up) = basis(f);
            let depths: Vec<f64> = corners.iter().map(|c| dot([c[0] - p[0], c[1] - p[1], c[2] - p[2]], f)).collect();
            let far = depths.iter().copied().fold(f64::NEG_INFINITY, f64::max) * 1.01 + 1.0;
            if far <= 1.0 {
                return None;
            }
            let near = depths.iter().copied().fold(f64::INFINITY, f64::min).clamp(1.0, far * 0.5) * 0.9;
            let half = (light.angle.clamp(1.0, 170.0) / 2.0 + 5.0).min(85.0).to_radians();
            let t = half.tan();
            let row = |a: [f64; 3], s: f64| [a[0] * s, a[1] * s, a[2] * s, -dot(a, p) * s];
            let w = row(f, 1.0);
            let k = far / (far - near);
            Some(([row(r, 1.0 / t), row(up, 1.0 / t), [w[0] * k, w[1] * k, w[2] * k, w[3] * k - near * k], w], 2.0 * t, true))
        }
        _ => None,
    }
}

/// The mirror across the plane `n · p + d = 0`.
fn mirror(n: [f64; 3], d: f64) -> M4 {
    let mut m = IDENTITY;
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] -= 2.0 * n[r] * n[c];
        }
        m[r][3] = -2.0 * d * n[r];
    }
    m
}

/// A reflector's plane in the world (its first triangle's), its normal toward `eye`: `(n, d)`.
fn plane_of(item: &SolidItem<'_>, eye: [f64; 3]) -> Option<([f64; 3], f64)> {
    let part = item.mesh.parts.iter().find(|p| p.indices.len() >= 3)?;
    let at = |i: u32| {
        let q = part.positions[i as usize];
        apply(&item.model, [f64::from(q[0]), f64::from(q[1]), f64::from(q[2])])
    };
    let (a, b, c) = (at(part.indices[0]), at(part.indices[1]), at(part.indices[2]));
    let mut n = unit(cross([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]))?;
    if dot(n, eye) - dot(n, a) < 0.0 {
        n = [-n[0], -n[1], -n[2]];
    }
    Some((n, -dot(n, a)))
}

/// The solid pipeline of one device (made on the first solid).
pub struct SolidPipeline {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    prepass: wgpu::RenderPipeline,
    translucent: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
    shadow_u_layout: wgpu::BindGroupLayout,
    scene_layout: wgpu::BindGroupLayout,
    texture_layout: wgpu::BindGroupLayout,
    item_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    env_sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    white: wgpu::TextureView,
    blank_shadow: wgpu::TextureView,
    blank_env: wgpu::TextureView,
    blank_mirror: wgpu::TextureView,
    textures: RefCell<Vec<(Weak<super::Texture>, wgpu::TextureView)>>,
    environments: RefCell<Vec<(Weak<Environment>, wgpu::TextureView)>>,
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType, dim: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension: dim, multisampled: false },
        count: None,
    }
}

fn sampler_entry(binding: u32, ty: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(ty), count: None }
}

impl SolidPipeline {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, PixelError> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("solid"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let shadow_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("solid shadow"), source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()) });
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let d2 = wgpu::TextureViewDimension::D2;
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("solid scene"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                texture_entry(1, wgpu::TextureSampleType::Depth, wgpu::TextureViewDimension::D2Array),
                sampler_entry(2, wgpu::SamplerBindingType::Comparison),
                texture_entry(3, float, d2),
                sampler_entry(4, wgpu::SamplerBindingType::Filtering),
                texture_entry(5, wgpu::TextureSampleType::Float { filterable: false }, d2),
            ],
        });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("solid texture"),
            entries: &[texture_entry(0, float, d2), sampler_entry(1, wgpu::SamplerBindingType::Filtering)],
        });
        let item_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("solid item"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT)],
        });
        let shadow_u_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("solid shadow"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("solid"),
            bind_group_layouts: &[Some(&scene_layout), Some(&texture_layout), Some(&item_layout)],
            ..Default::default()
        });
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("solid shadow"),
            bind_group_layouts: &[Some(&shadow_u_layout), Some(&item_layout)],
            ..Default::default()
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4];
        let buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &attributes,
        }];
        // Opaque solids: drawn and depth-written. Translucent ones: a depth-only pass of the solid, then its colour
        // where its depth is the one drawn (its nearest surface, blended once).
        let make = |label: &str, write_mask: wgpu::ColorWrites, depth_write: bool, compare: wgpu::CompareFunction| {
            moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &buffers },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: Storage::F16.format(),
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(compare),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState { count: SAMPLES, ..Default::default() },
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = make("solid", wgpu::ColorWrites::ALL, true, wgpu::CompareFunction::Greater);
        let prepass = make("solid depth", wgpu::ColorWrites::empty(), true, wgpu::CompareFunction::Greater);
        let translucent = make("solid translucent", wgpu::ColorWrites::ALL, false, wgpu::CompareFunction::Equal);
        let shadow_attr = wgpu::vertex_attr_array![0 => Float32x3];
        let shadow = moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
            label: Some("solid shadow"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shadow_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &shadow_attr,
                }],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("solid texture"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("solid environment"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("solid shadow"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(PixelError::Invalid(format!("the solid pipeline: {e}")));
        }
        let white = Self::upload(device, queue, &super::Texture { width: 1, height: 1, rgba: vec![255; 4] });
        let blank_shadow = Self::shadow_target(device, 1, 1).create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let blank_env = Self::upload_env(device, queue, &[(1, 1, vec![[0.0; 4]])]);
        let blank_mirror = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("solid mirror (none)"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Storage::F16.format(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            prepass,
            translucent,
            shadow,
            shadow_u_layout,
            scene_layout,
            texture_layout,
            item_layout,
            sampler,
            env_sampler,
            shadow_sampler,
            white,
            blank_shadow,
            blank_env,
            blank_mirror,
            textures: RefCell::new(Vec::new()),
            environments: RefCell::new(Vec::new()),
        })
    }

    fn shadow_target(device: &wgpu::Device, side: u32, layers: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("solid shadow map"),
            size: wgpu::Extent3d { width: side, height: side, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    fn upload(device: &wgpu::Device, queue: &wgpu::Queue, t: &super::Texture) -> wgpu::TextureView {
        let size = wgpu::Extent3d { width: t.width.max(1), height: t.height.max(1), depth_or_array_layers: 1 };
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("solid texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &t.rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * size.width), rows_per_image: None },
            size,
        );
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// An environment's levels as one mipmapped `Rgba16Float` texture.
    fn upload_env(device: &wgpu::Device, queue: &wgpu::Queue, levels: &[(u32, u32, Vec<[f32; 4]>)]) -> wgpu::TextureView {
        let (w, h) = (levels[0].0, levels[0].1);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("solid environment"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (i, (lw, lh, px)) in levels.iter().enumerate() {
            let bytes: Vec<u8> = px.iter().flatten().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: i as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &bytes,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8 * lw), rows_per_image: None },
                wgpu::Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
            );
        }
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn texture_view(&self, t: &Arc<super::Texture>) -> wgpu::TextureView {
        let mut cache = self.textures.borrow_mut();
        cache.retain(|(w, _)| w.strong_count() > 0);
        if let Some((_, v)) = cache.iter().find(|(w, _)| w.upgrade().is_some_and(|x| Arc::ptr_eq(&x, t))) {
            return v.clone();
        }
        let v = Self::upload(&self.device, &self.queue, t);
        cache.push((Arc::downgrade(t), v.clone()));
        v
    }

    fn env_view(&self, e: &Arc<Environment>) -> wgpu::TextureView {
        let mut cache = self.environments.borrow_mut();
        cache.retain(|(w, _)| w.strong_count() > 0);
        if let Some((_, v)) = cache.iter().find(|(w, _)| w.upgrade().is_some_and(|x| Arc::ptr_eq(&x, e))) {
            return v.clone();
        }
        let v = Self::upload_env(&self.device, &self.queue, &e.levels);
        cache.push((Arc::downgrade(e), v.clone()));
        v
    }

    fn buffer(&self, label: &str, contents: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        wgpu::util::DeviceExt::create_buffer_init(&self.device, &wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
    }

    fn item_bind(&self, item: &SolidItem<'_>, mirror_image: bool) -> wgpu::BindGroup {
        let m = &item.material;
        let data = ItemData {
            model: cols(&item.model),
            nrm: cols(&normal_matrix(&item.model)),
            mat: [
                m.metalness as f32,
                m.roughness as f32,
                match m.shading {
                    Shading::Smooth => 0.0,
                    Shading::Flat => 1.0,
                    Shading::Toon => 2.0,
                },
                m.opacity as f32,
            ],
            emissive: [m.emissive[0] as f32, m.emissive[1] as f32, m.emissive[2] as f32, 0.0],
            flags: [f32::from(u8::from(m.receive_shadow)), m.reflect.clamp(0.0, 1.0) as f32, f32::from(u8::from(mirror_image)), 0.0],
        };
        let buf = self.buffer("solid item", bytemuck::bytes_of(&data), wgpu::BufferUsages::UNIFORM);
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("solid item"),
            layout: &self.item_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        })
    }

    /// Each part's vertex and index buffers and its texture's bind group.
    fn part_draws(&self, mesh: &Mesh) -> Vec<(wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, u32)> {
        let mut parts = Vec::new();
        for part in mesh.parts.iter().filter(|p| !p.indices.is_empty()) {
            let lin = [to_linear(part.color[0]), to_linear(part.color[1]), to_linear(part.color[2]), part.color[3]];
            let verts: Vec<Vertex> = (0..part.positions.len())
                .map(|i| Vertex { pos: part.positions[i], nrm: part.normals[i], uv: part.uvs.get(i).copied().unwrap_or([0.0, 0.0]), col: lin })
                .collect();
            let vb = self.buffer("solid vertices", bytemuck::cast_slice(&verts), wgpu::BufferUsages::VERTEX);
            let ib = self.buffer("solid indices", bytemuck::cast_slice(&part.indices), wgpu::BufferUsages::INDEX);
            let tview = match &part.texture {
                Some(t) if !part.uvs.is_empty() => self.texture_view(t),
                _ => self.white.clone(),
            };
            let tbind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("solid texture"),
                layout: &self.texture_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&tview) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            parts.push((vb, ib, tbind, part.indices.len() as u32));
        }
        parts
    }

    /// Draw `scene` over the output pixels `out` into a fresh pool texture.
    pub fn draw(&self, pool: &TexturePool, scene: &SolidScene<'_>, out: Domain) -> Result<PoolTexture, PixelError> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("solid") });
        // World → output clip space for this domain, depth reversed (NEAR / W).
        let (ow, oh) = (f64::from(out.w), f64::from(out.h));
        let p = &scene.project;
        let (dx, dy) = (f64::from(out.x), f64::from(out.y));
        let mut clip = [[0.0; 4]; 4];
        for c in 0..4 {
            let xr = p[0][c] - dx * p[2][c];
            let yr = p[1][c] - dy * p[2][c];
            clip[0][c] = 2.0 * xr / ow - p[2][c];
            clip[1][c] = p[2][c] - 2.0 * yr / oh;
            clip[3][c] = p[2][c];
        }
        clip[2][3] = NEAR;

        // Shadow maps: one layer per shadow-casting directional or spot light.
        let corners = caster_corners(&scene.casters);
        let mut shadows: Vec<(usize, M4, f64, bool)> = Vec::new();
        for (k, l) in scene.lights.iter().enumerate().take(MAX_LIGHTS) {
            if l.shadow && shadows.len() < MAX_SHADOWS {
                if let Some((m, width, persp)) = shadow_matrix(l, &corners) {
                    shadows.push((k, m, width, persp));
                }
            }
        }
        let shadow_view = if shadows.is_empty() {
            self.blank_shadow.clone()
        } else {
            let target = Self::shadow_target(&self.device, SHADOW_SIZE, shadows.len() as u32);
            let casters: Vec<(wgpu::BindGroup, Vec<(wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, u32)>)> = scene
                .casters
                .iter()
                .filter(|c| c.material.cast_shadow)
                .map(|c| (self.item_bind(c, false), self.part_draws(c.mesh)))
                .collect();
            for (layer, (_, m, _, _)) in shadows.iter().enumerate() {
                let ubuf = self.buffer("solid shadow", bytemuck::cast_slice(&cols(m)), wgpu::BufferUsages::UNIFORM);
                let ubind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("solid shadow"),
                    layout: &self.shadow_u_layout,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }],
                });
                let view = target.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let profile_ts = crate::gpu_profile::pass_timestamps();
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("solid shadow"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.shadow);
                pass.set_bind_group(0, &ubind, &[]);
                for (ibind, parts) in &casters {
                    pass.set_bind_group(1, ibind, &[]);
                    for (vb, ib, _, n) in parts {
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*n, 0, 0..1);
                    }
                }
            }
            target.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() })
        };

        let mut u = Uniforms {
            clip: cols(&clip),
            mirror: cols(&IDENTITY),
            shadow: [cols(&IDENTITY); MAX_SHADOWS],
            eye: [scene.eye[0] as f32, scene.eye[1] as f32, scene.eye[2] as f32, 1.0],
            count: [scene.lights.len().min(MAX_LIGHTS) as f32, 0.0, 0.0, 0.0],
            plane: [0.0; 4],
            env: [0.0; 4],
            sh: [[0.0; 4]; 9],
            lights: [LightData { color: [0.0; 4], v: [0.0; 4], axis: [0.0; 4], ground: [0.0; 4], shadow: [-1.0, 0.0, 0.0, 0.0] }; MAX_LIGHTS],
        };
        for (layer, (_, m, _, _)) in shadows.iter().enumerate() {
            u.shadow[layer] = cols(m);
        }
        for (k, (slot, l)) in u.lights.iter_mut().zip(&scene.lights).enumerate() {
            let kind = match l.kind {
                LightKind::Ambient => 0.0,
                LightKind::Directional => 1.0,
                LightKind::Point => 2.0,
                LightKind::Spot => 3.0,
                LightKind::Hemisphere => 4.0,
            };
            let half = (l.angle.clamp(0.0, 179.0) / 2.0).to_radians();
            let (outer, inner) = (half.cos(), (half * (1.0 - l.penumbra.clamp(0.0, 1.0))).cos());
            let shadow = match shadows.iter().position(|s| s.0 == k) {
                Some(layer) => [layer as f32, l.softness.max(0.0) as f32, f32::from(u8::from(shadows[layer].3)), shadows[layer].2 as f32],
                None => [-1.0, 0.0, 0.0, 0.0],
            };
            *slot = LightData {
                color: [l.color[0] as f32, l.color[1] as f32, l.color[2] as f32, kind],
                v: [l.vector[0] as f32, l.vector[1] as f32, l.vector[2] as f32, inner as f32],
                axis: [l.axis[0] as f32, l.axis[1] as f32, l.axis[2] as f32, outer as f32],
                ground: [l.ground[0] as f32, l.ground[1] as f32, l.ground[2] as f32, l.range.map_or(0.0, |r| r.max(1e-3)) as f32],
                shadow,
            };
        }
        let env_view = match &scene.environment {
            Some(e) => {
                let k = e.intensity.max(0.0);
                u.env = [k as f32, e.rotation.to_radians() as f32, (super::env::ROUGH_LEVELS).min(e.environment.levels.len() as f32 - 1.0), 1.0];
                for (slot, c) in u.sh.iter_mut().zip(&e.environment.sh) {
                    *slot = [c[0] * k as f32, c[1] * k as f32, c[2] * k as f32, 0.0];
                }
                self.env_view(&e.environment)
            }
            None => self.blank_env.clone(),
        };

        // Opaque solids first (any order: the depth buffer sorts them), then translucent ones far to near by their
        // centres, each blended over what is drawn.
        let centre_depth = |item: &SolidItem<'_>| {
            item.mesh.bounds().map_or(0.0, |(lo, hi)| {
                let c = apply(&item.model, [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0]);
                p[2][0] * c[0] + p[2][1] * c[1] + p[2][2] * c[2] + p[2][3]
            })
        };
        let mut order: Vec<usize> = (0..scene.items.len()).collect();
        let translucent = |i: usize| scene.items[i].material.opacity < 1.0;
        order.sort_by(|&a, &b| {
            translucent(a)
                .cmp(&translucent(b))
                .then_with(|| if translucent(a) { centre_depth(&scene.items[b]).total_cmp(&centre_depth(&scene.items[a])) } else { std::cmp::Ordering::Equal })
        });
        let parts: Vec<_> = scene.items.iter().map(|i| self.part_draws(i.mesh)).collect();

        // The first reflector (a plane with `reflect`): the scene's other solids drawn mirrored in its plane, linear.
        let reflector = scene.items.iter().position(|i| i.material.reflect > 0.0).and_then(|r| plane_of(&scene.items[r], scene.eye).map(|pl| (r, pl)));
        let size = wgpu::Extent3d { width: out.w, height: out.h, depth_or_array_layers: 1 };
        let ms = |format, label| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
                    mip_level_count: 1,
                    sample_count: SAMPLES,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let scene_bind = |u: &Uniforms, mirror_view: &wgpu::TextureView| {
            let buf = self.buffer("solid uniforms", bytemuck::bytes_of(u), wgpu::BufferUsages::UNIFORM);
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("solid scene"),
                layout: &self.scene_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.shadow_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&env_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.env_sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(mirror_view) },
                ],
            })
        };
        let pass_over = |encoder: &mut wgpu::CommandEncoder,
                         target: &wgpu::TextureView,
                         sbind: &wgpu::BindGroup,
                         draws: &[(bool, wgpu::BindGroup, usize)]| {
            let color = ms(Storage::F16.format(), "solid colour");
            let depth = ms(wgpu::TextureFormat::Depth32Float, "solid depth");
            let profile_ts = crate::gpu_profile::pass_timestamps();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("solid"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color,
                    depth_slice: None,
                    resolve_target: Some(target),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, sbind, &[]);
            for (translucent, ibind, i) in draws {
                pass.set_bind_group(2, ibind, &[]);
                let pipes: &[&wgpu::RenderPipeline] = if *translucent { &[&self.prepass, &self.translucent] } else { &[&self.pipeline] };
                for pipe in pipes {
                    pass.set_pipeline(pipe);
                    for (vb, ib, tbind, n) in &parts[*i] {
                        pass.set_bind_group(1, tbind, &[]);
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*n, 0, 0..1);
                    }
                }
            }
        };
        let mirror_tex = match reflector {
            Some((r, (n, d))) => {
                let mirrored = pool.acquire(out.w, out.h, Storage::F16)?;
                let m = mirror(n, d);
                let mut mu = u;
                mu.mirror = cols(&m);
                let e = apply(&m, scene.eye);
                mu.eye = [e[0] as f32, e[1] as f32, e[2] as f32, 1.0];
                mu.count[1] = 1.0;
                mu.plane = [n[0] as f32, n[1] as f32, n[2] as f32, d as f32];
                let sbind = scene_bind(&mu, &self.blank_mirror);
                let draws: Vec<(bool, wgpu::BindGroup, usize)> =
                    order.iter().filter(|&&i| i != r).map(|&i| (translucent(i), self.item_bind(&scene.items[i], false), i)).collect();
                pass_over(&mut encoder, mirrored.view(), &sbind, &draws);
                Some((r, mirrored))
            }
            None => None,
        };
        let target = pool.acquire(out.w, out.h, Storage::F16)?;
        let sbind = scene_bind(&u, mirror_tex.as_ref().map_or(&self.blank_mirror, |(_, t)| t.view()));
        let draws: Vec<(bool, wgpu::BindGroup, usize)> = order
            .iter()
            .map(|&i| (translucent(i), self.item_bind(&scene.items[i], mirror_tex.as_ref().is_some_and(|(r, _)| *r == i)), i))
            .collect();
        pass_over(&mut encoder, target.view(), &sbind, &draws);
        self.queue.submit([encoder.finish()]);
        Ok(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn light(kind: LightKind, vector: [f64; 3], axis: [f64; 3]) -> Light {
        Light { axis, angle: 40.0, shadow: true, softness: 4.0, ..Light::simple(kind, [1.0; 3], vector) }
    }

    fn to_clip(m: &M4, p: [f64; 3]) -> [f64; 3] {
        let c: Vec<f64> = (0..4).map(|r| m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3]).collect();
        [c[0] / c[3], c[1] / c[3], c[2] / c[3]]
    }

    #[test]
    fn a_directional_shadow_map_holds_every_caster_corner() {
        let corners = vec![[0.0, 0.0, 0.0], [100.0, 50.0, 20.0], [-40.0, 80.0, -30.0], [10.0, -20.0, 60.0]];
        let l = light(LightKind::Directional, unit([0.3, -1.0, -0.4]).unwrap(), [0.0; 3]);
        let (m, width, persp) = shadow_matrix(&l, &corners).unwrap();
        assert!(!persp && width > 100.0);
        for c in &corners {
            let q = to_clip(&m, *c);
            assert!(q[0].abs() < 1.0 && q[1].abs() < 1.0 && (0.0..=1.0).contains(&q[2]), "{c:?} → {q:?}");
        }
        // Nearer the light, smaller depth.
        let toward = [l.vector[0] * 10.0, l.vector[1] * 10.0, l.vector[2] * 10.0];
        assert!(to_clip(&m, toward)[2] < to_clip(&m, [0.0; 3])[2]);
    }

    #[test]
    fn a_spot_shadow_map_sees_its_target_at_its_centre() {
        let corners = vec![[-50.0, -50.0, -50.0], [50.0, 50.0, 50.0]];
        let l = light(LightKind::Spot, [0.0, -500.0, 0.0], [0.0, -1.0, 0.0]);
        let (m, width, persp) = shadow_matrix(&l, &corners).unwrap();
        assert!(persp && width > 0.0);
        let c = to_clip(&m, [0.0, 0.0, 0.0]);
        assert!(c[0].abs() < 1e-9 && c[1].abs() < 1e-9 && (0.0..=1.0).contains(&c[2]), "{c:?}");
        // Nothing to shadow: no map; a point light casts none.
        assert!(shadow_matrix(&l, &[]).is_none());
        assert!(shadow_matrix(&light(LightKind::Point, [0.0; 3], [0.0; 3]), &corners).is_none());
    }

    #[test]
    fn the_mirror_reflects_across_its_plane() {
        // The floor y = 100, its normal up (−y).
        let m = mirror([0.0, -1.0, 0.0], 100.0);
        assert_eq!(apply(&m, [5.0, 40.0, 7.0]), [5.0, 160.0, 7.0]);
        assert_eq!(apply(&m, [0.0, 100.0, 0.0]), [0.0, 100.0, 0.0]);
    }
}

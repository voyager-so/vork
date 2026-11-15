//! Shared numeric curve sampling for authored v2 tracks and imported glTF clips.
use anyhow::{Context, Result, ensure};
use glam::Quat;
use vork_scene::v2;
mod exact;
mod weights;
mod clocks;
mod sound;
use clocks::ClockSpec;
pub(crate) use clocks::{ClockNode, FootageClock};
pub(crate) use sound::SoundPlan;
use vork_time::exact::Germ;

/// A clock-valued track: its keys are exact seconds or rates, not IEEE numbers.
pub(super) fn is_clock_track(path: &[String]) -> bool {
    matches!(path, [root, _, field] if root == "images" && field == "time")
        || matches!(path, [.., time, speed] if time == "time" && speed == "speed")
}

fn hermite_basis<T>(u: T, one: T, two: T, three: T) -> [T; 4]
where T: Clone + std::ops::Add<Output=T> + std::ops::Sub<Output=T> + std::ops::Mul<Output=T> {
    let u2 = u.clone() * u.clone();
    let u3 = u2.clone() * u.clone();
    [two.clone()*u3.clone()-three.clone()*u2.clone()+one,
        u3.clone()-two.clone()*u2.clone()+u,
        three*u2.clone()-two*u3.clone(), u3-u2]
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Interpolation {
    Step,
    Linear,
    Cubic,
}
/// Cubic samples use glTF's [incoming tangent, value, outgoing tangent] packing.
pub(crate) struct Curve {
    pub times: Vec<f32>,
    pub values: Vec<f32>,
    pub width: usize,
    pub interpolation: Interpolation,
    pub rotation: bool,
    /// Rotation keyed as XYZ Euler angles: each angle interpolates on its own
    /// (so a spin past 180° is two keys) and converts to a quaternion per sample.
    /// `values` then holds three angles per key; samples are still quaternions.
    pub euler: bool,
}
impl Curve {
    pub fn duration(&self) -> f32 {
        *self.times.last().unwrap()
    }
    pub fn sample(&self, time: f32, out: &mut Vec<f32>) {
        self.sample_interval(time, self.times.partition_point(|v| *v <= time), out);
    }
    fn sample_interval(&self, time: f32, upper: usize, out: &mut Vec<f32>) {
        let a = upper.saturating_sub(1);
        let b = upper.min(self.times.len() - 1);
        let dt = self.times[b] - self.times[a];
        let u = if dt > 0. {
            ((time - self.times[a]) / dt).clamp(0., 1.)
        } else {
            0.
        };
        let cubic = self.interpolation == Interpolation::Cubic;
        let stride = if self.euler { 3 } else { self.width };
        let value = |key: usize, part: usize, k: usize| {
            self.values[(key * if cubic { 3 } else { 1 } + part) * stride + k]
        };
        if self.euler {
            let angle = |k: usize| if cubic {
                let [h00,h10,h01,h11] = hermite_basis(u,1.,2.,3.);
                h00 * value(a, 1, k) + h10 * dt * value(a, 2, k)
                    + h01 * value(b, 1, k) + h11 * dt * value(b, 0, k)
            } else if self.interpolation == Interpolation::Step {
                value(a, 0, k)
            } else {
                value(a, 0, k) * (1. - u) + value(b, 0, k) * u
            };
            let euler = v2::Rotation::Euler { euler: [angle(0), angle(1), angle(2)] };
            out.clear();
            out.extend(euler.quaternion());
            return;
        }
        out.resize(self.width, 0.);
        if self.rotation && self.interpolation == Interpolation::Linear {
            let q = |i| Quat::from_array(std::array::from_fn(|k| value(i, 0, k))).normalize();
            out.copy_from_slice(&q(a).slerp(q(b), u).normalize().to_array());
            return;
        }
        for (k, out) in out.iter_mut().enumerate() {
            *out = if cubic {
                let [h00,h10,h01,h11] = hermite_basis(u,1.,2.,3.);
                h00 * value(a, 1, k)
                    + h10 * dt * value(a, 2, k)
                    + h01 * value(b, 1, k)
                    + h11 * dt * value(b, 0, k)
            } else if self.interpolation == Interpolation::Step {
                value(a, 0, k)
            } else {
                value(a, 0, k) * (1. - u) + value(b, 0, k) * u
            };
        }
        if self.rotation {
            let q = Quat::from_array(out.as_slice().try_into().unwrap());
            let q = if q.length_squared() > 0. {
                q
            } else {
                Quat::from_array(std::array::from_fn(|k| value(a, usize::from(cubic), k)))
            };
            out.copy_from_slice(&q.normalize().to_array());
        }
    }

    /// Compile explicit glTF CUBICSPLINE derivatives once.
    pub fn authored(
        track: &v2::Track,
        rotation: bool,
        times: Vec<f32>,
        values: impl AsRef<[serde_json::Value]>,
    ) -> Result<Self> {
        let values = values.as_ref();
        ensure!(!values.is_empty(), "track needs values");
        let euler = rotation && values[0].get("euler").is_some();
        let angles = |value: &serde_json::Value| -> Result<Vec<f32>> {
            let angles = value.get("euler").context("a rotation track's keys are all [x,y,z,w] or all {\"euler\": [x,y,z]}")?;
            let angles = components(angles)?;
            ensure!(angles.len() == 3, "euler rotation needs [x,y,z] radians");
            Ok(angles)
        };
        let components = |value:&serde_json::Value| -> Result<Vec<f32>> {
            if is_clock_track(&track.path) {
                let value=value.as_array().filter(|v|v.len()==1).map_or(value,|v|&v[0]);
                Ok(vec![serde_json::from_value::<vork_time::Clock>(value.clone())?.as_f32()])
            } else {components(value)}
        };
        let width = if euler { 3 } else { components(&values[0])?.len() };
        let mut flat = Vec::with_capacity(values.len() * width);
        for value in values {
            let value = if euler { angles(value)? } else { components(value)? };
            ensure!(value.len() == width, "track value shape mismatch");
            flat.extend(value);
        }
        if euler {
            let mut curve = Self::numeric(track, false, times, flat, 3)?;
            curve.rotation = true;
            curve.euler = true;
            curve.width = 4;
            return Ok(curve);
        }
        Self::numeric(track, rotation, times, flat, width)
    }
    fn numeric(
        track: &v2::Track,
        rotation: bool,
        times: Vec<f32>,
        values: Vec<f32>,
        width: usize,
    ) -> Result<Self> {
        let path = v2::display_path(&track.path);
        let clock=is_clock_track(&track.path);
        ensure!(
            width > 0
                && !times.is_empty()
                && values.len() == times.len() * width * if matches!(track.interpolation, v2::Interpolation::Cubic) { 3 } else { 1 }
                && (clock || (times.iter().all(|v|v.is_finite()) && (if matches!(track.interpolation,v2::Interpolation::Step) {times.windows(2).all(|v|v[0]<=v[1])}else{times.windows(2).all(|v|v[0]<v[1])}) && values.iter().all(|v|v.is_finite()))),
            "invalid numeric track {path}"
        );
        ensure!(
            !rotation
                || (width == 4
                    && values
                        .chunks_exact(width)
                        .enumerate()
                        .filter(|(i, _)| !matches!(track.interpolation, v2::Interpolation::Cubic) || i % 3 == 1)
                        .all(|(_, v)| Quat::from_slice(v).length_squared() > 0.)),
            "invalid quaternion track"
        );
        let interpolation = match track.interpolation {
            v2::Interpolation::Step => Interpolation::Step,
            v2::Interpolation::Linear => Interpolation::Linear,
            v2::Interpolation::Cubic => Interpolation::Cubic,
        };
        Ok(Self {
            times,
            values,
            width,
            interpolation,
            rotation,
            euler: false,
        })
    }
}

fn components(value: &serde_json::Value) -> Result<Vec<f32>> {
    let values = if let Some(array) = value.as_array() {
        array
            .iter()
            .map(|v| v.as_f64().context("numeric track component"))
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![value.as_f64().context("numeric track value")?]
    };
    ensure!(
        values.iter().all(|v| (*v as f32).is_finite()),
        "track values must be finite"
    );
    Ok(values.into_iter().map(|v| v as f32).collect())
}

/// How an additive layer combines with the value below it (Blender NLA Combine).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Combine {
    /// Offsets add: translation, weights, scalars.
    Add,
    /// Scale multiplies: base × sample^weight.
    Multiply,
    /// Quaternions post-multiply: base × slerp(identity, sample, weight).
    Rotation,
}
/// Replace layers mix toward the sample (Blender NLA Replace);
/// additive layers follow Blender NLA Combine for the value's kind.
pub(crate) fn blend(base: &mut [f32], sample: &[f32], weight: f32, additive: bool, kind: Combine) {
    if kind == Combine::Rotation {
        let current = Quat::from_slice(base);
        let sampled = Quat::from_slice(sample).normalize();
        let result = if additive {
            current * Quat::IDENTITY.slerp(sampled, weight)
        } else {
            current.slerp(sampled, weight)
        };
        base.copy_from_slice(&result.normalize().to_array());
        return;
    }
    for (base, sample) in base.iter_mut().zip(sample) {
        *base = match (additive, kind) {
            (false, _) => *base + weight * (sample - *base),
            (true, Combine::Multiply) => *base * sample.signum() * sample.abs().powf(weight),
            (true, _) => *base + weight * sample,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn file_buffer_times_are_checked_after_decode() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-track-times-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let result = (|| -> Result<()> {
            {
                for (times, valid) in [
                    (vec![], false),
                    (vec![0., 1., 2.], false),
                    (vec![-1., 1.], false),
                    (vec![0., f32::NAN], false),
                    (vec![0., f32::INFINITY], false),
                    (vec![1., 0.], false),
                    (vec![0., 0.], false),
                    (vec![0., 1.], true),
                ] {
                    let bytes: Vec<_> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
                    std::fs::write(base.join("times.bin"), bytes)?;
                    // Buffer-backed keys are numeric; names and switches are inline pairs.
                    let scene: v2::Scene = vork_scene::v2::from_test_value(serde_json::json!({
                        "version":2,"nodes":[{"type":"camera","name":"cam","material":"optics"}],
                        "materials":{"optics":{"program":"std/camera"}},
                        "buffers":{"t":{"type":"f32","file":"times.bin"},"v":{"type":"f32","data":[0,1]}},
                        "animation":[{"tracks":[{"path":["materials","optics","inputs","lens"],"keys":{"times":"t","values":"v"},"interpolation":"STEP"}]}]
                    }))?;
                    scene.validate()?;
                    assert_eq!(
                        Plan::compile(&scene, &base).is_ok(),
                        valid,
                        "times={times:?}"
                    );
                }
            }
            Ok(())
        })();
        std::fs::remove_dir_all(base)?;
        result
    }

    #[test]
    fn cubic_derivatives_are_scaled_by_segment_duration() -> Result<()> {
        let track: v2::Track = vork_scene::v2::from_test_value(
            serde_json::json!({"path":["materials","optics","inputs","lens"],"keys":[[0,0,0,2],[2,2,0,0]],"interpolation":"CUBICSPLINE"}),
        )?;
        let v2::Values::Inline(values) = &track.values else { unreachable!() };
        let curve = Curve::authored(&track, false, vec![0., 2.], values)?;
        let mut value = vec![];
        curve.sample(1., &mut value);
        assert_eq!(value, [1.5]);
        curve.sample(-1., &mut value);
        assert_eq!(value, [0.]);
        curve.sample(3., &mut value);
        assert_eq!(value, [2.]);
        Ok(())
    }
    #[test]
    fn additive_layers_combine_like_nla_and_replacement_is_ordered() {
        let x = Quat::from_rotation_x(0.7);
        let y = Quat::from_rotation_y(0.4);
        let mut base = x.to_array();
        blend(&mut base, &y.to_array(), 0.5, true, Combine::Rotation);
        // The offset turns about the current local axes (post-multiply).
        assert!(
            Quat::from_array(base)
                .dot(x * Quat::IDENTITY.slerp(y, 0.5))
                .abs()
                > 0.99999
        );
        let mut base = [10.];
        blend(&mut base, &[20.], 0.5, false, Combine::Add);
        blend(&mut base, &[4.], 0.5, true, Combine::Add);
        assert_eq!(base, [17.]);
        let mut scale = [2.];
        blend(&mut scale, &[4.], 0.5, true, Combine::Multiply);
        assert_eq!(scale, [4.]);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Sample {
    Numeric(Vec<f32>),
    String(std::sync::Arc<str>),
    Bool(bool),
    Style(std::sync::Arc<super::style::StyleValue>),
}
#[derive(Clone)]
pub(crate) enum Channel {
    Style(super::style::StyleTrack),
    Numeric(std::sync::Arc<Curve>),
    Discrete {
        times: Vec<f32>,
        values: Vec<Sample>,
    },
}
#[derive(Clone, Copy)]
pub(crate) struct AssetTarget {
    pub asset: usize,
    pub node: usize,
    pub property: gltf::animation::Property,
    pub component: Option<usize>,
}
#[derive(Clone)]
pub(crate) struct PropertyTrack {
    /// The authored target; None for a glTF clip channel (asset targets only).
    pub target: Option<v2::TrackTarget>,
    channel: Channel,
    time_curve: Option<std::sync::Arc<exact::DeferredTimeCurve>>,
    camera_keys: Option<Vec<vork_time::exact::Rational>>,
    discrete_keys: Option<std::sync::Arc<exact::DiscreteKeys>>,
    weight_target: Option<usize>,
    asset_targets: Vec<AssetTarget>,
    asset_visibility: Option<(usize, usize)>,
    node_targets: Vec<NodeBinding>,
    material_targets: Option<Vec<usize>>,
}
/// How one track is read at one sample.
struct TrackRead {
    /// The key interval, when the track's keys are read exactly.
    upper: Option<usize>,
    /// The layer's exact local instant, for a channel that reads it.
    exact_local: Option<Germ>,
}
impl CompiledLayer {
    /// A looping layer's numeric local time, wrapped into its tracks' key span.
    fn looped(&self, local: f32) -> f32 {
        match self.span {
            Some((a, b)) if self.source.looping && b > a => a + (local - a).rem_euclid(b - a),
            _ => local,
        }
    }
}
#[derive(Clone)]
pub(crate) struct CompiledLayer {
    pub source: v2::Layer,
    pub tracks: Vec<PropertyTrack>,
    span: Option<(f32, f32)>,
    time_span: Option<std::sync::Arc<exact::TimeSpan>>,
    /// A node-owned layer: the owner's pre-order address (child indices from the root).
    owner: Option<Vec<usize>>,
    /// The owner's installed node index, bound by `mark_nodes`; sampled at its local time.
    pub owner_node: Option<usize>,
}
/// One immutable ordered layer plan. glTF import appends channels to these layers.
#[derive(Clone)]
pub(crate) struct MaterialBinding {
    pub slot: usize,
    pub parameter: usize,
    /// Reflected input storage; slot is an absolute word address when present.
    pub storage: Option<vork_slang::Type>,
    pub enum_tokens: Option<Vec<String>>,
    pub default: Sample,
    /// Instance time offset of this record copy: it is written at time + offset.
    pub offset: f32,
}
#[derive(Clone)]
pub(crate) struct Samples {
    values: Vec<Sample>,
    weights: Vec<f32>,
    layer_weights: Vec<f32>,
    pub render: Option<v2::Render>,
    material_values: Vec<Sample>,
    /// Each node's local instant at this sample; `None` is an absent node (A7 rule 1).
    pub clocks: Vec<Option<Germ>>,
    /// Per track, how long the value it was last sampled to stands ([`Held`]). It is read from the track's keys
    /// each time the track is sampled, and from nothing else.
    held: Vec<Held>,
    /// For a consumer that asked for it ([`Plan::samples_naming_changes`]): the tracks whose value or weight the
    /// last sampling call changed, as flat track indices, ascending: that call's account of what changed. Each
    /// sampling call starts it again. `None` for a consumer that reads every value it uses each time (the 3D
    /// renderer, the sound mix): nothing is compared to make an account nobody reads.
    changed: Option<Vec<u32>>,
    /// Scratch for one numeric sample when changes are named: a track is sampled here, compared with the value it
    /// holds, and copied into the track's own storage only when it differs. It grows to the widest track and is
    /// never handed to one.
    scratch: Vec<f32>,
}
impl Samples {
    /// The tracks whose value or weight the last sampling call changed (flat track indices, ascending). A track
    /// the call did not consume (weight zero before and after) is not among them: nothing reads its value.
    /// Only samples made by [`Plan::samples_naming_changes`] have this account.
    pub(crate) fn changed(&self) -> &[u32] {
        self.changed.as_deref().expect("these samples were not asked to name what changes")
    }
}
/// How long a track's sampled value stands, as its keys say at the time it was sampled. While it stands and the
/// track's weight is the same, the track is not sampled and is not reported as changed.
///
/// Only spans where the sampler's own arithmetic returns the same bits are held: before the first key, after
/// the last, and between the keys of a stepped track. Two equal keys joined by a line or a cubic are sampled:
/// `a * (1 - u) + a * u` is not `a` for every `u`.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Held {
    /// Not sampled yet, or the value moves with time.
    No,
    /// It stands while the layer's local time is in `[from, until)`: the keys around the time it was sampled at.
    Span { from: f32, until: f32 },
    /// A track whose keys are compared exactly: it stands while the same number of keys are at or before the
    /// layer's exact local time.
    Key(usize),
}
impl Channel {
    /// Whether the channel's value is the same at every time that has `upper` keys at or before it.
    fn constant_at(&self, upper: usize) -> bool {
        match self {
            Channel::Numeric(curve) => curve.interpolation == Interpolation::Step || upper == 0 || upper >= curve.times.len(),
            Channel::Style(track) => track.step || upper == 0 || upper >= track.times.len(),
            Channel::Discrete { .. } => true,
        }
    }
    /// The channel's key times.
    fn times(&self) -> &[f32] {
        match self {
            Channel::Numeric(curve) => &curve.times,
            Channel::Style(track) => &track.times,
            Channel::Discrete { times, .. } => times,
        }
    }
}
#[derive(Clone, Default)]
struct AssetBindings {
    transforms: Vec<(usize, AssetTarget, bool)>,
    visibility: Vec<(usize, usize)>,
}
struct LiveValues {
    generation: u64,
    values: std::collections::BTreeMap<usize, Vec<f32>>,
}
#[derive(Clone, Default)]
pub(crate) struct Plan {
    // Absent in offline plans. Clones retained by imported animation share the
    // publication, while each frame still samples into its own storage.
    live: std::sync::Arc<std::sync::OnceLock<std::sync::RwLock<LiveValues>>>,
    pub layers: Vec<CompiledLayer>,
    reset_bindings: Vec<NodeBinding>,
    asset_bindings: Vec<AssetBindings>,
    material_bindings: Vec<MaterialBinding>,
    material_offsets: Vec<f32>,
    cameras: std::collections::BTreeMap<String, usize>,
    camera_tracks: bool,
    camera_weight_animation: bool,
    discrete_tracks: bool,
    /// Node clocks (A7), one per installed node, bound by `mark_nodes`.
    clocks: Vec<ClockSpec>,
    clock_names: Vec<String>,
    /// Flat index of each layer's first track, for clock-consumed tracks.
    layer_starts: Vec<usize>,
    has_clocks: bool,
    /// Footage image -> the node whose clock selects its frames.
    footage_owner: std::collections::BTreeMap<String, usize>,
    /// Per flat track, the installed node it evaluates for; an absent node's tracks are not sampled.
    subjects: Vec<Option<usize>>,
    /// Compile-time tables keyed by pre-order node address, resolved in `mark_nodes`.
    image_samplers: std::collections::BTreeMap<String, Vec<Vec<usize>>>,
    images_with_time: std::collections::BTreeSet<String>,
    source_lengths: std::collections::BTreeMap<Vec<usize>, vork_time::exact::Rational>,
}
impl Plan {
    /// Resolve host inputs after all imported channels and material bindings
    /// have finalized. Targets were declared as varying before installation.
    pub(crate) fn bind_live(&self, targets: &[v2::TrackTarget], initial: &[Vec<f32>]) -> Result<Vec<usize>> {
        let mut indices = Vec::new();
        for target in targets {
            let matching: Vec<_> = self.layers.iter().flat_map(|l| &l.tracks).enumerate()
                .filter(|(_, track)| track.target.as_ref() == Some(target)).collect();
            ensure!(matching.len() == 1, "live input must have exactly one prepared binding; remove other tracks for {target:?}");
            let (index, track) = matching[0];
            ensure!(matches!(track.channel, Channel::Numeric(_)), "live input must be numeric: {target:?}");
            indices.push(index);
        }
        self.live.set(std::sync::RwLock::new(LiveValues {
            generation: 0,
            values: indices.iter().copied().zip(initial.iter().cloned()).collect(),
        }))
            .map_err(|_| anyhow::anyhow!("live inputs already bound"))?;
        Ok(indices)
    }

    pub(crate) fn publish_live<'a>(&self, updates: impl Iterator<Item = (usize, &'a [f32])>) {
        let mut values = self.live.get().expect("prepared live inputs").write().expect("live values lock poisoned");
        let mut changed = false;
        for (index, value) in updates {
            values.values.get_mut(&index).expect("prepared live binding").copy_from_slice(value);
            changed = true;
        }
        if changed { values.generation += 1; }
    }

    /// Time alone does not identify a pose when the host can publish values.
    pub(crate) fn live_generation(&self) -> u64 {
        self.live.get().map_or(0, |values| values.read().expect("live values lock poisoned").generation)
    }

    #[cfg(test)]
    pub fn compile(scene: &v2::Scene, base: &std::path::Path) -> Result<Self> {
        Self::compile_with_buffers(scene, base, None)
    }
    pub fn compile_with_buffers(
        scene: &v2::Scene,
        base: &std::path::Path,
        workspace: Option<&std::collections::BTreeMap<String, crate::buffers::AttributeData>>,
    ) -> Result<Self> {
        Self::compile_in(scene, base, workspace, &Default::default())
    }
    /// Compile within a document open: inferred spans come from the sources that open read
    /// (`opened`), the same ones installation uses. [`Self::compile_with_buffers`] is a
    /// compile on its own, reading the sources itself.
    pub(crate) fn compile_in(
        scene: &v2::Scene,
        base: &std::path::Path,
        workspace: Option<&std::collections::BTreeMap<String, crate::buffers::AttributeData>>,
        opened: &crate::footage::Opened,
    ) -> Result<Self> {
        use std::collections::HashMap;
        let names: HashMap<_, _> = scene
            .animation
            .iter()
            .enumerate()
            .filter_map(|(i, l)| l.name.as_ref().map(|n| (n.as_str(), i)))
            .collect();
        let targets = scene.track_targets()?;
        let mut layers = Vec::new();
        for (source, targets) in scene.animation.iter().zip(targets) {
            layers.push(Self::compile_layer(scene, base, workspace, &names, source, source.tracks.as_deref().unwrap_or(&[]), targets, None)?);
        }
        for owned in scene.node_owned_layers()? {
            let source = v2::Layer {
                source_span: None, source_clock_span: Vec::new(), name: owned.layer.name.clone(), tracks: None,
                clip: owned.layer.clip.clone(), weight: owned.layer.weight, blend: owned.layer.blend,
                offset: owned.layer.offset, speed: owned.layer.speed, looping: owned.layer.looping,
            };
            layers.push(Self::compile_layer(scene, base, workspace, &names, &source, owned.layer.tracks.as_deref().unwrap_or(&[]), owned.targets, Some(owned.path))?);
        }
        let (image_samplers, images_with_time, source_lengths) = clocks::scene_tables(scene, base, opened)?;
        Self::finish_compile(layers, image_samplers, images_with_time, source_lengths)
    }
    #[allow(clippy::too_many_arguments)]
    fn compile_layer(
        scene: &v2::Scene,
        base: &std::path::Path,
        workspace: Option<&std::collections::BTreeMap<String, crate::buffers::AttributeData>>,
        names: &std::collections::HashMap<&str, usize>,
        source: &v2::Layer,
        source_tracks: &[v2::Track],
        targets: Vec<v2::TrackTarget>,
        owner: Option<Vec<usize>>,
    ) -> Result<CompiledLayer> {
        {
            ensure!(
                source.weight.is_finite() && source.offset.is_finite() && source.speed.is_finite(),
                "layer weight, offset and speed must be finite"
            );
            let mut tracks = Vec::new();
            let mut span: Option<(f32, f32)> = source.source_span;
            let time_slice = targets.iter().zip(source_tracks).any(|(target,track)| matches!(target, v2::TrackTarget::ImageTime {..} | v2::TrackTarget::Render(v2::RenderProperty::Camera)) || target.is_speed()
                || target.style_interpolation().is_some_and(|kind| matches!(&track.values,v2::Values::Inline(values)
                    if !((matches!(kind,v2::StyleInterpolation::Number | v2::StyleInterpolation::FontWeight) && values.iter().all(|v|v.is_number()))
                    || (kind==v2::StyleInterpolation::Paint && values.first().is_some_and(|v|v.is_array()))))))
                || source_tracks.iter().any(|track| matches!(track.interpolation, v2::Interpolation::Step) || matches!(&track.values,v2::Values::Inline(values) if values.first().is_some_and(|v| v.is_boolean() || v.is_string())));
            let mut clock_endpoints=if time_slice {source.source_clock_span.clone()}else{Vec::new()};
            for (track, target) in source_tracks.iter().zip(targets) {
                use v2::{NodeProperty, TrackTarget};
                let rotation = matches!(
                    target,
                    TrackTarget::Node {
                        property: NodeProperty::Rotation,
                        ..
                    } | TrackTarget::Imported {
                        property: NodeProperty::Rotation,
                        ..
                    } | TrackTarget::Text {
                        property: v2::UnitProperty::Rotation,
                        ..
                    }
                );
                let times = match &track.times {
                    v2::Times::Inline(times) => times.clone(),
                    v2::Times::Buffer(name) => {
                        let (times, width) = numeric_buffer(scene, name, base, workspace)?;
                        ensure!(width == 1, "time buffer must be scalar");
                        times
                    }
                };
                if matches!(target,TrackTarget::ImageTime {..}|TrackTarget::Render(v2::RenderProperty::Camera)) || target.is_speed() || matches!(track.interpolation,v2::Interpolation::Step) {
                    exact::times(track,&times)?;
                } else {
                ensure!(
                    !times.is_empty()
                        && times.iter().all(|t| t.is_finite() && *t >= 0.)
                        && times.windows(2).all(|w| w[0] < w[1]),
                    "track {}: times must be non-empty, finite, >= 0 and strictly increasing",
                    v2::display_path(&track.path)
                );
                }
                let weight_target = if let TrackTarget::LayerWeight { layer } = &target {
                    Some(
                        *names
                            .get(layer.as_str())
                            .context("unknown animation layer weight target")?,
                    )
                } else {
                    None
                };
                let channel = match &track.values {
                    v2::Values::Inline(values) if target.style_interpolation().is_some()
                        && !(matches!(target.style_interpolation(), Some(v2::StyleInterpolation::Number | v2::StyleInterpolation::FontWeight)) && values.iter().all(|v|v.is_number()))
                        && !(matches!(target.style_interpolation(), Some(v2::StyleInterpolation::Paint)) && values.first().is_some_and(|v|v.is_array())) => {
                        let kind=target.style_interpolation().unwrap();
                        if kind==v2::StyleInterpolation::Paint && !matches!(track.interpolation,v2::Interpolation::Step) {
                            for pair in values.windows(2) {if let (Some(a),Some(b))=(pair[0].as_str(),pair[1].as_str()) {
                                if a.contains("gradient(") && b.contains("gradient(") {crate::canvas_draw::interpolate_gradient(a,b,0.5,[0.,0.,100.,100.])?;}
                            }}
                        }

                        let values=values.iter().map(|v|super::style::StyleValue::parse(kind,v).map(std::sync::Arc::new)).collect::<Result<Vec<_>>>()?;
                        let step=matches!(track.interpolation,v2::Interpolation::Step);
                        let clocks=(!step && values.windows(2).any(|p|p[0].discrete_with(&p[1],kind))).then(||exact::raw_times(track,&times)).transpose()?;
                        Channel::Style(super::style::StyleTrack {times:times.clone(),kind,step,values,clocks})
                    }
                    v2::Values::Inline(values)
                        if !matches!(target,TrackTarget::ImageTime {..}) && values
                            .first()
                            .is_some_and(|v| v.is_string() || v.is_boolean()) =>
                    {
                        ensure!(
                            values.len() == times.len(),
                            "track {}: discrete values must match the time count",
                            v2::display_path(&track.path)
                        );
                        let strings = values[0].is_string();
                        let values = values
                            .iter()
                            .map(|v| {
                                if strings {
                                    let text = v.as_str().context("mixed track value types")?;
                                    Ok(Sample::String(text.into()))
                                } else {
                                    v.as_bool()
                                        .map(Sample::Bool)
                                        .context("mixed track value types")
                                }
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Channel::Discrete {
                            times: times.clone(),
                            values,
                        }
                    }
                    values => {
                        let curve = match values {
                            v2::Values::Inline(values) => {
                                Curve::authored(track, rotation, times.clone(), values)?
                            }
                            v2::Values::Buffer(name) => {
                                let (values, width) = numeric_buffer(scene, name, base, workspace)?;
                                Curve::numeric(track, rotation, times.clone(), values, width)?
                            }
                        };
                        Channel::Numeric(std::sync::Arc::new(curve))
                    }
                };
                span = Some(span.map_or((times[0], *times.last().unwrap()), |(a, b)| {
                    (a.min(times[0]), b.max(*times.last().unwrap()))
                }));
                if time_slice {
                    let times=exact::raw_times(track,&times)?;
                    clock_endpoints.extend([times.first().unwrap().clone(),times.last().unwrap().clone()]);
                }
                let time_curve = if matches!(target, TrackTarget::ImageTime {..}) || target.is_speed() {
                    let Channel::Numeric(curve) = &channel else { anyhow::bail!("image time needs a scalar track") };
                    Some(std::sync::Arc::new(exact::DeferredTimeCurve::new(track,curve.clone())))
                } else { None };
                let discrete_keys = (!matches!(target, TrackTarget::ImageTime {..}|TrackTarget::Render(v2::RenderProperty::Camera)) && !target.is_speed() && match &channel {
                    Channel::Style(track) => track.step || track.clocks.is_some(),
                    Channel::Discrete {..} => true,
                    Channel::Numeric(curve) => curve.interpolation == Interpolation::Step,
                }).then(|| exact::raw_times(track,&times).map(exact::DiscreteKeys::new).map(std::sync::Arc::new)).transpose()?;
                tracks.push(PropertyTrack {
                    discrete_keys,
                    camera_keys: (target == TrackTarget::Render(v2::RenderProperty::Camera)).then(||exact::times(track,&times)).transpose()?,
                    target: Some(target),
                    channel,
                    time_curve,
                    weight_target,
                    asset_targets: vec![],
                    asset_visibility: None,
                    node_targets: vec![],
                    material_targets: None,
                });
            }
            Ok(CompiledLayer {
                source: v2::Layer {
                    source_span: source.source_span,
                    source_clock_span: source.source_clock_span.clone(),
                    name: source.name.clone(),
                    tracks: None,
                    clip: source.clip.clone(),
                    weight: source.weight,
                    blend: source.blend.clone(),
                    offset: source.offset,
                    speed: source.speed,
                    looping: source.looping,
                },
                tracks,
                span,
                time_span: time_slice.then(||std::sync::Arc::new(exact::TimeSpan::new(clock_endpoints))),
                owner,
                owner_node: None,
            })
        }
    }
    fn finish_compile(
        layers: Vec<CompiledLayer>,
        image_samplers: std::collections::BTreeMap<String, Vec<Vec<usize>>>,
        images_with_time: std::collections::BTreeSet<String>,
        source_lengths: std::collections::BTreeMap<Vec<usize>, vork_time::exact::Rational>,
    ) -> Result<Self> {
        let discrete_tracks=layers.iter().flat_map(|layer|&layer.tracks).any(|track|track.discrete_keys.is_some());
        let camera_tracks=layers.iter().flat_map(|layer|&layer.tracks).any(|track|track.camera_keys.is_some());
        let mut needed:std::collections::BTreeSet<_>=layers.iter().enumerate().filter(|(_,layer)|layer.tracks.iter().any(|t|t.camera_keys.is_some())).map(|(i,_)|i).collect();
        let mut camera_weight_animation=false;
        for (i,layer) in layers.iter().enumerate().rev() {
            for track in &layer.tracks {
                if track.weight_target.is_some_and(|target|needed.contains(&target)) {
                    needed.insert(i);
                    camera_weight_animation|=matches!(&track.channel,Channel::Numeric(curve) if curve.times.len()>1);
                }
            }
        }
        Ok(Self {
            live: Default::default(),
            layers,
            camera_tracks,
            discrete_tracks,
            camera_weight_animation,
            reset_bindings: vec![],
            asset_bindings: vec![],
            material_bindings: vec![],
            material_offsets: vec![],
            cameras: Default::default(),
            clocks: vec![],
            clock_names: vec![],
            layer_starts: vec![],
            has_clocks: false,
            footage_owner: Default::default(),
            subjects: vec![],
            image_samplers,
            images_with_time,
            source_lengths,
        })
    }
    fn refresh_layer_starts(&mut self) {
        let mut start = 0;
        self.layer_starts = self.layers.iter().map(|layer| { let s = start; start += layer.tracks.len(); s }).collect();
    }
    /// Clip declarations retain their global layer indices in the compiled plan.
    pub fn clips_for_asset<'a>(
        &'a self,
        root: &'a str,
    ) -> impl Iterator<Item = (usize, &'a v2::Clip)> + 'a {
        self.layers
            .iter()
            .enumerate()
            .filter_map(move |(index, layer)| {
                let clip = layer.source.clip.as_ref()?;
                (clip.target == root)
                    .then_some((index, clip))
            })
    }

    pub fn add_clip_curve(
        &mut self,
        layer: usize,
        curve: std::sync::Arc<Curve>,
        target: AssetTarget,
    ) {
        self.discrete_tracks |= curve.interpolation==Interpolation::Step;
        let span = (0., curve.duration());
        let layer = &mut self.layers[layer];
        layer.span = Some(
            layer
                .span
                .map_or(span, |(a, b)| (a.min(span.0), b.max(span.1))),
        );
        if layer.time_span.is_some() || curve.interpolation==Interpolation::Step {
            let mut endpoints=layer.time_span.as_ref().map(|span|span.endpoints.clone()).unwrap_or_default();
            let (a,b)=layer.span.unwrap();
            endpoints.extend([vork_time::Clock::binary(a),vork_time::Clock::binary(b)]);
            layer.time_span=Some(std::sync::Arc::new(exact::TimeSpan::new(endpoints)));
        }
        layer.tracks.push(PropertyTrack {
            discrete_keys: (curve.interpolation == Interpolation::Step).then(||std::sync::Arc::new(exact::DiscreteKeys::new(curve.times.iter().map(|&t|vork_time::Clock::binary(t)).collect()))),
            target: None,
            channel: Channel::Numeric(curve),
            time_curve: None,
            camera_keys: None,
            weight_target: None,
            asset_targets: vec![target],
            asset_visibility: None,
            node_targets: vec![],
            material_targets: None,
        });
    }
    pub fn bind_asset(&mut self, doc: &gltf::Document, root: &str, asset: usize) -> Result<bool> {
        use v2::{NodeProperty, TrackTarget};
        let mut any = false;
        for track in self.layers.iter_mut().flat_map(|l| &mut l.tracks) {
            let Some(TrackTarget::Imported {
                asset: name,
                node,
                property,
            }) = &track.target
            else {
                continue;
            };
            if name != root {
                continue;
            }
            let path = || format!("[\"nodes\",{name:?},\"nodes\",{node:?}]");
            let imported = if let Some(index) = node.strip_prefix('#').and_then(|index| index.parse::<usize>().ok()) {
                doc.nodes()
                    .nth(index)
                    .with_context(|| {
                        format!("track {}: imported node index out of range", path())
                    })?
            } else {
                let nodes: Vec<_> = doc.nodes().filter(|n| n.name() == Some(node)).collect();
                ensure!(
                    nodes.len() == 1,
                    "track {}: segment 3: {}",
                    path(),
                    if nodes.is_empty() {
                        ::vork_scene::v2::diagnostics::unknown(
                            "imported node",
                            node,
                            doc.nodes().filter_map(|n| n.name()),
                        )
                    } else {
                        format!(
                            "{} imported nodes are named {node:?}; use #index",
                            nodes.len()
                        )
                    }
                );
                nodes[0].clone()
            };
            if matches!(property, NodeProperty::Visibility(v2::Ray::All)) {
                ensure!(matches!(&track.channel, Channel::Discrete {values,..} if matches!(values[0], Sample::Bool(_))), "imported visible requires boolean values");
                track.asset_visibility = Some((asset, imported.index()));
                any = true;
                continue;
            }
            let (property, component, width) = match property {
                NodeProperty::Translation => (gltf::animation::Property::Translation, None, 3),
                NodeProperty::Rotation => (gltf::animation::Property::Rotation, None, 4),
                NodeProperty::Scale => (gltf::animation::Property::Scale, None, 3),
                NodeProperty::MorphWeight(name) => {
                    let mesh = imported.mesh().context("morph track target needs a mesh")?;
                    let count = mesh
                        .primitives()
                        .next()
                        .context("morph track needs primitives")?
                        .morph_targets()
                        .len();
                    let extras: serde_json::Value = mesh
                        .extras()
                        .as_ref()
                        .map(|v| serde_json::from_str(v.get()))
                        .transpose()?
                        .unwrap_or_default();
                    let component = if let Some(index) = name.strip_prefix('#') {
                        index.parse::<usize>().context("invalid morph index")?
                    } else {
                        extras["targetNames"]
                            .as_array()
                            .and_then(|names| names.iter().position(|v| v.as_str() == Some(name)))
                            .with_context(|| {
                                format!(
                                    "unknown morph {name}; available: {}",
                                    extras["targetNames"]
                                )
                            })?
                    };
                    ensure!(component < count, "morph index out of range");
                    (
                        gltf::animation::Property::MorphTargetWeights,
                        Some(component),
                        1,
                    )
                }
                _ => anyhow::bail!("track {}: unsupported imported-node property", path()),
            };
            ensure!(
                matches!(&track.channel,Channel::Numeric(curve) if curve.width==width),
                "track {}: imported-node track shape mismatch",
                path()
            );
            track.asset_targets.push(AssetTarget {
                asset,
                node: imported.index(),
                property,
                component,
            });
            any = true;
        }
        Ok(any)
    }
    /// Freeze imported targets after every clip has appended its channels. Flat
    /// sample addresses are final only here, since later clips can extend earlier layers.
    pub fn finish_asset_bindings(&mut self) -> Result<()> {
        for track in self.layers.iter().flat_map(|l| &l.tracks) {
            if let Some(v2::TrackTarget::Imported { asset, node, .. }) = &track.target {
                ensure!(
                    !track.asset_targets.is_empty() || track.asset_visibility.is_some(),
                    "unbound imported-node track [\"nodes\",{asset:?},\"nodes\",{node:?}]"
                );
            }
        }
        self.asset_bindings.clear();
        let mut index = 0;
        for layer in &self.layers {
            let additive = matches!(layer.source.blend, v2::Blend::Add);
            for track in &layer.tracks {
                for &target in &track.asset_targets {
                    self.asset_bindings.resize_with(self.asset_bindings.len().max(target.asset + 1), AssetBindings::default);
                    self.asset_bindings[target.asset].transforms.push((index, target, additive));
                }
                if let Some((asset, node)) = track.asset_visibility {
                    self.asset_bindings.resize_with(self.asset_bindings.len().max(asset + 1), AssetBindings::default);
                    self.asset_bindings[asset].visibility.push((index, node));
                }
                index += 1;
            }
        }
        self.refresh_layer_starts();
        Ok(())
    }
    /// Content identity includes decoded buffer-backed keys, not their file names.
    pub fn input_identity(&self) -> blake3::Hash {
        let mut hash = blake3::Hasher::new();
        for layer in &self.layers {
            let relevant = |track: &&PropertyTrack| track.weight_target.is_some()
                || track.material_targets.as_ref().is_some_and(|targets| !targets.is_empty());
            if !layer.tracks.iter().any(|track| relevant(&track)) { continue; }
            for value in [layer.source.weight, layer.source.offset, layer.source.speed] { hash.update(&value.to_bits().to_le_bytes()); }
            hash.update(&[u8::from(layer.source.looping), u8::from(layer.source.blend == v2::Blend::Add)]);
            if layer.source.looping {
                if let Some((a,b)) = layer.span { hash.update(&a.to_bits().to_le_bytes()); hash.update(&b.to_bits().to_le_bytes()); }
            }
            for track in layer.tracks.iter().filter(relevant) {
                hash.update(track.target.as_ref().map(ToString::to_string).unwrap_or_default().as_bytes());
                match &track.channel {
                    Channel::Style(_) => unreachable!("style is not a material input"),
                    Channel::Numeric(curve) => {
                        hash.update(&[match curve.interpolation {Interpolation::Step=>0,Interpolation::Linear=>1,Interpolation::Cubic=>2}]);
                        hash.update(bytemuck::cast_slice(&curve.times));
                        hash.update(bytemuck::cast_slice(&curve.values));
                    },
                    Channel::Discrete { times, values } => {
                        hash.update(bytemuck::cast_slice(times));
                        hash.update(format!("{values:?}").as_bytes());
                    },
                }
            }
        }
        for binding in &self.material_bindings {
            hash.update(format!("{}:{:?}:{:?}", binding.slot, binding.storage, binding.default).as_bytes());
        }
        hash.finalize()
    }
    pub fn targets(&self) -> impl Iterator<Item = &v2::TrackTarget> {
        self.layers
            .iter()
            .flat_map(|l| l.tracks.iter().filter_map(|t| t.target.as_ref()))
    }
    /// All names a camera cut can select, after decoding inline or buffered keys.
    pub fn camera_names(&self) -> impl Iterator<Item = &str> {
        self.layers.iter().flat_map(|layer| &layer.tracks).filter_map(|track| {
            if matches!(track.target, Some(v2::TrackTarget::Render(v2::RenderProperty::Camera))) {
                if let Channel::Discrete { values, .. } = &track.channel { return Some(values); }
            }
            None
        }).flatten().filter_map(|value| match value {
            Sample::String(name) => Some(name.as_ref()), _ => None,
        })
    }
    /// Whether any track writes material records at frame time.
    pub fn has_material_bindings(&self) -> bool {
        !self.material_bindings.is_empty()
    }
    /// The (material, input) pairs material tracks write.
    pub fn material_inputs(&self) -> Vec<(String, String)> {
        let mut out: Vec<_> = self
            .targets()
            .filter_map(|t| match t {
                v2::TrackTarget::Material { material, input } => {
                    Some((material.clone(), input.clone()))
                }
                _ => None,
            })
            .collect();
        out.sort();
        out.dedup();
        out
    }
    pub fn bind_material(
        &mut self,
        (material, input): &(String, String),
        bindings: Vec<MaterialBinding>,
    ) -> Result<()> {
        let mut targets = Vec::new();
        for binding in bindings {
            let target = if let Some(index) = self
                .material_bindings
                .iter()
                .position(|b| b.slot == binding.slot && b.parameter == binding.parameter && b.storage == binding.storage)
            {
                index
            } else {
                let index = self.material_bindings.len();
                if binding.offset != 0. && !self.material_offsets.contains(&binding.offset) {
                    self.material_offsets.push(binding.offset);
                }
                self.material_bindings.push(binding);
                index
            };
            targets.push(target);
        }
        for track in self
            .layers
            .iter_mut()
            .flat_map(|l| &mut l.tracks)
            .filter(|t| {
                matches!(&t.target, Some(v2::TrackTarget::Material { material: m, input: i })
                    if m == material && i == input)
            })
        {
            for &target in &targets {
                let default = &self.material_bindings[target].default;
                ensure!(
                    match (&track.channel, default) {
                        (Channel::Numeric(curve), Sample::Numeric(v)) => curve.width == v.len(),
                        (Channel::Discrete { values, .. }, Sample::String(_)) =>
                            values.iter().all(|v| matches!(v, Sample::String(token) if self.material_bindings[target].enum_tokens.as_ref().is_some_and(|tokens| tokens.iter().any(|t| t == token.as_ref())))),
                        (Channel::Discrete { values, .. }, Sample::Bool(_)) =>
                            matches!(values[0], Sample::Bool(_)),
                        _ => false,
                    },
                    "material parameter type mismatch: [\"materials\",{material:?},\"inputs\",{input:?}]"
                );
            }
            track.material_targets.get_or_insert_with(Vec::new).extend(targets.iter().copied());
        }
        Ok(())
    }
    /// Record writes of the bindings at instance time `offset` (0: unshifted records).
    pub fn material_values<'a>(
        &'a self,
        samples: &'a Samples,
        offset: f32,
    ) -> impl Iterator<Item = (&'a MaterialBinding, &'a Sample)> {
        self.material_bindings
            .iter()
            .zip(&samples.material_values)
            .filter(move |(binding, _)| binding.offset == offset)
            .map(|(binding, value)| (binding, value))
    }
    /// Distinct nonzero instance time offsets with their own material record copies.
    pub fn material_offsets(&self) -> &[f32] {
        &self.material_offsets
    }
    /// Sample only the material tracks that write record copies at `offset`.
    pub fn sample_material_offset(&self, time: f32, offset: f32, samples: &mut Samples) {
        self.sample_into_where(time + offset, samples, |track| {
            track
                .material_targets
                .iter()
                .flatten()
                .any(|&t| self.material_bindings[t].offset == offset)
        });
    }
    pub fn samples(&self) -> Samples {
        Samples {
            values: self
                .layers
                .iter()
                .flat_map(|l| &l.tracks)
                .map(|track| match &track.channel {
                    Channel::Style(track) => Sample::Style(track.values[0].clone()),
                    Channel::Numeric(curve) => Sample::Numeric(vec![0.; curve.width]),
                    Channel::Discrete { values, .. } => values[0].clone(),
                })
                .collect(),
            weights: vec![0.; self.layers.iter().map(|l| l.tracks.len()).sum()],
            layer_weights: self.layers.iter().map(|l| l.source.weight).collect(),
            render: None,
            material_values: self
                .material_bindings
                .iter()
                .map(|b| b.default.clone())
                .collect(),
            clocks: vec![None; self.clocks.len()],
            held: vec![Held::No; self.layers.iter().map(|l| l.tracks.len()).sum()],
            changed: None,
            scratch: Vec::new(),
        }
    }
    /// Samples that also name, after each sampling call, the tracks it changed ([`Samples::changed`]): for a
    /// consumer that works only on what changed (the 2D canvas). Holding a constant track is the same for every
    /// consumer; naming costs a comparison of each sampled value with the one it replaces.
    pub(crate) fn samples_naming_changes(&self) -> Samples {
        Samples { changed: Some(Vec::new()), ..self.samples() }
    }
    pub fn has_discrete_tracks(&self)->bool {self.discrete_tracks}
    /// Every authored track's sampled value after `sample_into`, in layer then track order: its target,
    /// value, effective weight and whether its layer combines (additive). The 2D canvas applies these
    /// to its own nodes.
    pub(crate) fn sampled<'a>(&'a self, samples: &'a Samples) -> impl Iterator<Item = (&'a v2::TrackTarget, &'a Sample, f32, bool)> + 'a {
        self.layers.iter().flat_map(|layer| layer.tracks.iter().map(move |track| (layer, track))).enumerate()
            .filter_map(move |(index, (layer, track))| {
                let target = track.target.as_ref()?;
                Some((target, &samples.values[index], samples.weights[index], matches!(layer.source.blend, v2::Blend::Add)))
            })
    }
    /// The track at flat index `index` as [`Plan::sampled`] yields it; `None` for a track without an authored
    /// target. Flat indices are what [`Samples::changed`] names.
    pub(crate) fn sampled_at<'a>(&'a self, samples: &'a Samples, index: usize) -> Option<(&'a v2::TrackTarget, &'a Sample, f32, bool)> {
        let layer_index = self.layer_of(index);
        let layer = self.layers.get(layer_index)?;
        let track = layer.tracks.get(index.checked_sub(*self.layer_starts.get(layer_index)?)?)?;
        Some((track.target.as_ref()?, &samples.values[index], samples.weights[index], matches!(layer.source.blend, v2::Blend::Add)))
    }
    /// How many tracks the plan samples, authored or imported: one more than the largest flat index.
    pub(crate) fn track_count(&self) -> usize {
        self.layers.iter().map(|layer| layer.tracks.len()).sum()
    }
    /// Per sampled track, in [`Plan::sampled`] order: the one paint form every value it can take has (numeric
    /// channels are colours; a CSS paint track's keys must all be colours, or gradients of one form).
    pub(crate) fn sampled_paint_forms(&self) -> impl Iterator<Item = Option<crate::canvas_draw::PaintForm>> + '_ {
        self.layers.iter().flat_map(|layer| layer.tracks.iter()).filter(|track| track.target.is_some()).map(|track| match &track.channel {
            Channel::Numeric(_) => Some(crate::canvas_draw::PaintForm::Solid),
            Channel::Style(style) => {
                let mut forms = style.values.iter().map(|value| match value.as_ref() {
                    super::style::StyleValue::Paint(paint) => paint.form(),
                    _ => None,
                });
                let first = forms.next().flatten()?;
                forms.all(|form| form.as_ref() == Some(&first)).then_some(first)
            }
            _ => None,
        })
    }
    pub fn sample_into(&self, time: f32, samples: &mut Samples) {
        self.sample_into_where(time, samples, |_| true);
    }
    pub fn sample_request_into(&self, time:f32, request: &vork_time::Request, samples: &mut Samples) -> Result<()> {
        self.sample_where_request(time, Some(request), samples, |_| true)
    }
    fn sample_into_where(
        &self,
        time: f32,
        samples: &mut Samples,
        wanted: impl Fn(&PropertyTrack) -> bool,
    ) {
        self.sample_where_request(time, None, samples, wanted).expect("numeric sampling has no exact decisions");
    }
    fn sample_where_request(
        &self,
        time: f32,
        request: Option<&vork_time::Request>,
        samples: &mut Samples,
        wanted: impl Fn(&PropertyTrack) -> bool,
    ) -> Result<()> {
        let live = self.live.get().map(|values| values.read().expect("live values lock poisoned"));
        for (value, binding) in samples
            .material_values
            .iter_mut()
            .zip(&self.material_bindings)
        {
            value.clone_from(&binding.default);
        }
        self.start_sample(samples, 0..self.layers.len());
        if self.has_clocks {
            let scene = match request { Some(request) => request.exact()?, None => Germ::point(vork_time::exact::ieee(time)?) };
            samples.clocks = self.clocks(&scene)?;
        }
        let mut index = 0;
        let mut exact_request = None;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            // A node-owned layer runs on its owner's clock, retimed by its own offset, speed and loop.
            let mut exact_local = match layer.owner_node.and_then(|owner| samples.clocks.get(owner).cloned().flatten()) {
                Some(instant) => Some(layer.exact_local(&instant)?),
                None => None,
            };
            let owner_local = exact_local.as_ref().map(|local| local.numeric()).transpose()?;
            let count = layer.tracks.len();
            self.sample_layer(layer_index, index, time, owner_local, (index..index + count).map(|flat| (flat, ())), |samples, flat, track, ()| {
                // Rule 1: a track on an absent node, or inside an absent subtree, is not evaluated.
                let absent = self.subjects.get(flat).copied().flatten().is_some_and(|node| self.absent(samples, node));
                if absent || !(track.weight_target.is_some() || wanted(track)) { return Ok(None); }
                let upper = match track.discrete_keys.as_ref() {
                    Some(keys) => {
                        if exact_local.is_none() {
                            if let Some(request) = request {
                                if exact_request.is_none() {exact_request=Some(request.exact()?);}
                                exact_local=Some(layer.exact_local(exact_request.as_ref().unwrap())?);
                            }
                        }
                        exact_local.as_ref().map(|local| keys.upper(local))
                    }
                    None => None,
                };
                // Only a style channel reads the exact instant itself.
                let exact_local = if matches!(track.channel, Channel::Style(_)) { exact_local.clone() } else { None };
                Ok(Some(TrackRead { upper, exact_local }))
            }, samples, live.as_deref())?;
            index += count;
        }
        Ok(())
    }
    /// The start of one sample, for pictures and for sound: layer weights start at their authored values (weight
    /// tracks then key them), and the account of which tracks changed ([`Samples::changed`]) starts again.
    fn start_sample(&self, samples: &mut Samples, layers: impl IntoIterator<Item = usize>) {
        for layer in layers { samples.layer_weights[layer] = self.layers[layer].source.weight; }
        if let Some(changed) = &mut samples.changed { changed.clear(); }
    }
    /// One layer at one sample: what a layer does with its clock, its weight and its tracks,
    /// for pictures and for sound. `first` is the flat index of the layer's first track and
    /// `tracks` the flat indices to evaluate, in order, each with whatever `read` needs.
    /// A scene layer reads the sample's `time`; a node-owned layer reads `owner_local`, its
    /// owner's instant as the layer's numeric local time, and with an absent owner (`None`)
    /// evaluates nothing. `read` says how a track is read, or `None` when it is not evaluated
    /// (its node is absent, or the caller does not consume it).
    #[allow(clippy::too_many_arguments)]
    fn sample_layer<T>(
        &self,
        layer_index: usize,
        first: usize,
        time: f32,
        owner_local: Option<f32>,
        tracks: impl IntoIterator<Item = (usize, T)>,
        mut read: impl FnMut(&Samples, usize, &PropertyTrack, T) -> Result<Option<TrackRead>>,
        samples: &mut Samples,
        live: Option<&LiveValues>,
    ) -> Result<()> {
        let layer = &self.layers[layer_index];
        let weight = samples.layer_weights[layer_index];
        let local = match layer.owner_node {
            Some(_) => owner_local,
            None => Some(layer.source.offset + time * layer.source.speed),
        }.map(|local| layer.looped(local));
        let additive = matches!(layer.source.blend, v2::Blend::Add);
        for (index, given) in tracks {
            let track = &layer.tracks[index - first];
            let read = match local {
                Some(_) if weight != 0. => read(&*samples, index, track, given)?,
                _ => None,
            };
            match (local, read) {
                (Some(local), Some(read)) => self.sample_track(index, track, local, read.upper, read.exact_local.as_ref(), weight, additive, samples, live),
                _ => {
                    // A track that is no longer consumed changed: its weight went to zero.
                    if samples.weights[index].to_bits() != 0f32.to_bits() {
                        if let Some(named) = &mut samples.changed { named.push(index as u32); }
                    }
                    samples.weights[index] = 0.;
                }
            }
        }
        Ok(())
    }
    /// One track at its layer's local time, with the weight its layer and its node's presence
    /// give it: the channel's value (inside key interval `upper` when its keys are read exactly),
    /// then the material inputs and the layer weight it keys. Pictures and sound both sample here.
    ///
    /// The track is sampled when its weight changed or the value it holds no longer stands ([`Held`]): a track
    /// past its last key, before its first, or between the keys of a step is left as it is and costs one
    /// comparison. It is named in [`Samples::changed`] when its weight changed or its new value differs from the
    /// one it held. What a sample sums afresh from every weighted track (material inputs, layer weights) is
    /// summed from each of them, with the value it holds.
    #[allow(clippy::too_many_arguments)]
    fn sample_track(
        &self,
        index: usize,
        track: &PropertyTrack,
        local: f32,
        upper: Option<usize>,
        exact_local: Option<&Germ>,
        weight: f32,
        additive: bool,
        samples: &mut Samples,
        live: Option<&LiveValues>,
    ) {
        let mut changed = samples.weights[index].to_bits() != weight.to_bits();
        samples.weights[index] = weight;
        if weight == 0. {
            if changed { if let Some(named) = &mut samples.changed { named.push(index as u32); } }
            return;
        }
        // An image's time and a clock's speed are read from their curves by the clocks, and a camera cut from
        // its exact keys: no value is written here for them.
        let keyed = match &track.channel {
            Channel::Numeric(_) => track.time_curve.is_none(),
            Channel::Discrete { .. } => track.camera_keys.is_none(),
            Channel::Style(_) => true,
        };
        if let Some(value) = live.and_then(|values| values.values.get(&index)) {
            // A published live value stands in for the track's keys.
            let Sample::Numeric(sample) = &mut samples.values[index] else { unreachable!() };
            if sample.len() != value.len() || sample.iter().zip(value).any(|(a, b)| a.to_bits() != b.to_bits()) {
                sample.clone_from(value);
                changed = true;
            }
            samples.held[index] = Held::No;
        } else if keyed {
            // The value stands when the weight does and the time is still where the keys said the value is
            // constant.
            let stands = !changed && match (samples.held[index], upper) {
                (Held::Key(held), Some(upper)) => held == upper,
                (Held::Span { from, until }, None) => from <= local && local < until,
                _ => false,
            };
            if !stands {
                let times = track.channel.times();
                let exact = upper.is_some();
                let upper = upper.unwrap_or_else(|| times.partition_point(|t| *t <= local));
                match &track.channel {
                    Channel::Style(style) => {
                        let value = style.sample(local, Some(upper), exact_local);
                        let Sample::Style(held) = &mut samples.values[index] else { unreachable!() };
                        if samples.changed.is_some() && !std::sync::Arc::ptr_eq(held, &value) && **held != *value { changed = true; }
                        *held = value;
                    }
                    Channel::Numeric(curve) => {
                        let Sample::Numeric(value) = &mut samples.values[index] else {
                            unreachable!()
                        };
                        // Into the scratch, then into the track's own storage when it differs: each track
                        // keeps the allocation it was given, of its own width.
                        if samples.changed.is_some() {
                            curve.sample_interval(local, upper, &mut samples.scratch);
                            if value.len() != samples.scratch.len() || value.iter().zip(&samples.scratch).any(|(a, b)| a.to_bits() != b.to_bits()) {
                                value.clear();
                                value.extend_from_slice(&samples.scratch);
                                changed = true;
                            }
                        } else {
                            // Nobody reads this consumer's account: the value is sampled where it is kept.
                            curve.sample_interval(local, upper, value);
                        }
                    }
                    Channel::Discrete { values, .. } => {
                        let value = &values[upper.saturating_sub(1)];
                        if samples.values[index] != *value {
                            samples.values[index] = value.clone();
                            changed = true;
                        }
                    }
                }
                samples.held[index] = if !track.channel.constant_at(upper) {
                    Held::No
                } else if exact {
                    Held::Key(upper)
                } else {
                    Held::Span {
                        from: upper.checked_sub(1).map_or(f32::NEG_INFINITY, |key| times[key]),
                        until: times.get(upper).copied().unwrap_or(f32::INFINITY),
                    }
                };
            }
        }
        for &target in track.material_targets.iter().flatten() {
            blend_sample(
                &mut samples.material_values[target],
                &samples.values[index],
                weight,
                additive,
            );
        }
        if let Some(target) = track.weight_target {
            if let Sample::Numeric(value) = &samples.values[index] {
                blend(
                    std::slice::from_mut(&mut samples.layer_weights[target]),
                    value,
                    weight,
                    additive,
                    Combine::Add,
                );
            }
        }
        if changed { if let Some(named) = &mut samples.changed { named.push(index as u32); } }
    }
    pub fn apply_asset_visibility(&self, asset: usize, samples: &Samples, visibility: &mut [v2::Visibility]) {
        let Some(bindings) = self.asset_bindings.get(asset) else { return; };
        for &(index, node) in &bindings.visibility {
            if samples.weights[index] >= 0.5 {
                let Sample::Bool(value) = samples.values[index] else { unreachable!() };
                visibility[node].visible = value;
            }
        }
    }
    pub fn apply_asset(
        &self,
        asset: usize,
        samples: &Samples,
        mut apply: impl FnMut(AssetTarget, &Sample, f32, bool),
    ) {
        let Some(bindings) = self.asset_bindings.get(asset) else { return; };
        for &(index, target, additive) in &bindings.transforms {
            let weight = samples.weights[index];
            if weight != 0. {
                apply(target, &samples.values[index], weight, additive);
            }
        }
    }
    pub fn apply_nodes(&self, samples: &Samples, values: &mut [Option<super::super::EvalNode>]) {
        self.apply_nodes_where(samples, values, |_| true);
    }
    pub fn apply_nodes_subset(
        &self,
        samples: &Samples,
        values: &mut [Option<super::super::EvalNode>],
        nodes: &[usize],
    ) {
        self.apply_nodes_where(samples, values, |node| nodes.binary_search(&node).is_ok());
    }
    fn apply_nodes_where(
        &self,
        samples: &Samples,
        values: &mut [Option<super::super::EvalNode>],
        wanted: impl Fn(usize) -> bool,
    ) {
        for binding in self
            .reset_bindings
            .iter()
            .filter(|binding| wanted(binding.node))
        {
            reset_node(&mut values[binding.node].as_mut().unwrap().node, binding);
        }
        let mut index = 0;
        for layer in &self.layers {
            let additive = matches!(layer.source.blend, v2::Blend::Add);
            for track in &layer.tracks {
                if samples.weights[index] != 0. {
                    for binding in track
                        .node_targets
                        .iter()
                        .filter(|binding| wanted(binding.node))
                    {
                        apply_node(
                            &mut values[binding.node].as_mut().unwrap().node,
                            &binding.property,
                            &samples.values[index],
                            samples.weights[index],
                            additive,
                        );
                    }
                }
                index += 1;
            }
        }
        // Rule 1: an absent node has no picture and lights nothing, its subtree included.
        for (node, spec) in self.clocks.iter().enumerate() {
            if spec.gated && wanted(node) && samples.clocks.get(node).is_some_and(Option::is_none) {
                if let Some(value) = values[node].as_mut() { value.node.visibility.visible = false; }
            }
        }
    }
    pub fn start_camera(&self, base: &v2::Render, time: impl Into<vork_time::Request>) -> Option<String> {
        let time=time.into();
        let mut samples = self.samples();
        self.sample_into(time.numeric().expect("validated initial time"), &mut samples);
        self.camera_selection(&samples,&time).expect("validated camera request")
            .map(str::to_owned).or_else(||base.camera.clone())
    }
    pub(crate) fn has_shutter_animation(&self)->bool {
        self.targets().any(|target|matches!(target,v2::TrackTarget::Render(v2::RenderProperty::MotionBlurShutter)))
    }
    pub(crate) fn sample_shutter(&self,time:f32,samples:&mut Samples,base:&v2::Render) {
        self.sample_into_where(time,samples,|track|matches!(track.target,Some(v2::TrackTarget::Render(v2::RenderProperty::MotionBlurShutter))));
        self.apply_render(samples,base);
    }
    pub fn apply_render(&self, samples: &mut Samples, base: &v2::Render) {
        let render = samples.render.get_or_insert_with(|| base.clone());
        render.clone_from(base);
        let mut index = 0;
        for layer in &self.layers {
            let additive = matches!(layer.source.blend, v2::Blend::Add);
            for track in &layer.tracks {
                let weight = samples.weights[index];
                let property = match &track.target {
                    Some(v2::TrackTarget::Render(property)) => Some(*property),
                    _ => None,
                };
                if let (Some(property), true) = (property, weight != 0.) {
                    use v2::RenderProperty as R;
                    match (property, &samples.values[index]) {
                        (R::Background, Sample::Numeric(value)) => {
                            if let v2::Background::Color(color) = &mut render.background {
                                blend(color, value, weight, additive, Combine::Add);
                            }
                        },
                        (R::MotionBlurShutter, Sample::Numeric(value)) => {
                            if let Some(m) = &mut render.motion_blur {
                                blend(std::slice::from_mut(&mut m.shutter), value, weight, additive, Combine::Add);
                            }
                        }
                        _ => {}
                    }
                }
                index += 1;
            }
        }
    }
    fn apply_filtered(
        &self,
        samples: &Samples,
        wanted: impl Fn(&PropertyTrack) -> bool,
        mut apply: impl FnMut(&PropertyTrack, &Sample, f32, bool),
    ) {
        let mut index = 0;
        for layer in &self.layers {
            let additive = matches!(layer.source.blend, v2::Blend::Add);
            for track in &layer.tracks {
                if track.weight_target.is_none() && samples.weights[index] != 0. && wanted(track) {
                    apply(
                        track,
                        &samples.values[index],
                        samples.weights[index],
                        additive,
                    );
                }
                index += 1;
            }
        }
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;
    #[test]
    fn live_publication_replaces_sampling_in_cloned_plans_without_rebinding() -> Result<()> {
        let scene = v2::Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera","name":"camera"}],
            "animation":[{"tracks":[{"path":["nodes","camera","translation"],"keys":[[0,[-10,-10,-10]],[1,[10,10,10]]]}]}]}"#)?;
        let plan = Plan::compile(&scene, std::path::Path::new("."))?;
        let imported_copy = plan.clone();
        let target = scene.track_targets()?.remove(0);
        let binding = plan.bind_live(&target, &[vec![1.,2.,3.]])?[0];
        assert_eq!(imported_copy.live_generation(), 0);
        let mut samples = imported_copy.samples();
        imported_copy.sample_into(0.75, &mut samples);
        assert_eq!(samples.values[0], Sample::Numeric(vec![1.,2.,3.]));
        plan.publish_live([(binding, &[4.,5.,6.][..])].into_iter());
        assert_eq!(imported_copy.live_generation(), 1);
        plan.publish_live(std::iter::empty());
        assert_eq!(imported_copy.live_generation(), 1);
        imported_copy.sample_into(0.25, &mut samples);
        assert_eq!(samples.values[0], Sample::Numeric(vec![4.,5.,6.]));
        Ok(())
    }
}
#[cfg(test)]
mod held_tests {
    use super::*;
    /// Equal to the bit, for numeric samples.
    fn same_bits(a: &Sample, b: &Sample) -> bool {
        match (a, b) {
            (Sample::Numeric(a), Sample::Numeric(b)) => a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
            _ => a == b,
        }
    }
    /// Sampling with holds equals sampling everything, and the tracks it names are exactly those whose value or
    /// weight differs from the call before: before the first key, after the last and between step keys a track
    /// is not named; on a ramp it is.
    #[test]
    fn a_constant_track_is_not_named_and_the_values_equal_a_fresh_sampling() -> Result<()> {
        let scene = v2::Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera","name":"camera"}],
            "animation":[
                {"tracks":[
                    {"path":["nodes","camera","translation"],"keys":[[1,[0,0,0]],[2,[10,10,10]]]},
                    {"path":["nodes","camera","scale"],"keys":[[0,[1,1,1]],[1,[2,2,2]],[2,[3,3,3]]],"interpolation":"STEP"},
                    {"path":["animation","late","weight"],"keys":[[0,0],[3,0],[3.5,1]]}]},
                {"name":"late","weight":0,"tracks":[
                    {"path":["nodes","camera","translation"],"keys":[[0,[5,5,5]],[1,[5,5,5]]]}]}]}"#)?;
        let plan = Plan::compile(&scene, std::path::Path::new("."))?;
        let mut samples = plan.samples_naming_changes();
        // The same sampling for a consumer that asks for no account: the same holds, the same values.
        let mut unnamed = plan.samples();
        let mut named = Vec::new();
        for time in [0., 0.5, 0.75, 1., 1.5, 1.75, 2., 2.5, 3., 3.25, 3.5, 4., 1.5, 1.5] {
            let before = (samples.values.clone(), samples.weights.clone());
            plan.sample_into(time, &mut samples);
            plan.sample_into(time, &mut unnamed);
            // A sampling that holds nothing: the same plan from fresh samples.
            let mut fresh = plan.samples();
            plan.sample_into(time, &mut fresh);
            for index in 0..samples.values.len() {
                assert_eq!(samples.weights[index].to_bits(), fresh.weights[index].to_bits(), "weight of track {index} at {time}");
                assert_eq!(unnamed.weights[index].to_bits(), fresh.weights[index].to_bits(), "unnamed weight of track {index} at {time}");
                if samples.weights[index] != 0. {
                    assert!(same_bits(&samples.values[index], &fresh.values[index]), "track {index} at {time}");
                    assert!(same_bits(&unnamed.values[index], &fresh.values[index]), "unnamed track {index} at {time}");
                }
            }
            let differs: Vec<u32> = (0..samples.values.len())
                .filter(|&index| before.1[index].to_bits() != samples.weights[index].to_bits()
                    || (samples.weights[index] != 0. && !same_bits(&before.0[index], &samples.values[index])))
                .map(|index| index as u32).collect();
            assert_eq!(samples.changed(), differs, "at {time}");
            named.push(samples.changed().to_vec());
        }
        // The ramp is named while it ramps (1 to 2); the step when a key is crossed; the weight track on its ramp,
        // with the track it drives.
        assert_eq!(named[1], Vec::<u32>::new(), "nothing moves before the ramp's first key");
        assert_eq!(named[4], vec![0], "only the ramp moves at 1.5");
        assert_eq!(named[7], Vec::<u32>::new(), "everything stands after the last keys");
        assert_eq!(named[9], vec![2, 3], "the weight ramps and the layer it drives arrives");
        assert_eq!(named[13], Vec::<u32>::new(), "the same time again changes nothing");
        Ok(())
    }
}

pub(super) fn numeric_buffer(
    scene: &v2::Scene,
    name: &str,
    base: &std::path::Path,
    workspace: Option<&std::collections::BTreeMap<String, crate::buffers::AttributeData>>,
) -> Result<(Vec<f32>, usize)> {
    let decoded;
    let data = if let Some(data) = workspace.and_then(|data| data.get(name)) {
        data
    } else {
        decoded = crate::buffers::AttributeData::decode(
            scene
                .buffers
                .get(name)
                .with_context(|| format!("unknown animation buffer {name}"))?,
            base,
        )?;
        &decoded
    };
    let values = data
        .words
        .iter()
        .map(|v| match data.element_type {
            v2::BufferType::I32 => (*v as i32) as f32,
            v2::BufferType::U32 | v2::BufferType::Uvec4 => *v as f32,
            _ => f32::from_bits(*v),
        })
        .collect();
    Ok((values, data.width))
}

/// Bind paths once; frame sampling touches only numeric values on installed nodes.
#[derive(Clone)]
struct NodeBinding {
    node: usize,
    property: v2::NodeProperty,
    base: Sample,
}
impl Plan {
    /// Whether a track animates this node's own translation, rotation or scale.
    pub fn transforms_node(&self, node: usize) -> bool {
        self.reset_bindings.iter().any(|b| b.node == node
            && matches!(b.property, v2::NodeProperty::Translation | v2::NodeProperty::Rotation | v2::NodeProperty::Scale))
    }
    pub fn mark_nodes(&mut self, nodes: &mut [super::super::EvalNode], default_camera: usize) -> Result<()> {
        self.reset_bindings.clear();
        self.cameras = nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| {
                matches!(n.node.kind, v2::NodeKind::Camera { .. }).then(|| (n.node.name.clone(), i))
            })
            .collect();
        let by_name: std::collections::HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.node.name.is_empty())
            .map(|(i, n)| (n.node.name.clone(), i))
            .collect();
        self.bind_clocks(nodes)?;
        let mut flat = 0;
        for layer in &mut self.layers {
            for track in &mut layer.tracks {
                let flat_index = flat;
                flat += 1;
                if track.weight_target.is_some() {
                    continue;
                }
                if let Some(v2::TrackTarget::Node { node, property: property @ (v2::NodeProperty::Active | v2::NodeProperty::Speed) }) = &track.target {
                    // Read by the owning clock, never blended into the node's values.
                    let index = *by_name.get(node).with_context(|| format!("unknown track node {node}"))?;
                    match property {
                        v2::NodeProperty::Active => { self.clocks[index].active_track = Some(flat_index); self.clocks[index].gated = true; }
                        _ => self.clocks[index].speed_track = Some(flat_index),
                    }
                    self.has_clocks = true;
                    nodes[index].dynamic = true;
                    continue;
                }
                let path = || {
                    track
                        .target
                        .as_ref()
                        .map_or("glTF clip channel".into(), |t| format!("track {t}"))
                };
                let mut binding = match &track.target {
                    None | Some(v2::TrackTarget::Imported { .. }) => {
                        ensure!(!track.asset_targets.is_empty() || track.asset_visibility.is_some(), "unbound {}", path());
                        continue;
                    }
                    Some(v2::TrackTarget::Material { .. }) => {
                        ensure!(
                            track.material_targets.is_some(),
                            "material track requires program parameter binding: {}",
                            path()
                        );
                        continue;
                    }
                    Some(v2::TrackTarget::LayerWeight { .. } | v2::TrackTarget::ImageTime { .. }) => continue,
                    // Text tracks belong to a 2D canvas, which installs no 3D scene (canvas.rs applies them).
                    Some(v2::TrackTarget::Text { .. }) => anyhow::bail!("{}: a 2D text track in a 3D scene", path()),
                    Some(v2::TrackTarget::Render(property)) => {
                        use v2::RenderProperty as R;
                        if *property == R::Camera {
                            if let Channel::Discrete { values, .. } = &track.channel {
                                for value in values {
                                    ensure!(
                                        matches!(value,Sample::String(name) if self.cameras.contains_key(name.as_ref())),
                                        "camera track must reference installed cameras"
                                    );
                                }
                            }
                        }
                        continue;
                    }
                    Some(v2::TrackTarget::Node { node, property }) => NodeBinding {
                        node: *by_name
                            .get(node)
                            .with_context(|| format!("unknown track node {node}"))?,
                        property: property.clone(),
                        base: Sample::Numeric(vec![]),
                    },
                };
                let node = &mut nodes[binding.node];
                if let Channel::Numeric(curve) = &track.channel {
                    let mut base = Vec::new();
                    ensure!(
                        node_numeric(&mut node.node, &binding.property, |value, _| base
                            .extend_from_slice(value))
                            && base.len() == curve.width,
                        "unsupported property or value shape: {}",
                        path()
                    );
                    binding.base = Sample::Numeric(base);
                } else {
                    let v2::NodeProperty::Visibility(ray) = binding.property else {
                        anyhow::bail!("{} requires numeric values", path());
                    };
                    ensure!(
                        matches!(&track.channel,Channel::Discrete{values,..} if matches!(values[0],Sample::Bool(_))),
                        "{} requires boolean values",
                        path()
                    );
                    binding.base = Sample::Bool(*node_bool(&mut node.node, ray));
                }
                node.dynamic = true;
                if !self
                    .reset_bindings
                    .iter()
                    .any(|b| b.node == binding.node && b.property == binding.property)
                {
                    self.reset_bindings.push(binding.clone());
                }
                track.node_targets = vec![binding];
            }
        }
        for i in 0..nodes.len() {
            nodes[i].dynamic |= nodes[i].parent.is_some_and(|p| nodes[p].dynamic);
            // Existence is inherited: a gated ancestor gates the subtree.
            if nodes[i].parent.is_some_and(|p| self.clocks[p].gated) { self.clocks[i].gated = true; }
        }
        for i in 0..nodes.len() {
            if self.clocks[i].gated {
                let binding = NodeBinding { node: i, property: v2::NodeProperty::Visibility(v2::Ray::All), base: Sample::Bool(nodes[i].node.visibility.visible) };
                if !self.reset_bindings.iter().any(|b| b.node == i && b.property == binding.property) { self.reset_bindings.push(binding); }
            }
        }
        self.refresh_layer_starts();
        self.subjects = self.layers.iter().flat_map(|layer| layer.tracks.iter().map(move |track| (layer, track)))
            .map(|(layer, track)| match &track.target {
                Some(v2::TrackTarget::Node { node, .. } | v2::TrackTarget::Imported { asset: node, .. }) => by_name.get(node).copied(),
                None => layer.source.clip.as_ref().and_then(|clip| by_name.get(&clip.target).copied()),
                Some(_) => None,
            })
            .collect();
        let _=default_camera;
        Ok(())
    }
    /// The 2D canvas's binding (§8): its nodes' clocks, the parent- or scene-owned tracks that key a node's
    /// `active` or `time.speed`, and each track's subject, so a track on an absent node is not evaluated. The
    /// canvas applies the other tracks' values to its own nodes (`sampled`).
    pub(crate) fn mark_canvas_nodes<N: ClockNode>(&mut self, nodes: &mut [N]) -> Result<()> {
        self.bind_clocks(nodes)?;
        let by_name: std::collections::HashMap<&str, usize> = nodes.iter().enumerate()
            .filter(|(_, n)| !n.node().name.is_empty()).map(|(i, n)| (n.node().name.as_str(), i)).collect();
        let mut flat = 0;
        for layer in &mut self.layers {
            for track in &mut layer.tracks {
                let flat_index = flat;
                flat += 1;
                if let Some(v2::TrackTarget::Node { node, property: property @ (v2::NodeProperty::Active | v2::NodeProperty::Speed) }) = &track.target {
                    let index = *by_name.get(node.as_str()).with_context(|| format!("unknown track node {node}"))?;
                    match property {
                        v2::NodeProperty::Active => { self.clocks[index].active_track = Some(flat_index); self.clocks[index].gated = true; }
                        _ => self.clocks[index].speed_track = Some(flat_index),
                    }
                    self.has_clocks = true;
                }
            }
        }
        for i in 0..nodes.len() {
            if nodes[i].parent().is_some_and(|p| self.clocks[p].gated) { self.clocks[i].gated = true; }
        }
        self.refresh_layer_starts();
        self.subjects = self.layers.iter().flat_map(|layer| layer.tracks.iter())
            .map(|track| match &track.target {
                Some(v2::TrackTarget::Node { node, .. } | v2::TrackTarget::Text { node, .. }) => by_name.get(node.as_str()).copied(),
                _ => None,
            })
            .collect();
        Ok(())
    }
    pub fn camera_at(&self, time: f32, fallback: usize) -> usize {
        self.camera_at_request(&vork_time::Request::Numeric(time),fallback).expect("validated camera request")
    }
}

fn blend_sample(base: &mut Sample, value: &Sample, weight: f32, additive: bool) {
    match (base, value) {
        (Sample::Numeric(base), Sample::Numeric(value)) => {
            blend(base, value, weight, additive, Combine::Add)
        }
        (Sample::Bool(base), Sample::Bool(value)) if weight >= 0.5 && !additive => *base = *value,
        (Sample::String(base), Sample::String(value)) if weight >= 0.5 && !additive => {
            base.clone_from(value)
        }
        _ => {}
    }
}
fn reset_node(node: &mut v2::Node, binding: &NodeBinding) {
    match (&binding.base, &binding.property) {
        (Sample::Numeric(base), property) => {
            node_numeric(node, property, |value, _| value.copy_from_slice(base));
        }
        (Sample::Bool(base), v2::NodeProperty::Visibility(ray)) => {
            *node_bool(node, *ray) = *base;
        }
        _ => {}
    }
}
fn apply_node(
    node: &mut v2::Node,
    property: &v2::NodeProperty,
    sample: &Sample,
    weight: f32,
    additive: bool,
) {
    match (sample, property) {
        (Sample::Numeric(sample), property) => {
            node_numeric(node, property, |base, kind| {
                blend(base, sample, weight, additive, kind)
            });
        }
        (Sample::Bool(sample), v2::NodeProperty::Visibility(ray)) if weight >= 0.5 && !additive => {
            *node_bool(node, *ray) = *sample;
        }
        _ => {}
    }
}
fn node_bool(node: &mut v2::Node, ray: v2::Ray) -> &mut bool {
    let v = &mut node.visibility;
    match ray {
        v2::Ray::All => &mut v.visible,
        v2::Ray::Camera => &mut v.camera,
        v2::Ray::Shadow => &mut v.shadow,
        v2::Ray::Diffuse => &mut v.diffuse,
        v2::Ray::Glossy => &mut v.glossy,
        v2::Ray::Transmission => &mut v.transmission,
    }
}
/// Apply `apply` to the numeric value of `property`; false when the node has none.
fn node_numeric(
    node: &mut v2::Node,
    property: &v2::NodeProperty,
    mut apply: impl FnMut(&mut [f32], Combine),
) -> bool {
    use v2::{NodeKind as K, NodeProperty as P};
    macro_rules! scalar {
        ($v:expr) => {{
            apply(std::slice::from_mut($v), Combine::Add);
            true
        }};
    }
    match (&mut node.kind, property) {
        (_, P::AudioGain) => match &mut node.audio {
            Some(audio) => scalar!(&mut audio.gain),
            None => false,
        },
        (_, P::Translation) => {
            apply(&mut node.transform.translation, Combine::Add);
            true
        }
        (_, P::Rotation) => {
            apply(node.transform.rotation_mut(), Combine::Rotation);
            true
        }
        (_, P::Scale) => {
            apply(&mut node.transform.scale, Combine::Multiply);
            true
        }
        (K::Mesh { morph_weights, .. }, P::MorphWeight(name)) => {
            match morph_weights.get_mut(name) {
                Some(value) => scalar!(value),
                None => false,
            }
        }
        (
            K::Light {
                color,
                shape,
                image, material,
                ..
            },
            P::Color,
        ) => {
            apply(
                color.get_or_insert(shape.default_color(image.is_some() || material.is_some())),
                Combine::Add,
            );
            true
        }
        (
            K::Light {
                energy: Some(value),
                ..
            },
            P::Energy,
        ) => scalar!(value),
        (K::Light { temperature: Some(value), .. }, P::Temperature) => scalar!(value),
        (K::Light { angle, .. }, P::Angle) => scalar!(angle),
        (K::Light { radius, .. }, P::Radius) => scalar!(radius),
        (K::Light { spot_blend, .. }, P::SpotBlend) => scalar!(spot_blend),
        (K::Light { spot_size, .. }, P::SpotSize) => scalar!(spot_size),
        (K::Light { size: width, .. }, P::Width) => scalar!(width),
        (K::Light { size: width, size_y: height, .. }, P::Height) => {
            let height = height.get_or_insert(*width);
            scalar!(height)
        },
        _ => false,
    }
}

#[cfg(test)]
mod quaternion_tests {
    use super::*;
    #[test]
    fn antipodal_cubic_midpoint_retains_valid_direction() -> Result<()> {
        let track: v2::Track = vork_scene::v2::from_test_value(
            serde_json::json!({"path":["nodes","actor","rotation"],"keys":[[0,[0,0,0,1],[0,0,0,0],[0,0,0,0]],[1,[0,0,0,-1],[0,0,0,0],[0,0,0,0]]],"interpolation":"CUBICSPLINE"}),
        )?;
        let v2::Values::Inline(values) = &track.values else { unreachable!() };
        let curve = Curve::authored(&track, true, vec![0., 1.], values)?;
        let mut value = vec![];
        curve.sample(0.5, &mut value);
        assert_eq!(value, [0., 0., 0., 1.]);
        Ok(())
    }
}

#[cfg(test)]
mod camera_cut_tests {
    use super::*;
    #[test]
    fn node_radius_blends_point_and_spot_emitters() {
        for shape in ["point", "spot"] {
            let mut node: v2::Node = vork_scene::v2::from_test_value(serde_json::json!({"type":"light","shape":shape,"radius":0.1})).unwrap();
            apply_node(&mut node, &v2::NodeProperty::Radius, &Sample::Numeric(vec![0.3]), 0.5, false);
            let v2::NodeKind::Light { radius, .. } = node.kind else { unreachable!() };
            assert!((radius-0.2).abs()<1e-6);
        }
    }
    fn plan(animation: serde_json::Value) -> Result<Plan> {
        let input: v2::Scene = vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,"render":{"duration":2,"camera":"a"},
            "nodes":[{"type":"camera","name":"a","material":{"program":"std/camera","inputs":{"projection":"persp"}}},{"type":"camera","name":"b","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],
            "animation":animation
        }))?;
        input.validate()?;
        let mut plan = Plan::compile(&input, std::path::Path::new("."))?;
        let mut nodes: Vec<_> = ["a", "b"]
            .into_iter()
            .map(|name| crate::EvalNode { camera: None,
                node: vork_scene::v2::from_test_value(serde_json::json!({"type":"camera","name":name}))
                    .unwrap(),
                parent: None,
                world: glam::Mat4::IDENTITY,
                dynamic: false,
                embedded: None,
            })
            .collect();
        plan.mark_nodes(&mut nodes, 0)?;
        Ok(plan)
    }
    #[test]
    fn start_camera_uses_negative_offset_loop_and_ordered_layers() -> Result<()> {
        let plan = plan(serde_json::json!([
            {"tracks":[{"path":["render","camera"],"keys":[[0,"a"]]}]},
            {"offset":-0.25,"loop":true,"tracks":[{
                "path":["render","camera"],"keys":[[0,"a"],[0.5,"b"],[1,"a"]]
            }]}
        ]))?;
        assert_eq!(plan.start_camera(&v2::Render::default(), 0.).as_deref(), Some("b"));
        assert_eq!(plan.start_camera(&v2::Render::default(), 0.5).as_deref(), Some("a"));
        assert_eq!(plan.camera_at(0., 0), 1);
        Ok(())
    }
    #[test]
    fn start_camera_uses_buffer_backed_layer_weight() -> Result<()> {
        let scene: v2::Scene = vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,"nodes":[{"type":"camera","name":"a"},{"type":"camera","name":"b"}],
            "buffers":{"times":{"type":"f32","data":[0,1]},"weights":{"type":"f32","data":[1,0]}},
            "animation":[
                {"tracks":[{"path":["animation","cut","weight"],"keys":{"times":"times","values":"weights"}}]},
                {"name":"cut","weight":0,"tracks":[{"path":["render","camera"],"keys":[[0,"b"]]}]}
            ]
        }))?;
        let plan = Plan::compile(&scene, std::path::Path::new("."))?;
        assert_eq!(plan.start_camera(&scene.render, 0.).as_deref(), Some("b"));
        Ok(())
    }
    #[test]
    fn numeric_clock_controls_accept_scalar_arrays_tiny_speed_and_cubics() -> Result<()> {
        for weight in [serde_json::json!(0.5), serde_json::json!([0.5])] {
            let plan = plan(serde_json::json!([
                {"tracks":[{"path":["animation","cut","weight"],"keys":[[0,weight]]}]},
                {"name":"cut","weight":0,"speed":1e-20,"tracks":[{"path":["render","camera"],"keys":[[0,"b"]]}]}
            ]))?;
            assert_eq!(plan.camera_at(0.1234567,0),1);
        }
        let scene: v2::Scene = vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,"nodes":[{"type":"camera","name":"a"}],
            "images":{"clip":{"file":"missing%04d.png"}},
            "animation":[{"tracks":[{"path":["images","clip","time"],"keys":[[0,0],[1,1]],"interpolation":"CUBICSPLINE"}]}]
        }))?;
        scene.validate()?;
        let plan = Plan::compile(&scene,std::path::Path::new("."))?;
        let mut samples=plan.samples();
        let t=0.1234567;
        plan.sample_into(t,&mut samples);
        let request=vork_time::Request::Numeric(t).exact()?;
        let actual=plan.image_time_exact(&samples,&request,"clip",vork_time::Request::Numeric(0.).exact()?)?.numeric()?;
        assert!((actual-(3.*t*t-2.*t*t*t)).abs()<1e-7);
        Ok(())
    }
    #[test]
    fn camera_cut_shutter_interval_respects_offset_speed_and_loop() -> Result<()> {
        let plan = plan(
            serde_json::json!([{"offset":0.2,"speed":2,"loop":true,"tracks":[{
                "path":["render","camera"],"interpolation":"STEP","keys":[[0,"a"],[0.4,"b"],[1,"a"]]
            }]}]),
        )?;
        for nominal in [0.25_f32,0.75,1.75,100.75] {
            let point=vork_time::Request::Numeric(nominal).exact()?;
            for target in [nominal-0.2,nominal+0.2] {
                let clamped=plan.clamp_camera_request(&point,&vork_time::Request::Numeric(target).exact()?,0)?;
                assert_eq!(plan.camera_at_request(&vork_time::Request::Exact(clamped),0)?,1);
            }
        }
        Ok(())
    }
    #[test]
    fn rational_cut_initialization_and_one_sided_history_share_identity()->Result<()> {
        use vork_time::{Request,Time,exact::{Germ,decimal}};
        let plan=plan(serde_json::json!([{"tracks":[{"path":["render","camera"],"keys":[[0,"a"],["1/24","b"]]}]}]))?;
        let cut=Request::Frame(1,Time::integer(24));
        assert_eq!(plan.start_camera(&v2::Render::default(),&cut).as_deref(),Some("b"));
        assert_eq!(plan.camera_at_request(&cut,0)?,1);
        let nominal=Germ::point(decimal("1/48")?);
        let clamped=plan.clamp_camera_request(&nominal,&Germ::point(decimal("1/12")?),0)?;
        assert_eq!(clamped,Germ::left(decimal("1/24")?));
        assert_eq!(Request::Exact(clamped).previous(Time::integer(48))?.exact()?,Germ::left(decimal("1/48")?));
        Ok(())
    }
    #[test]
    fn returning_to_a_camera_starts_a_distinct_owning_interval()->Result<()> {
        let plan=plan(serde_json::json!([{"tracks":[{"path":["render","camera"],"keys":[[0,"a"],[0.4,"b"],[0.6,"a"]]}]}]))?;
        let current=vork_time::exact::Germ::point(vork_time::exact::decimal("3/4")?);
        let previous=vork_time::exact::Germ::point(vork_time::exact::decimal("1/4")?);
        assert_eq!(plan.camera_at_request(&vork_time::Request::Exact(current.clone()),0)?,plan.camera_at_request(&vork_time::Request::Exact(previous.clone()),0)?);
        assert_ne!(plan.clamp_camera_request(&current,&previous,0)?,previous);
        Ok(())
    }
    #[test]
    fn cubic_layer_weight_keeps_both_camera_threshold_crossings() -> Result<()> {
        let plan = plan(serde_json::json!([
            {"tracks":[{"path":["animation","cut","weight"],"keys":[[0,0,0,4],[1,0,-4,0]],"interpolation":"CUBICSPLINE"}]},
            {"name":"cut","weight":0,"tracks":[{"path":["render","camera"],"keys":[[0,"b"]],"interpolation":"STEP"}]}
        ]))?;
        for (nominal,target) in [(0.15_f32,0.14_f32),(0.85,0.86)] {
            let nominal=vork_time::Request::Numeric(nominal).exact()?;
            let target=vork_time::Request::Numeric(target).exact()?;
            let clamped=plan.clamp_camera_request(&nominal,&target,0)?;
            assert_ne!(clamped,target);
            assert_eq!(plan.camera_at_request(&vork_time::Request::Exact(clamped),0)?,1);
        }
        Ok(())
    }
    #[test]
    fn visible_defaults_to_step_and_rejects_continuous_or_additive_blends() -> Result<()> {
        plan(serde_json::json!([{"tracks":[{
            "path":["nodes","a","visible"],"keys":[[0,false],[1,true]]
        }]}]))?;
        for interpolation in ["LINEAR", "CUBICSPLINE"] {
            assert!(plan(serde_json::json!([{"tracks":[{
                "path":["nodes","a","visible"],"keys":[[0,false],[1,true]],"interpolation":interpolation
            }]}])).is_err());
        }
        assert!(plan(serde_json::json!([{"blend":"combine","tracks":[{
            "path":["nodes","a","visible"],"keys":[[0,false],[1,true]]
        }]}])).is_err());
        Ok(())
    }
    #[test]
    fn additive_boolean_tracks_fail_validation() {
        assert!(plan(serde_json::json!([{"blend":"combine","tracks":[{
            "path":["nodes","a","visible_camera"],"keys":[[0,true],[1,false]],"interpolation":"STEP"
        }]}])).is_err());
    }
}

/// Conservative component envelopes, including cubic Bezier control hulls and
/// ordered layer weights. Used only while installing projected curve detail.
impl Plan {
    pub(crate) fn node_envelope(
        &self,
        node: &str,
        property: &v2::NodeProperty,
        base: &[f32],
    ) -> Vec<[f32; 2]> {
        fn add(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
            [a[0] + b[0], a[1] + b[1]]
        }
        fn mul(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
            let v = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]];
            [
                v.into_iter().fold(f32::INFINITY, f32::min),
                v.into_iter().fold(f32::NEG_INFINITY, f32::max),
            ]
        }
        fn combine(a: [f32; 2], b: [f32; 2], w: [f32; 2], additive: bool, scale: bool) -> [f32; 2] {
            if additive && scale {
                // A signed power; retain both signs if the sample crosses zero.
                let magnitude = [b[0].abs(), b[1].abs()];
                let low = if b[0] <= 0. && b[1] >= 0. {
                    0.
                } else {
                    magnitude[0].min(magnitude[1])
                };
                let high = magnitude[0].max(magnitude[1]);
                let powers = [
                    low.powf(w[0]),
                    low.powf(w[1]),
                    high.powf(w[0]),
                    high.powf(w[1]),
                ];
                let lo = powers.into_iter().fold(f32::INFINITY, f32::min);
                let hi = powers.into_iter().fold(f32::NEG_INFINITY, f32::max);
                let factor = if b[0] >= 0. {
                    [lo, hi]
                } else if b[1] <= 0. {
                    [-hi, -lo]
                } else {
                    [-hi, hi]
                };
                return mul(a, factor);
            }
            if additive {
                return add(a, mul(w, b));
            }
            if w == [1.; 2] {
                return b;
            }
            if w[0] >= 0. && w[1] <= 1. {
                return [a[0].min(b[0]), a[1].max(b[1])];
            }
            add(mul([1. - w[1], 1. - w[0]], a), mul(w, b))
        }
        fn hull(curve: &Curve) -> Vec<[f32; 2]> {
            // Euler storage holds three angles; sampled values are unit quaternions.
            if curve.euler {
                return vec![[-1., 1.]; 4];
            }
            let mut result = vec![[f32::INFINITY, f32::NEG_INFINITY]; curve.width];
            let cubic = curve.interpolation == Interpolation::Cubic;
            for key in 0..curve.times.len() {
                for (component, bound) in result.iter_mut().enumerate() {
                    let at = |key: usize, part: usize| {
                        curve.values
                            [(key * if cubic { 3 } else { 1 } + part) * curve.width + component]
                    };
                    let value = at(key, usize::from(cubic));
                    let mut include = |v: f32| {
                        bound[0] = bound[0].min(v);
                        bound[1] = bound[1].max(v);
                    };
                    include(value);
                    if cubic && key + 1 < curve.times.len() {
                        let dt = (curve.times[key + 1] - curve.times[key]) / 3.;
                        include(value + dt * at(key, 2));
                        include(at(key + 1, 1) - dt * at(key + 1, 0));
                    }
                }
            }
            result
        }
        let mut result: Vec<_> = base.iter().map(|v| [*v; 2]).collect();
        let mut weights: Vec<_> = self.layers.iter().map(|l| [l.source.weight; 2]).collect();
        for (index, layer) in self.layers.iter().enumerate() {
            let weight = weights[index];
            for track in &layer.tracks {
                let matches_node = matches!(&track.target, Some(v2::TrackTarget::Node { node: n, property: p }) if n == node && p == property);
                if track.weight_target.is_none() && !matches_node {
                    continue;
                }
                let Channel::Numeric(curve) = &track.channel else {
                    continue;
                };
                let bound = hull(curve);
                let additive = matches!(layer.source.blend, v2::Blend::Add);
                if let Some(target) = track.weight_target {
                    weights[target] = combine(weights[target], bound[0], weight, additive, false);
                }
                if matches_node {
                    for (out, bound) in result.iter_mut().zip(bound) {
                        *out = combine(
                            *out,
                            bound,
                            weight,
                            additive,
                            *property == v2::NodeProperty::Scale,
                        );
                    }
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod envelope_tests {
    use super::*;
    #[test]
    fn cubic_envelope_covers_interior_overshoot_and_layer_combination() -> Result<()> {
        let scene: v2::Scene = vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "nodes":[{"type":"camera","name":"camera"}],
            "animation":[{"tracks":[{"path":["nodes","camera","scale"],"keys":[[0,[35,35,35],[0,0,0],[50,50,50]],[2,[35,35,35],[-50,-50,-50],[0,0,0]]],"interpolation":"CUBICSPLINE"}]},{"weight":0.5,"blend":"combine","tracks":[{"path":["nodes","camera","scale"],"keys":[[0,[2,2,2]],[2,[4,4,4]]]}]}]
        }))?;
        let plan = Plan::compile(&scene, std::path::Path::new("."))?;
        let bounds = plan.node_envelope("camera", &v2::NodeProperty::Scale, &[50.;3])[0];
        assert!(bounds[1] > 50.);
        for step in 0..=100 {
            let time = step as f32 / 50.;
            let mut value = vec![];
            let Channel::Numeric(curve) = &plan.layers[0].tracks[0].channel else {
                unreachable!()
            };
            curve.sample(time, &mut value);
            let actual = value[0] * (2. + time).powf(0.5);
            assert!(actual >= bounds[0] && actual <= bounds[1]);
        }
        Ok(())
    }
}

/// Bake time-invariant material tracks before any stage installs resources. Keep
/// layer spans so removing a constant endpoint cannot change another track's loop.
pub(crate) fn fold_material_inputs(scene: &mut v2::Scene, base: &std::path::Path, opened: &crate::footage::Opened) -> Result<()> {
    if !scene.animation.iter().any(|l| l.tracks.iter().flatten().any(|t| t.path.first().is_some_and(|p| p == "materials"))) { return Ok(()); }
    let mut plan = Plan::compile_in(scene, base, None, opened)?;
    let constant = |channel: &Channel| match channel {
        Channel::Style(track) => track.values.iter().all(|v|v==&track.values[0]),
        Channel::Numeric(curve) => {
            let cubic = curve.interpolation == Interpolation::Cubic;
            let stride = curve.width * if cubic {3} else {1};
            let first = &curve.values[if cubic {curve.width} else {0}..if cubic {curve.width * 2} else {curve.width}];
            curve.values.chunks_exact(stride).all(|key| {
                if cubic { &key[curve.width..curve.width * 2] == first && key[..curve.width].iter().chain(&key[curve.width * 2..]).all(|v| *v == 0.) }
                else { key == first }
            })
        },
        Channel::Discrete { values, .. } => values.iter().all(|v| v == &values[0]),
    };
    // A changing layer weight can make even a constant key time-varying.
    if plan.layers.iter().flat_map(|l| &l.tracks).any(|t| t.weight_target.is_some() && !constant(&t.channel)) { return Ok(()); }
    let mut fields = std::collections::BTreeMap::<(String, String), bool>::new();
    for track in plan.layers.iter().flat_map(|l| &l.tracks) {
        if let Some(v2::TrackTarget::Material {material, input}) = &track.target {
            *fields.entry((material.clone(), input.clone())).or_insert(true) &= constant(&track.channel);
        }
    }
    let fields: Vec<_> = fields.into_iter().filter_map(|(field, fixed)| fixed.then_some(field)).collect();
    for (slot, (name, input)) in fields.iter().enumerate() {
        let value = scene.materials[name].inputs.get(input);
        let default = match value {
            Some(v2::Input::Number(v)) => vec![*v],
            Some(v2::Input::Vector(v)) => v.clone(),
            Some(v2::Input::Bool(_)) => continue,
            Some(v2::Input::Attribute(binding)) if scene.materials[name].program == "std/openpbr" => match &binding.factor {
                Some(v2::Factor::Number(v)) => vec![*v],
                Some(v2::Factor::Vector(v)) => v.clone(),
                None => vec![1.; if matches!(v2::OpenPbrParam::find(input).unwrap().kind, v2::OpenPbrKind::Color3 | v2::OpenPbrKind::Vector3) {3} else {1}],
            },
            Some(_) => continue,
            None if scene.materials[name].program == "std/openpbr" => {
                let p = v2::OpenPbrParam::find(input).unwrap();
                if p.kind == v2::OpenPbrKind::Vector3 {vec![0.;3]} else {p.default.to_vec()}
            },
            None => {
                let track = plan.layers.iter().flat_map(|l| &l.tracks).find(|t| matches!(&t.target, Some(v2::TrackTarget::Material {material, input: i}) if material == name && i == input)).unwrap();
                let Channel::Numeric(curve) = &track.channel else {continue};
                vec![0.; curve.width]
            }
        };
        plan.bind_material(&(name.clone(), input.clone()), vec![MaterialBinding {slot, parameter: 0, storage: None, enum_tokens: None, default: Sample::Numeric(default), offset: 0.}])?;
    }
    let mut samples = plan.samples();
    plan.sample_material_offset(0., 0., &mut samples);
    let mut folded = std::collections::BTreeSet::new();
    for (binding, value) in plan.material_values(&samples, 0.) {
        let (name, input) = &fields[binding.slot];
        let Sample::Numeric(value) = value else {continue};
        let value = if value.len() == 1 {v2::Input::Number(value[0])} else {v2::Input::Vector(value.clone())};
        let material = scene.materials.get_mut(name).unwrap();
        let factor = match &value {
            v2::Input::Number(v) => v2::Factor::Number(*v),
            v2::Input::Vector(v) => v2::Factor::Vector(v.clone()),
            _ => unreachable!(),
        };
        match material.inputs.get_mut(input) {
            Some(v2::Input::Attribute(binding)) => binding.factor = Some(factor),
            _ => { material.inputs.insert(input.clone(), value); }
        }
        folded.insert((name.clone(), input.clone()));
    }
    for (layer, compiled) in scene.animation.iter_mut().zip(&plan.layers) {
        layer.source_span = compiled.span;
        layer.source_clock_span = compiled.time_span.as_ref().map(|span|span.endpoints.clone()).unwrap_or_default();
        if let Some(tracks) = &mut layer.tracks {
            tracks.retain(|track| !matches!(track.path.as_slice(), [root, name, inputs, input] if root == "materials" && inputs == "inputs" && folded.contains(&(name.clone(), input.clone()))));
        }
    }
    let referenced: std::collections::BTreeSet<_> = scene.animation.iter().flat_map(|l| l.tracks.iter().flatten())
        .filter_map(|t| match t.path.as_slice() { [root, name, field] if root == "animation" && field == "weight" => Some(name.clone()), _ => None }).collect();
    scene.animation.retain(|layer| layer.clip.is_some() || layer.tracks.as_ref().is_some_and(|t| !t.is_empty()) || layer.name.as_ref().is_some_and(|name| referenced.contains(name)));
    Ok(())
}

#[cfg(test)]
mod asset_index_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn asset_index_work_is_independent_of_unrelated_population() -> Result<()> {
        let mut plan = Plan::default();
        plan.layers.push(CompiledLayer { owner: None, owner_node: None, source: serde_json::from_value(json!({"tracks":[]}))?, tracks: vec![], span: None, time_span: None });
        let curve = std::sync::Arc::new(Curve { times: vec![0.,1.], values: vec![0.,0.,0.,1.,2.,3.], width: 3,
            interpolation: Interpolation::Linear, rotation: false, euler: false });
        for asset in 0..1024 {
            plan.add_clip_curve(0, curve.clone(), AssetTarget { asset, node: 0, property: gltf::animation::Property::Translation, component: None });
        }
        plan.finish_asset_bindings()?;
        assert_eq!(plan.asset_bindings.len(),1024);
        assert!(plan.asset_bindings.iter().all(|bindings|bindings.transforms.len()==1));
        let mut samples=plan.samples();
        plan.sample_into(0.5,&mut samples);
        let mut visits=0;
        plan.apply_asset(17,&samples,|target,value,_,_| {
            visits+=1;
            assert_eq!(target.asset,17);
            assert_eq!(*value,Sample::Numeric(vec![0.5,1.,1.5]));
        });
        assert_eq!(visits,1);
        Ok(())
    }

    #[test]
    fn indexed_assets_match_ordered_scan_across_slots_seeks_and_rebinding() -> Result<()> {
        let scene: v2::Scene = vork_scene::v2::from_test_value(json!({"version":2,
            "nodes":[{"type":"asset","name":"a","file":"a.gltf"},{"type":"asset","name":"b","file":"b.gltf"}],
            "animation":[
                {"tracks":[{"path":["animation","add","weight"],"keys":[[0,0],[2,1]]}]},
                {"name":"base","loop":true,"offset":-0.2,"tracks":[
                    {"path":["nodes","a","nodes","joint","translation"],"keys":[[0,[1,2,3]],[2,[3,4,5]]]},
                    {"path":["nodes","b","nodes","joint","scale"],"keys":[[0,[1,1,1]],[2,[2,3,4]]]},
                    {"path":["nodes","a","nodes","joint","visible"],"keys":[[0,false],[1,true]]}]},
                {"name":"add","blend":"combine","weight":0.75,"tracks":[
                    {"path":["nodes","a","nodes","joint","translation"],"keys":[[0,[0,0,0]],[2,[0,2,0]]]}]}
            ]}))?;
        let doc = gltf::Gltf::from_slice(br#"{"asset":{"version":"2.0"},"nodes":[{"name":"joint"}]}"#)?;
        let mut plan = Plan::compile(&scene, std::path::Path::new("."))?;
        plan.bind_asset(&doc.document, "a", 0)?;
        plan.bind_asset(&doc.document, "b", 1)?;
        // Importing a later asset extends an earlier layer, shifting every
        // following layer's sample address before the index is finalized.
        plan.add_clip_curve(1, std::sync::Arc::new(Curve { times: vec![0.,2.], values: vec![1.,1.,1.,2.,2.,2.], width: 3,
            interpolation: Interpolation::Linear, rotation: false, euler: false }),
            AssetTarget { asset: 1, node: 0, property: gltf::animation::Property::Scale, component: None });
        plan.finish_asset_bindings()?;
        assert_eq!(plan.asset_bindings[0].transforms.len(), 2);
        assert_eq!(plan.asset_bindings[0].visibility.len(), 1);
        assert_eq!(plan.asset_bindings[1].transforms.len(), 2);
        let mut slots = [plan.samples(), plan.samples()];
        for (slot, time) in [(0,0.),(1,1.25),(0,0.5),(1,0.),(0,1.75),(1,0.5),(0,0.5)] {
            plan.sample_into(time, &mut slots[slot]);
            let samples = &slots[slot];
            for asset in 0..3 {
                let mut expected = Vec::new();
                let mut visibility = vec![v2::Visibility::default()];
                let mut index = 0;
                for layer in &plan.layers {
                    let additive = matches!(layer.source.blend,v2::Blend::Add);
                    for track in &layer.tracks {
                        for target in &track.asset_targets {
                            if target.asset == asset && samples.weights[index] != 0. {
                                expected.push((target.node,target.property,target.component,samples.values[index].clone(),samples.weights[index],additive));
                            }
                        }
                        if track.asset_visibility.is_some_and(|(a,_)| a == asset) && samples.weights[index] >= 0.5 {
                            let Sample::Bool(value) = samples.values[index] else {unreachable!()};
                            visibility[0].visible = value;
                        }
                        index += 1;
                    }
                }
                let mut actual = Vec::new();
                plan.apply_asset(asset,samples,|target,value,weight,additive| actual.push((target.node,target.property,target.component,value.clone(),weight,additive)));
                let mut actual_visibility = vec![v2::Visibility::default()];
                plan.apply_asset_visibility(asset,samples,&mut actual_visibility);
                assert_eq!(actual,expected,"asset {asset}, slot {slot}, time {time}");
                assert_eq!(actual_visibility[0].visible,visibility[0].visible);
            }
        }
        // Re-finishing after binding edits replaces the index rather than appending.
        plan.layers[1].tracks[0].asset_targets[0].asset = 1;
        plan.finish_asset_bindings()?;
        assert_eq!(plan.asset_bindings[0].transforms.len(),1);
        assert_eq!(plan.asset_bindings[1].transforms.len(),3);
        Ok(())
    }
}

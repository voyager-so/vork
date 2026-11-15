//! Node clocks (A7): existence and local time per node, evaluated top-down from
//! the scene instant with exact arithmetic. A value is a `Germ`: a rational and
//! the side it is read from, so reverse, wraps and holds follow rule 9 by
//! construction. Node-owned layers sample at their owner's local instant.
use super::*;
use vork_time::exact::{Germ, Rational};
use std::cmp::Ordering;

/// A node as clock binding reads it, in pre-order: the authored node, its parent's index, and the mark a node
/// takes when its clock or an owned layer can change it. The 3D installer's nodes and the canvas's are both
/// read this way, each where it already holds them.
pub(crate) trait ClockNode {
    fn node(&self) -> &v2::Node;
    fn parent(&self) -> Option<usize>;
    fn set_dynamic(&mut self);
}
impl ClockNode for crate::install::EvalNode {
    fn node(&self) -> &v2::Node { &self.node }
    fn parent(&self) -> Option<usize> { self.parent }
    fn set_dynamic(&mut self) { self.dynamic = true; }
}
/// What decides a node's existence at its parent's instant.
#[derive(Clone, Debug)]
pub(crate) enum ActiveSpec {
    On(bool),
    Window(Rational, Rational),
}
/// One node's clock constants, resolved at `mark_nodes`.
#[derive(Clone, Debug)]
pub(crate) struct ClockSpec {
    pub parent: Option<usize>,
    pub active: ActiveSpec,
    /// Flat track index of a parent- or scene-owned boolean `active` track.
    pub active_track: Option<usize>,
    pub time: Option<TimeSpec>,
    /// Flat track index of a keyed `time.speed` curve.
    pub speed_track: Option<usize>,
    /// Whether existence can ever be false here or above: visibility is gated.
    pub gated: bool,
}
#[derive(Clone, Debug)]
pub(crate) struct TimeSpec {
    pub start: Rational,
    pub speed: Rational,
    pub section: v2::Section,
}
/// Scene time to a node's local within the current traversal: `local = rate * t + org`,
/// defined while every clock from the scene down to the node has a constant speed.
#[derive(Clone, Debug)]
struct Affine {
    rate: Rational,
    org: Rational,
    /// Scene time at which the node's current traversal of its section began (rule 2's
    /// "where the traversal begins"); none without a section.
    begin: Option<Rational>,
}
/// One node's clock at an instant. `local` is its local instant, `None` while it is absent;
/// `audible` is whether sound travels there: no clock from the scene down stands or is held.
#[derive(Clone, Debug)]
pub(super) struct State {
    pub(super) audible: bool,
    pub(super) local: Option<Germ>,
    map: Option<Affine>,
    /// For a sound block that spans samples: a clock's raw forward value before its section
    /// trims it, and whether the section holds it there (rule 5), so the clock stands however
    /// its germ approaches.
    pub(super) raw: Option<Germ>,
    pub(super) standing: bool,
}
impl State {
    pub(super) fn absent() -> Self { Self { local: None, map: None, audible: false, raw: None, standing: false } }
}
/// What a sound block reads of a keyed speed at one instant.
pub(super) struct KeyedSpeed {
    /// The parent-local instants nearest above and below at which the integral changes
    /// piece: the curve's neighbouring keys and where the parent's traversal begins.
    pub(super) edges: Vec<Rational>,
    /// No key is faster than this, so neither is the curve between them.
    pub(super) ceiling: Rational,
    /// Whether the speed changes along its current piece: then the rate is a line in the
    /// reading clock and the integral a quadratic; on a flat piece, a constant and a line.
    pub(super) sloped: bool,
}

/// Which clock a footage image follows.
pub(crate) enum FootageClock {
    /// No node clock shows it: images.X.time or scene time.
    Scene,
    /// The owning node's local instant.
    Node(Germ),
    /// The owning node is absent, and with it every node that shows the image (rule 1).
    Absent,
}

fn zero() -> Rational { Rational::from_integer(0.into()) }
fn one() -> Rational { Rational::from_integer(1.into()) }

impl Plan {
    pub(crate) fn has_clocks(&self) -> bool { self.has_clocks }
    /// Whether `node` or an ancestor has a clock, so its local time is not the scene's.
    pub(crate) fn under_clock(&self, mut node: usize) -> bool {
        loop {
            if self.clocks.get(node).is_some_and(|spec| spec.time.is_some()) { return true; }
            match self.clocks.get(node).and_then(|spec| spec.parent) { Some(parent) => node = parent, None => return false }
        }
    }
    /// Whether installed node `node` is absent at this sample (rule 1).
    pub(crate) fn absent(&self, samples: &Samples, node: usize) -> bool {
        // Samples start with every clock unset; they are evaluated only when the scene has clocks.
        self.has_clocks && samples.clocks.get(node).is_some_and(Option::is_none)
    }
    /// Node `node`'s local instant at scene `request`; `None` while it is absent (rule 1).
    pub(crate) fn node_instant(&self, request: &vork_time::Request, node: usize) -> Result<Option<Germ>> {
        let scene = request.exact()?;
        if !self.has_clocks { return Ok(Some(scene)); }
        Ok(self.clocks(&scene)?.swap_remove(node))
    }
    /// Node `node`'s local instant at this sample (the scene instant `scene` when the plan has no clocks); `None`
    /// while it is absent (rule 1). A composition's content reads its node's instant (rule 7).
    pub(crate) fn local_instant(&self, samples: &Samples, node: usize, scene: &Germ) -> Option<Germ> {
        if !self.has_clocks { return Some(scene.clone()); }
        samples.clocks.get(node).cloned().flatten()
    }
    /// The clock that selects a footage image's frames at this sample.
    pub(crate) fn footage_clock(&self, samples: &Samples, image: &str) -> FootageClock {
        match self.footage_owner.get(image) {
            None => FootageClock::Scene,
            Some(&node) => samples.clocks.get(node).cloned().flatten().map_or(FootageClock::Absent, FootageClock::Node),
        }
    }
    /// Every node's local instant at scene instant `scene`, parents before children;
    /// `None` is an absent node (rule 1), its subtree included.
    pub(crate) fn clocks(&self, scene: &Germ) -> Result<Vec<Option<Germ>>> {
        let _profile = moda_engine::prof::span("animation.clocks");
        let mut states: Vec<State> = Vec::with_capacity(self.clocks.len());
        for (index, spec) in self.clocks.iter().enumerate() {
            let state = self.clock_state(index, spec, scene, &states, false, false)?;
            states.push(state);
        }
        Ok(states.into_iter().map(|state| state.local).collect())
    }
    /// The clocks of `nodes` alone (ascending, each node's parent and the owner of its keyed
    /// `active` among them), written into `states` by installed node; no other entry is touched.
    /// Sound is absent while a clock or any ancestor is held: the picture's left-limit endpoint
    /// germ is a sampling side, not evidence that sound is still travelling.
    /// `motion` also keeps what a block needs to span samples: each clock's raw value and
    /// whether its section holds it.
    pub(super) fn sound_states(&self, nodes: &[usize], scene: &Germ, states: &mut [State], motion: bool) -> Result<()> {
        for &node in nodes {
            let state = self.clock_state(node, &self.clocks[node], scene, states, true, motion)?;
            states[node] = state;
        }
        Ok(())
    }
    fn clock_state(&self, index: usize, spec: &ClockSpec, scene: &Germ, states: &[State], sound: bool, motion: bool) -> Result<State> {
        let absent = State::absent();
        let parent_audible = spec.parent.is_none_or(|p| states[p].audible);
        let (parent, parent_map) = match spec.parent {
            Some(p) => match &states[p].local { Some(local) => (local.clone(), states[p].map.clone()), None => return Ok(absent) },
            None => (scene.clone(), Some(Affine { rate: one(), org: zero(), begin: None })),
        };
        let exists = match (&spec.active, spec.active_track) {
            (_, Some(track)) => self.discrete_bool(track, spec.parent, &parent, scene, states)?,
            (ActiveSpec::On(on), None) => *on,
            (ActiveSpec::Window(a, b), None) => parent.compare_point(a) != Ordering::Less && parent.compare_point(b) == Ordering::Less,
        };
        if !exists { return Ok(absent); }
        let Some(time) = &spec.time else { return Ok(State { local: Some(parent), map: parent_map, audible: parent_audible, raw: None, standing: false }) };
        let section = &time.section;
        let node_name = |plan: &Plan| plan.clock_names.get(index).cloned().unwrap_or_default();
        if let Some(track) = spec.speed_track {
            // Rule 2: a keyed speed integrated against the parent clock from `start`,
            // read in its owner's seconds along the parent's current traversal.
            let curve = self.track_at(track).time_curve.as_ref().context("speed curve is compiled")?.get()?;
            let owner = self.layers[self.layer_of(track)].owner_node;
            let raw = if owner == spec.parent && owner.is_some() || owner.is_none() && spec.parent.is_none() {
                // Owner seconds are the parent's seconds: integrate directly. The side is the
                // integral of the rate's germ (rule 9: composition, not motion), so a zero rate
                // over a stretch reads exact while an isolated zero keeps the parent's side.
                let integral = curve.integrate(&time.start, parent.value());
                Germ::point(&section.offset + integral).add(&side_integral(&parent, &curve.sample(&parent)))
            } else {
                let map = parent_map.as_ref().with_context(|| format!("node {:?}: time.speed is keyed by the scene under a parent with a keyed speed; key it from the parent", node_name(self)))?;
                ensure!(map.rate != zero(), "node {:?}: time.speed keyed by the scene under a standing parent", node_name(self));
                let at = |p: &Rational| (p - &map.org) / &map.rate;
                let (ta, tb) = (at(&time.start), at(parent.value()));
                let integral = &map.rate * curve.integrate_from(&ta, &tb, map.begin.as_ref());
                // The scene instant as a germ: (p - org) / rate keeps p's side, flipped by a negative rate.
                let scene_germ = parent.sub(&Germ::point(map.org.clone())).scale(&map.rate.recip());
                let rate_germ = if map.begin.as_ref().is_some_and(|begin| scene_germ.compare_point(begin) == Ordering::Less) {
                    curve.sample(&Germ::point(map.begin.clone().unwrap()))
                } else { curve.sample(&scene_germ) };
                // The parent's side already carries the direction of travel (dτ = rate dt), so the
                // side term is not scaled by the map's rate again.
                Germ::point(&section.offset + integral).add(&side_integral(&parent, &rate_germ))
            };
            let standing = motion && section.holds(&raw);
            let audible = !sound || (parent_audible && !(if motion { standing } else { section.holds(&raw) }) && raw.rate() != zero());
            let kept = motion.then(|| raw.clone());
            return Ok(State { local: Some(section.local(raw)), map: None, audible, raw: kept, standing });
        }
        let raw = section.constant_raw(&parent, &time.start, &time.speed);
        let standing = motion && section.holds(&raw);
        let audible = !sound || (parent_audible && !(if motion { standing } else { section.holds(&raw) }) && raw.rate() != zero());
        let kept = motion.then(|| raw.clone());
        let local = section.local(raw);
        let raw = kept;
        // The affine traversal map for descendants keyed from the scene.
        let map = parent_map.and_then(|parent_map| {
            if time.speed == zero() { return None; }
            let magnitude = if time.speed < zero() { -&time.speed } else { time.speed.clone() };
            let rate = &parent_map.rate * &magnitude;
            let org = &section.offset + (&parent_map.org - &time.start) * &magnitude;
            // Raw forward clock = rate * t + org; the wrap subtracts whole periods.
            let (raw_at_scene, cycle_shift) = {
                let raw = &rate * scene.value() + &org;
                match (&section.end, section.looping) {
                    (Some(end), true) => {
                        let period = end - &section.offset;
                        let cycle = ((&raw - &section.offset) / &period).floor();
                        (raw, cycle * period)
                    }
                    _ => (raw, zero()),
                }
            };
            let _ = raw_at_scene;
            let f_org = &org - &cycle_shift;
            // The traversal begins at the earlier of the scene times where the forward clock
            // reads offset and E: offset for a rising clock, E for a falling one (xtimerev 16).
            let at_offset = (&section.offset - &f_org) / &rate;
            let begin = Some(match &section.end {
                Some(end) => { let at_end = (end - &f_org) / &rate; if at_end < at_offset { at_end } else { at_offset } }
                None => at_offset,
            });
            Some(if section.reverse {
                let end = section.end.clone()?;
                Affine { rate: -&rate, org: end + &section.offset - f_org, begin }
            } else {
                Affine { rate, org: f_org, begin }
            })
        });
        Ok(State { local: Some(local), map, audible, raw, standing })
    }
    /// Where the keyed speed `track` of a present node changes piece around its parent's local
    /// instant `parent`, and how fast it can be: `None` when its integral is not a polynomial a
    /// block can carry exactly between those instants (a cubic curve can stand inside a piece).
    pub(super) fn keyed_speed(&self, spec: &ClockSpec, track: usize, parent: &Germ, parent_state: Option<&State>) -> Result<Option<KeyedSpeed>> {
        let curve = self.track_at(track).time_curve.as_ref().context("speed curve is compiled")?.get()?;
        if curve.is_cubic() { return Ok(None); }
        let (times, values) = curve.key_tables();
        let owner = self.layers[self.layer_of(track)].owner_node;
        // As `clock_state` reads it: in the parent's seconds, or in the scene's along the
        // parent's current traversal `parent = org + rate * scene`.
        let map = if owner == spec.parent && owner.is_some() || owner.is_none() && spec.parent.is_none() { None } else {
            let map = parent_state.and_then(|state| state.map.as_ref()).context("a scene-keyed speed reads its parent's traversal")?;
            ensure!(map.rate != zero(), "a scene-keyed speed under a standing parent");
            Some(map)
        };
        // The instant the curve is read at, with the parent's side.
        let reading = match map { Some(map) => parent.sub(&Germ::point(map.org.clone())).scale(&map.rate.recip()), None => parent.clone() };
        let back = |instant: &Rational| match map { Some(map) => &map.org + &map.rate * instant, None => instant.clone() };
        let mut edges = Vec::new();
        if let Some(above) = times.get(times.partition_point(|key| key <= reading.value())) { edges.push(back(above)); }
        if let Some(below) = times.partition_point(|key| key < reading.value()).checked_sub(1) { edges.push(back(&times[below])); }
        let begin = map.and_then(|map| map.begin.as_ref());
        if let Some(begin) = begin { edges.push(back(begin)); }
        let ceiling = values.iter().map(|value| if value < &zero() { -value } else { value.clone() }).max().unwrap_or_else(zero);
        // The piece `TimeCurve::sample` reads: between the keys either side of the instant,
        // held outside them and before the traversal begins.
        let upper = times.partition_point(|key| reading.compare_point(key).is_ge());
        let (a, b) = (upper.saturating_sub(1), upper.min(times.len() - 1));
        let held = begin.is_some_and(|begin| reading.compare_point(begin) == Ordering::Less);
        let sloped = !curve.is_step() && !held && values[a] != values[b];
        Ok(Some(KeyedSpeed { edges, ceiling, sloped }))
    }
    /// A boolean STEP track read at its owner's instant: the parent's local or the scene.
    fn discrete_bool(&self, track: usize, parent: Option<usize>, parent_local: &Germ, scene: &Germ, states: &[State]) -> Result<bool> {
        let layer = &self.layers[self.layer_of(track)];
        let owner_instant = match layer.owner_node {
            None => scene.clone(),
            Some(owner) if Some(owner) == parent => parent_local.clone(),
            Some(owner) => match &states[owner].local { Some(local) => local.clone(), None => return Ok(false) },
        };
        let property = self.track_at(track);
        let Channel::Discrete { values, .. } = &property.channel else { anyhow::bail!("active track must be boolean") };
        let keys = property.discrete_keys.as_ref().context("active keys are compiled")?;
        let upper = keys.upper(&owner_instant);
        if layer.source.weight < 0.5 { return Ok(true); }
        match values[upper.saturating_sub(1)] { Sample::Bool(value) => Ok(value), _ => anyhow::bail!("active track must be boolean") }
    }
    pub(super) fn track_at(&self, flat: usize) -> &PropertyTrack {
        let layer = self.layer_of(flat);
        &self.layers[layer].tracks[flat - self.layer_starts[layer]]
    }
    pub(super) fn layer_of(&self, flat: usize) -> usize {
        self.layer_starts.partition_point(|&start| start <= flat).saturating_sub(1)
    }
}
/// `∫_p^{p+side}` of a rate germ expanded about `p`: the parent's side carries the
/// orientation (dτ = ±dε), and the rate's own expansion decides whether a zero keeps it.
fn side_integral(p: &Germ, rate: &Germ) -> Germ {
    let side = p.sub(&Germ::point(p.value().clone()));
    match side.compare_point(&zero()) {
        Ordering::Equal => Germ::point(zero()),
        Ordering::Greater => rate.integrate_epsilon(),
        Ordering::Less => rate.integrate_epsilon().scale(&-one()),
    }
}

impl exact::TimeCurve {
    /// `∫_a^b` of the curve, holding its end values outside the keys; signed.
    pub(super) fn integrate(&self, a: &Rational, b: &Rational) -> Rational {
        self.integrate_from(a, b, None)
    }
    /// As `integrate`, reading the curve where the traversal begins for times before `begin`.
    pub(super) fn integrate_from(&self, a: &Rational, b: &Rational, begin: Option<&Rational>) -> Rational {
        if a > b { return -self.integrate_from(b, a, begin); }
        let (a, b) = match begin {
            Some(begin) if a < begin => {
                let held = self.sample(&Germ::point(begin.clone())).value().clone();
                let clamped_b = if b < begin { b.clone() } else { begin.clone() };
                let before = held * (&clamped_b - a);
                return before + if b > begin { self.integrate_from(begin, b, None) } else { zero() };
            }
            _ => (a.clone(), b.clone()),
        };
        self.antiderivative(&b) - self.antiderivative(&a)
    }
    /// An antiderivative with hold outside the key range.
    fn antiderivative(&self, t: &Rational) -> Rational {
        let (times, values) = self.key_tables();
        let n = times.len();
        let cubic = self.is_cubic();
        let value = |key: usize, part: usize| values[key * if cubic { 3 } else { 1 } + part].clone();
        let first = value(0, usize::from(cubic));
        if n == 1 || t <= &times[0] { return (t - &times[0]) * first; }
        let mut total = zero();
        for i in 0..n - 1 {
            let (t0, t1) = (&times[i], &times[i + 1]);
            if t <= t0 { break; }
            let dt = t1 - t0;
            let u = ((t.min(t1) - t0) / &dt).min(one());
            let (p0, p1) = (value(i, usize::from(cubic)), value(i + 1, usize::from(cubic)));
            let segment = if self.is_step() {
                &p0 * &u
            } else if cubic {
                let (m0, m1) = (value(i, 2) * &dt, value(i + 1, 0) * &dt);
                let u2 = &u * &u; let u3 = &u2 * &u; let u4 = &u3 * &u;
                let h00 = &u4 / Rational::from_integer(2.into()) - &u3 + &u;
                let h10 = &u4 / Rational::from_integer(4.into()) - &u3 * Rational::new(2.into(), 3.into()) + &u2 / Rational::from_integer(2.into());
                let h01 = -&u4 / Rational::from_integer(2.into()) + &u3;
                let h11 = &u4 / Rational::from_integer(4.into()) - &u3 / Rational::from_integer(3.into());
                h00 * &p0 + h10 * m0 + h01 * &p1 + h11 * m1
            } else {
                &p0 * (&u - &u * &u / Rational::from_integer(2.into())) + &p1 * (&u * &u / Rational::from_integer(2.into()))
            };
            total += segment * &dt;
        }
        if t > &times[n - 1] { total += (t - &times[n - 1]) * value(n - 1, usize::from(cubic)); }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn q(text: &str) -> Rational { vork_time::exact::decimal(text).unwrap() }
    fn reading(g: &Germ) -> String {
        let value = v2::time::rational(g.value());
        match g.compare_point(g.value()) { Ordering::Less => format!("{value}-"), Ordering::Greater => format!("{value}+"), Ordering::Equal => value }
    }
    /// The test path: the document path refuses these fields in this release.
    fn open(scene: serde_json::Value) -> Result<v2::Scene> {
        let scene: v2::Scene = vork_scene::v2::from_test_value(scene)?;
        scene.validate()?;
        Ok(scene)
    }
    fn plan(scene: serde_json::Value) -> Result<Plan> {
        let scene = open(scene)?;
        let mut plan = Plan::compile(&scene, std::path::Path::new("."))?;
        let mut nodes = Vec::new();
        fn flatten(input: &[v2::Node], parent: Option<usize>, nodes: &mut Vec<crate::install::EvalNode>) {
            for n in input {
                let id = nodes.len();
                nodes.push(crate::install::EvalNode { camera: None, node: n.without_children(), parent, world: glam::Mat4::IDENTITY, dynamic: false, embedded: None });
                flatten(&n.children, Some(id), nodes);
            }
        }
        flatten(&scene.nodes, None, &mut nodes);
        plan.mark_nodes(&mut nodes, 0)?;
        Ok(plan)
    }
    fn locals(plan: &Plan, at: &str) -> Vec<String> {
        plan.clocks(&Germ::point(q(at))).unwrap().iter().map(|g| g.as_ref().map_or("absent".into(), reading)).collect()
    }
    #[test]
    fn part_b_ramp_integrates_to_12_14_13_13() -> Result<()> {
        // A scene-owned speed curve keyed [0,1],[2,1],[4,2],[10,2] on a plate trimmed 12..16, looping.
        let plan = plan(serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam"},
            {"name":"plate","time":{"offset":12,"duration":4,"loop":true}}],
            "animation":[{"tracks":[{"path":["nodes","plate","time","speed"],"keys":[[0,1],[2,1],[4,2],[10,2]]}]}]}))?;
        let plate: Vec<String> = ["0","2","4","6"].iter().map(|at| locals(&plan, at)[1].clone()).collect();
        assert_eq!(plate, ["12","14","13","13"]);
        Ok(())
    }
    #[test]
    fn an_absent_subtree_is_not_sampled() -> Result<()> {
        // Rule 1: no evaluation, children included. The scene keys a child of an inactive node,
        // a node with a window and a present node; only tracks on present nodes are sampled.
        let plan = plan(serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam"},
            {"name":"gone","active":false,"children":[{"name":"kid"}]},
            {"name":"later","active":[1,2]},
            {"name":"here"}],
            "animation":[{"tracks":[
                {"path":["nodes","kid","translation"],"keys":[[0,[0,0,0]],[1,[1,0,0]]]},
                {"path":["nodes","later","translation"],"keys":[[0,[0,0,0]],[1,[1,0,0]]]},
                {"path":["nodes","here","translation"],"keys":[[0,[0,0,0]],[1,[1,0,0]]]}]}]}))?;
        let weights = |t: &str| -> Result<Vec<f32>> {
            let mut samples = plan.samples();
            let request = vork_time::Request::Exact(Germ::point(q(t)));
            plan.sample_request_into(request.numeric()?, &request, &mut samples)?;
            assert!(plan.absent(&samples, 1) && plan.absent(&samples, 2), "gone and its child are absent");
            Ok(samples.weights[..3].to_vec())
        };
        assert_eq!(weights("0.5")?, [0., 0., 1.]);
        assert_eq!(weights("1.5")?, [0., 1., 1.]);
        assert_eq!(plan.node_instant(&vork_time::Request::Exact(Germ::point(q("1.5"))), 3)?.map(|g| reading(&g)), Some("0.5".into()));
        assert!(plan.node_instant(&vork_time::Request::Exact(Germ::point(q("2"))), 3)?.is_none());
        Ok(())
    }
    #[test]
    fn node_layers_keep_their_own_offset_speed_and_loop() -> Result<()> {
        // Section 8: a node's layer keeps its own clock on top of the node's. The shot reads
        // 0.5 + t; its layer reads source = 1 + 2 * (0.5 + t), looped over the keys' [0, 2).
        let plan = plan(serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam"},
            {"name":"shot","time":{"offset":0.5},"children":[{"name":"box"}],
             "animation":[{"offset":1,"speed":2,"loop":true,"tracks":[{"path":["children","box","translation"],"keys":[[0,[0,0,0]],[2,[2,0,0]]]}]}]}]}))?;
        let x = |t: f32| -> Result<f32> {
            let mut samples = plan.samples();
            plan.sample_request_into(t, &vork_time::Request::Exact(Germ::point(vork_time::exact::ieee(t)?)), &mut samples)?;
            let Sample::Numeric(value) = &samples.values[0] else { anyhow::bail!("translation is numeric") };
            Ok(value[0])
        };
        for (t, expected) in [(0., 0.), (0.25, 0.5), (0.75, 1.5), (1.25, 0.5)] {
            assert!((x(t)? - expected).abs() < 1e-5, "scene {t}: x {} != {expected}", x(t)?);
        }
        Ok(())
    }
    #[test]
    fn reversed_section_inside_a_looping_parent_and_a_keyed_boolean() -> Result<()> {
        let plan = plan(serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam"},
            {"name":"loop","time":{"duration":4,"loop":true},"children":[
                {"name":"rev","time":{"offset":2,"duration":6,"speed":-1},"children":[{"name":"win","active":[0,6]}]},
                {"name":"flag","active":true}]},
            {"name":"later","active":[2,10],"children":[{"name":"inner","time":{"duration":2,"loop":true}}]}],
            "animation":[{"tracks":[{"path":["nodes","flag","active"],"keys":[[0,false],[1,true]]}]}]}))?;
        // indices: 0 cam, 1 loop, 2 rev, 3 win, 4 flag, 5 later, 6 inner
        assert_eq!(locals(&plan, "0"), ["0","0","8-","absent","absent","absent","absent"], "win [0,6) is absent while rev reads 8 from below");
        assert_eq!(locals(&plan, "3"), ["3","3","5-","5-","3","1","1"]);
        assert_eq!(locals(&plan, "4"), ["4","0","8-","absent","0","2","0"]);
        assert_eq!(locals(&plan, "7"), ["7","3","5-","5-","3","5","1"]);
        assert_eq!(locals(&plan, "10"), ["10","2","6-","6-","2","absent","absent"]);
        Ok(())
    }
    #[test]
    fn scene_owned_speed_under_a_retimed_parent_reads_the_traversal() -> Result<()> {
        // amend-3 §4: P(t) = 2t, scene-owned s(t) = 1 + t gives child 2t + t²; under
        // P(t) = mod(t, 4) the two traversals read s(1) and s(5).
        let plan = plan(serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam"},
            {"name":"fast","time":{"speed":2},"children":[{"name":"child","time":{}}]},
            {"name":"wrap","time":{"duration":4,"loop":true},"children":[{"name":"kid","time":{}}]}],
            "animation":[{"tracks":[{"path":["nodes","child","time","speed"],"keys":[[0,1],[100,101]]},
                                    {"path":["nodes","kid","time","speed"],"keys":[[0,1],[100,101]]}]}]}))?;
        for t in ["0","1","2","3"] {
            let tt = q(t);
            assert_eq!(q(&locals(&plan, t)[2]), &tt * q("2") + &tt * &tt, "at {t}");
        }
        // kid: first traversal 0..4 reads s(t) = 1 + t: ∫_0^1 (1+τ)dτ = 1.5 at scene 1;
        // second traversal from scene 4 reads s from 4 on: at scene 5, ∫_4^5 (1+τ)dτ = 5.5.
        assert_eq!(locals(&plan, "1")[4], "1.5");
        assert_eq!(locals(&plan, "5")[4], "5.5");
        Ok(())
    }
    /// A 10 s, 30 fps movie beside a scene that shows it on `plate` (and on `other`).
    fn footage_scene(dir: &std::path::Path, plate_time: serde_json::Value, extra: serde_json::Value) -> serde_json::Value {
        let path = dir.join("clip.mp4");
        if !path.exists() {
            let status = std::process::Command::new("ffmpeg").args(["-v","error","-y","-f","lavfi","-i","testsrc2=size=16x16:rate=30:duration=10","-pix_fmt","yuv420p"]).arg(&path).status().unwrap();
            assert!(status.success());
        }
        let mut scene = serde_json::json!({"version":2,"render":{"fps":30},
            "buffers":{"p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0]]}},
            "geometry":{"tri":{"attributes":{"POSITION":"p"}}},
            "images":{"clip":{"file":"clip.mp4"}},
            "materials":{"m":{"inputs":{"base_color":{"image":"clip"}}}},
            "nodes":[{"type":"camera","name":"cam"},
                {"name":"plate","time":plate_time,"children":[{"type":"mesh","name":"screen","geometry":"tri","materials":["m"]}]}]});
        if let Some(object) = extra.as_object() { for (key, value) in object { scene[key] = value.clone(); } }
        scene
    }
    fn compile_in(dir: &std::path::Path, scene: serde_json::Value) -> Result<Plan> {
        let scene = open(scene)?;
        let mut plan = Plan::compile_with_buffers(&scene, dir, None)?;
        let mut nodes = Vec::new();
        fn flatten(input: &[v2::Node], parent: Option<usize>, nodes: &mut Vec<crate::install::EvalNode>) {
            for n in input {
                let id = nodes.len();
                nodes.push(crate::install::EvalNode { camera: None, node: n.without_children(), parent, world: glam::Mat4::IDENTITY, dynamic: false, embedded: None });
                flatten(&n.children, Some(id), nodes);
            }
        }
        flatten(&scene.nodes, None, &mut nodes);
        plan.mark_nodes(&mut nodes, 0)?;
        Ok(plan)
    }
    #[test]
    fn footage_spans_are_inferred_from_the_one_source_and_one_clock_owns_an_image() -> Result<()> {
        if std::process::Command::new("ffmpeg").arg("-version").output().is_err() { return Ok(()); }
        let dir = std::env::temp_dir().join(format!("vork-clocks-footage-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        // amend-1-review 2: offset at the source's end leaves nothing to loop; refused naming the length.
        let error = compile_in(&dir, footage_scene(&dir, serde_json::json!({"offset":10,"loop":true}), serde_json::json!({}))).map(|_| ()).map_err(|e| format!("{e:#}")).unwrap_err();
        assert!(error.contains("nothing after offset: the source is 10 s; write offset < 10 or time.duration"), "{error}");
        // Inferred E = 10 with offset 2: the loop period is 8, and the movie follows the plate's clock.
        let plan = compile_in(&dir, footage_scene(&dir, serde_json::json!({"offset":2,"loop":true}), serde_json::json!({})))?;
        assert_eq!(locals(&plan, "9")[1], "3");
        let mut samples = plan.samples();
        plan.sample_request_into(9., &vork_time::Request::Numeric(9.), &mut samples)?;
        let FootageClock::Node(clock) = plan.footage_clock(&samples, "clip") else { anyhow::bail!("the plate's clock shows the clip") };
        assert_eq!(reading(&clock), "3");
        // Two clocks for one image, and a written images.X.time beside a clock: refused, one owner.
        let two = footage_scene(&dir, serde_json::json!({"offset":2,"loop":true}), serde_json::json!({}));
        let mut two_nodes = two.clone();
        two_nodes["nodes"].as_array_mut().unwrap().push(serde_json::json!({"name":"other","time":{"speed":2},"children":[{"type":"mesh","name":"screen2","geometry":"tri","materials":["m"]}]}));
        let error = compile_in(&dir, two_nodes).map(|_| ()).map_err(|e| format!("{e:#}")).unwrap_err();
        assert!(error.contains("shown under two clocks"), "{error}");
        let mut timed = two.clone();
        timed["images"]["clip"]["time"] = serde_json::json!(3);
        let error = compile_in(&dir, timed).map(|_| ()).map_err(|e| format!("{e:#}")).unwrap_err();
        assert!(error.contains("two owners"), "{error}");
        // No source and no duration: refused at open before any probe.
        let scene = open(serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam"},{"name":"spin","time":{"loop":true}}]}));
        let error = scene.err().map(|e| format!("{e:#}")).unwrap_or_default();
        assert!(error.contains("loop and speed < 0 need an end; write time.duration"), "{error}");
        // xtimerev 14: a one-frame movie is as long as its stream says, not one output frame.
        let one = dir.join("one.mp4");
        let status = std::process::Command::new("ffmpeg").args(["-v","error","-y","-f","lavfi","-i","testsrc2=size=16x16:rate=1:duration=1","-pix_fmt","yuv420p"]).arg(&one).status()?;
        assert!(status.success());
        let length = crate::footage::Source::open(&one)?.length("24".parse()?)?;
        assert_eq!(length, q("1"), "stream duration 1 s at 1 fps, not 1/24");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
    #[test]
    fn an_inferred_sequence_length_follows_the_frames_on_disk() -> Result<()> {
        // audit-core 5: the pattern is not a file, so its own metadata never changed and a worker kept the first length.
        let dir = std::env::temp_dir().join(format!("vork-sequence-length-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        let frame = |n: u32| image::RgbaImage::from_pixel(1, 1, image::Rgba([n as u8, 0, 0, 255])).save(dir.join(format!("{n:04}.png")));
        frame(1)?;
        frame(2)?;
        let pattern = dir.join("%04d.png");
        let fps = vork_time::Time::integer(2);
        // One document open reads the source once and keeps to it; the next open reads again.
        let open = crate::footage::Opened::default();
        assert_eq!(source_length(&pattern, fps, &open)?, q("1"));
        frame(3)?;
        assert_eq!(source_length(&pattern, fps, &open)?, q("1"), "within one open, the source as first read");
        // Other spellings of its path are the same source: through `.`, through `..`, and
        // through a link to its directory.
        std::fs::create_dir_all(dir.join("beside"))?;
        for spelled in [dir.join(".").join("%04d.png"), dir.join("beside").join("..").join("%04d.png")] {
            assert_eq!(source_length(&spelled, fps, &open)?, q("1"), "{}", spelled.display());
        }
        #[cfg(unix)]
        {
            let linked = dir.join("beside").join("linked");
            std::os::unix::fs::symlink(&dir, &linked)?;
            assert_eq!(source_length(&linked.join("%04d.png"), fps, &open)?, q("1"), "through a link");
            std::fs::remove_file(&linked)?;
        }
        assert_eq!(source_length(&pattern, fps, &crate::footage::Opened::default())?, q("1.5"), "an appended frame lengthens the source");
        std::fs::remove_file(dir.join("0003.png"))?;
        std::fs::remove_file(dir.join("0002.png"))?;
        assert_eq!(source_length(&pattern, fps, &crate::footage::Opened::default())?, q("0.5"), "removed frames shorten it");
        std::fs::remove_file(dir.join("0001.png"))?;
        let error = source_length(&pattern, fps, &crate::footage::Opened::default()).map_err(|e| format!("{e:#}")).unwrap_err();
        assert!(error.contains("write time.duration") && error.contains("has no frames"), "{error}");
        // A movie is read each time too: a replaced movie gives its own length.
        let movie = dir.join("clip.mp4");
        let make = |seconds: &str| -> Result<()> {
            let status = std::process::Command::new("ffmpeg").args(["-v","error","-y","-f","lavfi","-i"])
                .arg(format!("testsrc2=size=16x16:rate=4:duration={seconds}")).args(["-pix_fmt","yuv420p"]).arg(&movie).status()?;
            anyhow::ensure!(status.success(), "make a {seconds} s movie");
            Ok(())
        };
        make("1")?;
        assert_eq!(source_length(&movie, fps, &crate::footage::Opened::default())?, q("1"));
        assert_eq!(source_length(&movie, vork_time::Time::integer(24), &crate::footage::Opened::default())?, q("1"), "the stream's own duration at any rate asked");
        make("3")?;
        assert_eq!(source_length(&movie, fps, &crate::footage::Opened::default())?, q("3"), "a replaced movie is read again");
        // A document whose two clocks each show the sequence through their own image: it
        // compiles and binds, and a recompile after an appended frame carries the new length.
        frame(1)?;
        frame(2)?;
        let document = serde_json::json!({"version":2,"render":{"fps":2},
            "buffers":{"p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0]]}},
            "geometry":{"tri":{"attributes":{"POSITION":"p"}}},
            "images":{"first":{"file":"%04d.png"},"second":{"file":"%04d.png"}},
            "materials":{"ma":{"inputs":{"base_color":{"image":"first"}}},"mb":{"inputs":{"base_color":{"image":"second"}}}},
            "nodes":[{"type":"camera","name":"cam"},
                {"name":"a","time":{"loop":true},"children":[{"type":"mesh","name":"sa","geometry":"tri","materials":["ma"]}]},
                {"name":"b","time":{"loop":true},"children":[{"type":"mesh","name":"sb","geometry":"tri","materials":["mb"]}]}]});
        let lengths = |plan: &Plan| plan.source_lengths.values().cloned().collect::<Vec<_>>();
        assert_eq!(lengths(&compile_in(&dir, document.clone())?), vec![q("1"), q("1")]);
        frame(3)?;
        assert_eq!(lengths(&compile_in(&dir, document)?), vec![q("1.5"), q("1.5")], "a recompile reads the appended frame");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
    #[test]
    fn scene_owned_speed_under_a_reversed_looping_chain_reads_its_own_traversal() -> Result<()> {
        // xtimerev 16: A {duration 4, speed -1, loop}, B {duration 2, loop}, C with scene-owned
        // s(t) = 1 + t. At scene 1, B reads 1 from below on its traversal B(t) = 2 - t over
        // scene [0, 2]; C integrates that traversal's own instants: 2.5, from below.
        let chain = plan(serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam"},
            {"name":"a","time":{"duration":4,"speed":-1,"loop":true},"children":[
                {"name":"b","time":{"duration":2,"loop":true},"children":[{"name":"c","time":{"duration":100}}]}]}],
            "animation":[{"tracks":[{"path":["nodes","c","time","speed"],"keys":[[0,1],[100,101]]}]}]}))?;
        assert_eq!(locals(&chain, "1")[1..4], ["3-", "1-", "2.5-"]);
        // The reviewer's O4: P(t) = 6 - mod(t, 6), child start 0, duration 100: 45/2 at 1, 105/2 at 7.
        let mirrored = plan(serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam"},
            {"name":"p","time":{"duration":6,"speed":-1,"loop":true},"children":[{"name":"k","time":{"duration":100}}]}],
            "animation":[{"tracks":[{"path":["nodes","k","time","speed"],"keys":[[0,1],[100,101]]}]}]}))?;
        assert_eq!(locals(&mirrored, "1")[2], "22.5-");
        assert_eq!(locals(&mirrored, "7")[2], "52.5-");
        Ok(())
    }
    #[test]
    fn holds_pass_their_side_down_and_a_flat_rate_reads_exact() -> Result<()> {
        // amend-4 cases H1..H8 and F1..F4 (test-cases.md): the side is set by composition.
        let plan = plan(serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam"},
            {"name":"hold","time":{"duration":6},"children":[
                {"name":"ident","time":{"duration":10},"children":[{"name":"gw","active":[0,6]}]},
                {"name":"frozen","time":{"duration":4,"speed":0,"loop":true}},
                {"name":"stepped","time":{"duration":10},"children":[{"name":"gw2","active":[2,3]}]},
                {"name":"revkid","time":{"duration":10,"speed":-1}},
                {"name":"bowl","time":{"duration":100}}],
             "animation":[{"tracks":[
                {"path":["children","stepped","time","speed"],"keys":[[0,1],[2,0]],"interpolation":"STEP"},
                {"path":["children","bowl","time","speed"],"keys":[[0,36,0,-12],[6,0,0,0]],"interpolation":"CUBICSPLINE"}]}]},
            {"name":"late","time":{"start":10,"duration":6,"speed":-1},"children":[{"name":"ident2","time":{"duration":10}}]},
            {"name":"rev","time":{"duration":6,"speed":-1,"loop":true},"children":[{"name":"rev2","time":{"duration":6,"speed":-1,"loop":true}}]}],
            "animation":[]}))?;
        // indices: 0 cam, 1 hold, 2 ident, 3 gw, 4 frozen, 5 stepped, 6 gw2, 7 revkid, 8 bowl, 9 late, 10 ident2, 11 rev, 12 rev2
        let at = |t: &str| locals(&plan, t);
        assert_eq!(at("5")[1..3], ["5", "5"]);
        for t in ["6", "7", "100"] {
            let l = at(t);
            assert_eq!(&l[1..3], ["6-", "6-"], "H1 at {t}: the hold's side passes through the identity child");
            assert_ne!(l[3], "absent", "H3: the grandchild window [0,6) stays through the hold");
            assert_eq!(l[4], "0", "F1: a frozen child reads offset exactly");
            assert_eq!(l[5], "2", "F2: a STEP rate that is flat zero reads the integral exactly");
            assert_ne!(l[6], "absent", "F4: the grandchild window [2,3) under the flat rate stays");
            assert_eq!(l[7], "4+", "H7: a reversed child under a held parent is exact at 4");
            assert_eq!(l[8], "72-", "H8: an isolated zero of (p-6)^2 keeps the side; ∫ = 72");
        }
        // H4: a reversed clock before its start stands at E from below; its identity child keeps it.
        for t in ["0", "5", "10"] { assert_eq!(at(t)[9..11], ["6-", "6-"], "{t}"); }
        assert_eq!(at("13")[9..11], ["3-", "3-"]);
        for t in ["16", "20"] { assert_eq!(at(t)[9..11], ["0", "0"], "{t}"); }
        // H5: the mirror of a mirror rises and reads exactly; at the wrap it is 0, not 6 from below.
        let expect = [("0", "6-", "0+"), ("1/30", "179/30-", "1/30+"), ("1", "5-", "1+"), ("3", "3-", "3+"), ("6", "6-", "0+"), ("7", "5-", "1+")];
        for (t, parent, child) in expect { let l = at(t); assert_eq!((l[11].as_str(), l[12].as_str()), (parent, child), "H5 at {t}"); }
        Ok(())
    }
    #[test]
    fn integrals_match_closed_forms() -> Result<()> {
        let curve = |json: serde_json::Value| -> Result<exact::TimeCurve> {
            let track: v2::Track = serde_json::from_value(serde_json::json!({"path":["nodes","n","time","speed"],"keys":json["keys"],"interpolation":json["interpolation"]}))?;
            let times = match &track.times { v2::Times::Inline(t) => t.clone(), _ => unreachable!() };
            let decoded = Curve::authored(&track, false, times, match &track.values { v2::Values::Inline(v) => v.clone(), _ => unreachable!() })?;
            exact::TimeCurve::compile(&track, &decoded)
        };
        let step = curve(serde_json::json!({"keys":[[0,1],[2,3]],"interpolation":"STEP"}))?;
        assert_eq!(step.integrate(&q("0"), &q("5")), q("11"));
        assert_eq!(step.integrate(&q("-1"), &q("1")), q("2"));
        let linear = curve(serde_json::json!({"keys":[[0,0],[2,2]],"interpolation":"LINEAR"}))?;
        assert_eq!(linear.integrate(&q("0"), &q("2")), q("2"));
        assert_eq!(linear.integrate(&q("0"), &q("4")), q("6"));
        assert_eq!(linear.integrate(&q("4"), &q("0")), q("-6"));
        // A cubic with zero tangents between equal values is flat; with tangents it bows.
        let flat = curve(serde_json::json!({"keys":[[0,2],[4,2]],"interpolation":"CUBICSPLINE"}))?;
        assert_eq!(flat.integrate(&q("0"), &q("4")), q("8"));
        let bow = curve(serde_json::json!({"keys":[[0,0,0,1],[1,0,1,0]],"interpolation":"CUBICSPLINE"}))?;
        // h10 integrates to 1/12 and h11 to -1/12 over the unit segment: ∫ = (1 - 1)/12 = 0 for m0 = m1 = 1.
        assert_eq!(bow.integrate(&q("0"), &q("1")), q("0"));
        let bump = curve(serde_json::json!({"keys":[[0,0,0,1],[1,0,-1,0]],"interpolation":"CUBICSPLINE"}))?;
        assert_eq!(bump.integrate(&q("0"), &q("1")), q("1/6"));
        Ok(())
    }
}

/// Compile-time tables for the clocks: which media each mesh node shows (by pre-order
/// address), which images carry their own `time`, and the length of the one source of
/// every node whose span is inferred (A7 rule 3), from the sources the document open
/// read ([`source_length`]).
pub(super) type SceneTables = (
    std::collections::BTreeMap<String, Vec<Vec<usize>>>,
    std::collections::BTreeSet<String>,
    std::collections::BTreeMap<Vec<usize>, Rational>,
);
pub(super) fn scene_tables(scene: &v2::Scene, base: &std::path::Path, opened: &crate::footage::Opened) -> Result<SceneTables> {
    let mut samplers: std::collections::BTreeMap<String, Vec<Vec<usize>>> = Default::default();
    let mut lengths = std::collections::BTreeMap::new();
    let with_time: std::collections::BTreeSet<String> = scene.images.iter().filter(|(_, image)| image.time.is_some()).map(|(name, _)| name.clone()).collect();
    fn visit(scene: &v2::Scene, base: &std::path::Path, list: &[v2::Node], path: &mut Vec<usize>, samplers: &mut std::collections::BTreeMap<String, Vec<Vec<usize>>>, lengths: &mut std::collections::BTreeMap<Vec<usize>, Rational>, opened: &crate::footage::Opened) -> Result<()> {
        for (index, node) in list.iter().enumerate() {
            path.push(index);
            for image in scene.node_media(node) {
                let entry = samplers.entry(image).or_default();
                if !entry.contains(path) { entry.push(path.clone()); }
            }
            if node.has_clock() && node.time.as_ref().is_none_or(|time| time.duration.is_none()) {
                let sources = scene.media_sources(node);
                if sources.len() == 1 && scene.audio_sources(node).is_empty() {
                    let name = sources.into_iter().next().unwrap();
                    let file = base.join(scene.images[&name].file.as_deref().unwrap());
                    lengths.insert(path.clone(), source_length(&file, scene.render.fps, opened)?);
                }
            }
            if node.has_clock()
                && node
                    .time
                    .as_ref()
                    .is_none_or(|time| time.duration.is_none())
                && scene.media_sources(node).is_empty()
            {
                let sources = scene.audio_sources(node);
                if sources.len() == 1 {
                    let source = crate::audio::load(&base.join(sources.first().unwrap()))?;
                    lengths.insert(
                        path.clone(),
                        Rational::new(source.frames()?.into(), source.rate.into()),
                    );
                }
            }
            visit(scene, base, &node.children, path, samplers, lengths, opened)?;
            path.pop();
        }
        Ok(())
    }
    visit(scene, base, &scene.nodes, &mut Vec::new(), &mut samplers, &mut lengths, opened)?;
    Ok((samplers, with_time, lengths))
}
/// The one source's length, from the source as this document open read it
/// ([`crate::footage::Opened`]): a sequence's frames on disk and its first frame's rate, a
/// movie's stream. The open reads each source once, shares it with installation, and keeps
/// nothing for the next open, so a source that changed is never answered from before.
fn source_length(path: &std::path::Path, fps: vork_time::Time, opened: &crate::footage::Opened) -> Result<Rational> {
    opened.length(path, fps)
        .with_context(|| format!("time: the one source {} must be readable to infer its length; or write time.duration", path.display()))
}

impl Plan {
    /// Bind every node's clock from the installed tree (`mark_nodes`): resolve owned
    /// layers and inferred spans by pre-order address, refuse what rule 4 and the
    /// one-owner rule refuse, and decide which node selects each movie's frames.
    pub(super) fn bind_clocks<N: ClockNode>(&mut self, nodes: &mut [N]) -> Result<()> {
        // Pre-order addresses: a node's path is its parent's plus its index among siblings.
        let mut paths: Vec<Vec<usize>> = Vec::with_capacity(nodes.len());
        let mut sibling_counts: std::collections::HashMap<Option<usize>, usize> = Default::default();
        for node in nodes.iter() {
            let count = sibling_counts.entry(node.parent()).or_default();
            let mut path = node.parent().map(|p| paths[p].clone()).unwrap_or_default();
            path.push(*count);
            *count += 1;
            paths.push(path);
        }
        let by_path: std::collections::HashMap<&[usize], usize> = paths.iter().enumerate().map(|(i, p)| (p.as_slice(), i)).collect();
        self.clock_names = nodes.iter().map(|n| n.node().name.clone()).collect();
        let mut specs = Vec::with_capacity(nodes.len());
        let mut any = false;
        for (index, eval) in nodes.iter().enumerate() {
            let node = eval.node();
            let active = match &node.active {
                v2::Active::On(on) => ActiveSpec::On(*on),
                v2::Active::Window([a, b]) => ActiveSpec::Window(a.exact(), b.exact()),
            };
            // A window `active` alone is a clock too: it starts at active[0] (rule 2).
            let clock = node.time.clone().or_else(|| node.has_clock().then(v2::NodeClock::default));
            let time = clock.as_ref().map(|time| -> Result<TimeSpec> {
                let length = self.source_lengths.get(&paths[index]);
                let section = time.section(length);
                if let Some(line) = section.span_refusal(length) { anyhow::bail!("node {:?}: time: {line}", node.name); }
                Ok(TimeSpec { start: time.start(&node.active), speed: time.speed.exact(), section })
            }).transpose()?;
            let gated = !matches!(node.active, v2::Active::On(true));
            any |= gated || time.is_some();
            specs.push(ClockSpec { parent: eval.parent(), active, active_track: None, time, speed_track: None, gated });
        }
        self.clocks = specs;
        for layer in &mut self.layers {
            if let Some(path) = &layer.owner {
                let owner = *by_path.get(path.as_slice()).context("node animation owner is not an installed node")?;
                layer.owner_node = Some(owner);
                nodes[owner].set_dynamic();
                any = true;
            }
        }
        for index in 0..nodes.len() {
            if self.clocks[index].time.is_some() || self.clocks[index].gated { nodes[index].set_dynamic(); }
        }
        // Footage shown by a mesh under a clock follows that clock; one clock per image.
        let clock_of = |mut index: usize, clocks: &[ClockSpec], nodes: &[N]| -> Option<usize> {
            loop {
                if nodes[index].node().has_clock() { return Some(index); }
                index = clocks[index].parent?;
            }
        };
        self.footage_owner.clear();
        let timed_tracks: std::collections::BTreeSet<String> = self.layers.iter().flat_map(|l| &l.tracks)
            .filter_map(|t| match &t.target { Some(v2::TrackTarget::ImageTime { image }) => Some(image.clone()), _ => None }).collect();
        for (image, samplers) in &self.image_samplers {
            let owners: std::collections::BTreeSet<Option<usize>> = samplers.iter()
                .filter_map(|path| by_path.get(path.as_slice()).copied())
                .map(|node| clock_of(node, &self.clocks, nodes)).collect();
            ensure!(owners.len() <= 1, "image {image:?} is shown under two clocks; give each node its own image");
            if let Some(Some(owner)) = owners.into_iter().next() {
                ensure!(!self.images_with_time.contains(image) && !timed_tracks.contains(image),
                    "image {image:?}: time and a node clock are two owners; remove images.{image}.time, the node's clock shows it");
                self.footage_owner.insert(image.clone(), owner);
            }
        }
        self.has_clocks = any;
        Ok(())
    }
    /// Discrete clock decisions for pose reuse: existence per gated node and the key
    /// interval of every owned discrete track at its owner's instant.
    pub(super) fn append_clock_key(&self, scene: &Germ, out: &mut Vec<u8>) -> Result<()> {
        let clocks = self.clocks(scene)?;
        for (index, spec) in self.clocks.iter().enumerate() {
            if spec.gated { out.push(u8::from(clocks[index].is_some())); }
        }
        for layer in &self.layers {
            let Some(owner) = layer.owner_node else { continue };
            let Some(owner_local) = &clocks[owner] else { out.push(0xff); continue };
            let local = layer.exact_local(owner_local)?;
            for track in &layer.tracks {
                if let Some(keys) = &track.discrete_keys { out.extend_from_slice(&(keys.upper(&local) as u64).to_le_bytes()); }
            }
        }
        Ok(())
    }
}

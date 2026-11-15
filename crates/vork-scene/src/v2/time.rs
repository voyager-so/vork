//! Node time: existence (`active`), the node clock (`time`) and node-owned
//! animation layers. Two questions, two fields: `active` says when the node
//! exists, in parent seconds; `time` says what the content shows while it exists.
//! A clock value is a `Germ`: a rational instant and the side it is read from.
use super::*;
use vork_time::exact::{Germ, Rational};
use std::cmp::Ordering;

/// When the node exists, in parent seconds: always, never, or a half-open window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Active {
    On(bool),
    /// `[start, end)`: present when start <= p < end.
    Window([vork_time::Clock; 2]),
}
impl Default for Active {
    fn default() -> Self { Self::On(true) }
}
impl Active {
    pub fn is_default(&self) -> bool { matches!(self, Self::On(true)) }
    pub fn window(&self) -> Option<(Rational, Rational)> {
        match self { Self::Window([a, b]) => Some((a.exact(), b.exact())), Self::On(_) => None }
    }
    /// Rule 1: at parent instant `p` the node exists when `active` is true or
    /// `start <= p < end`, read in p's side (a falling clock holds `start < p <= end`).
    pub fn exists(&self, p: &Germ) -> bool {
        match self {
            Self::On(on) => *on,
            Self::Window([a, b]) => p.compare_point(&a.exact()) != Ordering::Less && p.compare_point(&b.exact()) == Ordering::Less,
        }
    }
}

/// How sound follows this node's clock speed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Pitch {
    #[default]
    Follow,
    Keep,
}

/// The node clock: where in the content to begin, how much to use, how fast,
/// whether to repeat, and when on the parent's clock it starts.
#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeClock {
    /// Parent seconds at which local time is `offset`; omitted: the window's start, else 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "vork (node clock)", "x-help-path" = "node.time.start", "x-help" = "number|rational string|null; parent seconds where local = offset; null = active window start, else 0; shifts loop phase"))]
    pub start: Option<vork_time::Clock>,
    #[serde(default = "constants::clock_zero")]
    #[schemars(extend("x-source" = "vork (node clock)", "x-help-path" = "node.time.offset", "x-help" = "number|rational string; content seconds where the clip begins; >= 0"))]
    pub offset: vork_time::Clock,
    /// Content seconds used; omitted: the rest of the one source after `offset`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "vork (node clock)", "x-help-path" = "node.time.duration", "x-help" = "number|rational string|null; content seconds used from offset; > 0; null = the rest of the one source; only active ends the node"))]
    pub duration: Option<vork_time::Clock>,
    #[serde(default = "constants::clock_one")]
    #[schemars(extend("x-source" = "vork (node clock)", "x-help-path" = "node.time.speed", "x-help" = "number|rational string; keyable >= 0 and integrated; a constant < 0 plays the trimmed section backwards from its end"))]
    pub speed: vork_time::Clock,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (node clock)", "x-help-path" = "node.time.pitch", "x-help" = "follow|keep; keep holds sound pitch when speed changes; keep is not built yet"))]
    pub pitch: Pitch,
    #[serde(default, rename = "loop")]
    #[schemars(extend("x-source" = "vork (node clock)", "x-help-path" = "node.time.loop", "x-help" = "bool; wraps inside [offset, offset+duration); needs an end"))]
    pub looping: bool,
}
impl Default for NodeClock {
    fn default() -> Self {
        Self { start: None, offset: constants::clock_zero(), duration: None, speed: constants::clock_one(), pitch: Pitch::Follow, looping: false }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthoredTime {
    #[serde(default)]
    start: Option<vork_time::Clock>,
    #[serde(default = "constants::clock_zero")]
    offset: vork_time::Clock,
    #[serde(default)]
    duration: Option<vork_time::Clock>,
    #[serde(default = "constants::clock_one")]
    speed: vork_time::Clock,
    #[serde(default)]
    pitch: Pitch,
    #[serde(default, rename = "loop")]
    looping: bool,
    #[serde(default)]
    program: Option<serde_json::Value>,
    #[serde(default)]
    inputs: Option<serde_json::Value>,
}
impl<'de> Deserialize<'de> for NodeClock {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let t = AuthoredTime::deserialize(deserializer)?;
        if t.program.is_some() || t.inputs.is_some() {
            return Err(serde::de::Error::custom("clock programs are not built yet; write time: {start, offset, duration, speed, loop}"));
        }
        Ok(Self { start: t.start, offset: t.offset, duration: t.duration, speed: t.speed, pitch: t.pitch, looping: t.looping })
    }
}

/// An animation layer owned by a node: tracks on this node's clock, paths starting at the node.
/// Like a scene layer it keeps its own clock: source time = offset + node time * speed.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeLayer {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.animation.layer.name", "x-help" = "string|null"))]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.animation.layer.tracks", "x-help" = "track[]|null; XOR clip; paths start at this node: [\"translation\"], [\"children\",name,...]; keys in this node's seconds"))]
    pub tracks: Option<Vec<Track>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.animation.layer.clip", "x-help" = "clip|null; XOR tracks; target = this node or a descendant asset"))]
    pub clip: Option<Clip>,
    #[serde(default = "constants::layer_unit")]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.animation.layer.weight", "x-help" = "number"))]
    pub weight: f32,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.animation.layer.blend", "x-help" = "replace|combine"))]
    pub blend: Blend,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.animation.layer.offset", "x-help" = "number; source time = offset + node time*speed"))]
    pub offset: f32,
    #[serde(default = "constants::layer_unit")]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.animation.layer.speed", "x-help" = "number"))]
    pub speed: f32,
    #[serde(default, rename = "loop")]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.animation.layer.loop", "x-help" = "bool"))]
    pub looping: bool,
}

/// Rules 3 to 5 and 10: the trimmed section `[offset, E)` and how a raw forward
/// clock value (offset plus the integral of |speed|) becomes the node's local time.
#[derive(Clone, Debug, PartialEq)]
pub struct Section {
    pub offset: Rational,
    /// E: written `offset + duration`, else inferred from the one source, else none.
    pub end: Option<Rational>,
    pub looping: bool,
    /// A constant negative speed: the section plays mirrored from its end.
    pub reverse: bool,
}
impl Section {
    /// E from a written duration, else from the single source's length (zero at or past its end).
    pub fn end(offset: &Rational, duration: Option<&Rational>, source_length: Option<&Rational>) -> Option<Rational> {
        match (duration, source_length) {
            (Some(duration), _) => Some(offset + duration),
            (None, Some(length)) => Some(if length > offset { length.clone() } else { offset.clone() }),
            (None, None) => None,
        }
    }
    /// Rule 4's span rule: `loop` and reverse need E to exist and exceed `offset`.
    pub fn span_refusal(&self, source_length: Option<&Rational>) -> Option<String> {
        if !self.looping && !self.reverse { return None; }
        match &self.end {
            None => Some("write time.duration".into()),
            Some(end) if end <= &self.offset => Some(match source_length {
                Some(length) => format!("nothing after offset: the source is {} s; write offset < {} or time.duration", rational(length), rational(length)),
                None => "offset >= 0; duration > 0".into(),
            }),
            Some(_) => None,
        }
    }
    /// The forward clock `f` from `raw`, and whether it stands at an end (rule 5).
    fn forward(&self, raw: Germ) -> (Germ, bool) {
        if let (Some(end), true) = (&self.end, self.looping) {
            // Rule 4: a loop wraps everywhere, before `start` included (no hold).
            return (raw.euclidean_loop(&self.offset, end).expect("validated span"), false);
        }
        if raw.compare_point(&self.offset) == Ordering::Less {
            return (Germ::point(self.offset.clone()), true);
        }
        match &self.end {
            Some(end) if raw.compare_point(end) != Ordering::Less => (Germ::left(end.clone()), true),
            _ => (raw, false),
        }
    }
    /// Whether the content clock is clamped, even when its picture reads an endpoint from below.
    pub fn holds(&self, raw: &Germ) -> bool {
        self.forward(raw.clone()).1
    }
    /// Local time from the raw forward value, read in the direction of travel (rule 9).
    pub fn local(&self, raw: Germ) -> Germ {
        let (f, standing) = self.forward(raw);
        if !self.reverse { return f; }
        let end = self.end.as_ref().expect("reverse needs a span");
        if standing {
            // The reversed clip stands at its first state exactly, or at E from below.
            return if f.compare_point(&self.offset) == Ordering::Equal { Germ::left(end.clone()) } else { Germ::point(self.offset.clone()) };
        }
        Germ::point(end.clone()).sub(&f.sub(&Germ::point(self.offset.clone())))
    }
    /// Rule 2 for a constant speed: `raw = offset + (p - start) * |speed|`. A reversed
    /// node reads a rising parent instant from above so its mirror falls (rule 9).
    pub fn constant_raw(&self, p: &Germ, start: &Rational, speed: &Rational) -> Germ {
        let p = if self.reverse && p.compare_point(p.value()) == Ordering::Equal { Germ::right(p.value().clone()) } else { p.clone() };
        let magnitude = if speed < &Rational::from_integer(0.into()) { -speed } else { speed.clone() };
        Germ::point(self.offset.clone()).add(&p.sub(&Germ::point(start.clone())).scale(&magnitude))
    }
    /// Local time for a constant speed at parent instant `p` (rules 2, 4, 5, 9, 10).
    pub fn local_constant(&self, p: &Germ, start: &Rational, speed: &Rational) -> Germ {
        self.local(self.constant_raw(p, start, speed))
    }
}

/// A movie or frame-sequence file: the content with a length of its own.
pub fn is_media_file(path: &str) -> bool {
    let name = std::path::Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or("");
    name.contains('%') || std::path::Path::new(path).extension().is_some_and(|e| ["mov", "mp4", "mkv"].iter().any(|v| e.eq_ignore_ascii_case(v)))
}

/// A rational for messages: an integer, a decimal when exact, else `n/d`.
pub fn rational(value: &Rational) -> String {
    if value.is_integer() { return value.numer().to_string(); }
    let small = |n: &dyn std::fmt::Display| n.to_string().parse::<i64>().ok();
    match (small(value.numer()), small(value.denom())) {
        (Some(n), Some(d)) => vork_time::Time::new(n, d).map(|t| match t.json() { serde_json::Value::String(s) => s, v => v.to_string() }).unwrap_or_else(|_| format!("{n}/{d}")),
        _ => format!("{}/{}", value.numer(), value.denom()),
    }
}

impl NodeClock {
    /// `start` resolved against the node's `active` window (rule 2).
    pub fn start(&self, active: &Active) -> Rational {
        match (&self.start, active.window()) {
            (Some(start), _) => start.exact(),
            (None, Some((a, _))) => a,
            (None, None) => Rational::from_integer(0.into()),
        }
    }
    pub fn reverse(&self) -> bool { self.speed.exact() < Rational::from_integer(0.into()) }
    pub fn section(&self, source_length: Option<&Rational>) -> Section {
        let offset = self.offset.exact();
        let duration = self.duration.as_ref().map(vork_time::Clock::exact);
        Section { end: Section::end(&offset, duration.as_ref(), source_length), offset, looping: self.looping, reverse: self.reverse() }
    }
    /// Rule 12's constant checks, before any source length is known.
    pub fn refusal(&self) -> Option<String> {
        if self.pitch == Pitch::Keep {
            return Some("time.pitch: keep is not built yet; use time.pitch: follow".into());
        }
        let zero = Rational::from_integer(0.into());
        if self.offset.exact() < zero || self.duration.as_ref().is_some_and(|d| d.exact() <= zero) {
            return Some("time: offset >= 0; duration > 0".into());
        }
        if !self.speed.as_f64().is_finite() || self.start.as_ref().is_some_and(|s| !s.as_f64().is_finite()) {
            return Some("time: start and speed must be finite".into());
        }
        None
    }
}
impl Node {
    /// Whether this node carries a clock of its own: a `time` or a window `active`.
    pub fn has_clock(&self) -> bool { self.time.is_some() || matches!(self.active, Active::Window(_)) }
}

#[cfg(test)]
mod clip_model_tests {
    use super::*;
    fn q(text: &str) -> Rational { vork_time::exact::decimal(text).unwrap() }
    fn p(text: &str) -> Germ { Germ::point(q(text)) }
    /// A value and the side it is read from, as the clip-model tables write them.
    fn reading(g: &Germ) -> String {
        let value = rational(g.value());
        match g.compare_point(g.value()) { Ordering::Less => format!("{value}-"), Ordering::Greater => format!("{value}+"), Ordering::Equal => value }
    }
    fn time(json: serde_json::Value) -> NodeClock { serde_json::from_value(json).unwrap() }
    fn local(t: &NodeClock, active: &Active, source: Option<&str>, parent: &str) -> String {
        let section = t.section(source.map(q).as_ref());
        reading(&section.local_constant(&p(parent), &t.start(active), &t.speed.exact()))
    }

    #[test]
    fn boundary_table_forward() {
        // clip-model.md: active [2, 10), offset 12, duration 4, loop: 2 -> 12; 5.9 -> 15.9; 6 -> 12; 10 -> gone.
        let active: Active = serde_json::from_value(serde_json::json!([2, 10])).unwrap();
        let t = time(serde_json::json!({"offset": 12, "duration": 4, "loop": true}));
        assert_eq!(local(&t, &active, None, "2"), "12");
        assert_eq!(local(&t, &active, None, "5.9"), "15.9");
        assert_eq!(local(&t, &active, None, "6"), "12");
        assert!(active.exists(&p("2")) && active.exists(&p("9.999")) && !active.exists(&p("10")) && !active.exists(&p("1.5")));
        // Hold: without loop, local >= E stands at E from below; before start, at offset exactly.
        let hold = time(serde_json::json!({"offset": 12, "duration": 4}));
        assert_eq!(local(&hold, &active, None, "7"), "16-");
        assert_eq!(local(&hold, &active, None, "6"), "16-");
        assert_eq!(local(&hold, &active, None, "5.5"), "15.5");
        let late = time(serde_json::json!({"start": 4, "offset": 12, "duration": 4}));
        assert_eq!(local(&late, &active, None, "3"), "12", "before start: the first frame, exactly");
        // speed 0 holds the frame at offset and reads exactly whatever the side of p.
        let frozen = time(serde_json::json!({"offset": 3, "speed": 0}));
        assert_eq!(reading(&frozen.section(None).local_constant(&Germ::left(q("7")), &q("0"), &q("0"))), "3");
        // start shifts loop phase (rule 6): start 0 under [2, 10) begins at source 14.
        let phase = time(serde_json::json!({"start": 0, "offset": 12, "duration": 4, "loop": true}));
        assert_eq!(local(&phase, &active, None, "2"), "14");
        assert_eq!(local(&phase, &active, None, "4"), "12");
        // xtimerev 4: before a written start a loop wraps (Euclidean), it does not hold.
        let early = time(serde_json::json!({"start": 4, "offset": 2, "duration": 6, "loop": true}));
        assert_eq!(local(&early, &Active::default(), None, "1"), "5");
        assert_eq!(local(&early, &Active::default(), None, "4"), "2");
        // A window without `time` still starts the clock at the window (rule 2): local = p - start.
        let plain = time(serde_json::json!({}));
        assert_eq!(local(&plain, &active, None, "3"), "1");
        // A file shorter than its span: inferred E = source length; hold from there.
        let short = time(serde_json::json!({}));
        let window: Active = serde_json::from_value(serde_json::json!([0, 10])).unwrap();
        assert_eq!(local(&short, &window, Some("6"), "5"), "5");
        assert_eq!(local(&short, &window, Some("6"), "6"), "6-");
        assert_eq!(local(&short, &window, Some("6"), "9"), "6-");
    }

    #[test]
    fn boundary_table_reverse() {
        // A7.10 worked: 10 s source, offset 2, duration 6, speed -1: parent 0, 3, 6, 7.
        let on = Active::default();
        let t = time(serde_json::json!({"offset": 2, "duration": 6, "speed": -1}));
        // "8 and 5 from below; 2 standing, exact" (amend-3): a falling clock reads from below.
        let readings: Vec<_> = ["0", "3", "6", "7"].iter().map(|at| local(&t, &on, Some("10"), at)).collect();
        assert_eq!(readings, ["8-", "5-", "2", "2"]);
        let looped = time(serde_json::json!({"offset": 2, "duration": 6, "speed": -1, "loop": true}));
        let readings: Vec<_> = ["0", "3", "6", "7"].iter().map(|at| local(&looped, &on, Some("10"), at)).collect();
        assert_eq!(readings, ["8-", "5-", "8-", "7-"]);
        // Speed -2, as the review computed.
        let fast = time(serde_json::json!({"offset": 2, "duration": 6, "speed": -2}));
        let readings: Vec<_> = ["0", "3", "6", "7"].iter().map(|at| local(&fast, &on, Some("10"), at)).collect();
        assert_eq!(readings, ["8-", "2", "2", "2"]);
        let fast = time(serde_json::json!({"offset": 2, "duration": 6, "speed": -2, "loop": true}));
        let readings: Vec<_> = ["0", "3", "6", "7"].iter().map(|at| local(&fast, &on, Some("10"), at)).collect();
        assert_eq!(readings, ["8-", "8-", "8-", "6-"]);
        // A reversed child inside a forward parent looping over [0, 4): child 8-, 5, 8-, 5 at scene 0, 3, 4, 7.
        let parent = time(serde_json::json!({"duration": 4, "loop": true}));
        let readings: Vec<_> = ["0", "3", "4", "7"].iter().map(|at| {
            let parent_local = parent.section(None).local_constant(&p(at), &q("0"), &q("1"));
            reading(&t.section(Some(&q("10"))).local_constant(&parent_local, &q("0"), &q("-1")))
        }).collect();
        assert_eq!(readings, ["8-", "5-", "8-", "5-"]);
    }

    #[test]
    fn reviewer_subtree_at_the_wrap_instant() {
        // amend-1-review 1: parent {duration 6, speed -1, loop}; at scene 0, 1/30, 3, 6, 7
        // the parent reads 6-, 179/30, 3, 6-, 5, all from below.
        let parent = time(serde_json::json!({"duration": 6, "speed": -1, "loop": true}));
        let section = parent.section(None);
        let at = |t: &str| section.local_constant(&p(t), &q("0"), &q("-1"));
        assert_eq!(reading(&at("0")), "6-");
        assert_eq!(reading(&at("1/30")), "179/30-");
        assert_eq!(reading(&at("3")), "3-");
        assert_eq!(reading(&at("6")), "6-");
        assert_eq!(reading(&at("7")), "5-");
        // A child active on [0, 6) is present at all five instants: no hole at the wrap.
        let child: Active = serde_json::from_value(serde_json::json!([0, 6])).unwrap();
        for t in ["0", "1/30", "3", "6", "7"] { assert!(child.exists(&at(t)), "{t}"); }
        // A child looping over [0, 6) shows the end of its cycle at the wrap, not its start.
        let looping = time(serde_json::json!({"duration": 6, "loop": true}));
        assert_eq!(reading(&looping.section(None).local_constant(&at("0"), &q("0"), &q("1"))), "6-");
        // A STEP key at 6 has not fired: the value is below 6.
        assert_eq!(at("0").compare_point(&q("6")), Ordering::Less);
        // A reversed child under a reversed parent rises: read from above, which every
        // consumer (a <= p < b, floor, a STEP key at its time) reads as exactly.
        let reversed = time(serde_json::json!({"duration": 6, "speed": -1}));
        let rising = reversed.section(None).local_constant(&at("3"), &q("0"), &q("-1"));
        assert_eq!(reading(&rising), "3+");
        let window: Active = serde_json::from_value(serde_json::json!([3, 6])).unwrap();
        assert!(window.exists(&rising) && window.exists(&p("3")));
        assert_eq!(rising.floor(), p("3").floor());
        // 30 fps source trimmed to [2, 8) reversed: frames 239, 238, ... 60 once each, then 60 held.
        let trim = time(serde_json::json!({"offset": 2, "duration": 6, "speed": -1}));
        let frame = |t: &str| trim.section(Some(&q("10"))).local_constant(&p(t), &q("0"), &q("-1")).scale(&q("30")).floor();
        let frames: Vec<i64> = (0..=180).map(|i| frame(&format!("{i}/30")).to_string().parse().unwrap()).collect();
        assert_eq!(frames[0], 239);
        assert_eq!(frames[1], 238);
        assert_eq!(frames[179], 60);
        assert_eq!(frames[180], 60, "the hold shows the section's first frame");
        assert_eq!(frames[..180].windows(2).filter(|w| w[0] == w[1]).count(), 0, "every moving frame once");
    }

    #[test]
    fn span_rule_covers_inferred_spans() {
        // amend-1-review 2: 10 s source, offset 10, no duration, speed -1, loop: E = 10 = offset.
        let t = time(serde_json::json!({"offset": 10, "speed": -1, "loop": true}));
        let section = t.section(Some(&q("10")));
        assert_eq!(section.end, Some(q("10")));
        assert_eq!(section.span_refusal(Some(&q("10"))).unwrap(), "nothing after offset: the source is 10 s; write offset < 10 or time.duration");
        let unbounded = time(serde_json::json!({"loop": true}));
        assert_eq!(unbounded.section(None).span_refusal(None).unwrap(), "write time.duration");
        assert!(time(serde_json::json!({"duration": 2, "loop": true})).section(None).span_refusal(None).is_none());
        // Forward, an empty inferred span is a hold at E: the last frame.
        let forward = time(serde_json::json!({"offset": 10}));
        assert!(forward.section(Some(&q("10"))).span_refusal(Some(&q("10"))).is_none());
        assert_eq!(reading(&forward.section(Some(&q("10"))).local_constant(&p("3"), &q("0"), &q("1"))), "10-");
        assert_eq!(time(serde_json::json!({"offset": -1})).refusal().unwrap(), "time: offset >= 0; duration > 0");
        assert_eq!(time(serde_json::json!({"duration": 0})).refusal().unwrap(), "time: offset >= 0; duration > 0");
        assert!(time(serde_json::json!({"speed": 0})).refusal().is_none());
    }

    #[test]
    fn spellings_are_refused_with_one_line() {
        let program = serde_json::from_value::<NodeClock>(serde_json::json!({"program": "ramp", "inputs": {}})).unwrap_err().to_string();
        assert!(program.contains("clock programs are not built yet; write time: {start, offset, duration, speed, loop}"), "{program}");
        // A node's layers keep their own clock (section 8): two clips at different rates on one node.
        let layer: NodeLayer = serde_json::from_value(serde_json::json!({"tracks": [], "offset": 2, "speed": 0.5, "loop": true})).unwrap();
        assert!(layer.offset == 2. && layer.speed == 0.5 && layer.looping);
        assert!(serde_json::from_value::<NodeLayer>(serde_json::json!({"tracks": [], "time": {}})).is_err());
        assert!(serde_json::from_value::<NodeClock>(serde_json::json!({"looping": true})).is_err());
        let t: NodeClock = serde_json::from_value(serde_json::json!({"offset": "1/3", "speed": "24/25"})).unwrap();
        assert_eq!(t.offset.exact(), q("1/3"));
        assert_eq!(serde_json::to_value(&t).unwrap(), serde_json::json!({"offset": "1/3", "speed": "24/25", "pitch": "follow", "loop": false}));
        let active: Active = serde_json::from_value(serde_json::json!(false)).unwrap();
        assert!(!active.exists(&p("0")));
    }
}

//! Resource-report seam (WF5 round 3).
//!
//! USER DIRECTIVE: a native render that failed to fetch/decode any
//! image/SVG/font/icon must be impossible to mistake for a clean render.
//! The PIXELS keep browser-parity placeholder semantics (copy the bugs —
//! grey/rose 1×1, Canvas2D fallback class, transparent emoji); the screaming
//! happens in this reporting channel, which the CLI turns into a stderr
//! banner, `--resource-report` JSON, and `--fail-on-missing-assets` exit 3.
//!
//! Two channels feed the CLI's per-render report:
//! 1. **Prefetch results** — returned by value from
//!    [`crate::fetch::prefetch_blocking`] (`PrefetchReport`).
//! 2. **Runtime events** — recorded process-globally HERE from the draw
//!    paths (placeholder bakes in [`crate::engine_seam`], font-engine
//!    decisions and emoji misses in moda-engine's `text_draw`), because the
//!    M1-frozen seam fns carry no reporting parameter (same rationale as
//!    `engine_seam`'s process-global store).
//!
//! Events are deduped on record (one event per unique payload) so per-frame
//! re-resolution cannot flood the report.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "native-io")]
use crate::{AssetFailure, BlankAsset};

/// A resource observation from the render path.
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeEvent {
    /// A pattern / shader-image source resolved to a placeholder at draw
    /// time (`state`: `"failed"` | `"timed-out"` | `"never-prefetched"`).
    /// `failed`/`timed-out` normally duplicate a prefetch failure; a
    /// `never-prefetched` bake means the scene scan missed a source — a
    /// collection bug, and always a missing resource.
    PlaceholderBaked {
        src: String,
        state: String,
        shader_image: bool,
    },
    /// A source that DECODED SUCCESSFULLY but paints nothing — every
    /// pixel fully transparent (`coverage_permille == 0`). Not a decode failure:
    /// the pixels shipped, they are simply invisible, which until now was
    /// indistinguishable from a healthy asset on every channel we have.
    ///
    /// `src` is SANITIZED ([`sanitized_asset_ref`]) — the asset loaded, so this
    /// is telemetry about a working fetch and must not carry the signed-URL
    /// credentials that `AssetFailure::url` deliberately keeps for reproduction.
    ///
    /// `shader_image` exists for shape parity with [`Self::PlaceholderBaked`]
    /// and is ALWAYS `false` today: several shader processings re-encode
    /// channels (`PoissonMask` puts the source alpha in `g` and a distance field
    /// in `r`), so an all-zero alpha channel on a processed image is not
    /// evidence that it paints nothing, and a false blank on a trust signal is
    /// worse than no signal at all.
    BlankRasterDecoded {
        src: String,
        coverage_permille: u16,
        cause: String,
        shader_image: bool,
    },
    /// renderWithMsdf fallback-class routing (browser-parity: the browser
    /// Canvas2D-renders these too — NOT a missing resource, but reported so
    /// fallback typography is visible).
    FontFallbackClass {
        node_id: String,
        reason: String,
        /// A low-cardinality class for the decline (`coverage`,
        /// `font-unavailable:not-vendored`, …) — `reason` is free text that
        /// interpolates family names and error messages, so it can neither be
        /// grouped nor alerted on. Empty on lanes that don't classify.
        cause: String,
        families: Vec<String>,
    },
    /// A text node asks for a family the renderer can't draw (not in its font catalog, or no face of it can be
    /// read): its runs draw in `drawn` instead. Not a missing resource in the vendor-sanity sense (the text draws),
    /// but the author's font is not what they see.
    FontSubstituted {
        node_id: String,
        requested: String,
        drawn: String,
    },
    /// A font resource the catalog promises could not be used (vendor-sanity
    /// failure: msdf-ready family without vendored bytes) — a MISSING
    /// RESOURCE; the node was skipped or degraded.
    FontMissing { node_id: String, detail: String },
    /// Emoji cluster with no renderable SVG at draw time (the browser bakes
    /// a 1×1 transparent for genuinely-missing CDN emoji).
    EmojiMissing { cluster: String },
    /// Emoji cluster the bundled full Noto SVG set does not contain. Emitted by
    /// [`record_emoji_miss`] when the installed classifier (bundle-index
    /// membership, `moda_fonts::emoji_bundle`) marks the cluster a browser-parity
    /// gap: the bundle is the whole set the CDN serves, so this cluster's CDN
    /// fetch would 404 too and the browser bakes the SAME transparent hole — NOT
    /// a missing resource. Reports ingest it directly into their coverage-gap
    /// list (no cross-event demotion).
    EmojiCoverageGap { cluster: String },
    /// The render drew a node at lower quality than the scene asks for, by an
    /// engine rule the author can't see: a frame holding an active backdrop
    /// effect renders without MSAA (`backdrop_aliases_frame`, quirk Q-02), a
    /// backdrop blur reach past the target-px clamp draws at the clamp
    /// (`blur_radius_clamped`). The pixels shipped — NOT a missing resource;
    /// reports surface it as an export warning. `message` names the layer.
    RenderQuality {
        code: String,
        node_id: String,
        message: String,
    },
}

impl RuntimeEvent {
    /// True when the event represents a missing resource (vs. browser-parity
    /// degraded routing).
    pub fn is_missing_resource(&self) -> bool {
        match self {
            RuntimeEvent::PlaceholderBaked { .. } => true,
            // A blank decode IS a missing resource in the sense this predicate
            // names (the node has no visible content). It deliberately does NOT
            // reach either report's `failure_count` — see the comment there.
            RuntimeEvent::BlankRasterDecoded { .. } => true,
            RuntimeEvent::FontFallbackClass { .. } => false,
            RuntimeEvent::FontSubstituted { .. } => false,
            RuntimeEvent::FontMissing { .. } => true,
            RuntimeEvent::EmojiMissing { .. } => true,
            RuntimeEvent::EmojiCoverageGap { .. } => false,
            RuntimeEvent::RenderQuality { .. } => false,
        }
    }

    fn dedup_key(&self) -> String {
        match self {
            RuntimeEvent::PlaceholderBaked {
                src,
                state,
                shader_image,
            } => format!("placeholder\u{1}{src}\u{1}{state}\u{1}{shader_image}"),
            // Keyed on `src` ALONE, unlike the other events: the same source
            // decoded twice at different min-raster sizes is one blank asset,
            // not two, and `coverage_permille` is exactly the kind of field that
            // would fragment the key while saying nothing new.
            RuntimeEvent::BlankRasterDecoded { src, .. } => format!("blank\u{1}{src}"),
            RuntimeEvent::FontFallbackClass {
                node_id,
                reason,
                cause,
                families,
            } => format!(
                "font-fallback\u{1}{node_id}\u{1}{reason}\u{1}{cause}\u{1}{}",
                families.join(",")
            ),
            RuntimeEvent::FontSubstituted { node_id, requested, drawn } => {
                format!("font-substituted\u{1}{node_id}\u{1}{requested}\u{1}{drawn}")
            }
            RuntimeEvent::FontMissing { node_id, detail } => {
                format!("font-missing\u{1}{node_id}\u{1}{detail}")
            }
            RuntimeEvent::EmojiMissing { cluster } => format!("emoji-missing\u{1}{cluster}"),
            RuntimeEvent::EmojiCoverageGap { cluster } => format!("emoji-gap\u{1}{cluster}"),
            RuntimeEvent::RenderQuality { code, node_id, message } => {
                format!("render-quality\u{1}{code}\u{1}{node_id}\u{1}{message}")
            }
        }
    }
}

/// Credential-free, bounded form of an asset source for telemetry fields.
/// Strips the query/fragment (signed-URL credentials) AND any authority
/// userinfo (`user:token@host`), collapses `data:` URIs to
/// `data:<mime> (<N> bytes)`, and caps the remainder.
///
/// Lives HERE rather than in `fetch` because the `native-io`-free graph needs it
/// too: [`RuntimeEvent::BlankRasterDecoded`] is recorded from the wasm editor's
/// `decode_pattern_bytes`, and an unsanitized src there would put presigned-GCS
/// credentials into a telemetry channel.
pub fn sanitized_asset_ref(src: &str) -> String {
    const MAX: usize = 200;
    if let Some(rest) = src.strip_prefix("data:") {
        let mime: String = rest
            .chars()
            .take_while(|c| *c != ';' && *c != ',')
            .take(64)
            .collect();
        return format!("data:{mime} ({} bytes)", src.len());
    }
    let base = src.split(['?', '#']).next().unwrap_or(src);
    let query_stripped = base.len() != src.len();
    // Drop userinfo. URL semantics delimit userinfo at the LAST '@' inside the
    // AUTHORITY (everything before the first path '/'), so a raw-'@'-bearing
    // credential (user:p@ss@host) is cut whole; scheme-relative sources
    // ('//user:tok@host/…') get the same treatment as '<scheme>://'.
    let authority_start = base
        .find("://")
        .map(|i| i + 3)
        .or_else(|| base.starts_with("//").then_some(2));
    let base = match authority_start {
        Some(start) => {
            let authority_end = base[start..]
                .find('/')
                .map(|i| start + i)
                .unwrap_or(base.len());
            match base[start..authority_end].rfind('@') {
                Some(at) => format!("{}…@{}", &base[..start], &base[start + at + 1..]),
                None => base.to_string(),
            }
        }
        None => base.to_string(),
    };
    let mut out: String = base.chars().take(MAX).collect();
    if base.chars().count() > MAX {
        out.push('…');
    } else if query_stripped {
        out.push_str("?…"); // marks that a query/fragment was stripped
    }
    out
}

struct Collector {
    events: Vec<RuntimeEvent>,
    seen: HashSet<String>,
    /// Node ids the page-global VECTOR lane served — the denominator of the
    /// vector→MSDF share. A SET, not a counter, so it dedupes on node
    /// identity exactly as `seen` does for the fallback events: a node planned
    /// twice in one report window (a shared/master node across the pages of a
    /// multi-page render, or a `moda-cli batch` run) would otherwise count once in
    /// the numerator and twice in the denominator — biasing the share toward
    /// looking HEALTHY, which is the direction that hid this bug in the first
    /// place. Sized like the events set it lives beside, and only text nodes.
    vector_served: HashSet<String>,
    /// The first `FontMissing` recorded since [`reset`] — `(node_id, detail)`.
    /// Kept OUTSIDE `events` so it survives [`drain`]: the animated export lanes
    /// poll it between frames ([`first_font_missing`]) to stop rendering the
    /// moment a text node has drawn without its font, and a per-page drainer
    /// (`moda-cli batch`) must not be able to hide the miss from that poll.
    first_font_missing: Option<(String, String)>,
}

fn collector() -> &'static Mutex<Collector> {
    static SLOT: OnceLock<Mutex<Collector>> = OnceLock::new();
    SLOT.get_or_init(|| {
        Mutex::new(Collector {
            events: Vec::new(),
            seen: HashSet::new(),
            vector_served: HashSet::new(),
            first_font_missing: None,
        })
    })
}

/// Record a runtime resource event (deduped on payload).
///
/// Gated on [`COLLECTING`] exactly like [`record_vector_text_served`], and for
/// the same reason: only the lanes that later BUILD a report arm it (moda-py per
/// render, moda-cli per invocation), while the wasm editor, mobile and the
/// desktop shell never `reset` and never `drain`. Ungated, `events` and `seen`
/// grew for the life of the tab in the never-shrinking wasm arena — and the taxonomy
/// work enlarged that key space three ways at once: `cause` joined `dedup_key`,
/// `families` went from hardcoded-empty to populated (including on `emoji-run`,
/// in practice the most frequent decline of all), and a `not-vendored` reason now
/// embeds a per-url digest that changes on every re-signed team-font url.
///
/// The live editor arms a narrower mode instead ([`arm_events`]): events only, no
/// vector-served denominator, and a bounded dedup memory, so a host can drain the
/// (normally empty) event list and count its fallbacks without the unbounded growth
/// described above.
pub fn record(event: RuntimeEvent) {
    capture_push(|| Captured::Event(event.clone()));
    let collecting = COLLECTING.load(std::sync::atomic::Ordering::Relaxed);
    if !collecting && !EVENTS_ONLY.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let mut c = collector().lock().expect("resource report lock");
    if !collecting && c.seen.len() >= EVENTS_ONLY_SEEN_CAP {
        // Events-only lane: forget the dedup memory rather than grow it for the life
        // of the tab. The host dedupes what it drains, so a re-recorded event costs
        // one duplicate row, never a leak.
        c.seen.clear();
    }
    if c.seen.insert(event.dedup_key()) {
        if c.first_font_missing.is_none() {
            if let RuntimeEvent::FontMissing { node_id, detail } = &event {
                c.first_font_missing = Some((node_id.clone(), detail.clone()));
            }
        }
        c.events.push(event);
    }
}

/// The first text node that drew WITHOUT its font since the last [`reset`]
/// (`RuntimeEvent::FontMissing` — the vendor-sanity scream), as
/// `(node_id, detail)`, or `None` while every node has drawn with its bytes.
///
/// A non-draining poll for the animated export lanes: a `FontMissing` is a
/// reason to stop producing frames; retryability is ruled separately from the
/// consumed resource snapshot. A lane that pays per frame reads this after each
/// frame and aborts instead of rendering thousands more frames into a refusal
/// (2026-09-18: 8,410 frames / 212 s of GPU per attempt, nine attempts, for one
/// unvendored family the first frame had already screamed about).
pub fn first_font_missing() -> Option<(String, String)> {
    collector()
        .lock()
        .expect("resource report lock")
        .first_font_missing
        .clone()
}

/// Classifies a draw-time emoji miss as a **browser-parity coverage gap**
/// (`true`) vs a genuine **missing resource** (`false`). Installed by
/// `moda-fonts` from the bundled Noto pack: a cluster the PRESENT bundle's index
/// lacks is one the CDN 404s too (both bake the same transparent hole), whereas
/// a cluster the bundle indexes but can't decode (corrupt entry) is a real miss.
/// Absent on lanes with no bundle (wasm, a build without the pack) → every miss
/// stays `EmojiMissing`.
type EmojiMissClassifier = Box<dyn Fn(&str) -> bool + Send + Sync>;
static EMOJI_MISS_CLASSIFIER: OnceLock<EmojiMissClassifier> = OnceLock::new();

/// Install the emoji-miss classifier (first call wins).
pub fn install_emoji_miss_classifier(classifier: EmojiMissClassifier) {
    let _ = EMOJI_MISS_CLASSIFIER.set(classifier);
}

/// Record a draw-time emoji miss, CLASSIFIED per-event via the installed
/// classifier: a browser-parity [`RuntimeEvent::EmojiCoverageGap`] when the
/// bundle simply lacks the cluster, else a [`RuntimeEvent::EmojiMissing`]. Draw
/// sites call this instead of recording `EmojiMissing` directly, so the
/// classification is decided at every miss event (robust to the negative-result
/// caches whose lifetimes don't align with the per-render report reset) rather
/// than only when a cache is first populated.
pub fn record_emoji_miss(cluster: &str) {
    let is_gap = EMOJI_MISS_CLASSIFIER
        .get()
        .map(|c| c(cluster))
        .unwrap_or(false);
    record(if is_gap {
        RuntimeEvent::EmojiCoverageGap {
            cluster: cluster.to_string(),
        }
    } else {
        RuntimeEvent::EmojiMissing {
            cluster: cluster.to_string(),
        }
    });
}

/// One ingested browser-parity fallback-class routing event.
///
/// This replaced the old `(node, reason, families)` tuples the two reports
/// (`moda-cli`, `moda-py`) each kept privately: `reason` is free text — it
/// interpolates family names, variants and shaping-error messages — so the
/// taxonomy had to prefix-match it and nothing could group by cause. `cause` is
/// the low-cardinality class, and `families` is finally populated (the vector
/// emit site hardcoded it empty, so every production record read `families: []`).
///
/// Defined HERE, next to the event it ingests, so the two reports share one
/// shape and one [`font_fallback_taxonomy`] — the drift this ticket found
/// (moda-py's twin report had no taxonomy block at all, and moda-py is the
/// default warm lane) is exactly what a per-crate copy invites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FontFallbackRecord {
    pub node: String,
    pub reason: String,
    pub cause: String,
    pub families: Vec<String>,
}

#[cfg(feature = "native-io")]
#[derive(Default)]
pub struct RuntimeEventSummary {
    pub vector_nodes_served: u64,
    pub never_prefetched: Vec<(String, String, bool)>,
    pub blank: Vec<BlankAsset>,
    pub font_missing: Vec<(String, String)>,
    pub font_fallbacks: Vec<FontFallbackRecord>,
    /// `(node, requested family, drawn family)`.
    pub font_substituted: Vec<(String, String, String)>,
    pub emoji_draw_misses: Vec<String>,
    pub emoji_runtime_coverage_gaps: Vec<String>,
    /// `(code, node, message)` ([`RuntimeEvent::RenderQuality`]).
    pub render_quality: Vec<(String, String, String)>,
}

#[cfg(feature = "native-io")]
pub fn summarize_runtime_events(
    events: Vec<RuntimeEvent>,
    known_failures: &[AssetFailure],
    known_blank: &[BlankAsset],
) -> RuntimeEventSummary {
    let mut summary = RuntimeEventSummary {
        vector_nodes_served: vector_text_served(),
        ..RuntimeEventSummary::default()
    };
    for event in events {
        match event {
            RuntimeEvent::PlaceholderBaked {
                src,
                state,
                shader_image,
            } => {
                if !known_failures.iter().any(|failure| failure.url == src) {
                    summary.never_prefetched.push((src, state, shader_image));
                }
            }
            RuntimeEvent::BlankRasterDecoded {
                src,
                coverage_permille,
                cause,
                ..
            } => {
                if !known_blank.iter().any(|blank| blank.url == src)
                    && !summary.blank.iter().any(|blank| blank.url == src)
                {
                    summary.blank.push(BlankAsset {
                        url: src,
                        nodes: Vec::new(),
                        coverage_permille,
                        cause,
                    });
                }
            }
            RuntimeEvent::FontMissing { node_id, detail } => {
                summary.font_missing.push((node_id, detail));
            }
            RuntimeEvent::FontFallbackClass {
                node_id,
                reason,
                cause,
                families,
            } => summary.font_fallbacks.push(FontFallbackRecord {
                node: node_id,
                reason,
                cause,
                families,
            }),
            RuntimeEvent::FontSubstituted { node_id, requested, drawn } => {
                summary.font_substituted.push((node_id, requested, drawn));
            }
            RuntimeEvent::EmojiCoverageGap { cluster } => {
                summary.emoji_runtime_coverage_gaps.push(cluster);
            }
            RuntimeEvent::EmojiMissing { cluster } => summary.emoji_draw_misses.push(cluster),
            RuntimeEvent::RenderQuality { code, node_id, message } => {
                summary.render_quality.push((code, node_id, message));
            }
        }
    }
    summary
}

/// The one place the label's text lives. Both the public constant and the match
/// prefix below expand from it, because as two independent literals they can
/// DESYNC — and that failure is silent and maximally misleading: the numerator
/// matches nothing while the denominator keeps counting, so every render reads as
/// 100% healthy vector, the exact shape of the bug this file exists to measure.
macro_rules! vector_lane_label {
    () => {
        "vector→msdf"
    };
}

/// The label the page-global vector lane stamps on its declines
/// (`renderWithVector` → false → `renderWithMsdf`).
///
/// Load-bearing in TWO places that must agree: the numerator's prefix match here,
/// and the denominator's gate at the draw site (`moda_engine::text_draw`, which
/// counts a served node only for THIS lane). The hybrid arm and the
/// effects/kinetic reroute go through the same builder under different labels; if
/// they fed the same counters, the vector share would read healthier than reality
/// whenever `NATIVE_RENDER_TEXT_ENGINE=hybrid`.
pub const VECTOR_LANE_LABEL: &str = vector_lane_label!();

/// [`VECTOR_LANE_LABEL`] as it appears at the start of a recorded reason/cause.
const VECTOR_TO_MSDF_PREFIX: &str = concat!(vector_lane_label!(), ":");

/// The fallback-class taxonomy block shared by every report.
///
/// `vector_nodes_served` is the DENOMINATOR the report was missing: without it the
/// report could count vector→MSDF declines but not say what SHARE of the lane's
/// text they were, which is how a whole-document silent downgrade to the
/// deprecated engine stayed invisible in every dashboard. Passed in rather than
/// read from the global counter so this stays a PURE function of its inputs —
/// testable without touching process state, and impossible to pair with a count
/// from a different render.
pub fn font_fallback_taxonomy(
    records: &[FontFallbackRecord],
    vector_nodes_served: u64,
) -> moda_scene::value::Value {
    // Keyed on `cause`, not `reason`: `cause` exists precisely because `reason` is
    // free text that interpolates family names and shaping-error prose, and
    // `degraded_tier`'s "no lane arrow" test over interpolated prose is exactly the
    // fragility this ticket is about.
    let count = |prefix: &str| {
        records
            .iter()
            .filter(|f| f.cause.starts_with(prefix))
            .count()
    };
    // Per-cause split of the vector→MSDF bucket, e.g.
    // `{"font-unavailable:not-vendored": 189}`. Bounded cardinality by
    // construction (see `VectorEngineDecision::Fallback::cause`).
    let mut causes: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    // Counted, not just collected: every downstream consumer caps this list, and a
    // cap over an alphabetically-sorted set keeps the alphabetically-first families
    // rather than the most-affected ones (`causes` is already ranked by count).
    let mut family_counts: std::collections::BTreeMap<&str, usize> =
        std::collections::BTreeMap::new();
    for f in records
        .iter()
        .filter(|f| f.cause.starts_with(VECTOR_TO_MSDF_PREFIX))
    {
        // Bucket, never echo: falling back to the RAW cause as the key would make
        // this histogram unbounded the moment an emitter's `reason` and `cause`
        // disagree about the lane prefix — free text as a metric label.
        let cause = f
            .cause
            .strip_prefix(VECTOR_TO_MSDF_PREFIX)
            .map(str::trim)
            // An empty remainder is as unusable a metric label as a missing one.
            .filter(|c| !c.is_empty())
            .unwrap_or("unclassified");
        *causes.entry(cause).or_default() += 1;
        for family in &f.families {
            *family_counts.entry(family.as_str()).or_default() += 1;
        }
    }
    // Most-affected first, ties alphabetical — stable across renders.
    let mut families: Vec<&str> = family_counts.keys().copied().collect();
    families.sort_by_key(|f| (std::cmp::Reverse(family_counts[f]), *f));
    moda_scene::value::json!({
        "hybrid_vector_routed": count("hybrid→vector:"),
        "hybrid_vector_declined": count("hybrid→degraded:"),
        // DISTINCT NODES, to match `vector_lane_nodes_served`, which is a node-id
        // set. The events themselves dedupe on the whole payload (node + reason +
        // cause + families), so a node that declines twice for different reasons
        // contributes two records but only one served-slot — counting records here
        // would put a numerator and a denominator with different units into one
        // ratio, and name the result `..._nodes` (guidelines 6.2b: name the counter
        // after what it counts).
        // NAME CAVEAT: this counts nodes the vector lane DECLINED, not nodes that
        // provably drew through MSDF. MSDF is the usual destination, but a declined
        // node can also land in the degraded (cosmic-text) lane — `complex-script`
        // does — so read it as "lost the primary engine", which is the question
        // this counter asks. The router publishes it under the accurate
        // `text_vector_declined_*` names. The raw key keeps its spelling on purpose:
        // it is a WIRE field shared by two separately-built artifacts (the
        // `moda_render` wheel and the `moda-render` binary), and renaming it — or
        // dual-writing a second spelling — is precisely the producer/consumer skew
        // this file's `taxonomy_schema_status` detector exists to catch.
        "vector_lane_msdf_fallbacks": records
            .iter()
            .filter(|f| f.cause.starts_with(VECTOR_TO_MSDF_PREFIX))
            .map(|f| f.node.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        "degraded_tier": records.iter().filter(|f| !f.cause.contains('→')).count(),
        "vector_lane_nodes_served": vector_nodes_served,
        "vector_lane_msdf_fallback_causes": causes,
        "vector_lane_msdf_fallback_families": families,
    })
}

/// Record one text node served by the page-global vector lane
/// ([`VECTOR_LANE_LABEL`]) — the denominator of `vector_lane_msdf_fallbacks`.
///
/// [`RuntimeEvent::FontFallbackClass`] is the NUMERATOR of the
/// vector→MSDF degrade, but the report carried no denominator, so "what share of
/// native-lane text is actually drawn by the primary engine?" was unanswerable —
/// which is how a whole-document silent downgrade went unnoticed. Recorded as a
/// node id rather than an event: a served node is not a diagnostic, and one report
/// row per node would add hundreds per page. It is deduped like the events (see
/// `Collector::vector_served`).
///
/// COUPLING worth knowing: a text batch served from the residency cache
/// (`text_draw::build_richtext_item` returns before `try_build_vector_batch`)
/// re-plans neither arm, so neither this nor a fallback event fires for it on its
/// own. The one collecting lane on cached plans, an export job's caches
/// (`moda_engine::export_job`, moda-cli `batch` and `slices`), captures what each
/// batch's build recorded ([`capture_begin`]) and [`replay`]s it on every hit, so
/// its report is the uncached one. The editor's cache-driven lanes never `reset()`
/// and so never collect.
///
/// Any other collecting lane pointed at the retained path (moda-cli's `--retained`
/// benchmarking flag) under-reports, and the SHARE skews — do NOT assume it cancels
/// out. Cached and dirty nodes need not have the same engine mix: a page whose
/// vector-served nodes are cached while its declining nodes are dirty reads as a far
/// worse share than it is, and the reverse reads as an all-clear.
pub fn record_vector_text_served(node_id: &str) {
    capture_push(|| Captured::VectorServed);
    // Only while a report lane is actually collecting. `reset()` arms this, and the
    // only callers of `reset()` are the lanes that later BUILD a report (moda-py per
    // render, moda-cli per invocation). The wasm editor and the desktop shell never
    // reset, never drain, and never read the denominator — so without this gate the
    // set would grow by one `String` for EVERY text node ever planned, for the life
    // of the tab, in the never-shrinking wasm arena, and take a global mutex on the
    // per-node plan path of the PRIMARY renderer to do it.
    if !COLLECTING.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let mut c = collector().lock().expect("resource report lock");
    if !c.vector_served.contains(node_id) {
        c.vector_served.insert(node_id.to_string());
    }
}

/// Whether a report lane is collecting (armed by [`reset`]). See
/// [`record_vector_text_served`].
static COLLECTING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a host drains fallback events without building a report (armed by
/// [`arm_events`]). Gates [`record`] only — never the vector-served set.
static EVENTS_ONLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Dedup-memory bound for the events-only lane (see [`record`]).
const EVENTS_ONLY_SEEN_CAP: usize = 4096;

/// Arm event recording for a host that drains [`drain`] periodically to count
/// fallbacks (the live wasm editor's fallback monitor). Unlike [`reset`] it does
/// not arm the vector-served denominator, whose set grows by one entry per text
/// node planned; recording an event costs nothing on a document that has no
/// fallbacks. It clears the dedup memory (so a fallback recorded before surfaces
/// again when its node is re-planned) unless a report lane is collecting, whose
/// dedup it must not disturb. Pending events are kept.
pub fn arm_events() {
    EVENTS_ONLY.store(true, std::sync::atomic::Ordering::Relaxed);
    if !COLLECTING.load(std::sync::atomic::Ordering::Relaxed) {
        collector().lock().expect("resource report lock").seen.clear();
    }
}

/// Return to the never-armed state, so a test can assert what an UNARMED lane
/// (the wasm editor, mobile, the desktop shell) does. There is deliberately no
/// production disarm: a lane that armed the collector owns it until the process
/// ends, and a stray disarm would silently empty a report.
#[cfg(test)]
fn disarm_for_test() {
    COLLECTING.store(false, std::sync::atomic::Ordering::Relaxed);
}

/// Whether a specific node was recorded as vector-served since the last [`reset`].
///
/// For tests and ad-hoc diagnostics: asserting MEMBERSHIP rather than the count
/// is what lets a test about one node survive a concurrently-running sibling that
/// plans a different one, on a collector that is process-global by design.
pub fn vector_text_served_contains(node_id: &str) -> bool {
    collector()
        .lock()
        .expect("resource report lock")
        .vector_served
        .contains(node_id)
}

/// Distinct text nodes the vector lane served since the last [`reset`].
///
/// Survives [`drain`] for the same reason `seen` does: a caller that drains
/// per page and accumulates into ONE report (`moda-cli batch`) must keep reading a
/// denominator over the same window as the numerator it accumulated.
pub fn vector_text_served() -> u64 {
    collector()
        .lock()
        .expect("resource report lock")
        .vector_served
        .len() as u64
}

/// One observation recorded while a capture was open ([`capture_begin`]): what a cached result's build reported,
/// to [`replay`] when the result is served again.
#[derive(Debug, Clone, PartialEq)]
pub enum Captured {
    Event(RuntimeEvent),
    /// [`record_vector_text_served`] for the captured layer.
    VectorServed,
}

thread_local! {
    /// The open captures, innermost last (each sees what is recorded while it is open).
    static CAPTURES: std::cell::RefCell<Vec<Vec<Captured>>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn capture_push(item: impl FnOnce() -> Captured) {
    CAPTURES.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_empty() {
            return;
        }
        let item = item();
        for open in c.iter_mut() {
            open.push(item.clone());
        }
    });
}

/// Start capturing what this thread records (whether or not a report is collecting), for a cache that keeps a
/// built result and must report it again when it serves it ([`replay`]). Pair with [`capture_end`].
pub fn capture_begin() {
    CAPTURES.with(|c| c.borrow_mut().push(Vec::new()));
}

/// Close the innermost capture and return what it saw, in record order.
pub fn capture_end() -> Vec<Captured> {
    CAPTURES.with(|c| c.borrow_mut().pop().unwrap_or_default())
}

/// Record a captured build's observations again, as the layer `node_id` (the layer served the result now; the
/// build may have been another layer's with the same content). Events that name no layer are recorded as they
/// were. The same events, in the same order, that rebuilding the result for `node_id` records.
pub fn replay(captured: &[Captured], node_id: &str) {
    for c in captured {
        match c {
            Captured::VectorServed => record_vector_text_served(node_id),
            Captured::Event(e) => record(e.clone().for_node(node_id)),
        }
    }
}

impl RuntimeEvent {
    /// This event, naming `id` as its layer (events that name none are unchanged).
    fn for_node(mut self, id: &str) -> RuntimeEvent {
        match &mut self {
            RuntimeEvent::FontFallbackClass { node_id, .. }
            | RuntimeEvent::FontSubstituted { node_id, .. }
            | RuntimeEvent::FontMissing { node_id, .. }
            | RuntimeEvent::RenderQuality { node_id, .. } => *node_id = id.to_string(),
            RuntimeEvent::PlaceholderBaked { .. }
            | RuntimeEvent::BlankRasterDecoded { .. }
            | RuntimeEvent::EmojiMissing { .. }
            | RuntimeEvent::EmojiCoverageGap { .. } => {}
        }
        self
    }
}

/// Drain all recorded events (dedup memory is kept, so a re-recorded event
/// stays one event per process — use [`reset`] between independent renders).
pub fn drain() -> Vec<RuntimeEvent> {
    let mut c = collector().lock().expect("resource report lock");
    std::mem::take(&mut c.events)
}

/// Clear events AND dedup memory (start of an independent render). Also ARMS
/// vector-served recording — see [`record_vector_text_served`] for why the lanes
/// that never build a report must not pay for it.
pub fn reset() {
    COLLECTING.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut c = collector().lock().expect("resource report lock");
    c.events.clear();
    c.seen.clear();
    c.vector_served.clear();
    c.first_font_missing = None;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The collector is process-global, so any test that touches it must hold
    /// this for its whole body. A single test fn used to be the serialization
    /// strategy here; that silently stopped working the moment a SECOND
    /// globals-touching test was added (measured 14/30 runs failing at
    /// `--test-threads=32`), and unlike the `moda-engine` binary this crate is
    /// in CI's `--workspace` step at full parallelism. A lock scales; "there is
    /// only one test" does not.
    ///
    /// Poisoning is ignored deliberately: one failing test must report ITS own
    /// assertion, not cascade into unrelated lock-poisoning panics.
    pub(crate) fn serial_globals() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The animated lanes' per-frame poll: the first `FontMissing` is
    /// remembered outside `events`, so it survives a `drain` (a per-page
    /// drainer cannot hide it), is inert until a report lane arms the
    /// collector, and is cleared by `reset` like everything else.
    #[test]
    fn first_font_missing_survives_drain_and_clears_on_reset() {
        let _guard = serial_globals();
        reset();
        assert_eq!(first_font_missing(), None);
        record(RuntimeEvent::FontMissing {
            node_id: "richtext-1789721452468-145272323".into(),
            detail: "richtext-1789721452468-145272323: msdf family 'Good Sans' not vendored".into(),
        });
        record(RuntimeEvent::FontMissing {
            node_id: "n2".into(),
            detail: "n2: msdf family 'Outfit' not vendored".into(),
        });
        let first = first_font_missing().expect("memoized at record time");
        assert_eq!(first.0, "richtext-1789721452468-145272323");
        assert!(first.1.contains("'Good Sans' not vendored"));
        assert_eq!(drain().len(), 2);
        assert_eq!(
            first_font_missing().map(|m| m.0),
            Some("richtext-1789721452468-145272323".to_string()),
            "the memo must outlive drain"
        );
        reset();
        assert_eq!(first_font_missing(), None);
        // Disarmed (no reset since): an unarmed lane records nothing, memo included.
        COLLECTING.store(false, std::sync::atomic::Ordering::Relaxed);
        record(RuntimeEvent::FontMissing {
            node_id: "n3".into(),
            detail: "n3: msdf family 'X' not vendored".into(),
        });
        assert_eq!(first_font_missing(), None);
    }

    /// The live editor's events-only lane: events record and drain, the
    /// vector-served denominator stays off, and the dedup memory is bounded.
    #[test]
    fn events_only_lane_records_events_but_not_the_denominator() {
        let _guard = serial_globals();
        disarm_for_test();
        EVENTS_ONLY.store(false, std::sync::atomic::Ordering::Relaxed);
        {
            let mut c = collector().lock().unwrap();
            c.events.clear();
            c.seen.clear();
            c.vector_served.clear();
        }
        arm_events();
        record_vector_text_served("served-node");
        assert!(!vector_text_served_contains("served-node"));
        let fallback = |n: usize| RuntimeEvent::FontFallbackClass {
            node_id: format!("n{n}"),
            reason: "vector→msdf: x".into(),
            cause: "vector→msdf: font-unavailable:not-vendored".into(),
            families: vec!["Noto Sans".into()],
        };
        record(fallback(0));
        record(fallback(0));
        assert_eq!(drain().len(), 1, "deduped while under the cap");
        for n in 0..(EVENTS_ONLY_SEEN_CAP + 10) {
            record(fallback(n + 1));
        }
        assert!(collector().lock().unwrap().seen.len() <= EVENTS_ONLY_SEEN_CAP);
        assert_eq!(drain().len(), EVENTS_ONLY_SEEN_CAP + 10);
        // Re-arming forgets the dedup memory: a re-planned fallback surfaces again.
        let last = EVENTS_ONLY_SEEN_CAP + 10;
        record(fallback(last));
        assert_eq!(drain().len(), 0);
        arm_events();
        record(fallback(last));
        assert_eq!(drain().len(), 1);
        EVENTS_ONLY.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn sanitized_asset_ref_strips_credentials_and_bounds() {
        // Presigned-GCS / capability-token credentials live in the query
        // string — stripped, with a marker that something was removed.
        assert_eq!(
            sanitized_asset_ref(
                "https://storage.googleapis.com/b/icon.svg?X-Goog-Signature=abc123&X-Goog-Expires=900"
            ),
            "https://storage.googleapis.com/b/icon.svg?…"
        );
        // Fragments are stripped too; a plain URL passes through unchanged.
        assert_eq!(
            sanitized_asset_ref("https://x/a.svg#frag"),
            "https://x/a.svg?…"
        );
        assert_eq!(sanitized_asset_ref("https://x/a.svg"), "https://x/a.svg");
        // Authority userinfo (user:token@host) is credentials too — dropped
        // even though it precedes the path; a path-side '@' is untouched.
        assert_eq!(
            sanitized_asset_ref("https://user:tok3n@host/a.svg?sig=x"),
            "https://…@host/a.svg?…"
        );
        assert_eq!(
            sanitized_asset_ref("https://host/v1/@scope/a.svg"),
            "https://host/v1/@scope/a.svg"
        );
        // Userinfo is delimited by the LAST '@' in the authority: a raw '@'
        // inside the credential must not leak its tail segment.
        assert_eq!(
            sanitized_asset_ref("https://user:token@segment@host/a.svg"),
            "https://…@host/a.svg"
        );
        // Scheme-relative sources get the same userinfo treatment.
        assert_eq!(
            sanitized_asset_ref("//user:tok@host/a.svg"),
            "//…@host/a.svg"
        );
        // data: URIs collapse to a bounded scheme+size stub, regardless of size.
        let big = format!("data:image/svg+xml;utf8,{}", "x".repeat(100_000));
        let s = sanitized_asset_ref(&big);
        assert_eq!(s, format!("data:image/svg+xml ({} bytes)", big.len()));
        assert!(s.len() < 64);
        // Long credential-free paths are length-capped.
        let long = format!("https://x/{}", "p".repeat(500));
        assert!(sanitized_asset_ref(&long).chars().count() <= 201);
    }

    #[test]
    fn record_dedupes_and_drains() {
        let _guard = serial_globals();
        reset();
        let e = RuntimeEvent::PlaceholderBaked {
            src: "https://x/a.png".into(),
            state: "failed".into(),
            shader_image: false,
        };
        record(e.clone());
        record(e.clone()); // dedup
        record(RuntimeEvent::EmojiMissing {
            cluster: "🦀".into(),
        });
        let drained = drain();
        assert_eq!(drained.len(), 2);
        assert!(drained[0].is_missing_resource());
        assert!(matches!(
            &drained[1],
            RuntimeEvent::EmojiMissing { cluster } if cluster == "🦀"
        ));
        // dedup memory survives drain…
        record(e.clone());
        assert!(drain().is_empty());
        // …and reset clears it.
        reset();
        record(RuntimeEvent::FontFallbackClass {
            node_id: "n1".into(),
            reason: "no-msdf-runs".into(),
            cause: "no-msdf-runs".into(),
            families: vec!["Comic Sans MS".into()],
        });
        let drained = drain();
        assert_eq!(drained.len(), 1);
        assert!(!drained[0].is_missing_resource());
        reset();
    }

    /// The blank verdict rides the existing deduped channel, keyed on
    /// `src` ALONE — a source re-decoded at a second min-raster size is one blank
    /// asset, not two — and reports as a missing resource (the node draws
    /// nothing) even though it deliberately never reaches a `failure_count`.
    #[test]
    fn blank_raster_dedupes_on_src_and_is_a_missing_resource() {
        let _guard = serial_globals();
        reset();
        let blank = |cause: &str, permille: u16| RuntimeEvent::BlankRasterDecoded {
            src: "https://x/logo.svg".into(),
            coverage_permille: permille,
            cause: cause.into(),
            shader_image: false,
        };
        record(blank("svg_text_no_fontdb", 0));
        // Same src, different payload — still ONE row (unlike the other events,
        // whose dedup key spans the whole payload).
        record(blank("fully_transparent", 0));
        record(RuntimeEvent::BlankRasterDecoded {
            src: "https://x/other.svg".into(),
            coverage_permille: 0,
            cause: "fully_transparent".into(),
            shader_image: false,
        });
        let drained = drain();
        assert_eq!(drained.len(), 2);
        assert!(drained[0].is_missing_resource());
        assert!(matches!(
            &drained[0],
            RuntimeEvent::BlankRasterDecoded { src, cause, coverage_permille, shader_image }
                if src == "https://x/logo.svg"
                    && cause == "svg_text_no_fontdb"
                    && *coverage_permille == 0
                    && !*shader_image
        ));
        reset();
    }

    /// A capture sees what is recorded while it is open, collecting or not, and a replay records it again as the
    /// layer that serves the cached result: the report a rebuild for that layer would have made.
    #[test]
    fn a_capture_replays_as_the_serving_layer() {
        let _guard = serial_globals();
        reset();
        capture_begin();
        record(RuntimeEvent::FontSubstituted { node_id: "a".into(), requested: "Nope".into(), drawn: "Inter".into() });
        record_vector_text_served("a");
        record(RuntimeEvent::EmojiMissing { cluster: "x".into() });
        let captured = capture_end();
        assert_eq!(captured.len(), 3);
        assert!(capture_end().is_empty(), "no capture left open");
        let _ = drain();
        replay(&captured, "b");
        let replayed = drain();
        assert_eq!(
            replayed,
            vec![
                RuntimeEvent::FontSubstituted { node_id: "b".into(), requested: "Nope".into(), drawn: "Inter".into() },
                // Already recorded once since the reset: deduped, as a rebuild's would be.
            ]
        );
        assert!(vector_text_served_contains("a") && vector_text_served_contains("b"));
        assert_eq!(vector_text_served(), 2);
        reset();
    }

    /// The denominator is per-render and must be armed by a report lane.
    /// Deleting the `reset()` clear used to be invisible — and on a warm pooled
    /// worker that makes the share drift monotonically toward "healthy" the longer
    /// the process lives, which is the exact direction of bias that hid this bug.
    #[test]
    fn vector_served_is_armed_by_reset_survives_drain_and_clears_on_reset() {
        let _guard = serial_globals();
        reset();
        // Unarmed lanes (the wasm editor never resets) record nothing at all.
        assert_eq!(vector_text_served(), 0);
        record_vector_text_served("n1");
        record_vector_text_served("n1"); // deduped by node id, like the events
        record_vector_text_served("n2");
        assert_eq!(vector_text_served(), 2);
        assert!(vector_text_served_contains("n1"));
        // Survives a drain: `moda-cli batch` drains per page into ONE cumulative
        // report, so the denominator must span the same window as the numerator.
        let _ = drain();
        assert_eq!(vector_text_served(), 2);
        // …and is per-render: the next render starts from zero.
        reset();
        assert_eq!(vector_text_served(), 0);
    }

    /// The numerator must be DISTINCT NODES, like the denominator.
    ///
    /// The events dedupe on the whole payload (node + reason + cause + families),
    /// so one node declining for two different reasons banks two records but only
    /// one served-slot. Counting records would put a numerator and a denominator
    /// with different units into a single ratio — and call the result
    /// `vector_lane_msdf_fallbacks`, i.e. nodes.
    ///
    /// Touches no process global (the denominator is a parameter), so it needs no
    /// serialization guard.
    #[test]
    fn taxonomy_counts_distinct_nodes_not_records() {
        let rec = |node: &str, cause: &str| FontFallbackRecord {
            node: node.into(),
            reason: format!("vector\u{2192}msdf: {cause}"),
            cause: format!("vector\u{2192}msdf: {cause}"),
            families: vec!["Inter".into()],
        };
        let taxonomy = font_fallback_taxonomy(
            &[
                rec("n1", "font-unavailable:not-vendored"),
                rec("n1", "coverage"), // SAME node, second reason
                rec("n2", "coverage"),
            ],
            7,
        );
        // Two nodes declined, not three records.
        assert_eq!(taxonomy["vector_lane_msdf_fallbacks"], 2);
        assert_eq!(taxonomy["vector_lane_nodes_served"], 7);
        // The cause histogram stays RECORD-keyed on purpose: a node that declines
        // for two causes genuinely belongs in both buckets, and the buckets are a
        // breakdown of reasons, not a partition of nodes.
        assert_eq!(taxonomy["vector_lane_msdf_fallback_causes"]["coverage"], 2);
        assert_eq!(
            taxonomy["vector_lane_msdf_fallback_causes"]["font-unavailable:not-vendored"],
            1
        );
    }

    /// An UNARMED lane records nothing at all.
    ///
    /// Only the lanes that later build a report call `reset()`; the wasm editor,
    /// mobile and the desktop shell never reset and never drain, so an ungated
    /// `record` grows `events` + `seen` for the life of the tab in the
    /// never-shrinking wasm arena — and this ticket enlarged that key space three
    /// ways (a new `cause` component, populated `families`, and a per-url digest
    /// in the `not-vendored` reason that turns every re-signed team-font url into
    /// a fresh key).
    #[test]
    fn record_is_inert_until_a_report_lane_arms_it() {
        let _guard = serial_globals();
        reset();
        disarm_for_test();
        record(RuntimeEvent::EmojiMissing {
            cluster: "\u{1F980}".into(),
        });
        record_vector_text_served("n1");
        assert!(drain().is_empty(), "an unarmed lane must record nothing");
        assert_eq!(vector_text_served(), 0);
        // …and arming it makes both halves live again.
        reset();
        record(RuntimeEvent::EmojiMissing {
            cluster: "\u{1F980}".into(),
        });
        record_vector_text_served("n1");
        assert_eq!(drain().len(), 1);
        assert_eq!(vector_text_served(), 1);
        reset();
    }

    /// The taxonomy must split the lanes, report the per-cause
    /// breakdown of the vector→MSDF bucket, and carry the DENOMINATOR — the
    /// vector-served node count that makes the share computable.
    ///
    /// Deliberately touches NO process global (the denominator is a parameter),
    /// so it can run alongside `record_dedupes_and_drains` — which owns the
    /// collector and calls `reset()` — without either perturbing the other.
    #[test]
    fn taxonomy_splits_lanes_and_carries_the_denominator() {
        let rec = |reason: &str, cause: &str, family: &str| FontFallbackRecord {
            node: format!("n-{reason}-{family}"),
            reason: reason.into(),
            cause: cause.into(),
            families: if family.is_empty() {
                Vec::new()
            } else {
                vec![family.into()]
            },
        };
        let records = vec![
            rec(
                "vector→msdf: font-unavailable:Open Sans (700 normal; not-vendored h/x.ttf)",
                "vector→msdf: font-unavailable:not-vendored",
                "Open Sans",
            ),
            rec(
                "vector→msdf: font-unavailable:Roboto (400 normal; not-vendored h/y.ttf)",
                "vector→msdf: font-unavailable:not-vendored",
                "Roboto",
            ),
            // A coverage decline must carry its family too. The oracle-contract
            // `family` field is `None` here, so reporting through that alone is
            // what left these records with `families: []`.
            rec(
                "vector→msdf: coverage (Inter 400 normal)",
                "vector→msdf: coverage",
                "Inter",
            ),
            // A second `not-vendored` on a family already seen — families dedupe,
            // causes accumulate.
            rec(
                "vector→msdf: font-unavailable:Open Sans (400 normal; not-vendored h/z.ttf)",
                "vector→msdf: font-unavailable:not-vendored",
                "Open Sans",
            ),
            rec(
                "hybrid→vector: no-msdf-runs",
                "hybrid→vector: no-msdf-runs",
                "",
            ),
            rec(
                "hybrid→degraded: emoji-run",
                "hybrid→degraded: emoji-run",
                "",
            ),
            rec("missing-primary-atlas", "missing-primary-atlas", ""),
        ];
        let t = font_fallback_taxonomy(&records, 7);
        // CROSS-LANGUAGE CONTRACT. These key names are matched literally by the
        // Python worker (`render_worker.text_engine_summary`) across the moda_render
        // wheel boundary, so a rename here silently drops the vector-vs-MSDF share
        // from every decision record — the same "invisible by construction" failure
        // this taxonomy exists to remove. Rename only with the Python side in the same
        // commit.
        //
        // SCOPE, precisely: this pins the PRODUCER's schema — the keys this
        // function emits. It does NOT pin moda-py's use of it: the outer
        // `"font_fallback_taxonomy"` key and the `vector_nodes_served` argument are
        // literals in `moda_py::ResourceReport::to_json`, and moda-py sets
        // `[lib] test = false` while its Python test file is not wired into any
        // workflow, so nothing in CI covers that seam. What covers it at RUNTIME is
        // `render_worker._warn_once_on_taxonomy_skew`, which fires on a report that
        // carries `font_fallback_class_nodes` without a parseable taxonomy — i.e. a
        // moda-py rename is loud in the logs even though it is green in CI.
        for key in [
            "hybrid_vector_routed",
            "hybrid_vector_declined",
            "vector_lane_msdf_fallbacks",
            "degraded_tier",
            "vector_lane_nodes_served",
            "vector_lane_msdf_fallback_causes",
            "vector_lane_msdf_fallback_families",
        ] {
            assert!(!t[key].is_null(), "taxonomy key {key} missing from {t}");
        }
        assert_eq!(t["vector_lane_msdf_fallbacks"], 4);
        assert_eq!(t["hybrid_vector_routed"], 1);
        assert_eq!(t["hybrid_vector_declined"], 1);
        assert_eq!(t["degraded_tier"], 1);
        assert_eq!(t["vector_lane_nodes_served"], 7);
        assert_eq!(
            t["vector_lane_msdf_fallback_causes"]["font-unavailable:not-vendored"],
            3
        );
        // A `coverage` decline carries its family too — reporting it only through
        // the oracle-contract `family` field left these as `families: []`.
        assert_eq!(t["vector_lane_msdf_fallback_causes"]["coverage"], 1);
        // A record whose `cause` lost the lane prefix is BUCKETED, never echoed —
        // free text as a histogram key is unbounded cardinality.
        let odd = [
            FontFallbackRecord {
                node: "n-odd".into(),
                reason: "vector→msdf: something".into(),
                cause: "a free-text cause with a family name in it".into(),
                families: Vec::new(),
            },
            // A prefix with nothing after it is as unusable a label as none at all.
            FontFallbackRecord {
                node: "n-empty".into(),
                reason: "vector→msdf: something".into(),
                cause: "vector→msdf: ".into(),
                families: Vec::new(),
            },
        ];
        let odd_t = font_fallback_taxonomy(&odd, 0);
        // Only the record whose CAUSE carries the lane prefix is in the vector
        // bucket — the buckets key on `cause` (low-cardinality by construction),
        // not on the free-text `reason` that interpolates family names and error
        // prose. Its empty remainder is as unusable a label as a missing one.
        assert_eq!(
            odd_t["vector_lane_msdf_fallback_causes"],
            moda_scene::value::json!({ "unclassified": 1 })
        );
        assert_eq!(odd_t["vector_lane_msdf_fallbacks"], 1);
        // The prefix-less one falls to the degraded tier rather than vanishing.
        assert_eq!(odd_t["degraded_tier"], 1);
        // Families come only from the vector→MSDF bucket, deduped and sorted.
        assert_eq!(
            t["vector_lane_msdf_fallback_families"],
            moda_scene::value::json!(["Open Sans", "Inter", "Roboto"])
        );
        // A render that served nothing reports a MEASURED zero, not an absent field.
        assert_eq!(
            font_fallback_taxonomy(&[], 0)["vector_lane_nodes_served"],
            0
        );
    }
}

//! opentype.js-faithful font model for the vector text engine.
//!
//! The browser's vector engine shapes and kerns through **opentype.js 1.3.4**
//! (`VectorFontCache` → `opentype.parse`; `VectorGlyphShaper.shape` →
//! `Font.stringToGlyphs`; `placeGlyphs` → `Font.getKerningValue`). That
//! library has load-bearing quirks that real font crates (read-fonts/skrifa,
//! rustybuzz) deliberately do NOT reproduce, so — like `ttf.rs` for the
//! baseline metrics — this module hand-ports the exact opentype.js behavior,
//! citing its source per function. The quirks copied on purpose:
//!
//!  - `Font.getKerningValue` (font.js:252-261): if a GPOS table exists AT ALL,
//!    only the default-script 'kern' feature's **direct lookup-type-2**
//!    lookups are consulted (extension/type-9 lookups are invisible →
//!    Montserrat and Inter kern as 0); the legacy `kern` table is ONLY read
//!    when there is no GPOS table — an empty GPOS kern lookup list is truthy
//!    and still shadows it.
//!  - GPOS PairPos format 2 (position.js:52-57): a coverage hit on the left
//!    glyph **returns immediately** with `classRecords[c1][c2].value1.xAdvance
//!    || 0` — later subtables/lookups are never consulted, even for 0.
//!  - `Font.stringToGlyphs` (font.js:178-202) runs the bidi engine: one token
//!    per code point, **latn 'liga'** GSUB feature applied over "latin word"
//!    ranges (`/[A-z]/` — including `[\]^_`` between 'Z' and 'a', char.js:33),
//!    via `FeatureQuery.lookupFeature`. `getLookupMethod`
//!    (featureQuery.js:296-318) **throws** for any lookup type+format outside
//!    {1/1, 1/2, 2/1, 4/1, 6/3} — including type-7 extension wrappers (gsub.js
//!    parses them as `substFormat: 1` with the inner table never consulted,
//!    gsub.js:163-174), which is why Lato (liga → type 7) throws on every
//!    shape and the browser falls back to MSDF via `renderWithVector`'s catch.
//!  - GSUB feature resolution (featureQuery.js:207-244): 'latn' missing (or
//!    present without a defaultLangSys) falls back to the DFLT script's
//!    default-lang-sys feature indexes; the **last** 'liga' feature wins
//!    (`tags[tag] = feature` overwrites, featureQuery.js:251-259). GPOS kern
//!    feature resolution (layout.js:206-233) takes the **first** 'kern'.
//!  - Single substitution format 1 (featureQuery.js:66-69): `glyphIndex +
//!    deltaGlyphId` with the delta parsed **unsigned** and **no & 0xFFFF**.
//!  - Multiple substitution ('21') actions are matched and recorded but never
//!    applied (`SUBSTITUTIONS` has no 21 entry, applySubstitution.js:55-60) —
//!    they only block later subtables from rematching that index.
//!  - cmap (tables/cmap.js:84-116): sub-table records scanned **backwards**,
//!    first match of platform 3 (enc 0/1/10) or platform 0 (enc 0-4) wins;
//!    only formats 4 and 12 parse (anything else fails the font);
//!    `charToGlyphIndex` = `glyphIndexMap[cp] || 0` (encoding.js:165-167).
//!  - kern table (tables/kern.js): version 0 → first Windows subtable only;
//!    version 1 → first Mac subtable; anything else fails the font.
//!
//! MIRRORED IN PYTHON: `backend/app/services/pptx/font_metrics.py` reproduces
//! these quirks — the default-script `kern` walk, direct-type-2-only lookups,
//! the PairPos format-2 early return, last-`liga`-wins, and
//! `has_unsupported_required_feature`'s reroute — so the PDF importer can
//! predict this lane's widths. Change one of them here and that module is
//! silently wrong until someone renders a page.

use std::collections::HashMap;

use rustc_hash::FxHashMap;

use crate::ttf::{choose_baseline_metrics, parse_ttf_for_metrics, TtfMetrics};

// ───────────────────────────── errors ─────────────────────────────

/// A throw inside the shaping/kerning path. In the browser these unwind to
/// `WebGPUTextSystem.renderWithVector`'s catch (WebGPUTextSystem.ts:758-761)
/// → the node falls back to MSDF. `message()` must match the JS error text
/// byte-for-byte where the oracle records it.
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeError {
    /// `getLookupMethod`'s throw (featureQuery.js:296-318).
    UnsupportedLookup { lookup_type: u16, subst_format: u16 },
    /// JS TypeError-class failures (malformed tables) and not-ported scripts.
    Other(String),
}

impl ShapeError {
    pub fn message(&self) -> String {
        match self {
            ShapeError::UnsupportedLookup {
                lookup_type,
                subst_format,
            } => format!(
                "lookupType: {lookup_type} - substFormat: {subst_format} is not yet supported"
            ),
            ShapeError::Other(m) => m.clone(),
        }
    }
}

// ───────────────────────── byte readers ───────────────────────────

fn u16be(d: &[u8], o: usize) -> Result<u16, String> {
    match (d.get(o), d.get(o + 1)) {
        (Some(a), Some(b)) => Ok(u16::from_be_bytes([*a, *b])),
        _ => Err(format!("u16 read out of bounds at {o}")),
    }
}
fn i16be(d: &[u8], o: usize) -> Result<i16, String> {
    Ok(u16be(d, o)? as i16)
}
fn u32be(d: &[u8], o: usize) -> Result<u32, String> {
    match (d.get(o), d.get(o + 3)) {
        (Some(_), Some(_)) => Ok(u32::from_be_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])),
        _ => Err(format!("u32 read out of bounds at {o}")),
    }
}

// ─────────────────────── layout structures ────────────────────────

/// Coverage table (parse.js `Parser.coverage`).
#[derive(Debug, Clone)]
pub enum Coverage {
    /// format 1: the glyph list. `ascending`: strictly ascending, as the spec
    /// requires, so a binary search finds what `indexOf` finds.
    F1 { glyphs: Vec<u16>, ascending: bool },
    /// format 2: (start, end, startCoverageIndex). `ascending`: each range
    /// starts after the one before it ends, so a binary search finds the
    /// range the linear scan finds.
    F2 {
        ranges: Vec<(u16, u16, u16)>,
        ascending: bool,
    },
}

impl Coverage {
    /// GSUB-side `lookupCoverage` (featureQuery.js:39-59): `!glyphIndex` (0)
    /// never matches; format 1 `indexOf`, format 2 linear range scan. The
    /// GPOS-side `getCoverageIndex` (layout.js:302-311) binary-searches but
    /// agrees on every well-formed (sorted) font; the `!glyphIndex` guard is
    /// harmless there because gid 0 never reaches kerning (see
    /// `get_kerning_value`). A well-formed table is binary-searched (the same
    /// answer as the scan); only an unsorted one is scanned.
    pub fn index(&self, gid: u32) -> Option<u32> {
        if gid == 0 {
            return None;
        }
        match self {
            Coverage::F1 { glyphs, ascending } => {
                if *ascending {
                    let gid = u16::try_from(gid).ok()?;
                    glyphs.binary_search(&gid).ok().map(|i| i as u32)
                } else {
                    glyphs
                        .iter()
                        .position(|g| *g as u32 == gid)
                        .map(|i| i as u32)
                }
            }
            Coverage::F2 { ranges, ascending } => {
                let (start, _, index) = find_range(ranges, *ascending, gid)?;
                Some(index as u32 + gid - start as u32)
            }
        }
    }
}

/// The first `(start, end, value)` range holding `gid`: a binary search when
/// the ranges are `ascending` (sorted and disjoint, where it finds the same
/// range), else the linear scan.
fn find_range(ranges: &[(u16, u16, u16)], ascending: bool, gid: u32) -> Option<(u16, u16, u16)> {
    if !ascending {
        return ranges
            .iter()
            .find(|(start, end, _)| gid >= *start as u32 && gid <= *end as u32)
            .copied();
    }
    let i = ranges.partition_point(|(_, end, _)| (*end as u32) < gid);
    ranges
        .get(i)
        .filter(|(start, _, _)| *start as u32 <= gid)
        .copied()
}

/// Ranges sorted and disjoint: each `start <= end`, and starts after the
/// previous range's end.
fn ranges_ascending(ranges: &[(u16, u16, u16)]) -> bool {
    ranges.iter().all(|(start, end, _)| start <= end) && ranges.windows(2).all(|w| w[0].1 < w[1].0)
}

/// Class definition table (parse.js `Parser.classDef`); `get` mirrors
/// `Layout.getGlyphClass` (layout.js:282-293): not found → class 0.
#[derive(Debug, Clone)]
pub enum ClassDef {
    F1 {
        start_glyph: u16,
        classes: Vec<u16>,
    },
    /// (start, end, class). `ascending`: sorted and disjoint (see `Coverage::F2`).
    F2 {
        ranges: Vec<(u16, u16, u16)>,
        ascending: bool,
    },
}

impl ClassDef {
    pub fn get(&self, gid: u32) -> u32 {
        match self {
            ClassDef::F1 {
                start_glyph,
                classes,
            } => {
                let start = *start_glyph as u32;
                if gid >= start && gid < start + classes.len() as u32 {
                    classes[(gid - start) as usize] as u32
                } else {
                    0
                }
            }
            ClassDef::F2 { ranges, ascending } => {
                find_range(ranges, *ascending, gid).map_or(0, |(_, _, class_id)| class_id as u32)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LangSys {
    pub feature_indexes: Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct ScriptRecord {
    pub tag: [u8; 4],
    pub default_lang_sys: Option<LangSys>,
    pub lang_sys_records: Vec<([u8; 4], LangSys)>,
}

#[derive(Debug, Clone)]
pub struct FeatureRecord {
    pub tag: [u8; 4],
    pub lookup_list_indexes: Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct Lookup<T> {
    pub lookup_type: u16,
    pub subtables: Vec<T>,
}

#[derive(Debug, Clone)]
pub struct LayoutTable<T> {
    pub scripts: Vec<ScriptRecord>,
    pub features: Vec<FeatureRecord>,
    pub lookups: Vec<Lookup<T>>,
}

/// One ligature (gsub.js:53-66): `components` excludes the first glyph
/// (count − 1 entries, matched against the lookahead).
#[derive(Debug, Clone)]
pub struct Ligature {
    pub lig_glyph: u16,
    pub components: Vec<u16>,
}

/// GSUB subtables, parsed exactly as far as opentype.js's `subtableParsers`
/// go for the types `lookupFeature` can execute; everything else keeps just
/// the substFormat so `getLookupMethod`'s throw can be reproduced.
#[derive(Debug, Clone)]
pub enum GsubSubtable {
    /// 1/1 — `glyphIndex + deltaGlyphId`, delta UNSIGNED and unmasked
    /// (featureQuery.js:66-69 + gsub.js:14-19).
    Single1 {
        coverage: Coverage,
        delta_glyph_id: u16,
    },
    /// 1/2
    Single2 {
        coverage: Coverage,
        substitutes: Vec<u16>,
    },
    /// 2/1 — matched but never applied (applySubstitution.js:55-60).
    Multiple {
        coverage: Coverage,
        sequences: Vec<Vec<u16>>,
    },
    /// 4/1
    Ligature {
        coverage: Coverage,
        ligature_sets: Vec<Vec<Ligature>>,
    },
    /// 6/3
    Chain3 {
        backtrack: Vec<Coverage>,
        input: Vec<Coverage>,
        lookahead: Vec<Coverage>,
        /// (sequenceIndex, lookupListIndex)
        lookup_records: Vec<(u16, u16)>,
    },
    /// Anything `getLookupMethod` throws for — including type-7 extension
    /// wrappers, which gsub.js:163-174 parses as `substFormat: 1` (the inner
    /// table is never consulted by featureQuery).
    Other { subst_format: u16 },
}

#[derive(Debug, Clone)]
pub enum GposSubtable {
    /// PairPos format 1 (gpos.js:31-53): per-pair `value1.xAdvance` (None ⇔
    /// no value1 / no X_ADVANCE bit — both read back as 0).
    Pair1 {
        coverage: Coverage,
        pair_sets: Vec<Vec<(u16, Option<i16>)>>,
        /// Every pair set's second glyphs strictly ascending, as the spec
        /// requires: a binary search finds the pair the linear scan finds.
        pairs_ascending: bool,
    },
    /// PairPos format 2: `classRecords[c1][c2].value1.xAdvance`, flattened
    /// row-major.
    Pair2 {
        coverage: Coverage,
        class_def1: ClassDef,
        class_def2: ClassDef,
        class1_count: u16,
        class2_count: u16,
        records: Vec<Option<i16>>,
    },
    /// Types 1, 3-9 parse as `{error}` stubs in gpos.js and are never
    /// queried (only type-2 lookups are collected, position.js:71-75).
    Other,
}

// ───────────────────────────── the font ───────────────────────────

/// The opentype.js-visible surface of one parsed font.
pub struct OtjsFont {
    /// head.unitsPerEm; consumed as `font.unitsPerEm || 1000`
    /// (glyphPlacement.ts:27).
    pub units_per_em: u16,
    /// cmap `glyphIndexMap` (only the selected format-4/12 subtable).
    cmap: FxHashMap<u32, u32>,
    /// hmtx advance widths (one per hMetric; gid ≥ numberOfHMetrics reuses
    /// the last).
    advances: Vec<u16>,
    pub num_glyphs: u16,
    gsub: Option<LayoutTable<GsubSubtable>>,
    gpos: Option<LayoutTable<GposSubtable>>,
    /// `position.defaultKerningTables` (position.js:22-25): `None` ⇔ no GPOS
    /// table; `Some(vec)` (possibly EMPTY — still shadows the kern table,
    /// font.js:255-260) = indexes of direct type-2 lookups under the default
    /// script's first 'kern' feature.
    gpos_kern_lookups: Option<Vec<u16>>,
    /// `font.kerningPairs` from the legacy kern table (tables/kern.js).
    kerning_pairs: FxHashMap<(u32, u32), i16>,
    /// `chooseBaselineMetrics(parseTtfForMetrics(...))` — the same pair
    /// `VectorFontCache.loadFont` computes (VectorFontCache.ts:149-153).
    pub metrics: Option<TtfMetrics>,
    /// glyf/loca outline reader (Track C — the analytic fill's curve source,
    /// tables/glyf.js + tables/loca.js). `None` for CFF fonts: none in the
    /// catalog upstream; the desktop fork reads CFF outlines through `var`).
    glyf: Option<crate::vector::curves::GlyfOutlines>,
    /// `CURVE_CACHE` (glyphCurves.ts:200-201): one em-normalized entry per
    /// glyph. The browser keys `${fontKey}|${glyphIndex}` in one global map;
    /// a per-font map is the identical partition.
    curve_cache:
        std::cell::RefCell<FxHashMap<u32, std::rc::Rc<crate::vector::curves::GlyphQuadratics>>>,
    /// `CONTOUR_CACHE` (vectorTextExtrude.ts:41-42): em-normalized flattened
    /// contours for the extrude wall sweep, one entry per glyph — the same
    /// per-font partition of the browser's `${fontKey}|${glyphIndex}` map.
    contour_cache: std::cell::RefCell<FxHashMap<u32, std::rc::Rc<Vec<crate::vector::extrude::Ring>>>>,
    /// Desktop fork: the ttf-parser face that supplies outlines when the otjs
    /// glyf reader cannot (CFF/CFF2), and outlines + advances + metrics when
    /// the font is instanced away from its default coordinates
    /// (`vector/varfont.rs`). `None` for a static glyf font.
    var: Option<crate::vector::varfont::VarFace>,
    /// Vertical text: `vmtx` advances, `VORG` origins and the ascender / descender fallback (`crate::vertical`).
    vertical: Option<std::rc::Rc<crate::vertical::VerticalMetrics>>,
}

struct TableDir {
    entries: HashMap<[u8; 4], (usize, usize)>,
}

fn parse_table_directory(data: &[u8]) -> Result<TableDir, String> {
    let num_tables = u16be(data, 4)?;
    let mut entries = HashMap::new();
    let mut p = 12usize;
    for _ in 0..num_tables {
        let tag: [u8; 4] = data
            .get(p..p + 4)
            .ok_or("table dir truncated")?
            .try_into()
            .unwrap();
        let offset = u32be(data, p + 8)? as usize;
        let length = u32be(data, p + 12)? as usize;
        entries.insert(tag, (offset, length));
        p += 16;
    }
    Ok(TableDir { entries })
}

impl OtjsFont {
    /// `opentype.parse` for the tables the vector engine consumes. Any parse
    /// failure here corresponds to an `opentype.parse` throw in the browser —
    /// `VectorFontCache.loadFont` catches it, marks the URL failed, and the
    /// node falls back (font-unavailable class).
    pub fn parse(data: &[u8]) -> Result<OtjsFont, String> {
        let font = Self::parse_tables(data)?;
        if font.glyf.is_some() {
            return Ok(font);
        }
        // No glyf table: outlines come from ttf-parser, which needs owned bytes.
        font.with_var_face(std::sync::Arc::new(data.to_vec()), &[])
    }

    /// Desktop fork: parse `bytes` as the instance at `variations` (axis tag →
    /// user-space value). A static glyf font with no variations is exactly
    /// [`Self::parse`]; CFF/CFF2 faces and non-default instances read outlines
    /// (and, when instanced, advances and MVAR-adjusted metrics) through
    /// ttf-parser over the same bytes.
    pub fn parse_instance(
        bytes: &std::sync::Arc<Vec<u8>>,
        variations: &[crate::vector::varfont::Variation],
    ) -> Result<OtjsFont, String> {
        let font = Self::parse_tables(bytes)?;
        if font.glyf.is_some() && variations.is_empty() {
            return Ok(font);
        }
        font.with_var_face(bytes.clone(), variations)
    }

    fn with_var_face(
        mut self,
        bytes: std::sync::Arc<Vec<u8>>,
        variations: &[crate::vector::varfont::Variation],
    ) -> Result<OtjsFont, String> {
        let var = crate::vector::varfont::VarFace::new(bytes, variations)?;
        if var.instanced {
            self.metrics = self.metrics.as_ref().map(|m| var.vary_metrics(m));
        }
        self.var = Some(var);
        Ok(self)
    }

    /// True when outlines come from ttf-parser rather than the glyf port.
    fn outlines_from_var(&self) -> bool {
        self.var
            .as_ref()
            .is_some_and(|v| v.instanced || self.glyf.is_none())
    }

    /// Font-unit, y-up path commands for `gid` from whichever outline source
    /// describes this instance.
    fn glyph_commands(
        &self,
        gid: u32,
    ) -> Result<Vec<crate::vector::curves::PathCommand>, ShapeError> {
        if self.outlines_from_var() {
            return self
                .var
                .as_ref()
                .unwrap()
                .glyph_commands(gid)
                .map_err(crate::vector::curves::outline_error);
        }
        let Some(glyf) = &self.glyf else {
            return Err(ShapeError::Other("font has no outlines".to_string()));
        };
        glyf.glyph_commands(gid)
            .map_err(crate::vector::curves::outline_error)
    }

    fn parse_tables(data: &[u8]) -> Result<OtjsFont, String> {
        let dir = parse_table_directory(data)?;
        let get = |tag: &[u8; 4]| dir.entries.get(tag).copied();

        // head.unitsPerEm (tables/head.js; offset 18)
        let (head_off, _) = get(b"head").ok_or("missing head table")?;
        let units_per_em = u16be(data, head_off + 18)?;

        // maxp.numGlyphs
        let (maxp_off, _) = get(b"maxp").ok_or("missing maxp table")?;
        let num_glyphs = u16be(data, maxp_off + 4)?;

        // hhea.numberOfHMetrics (offset 34)
        let (hhea_off, _) = get(b"hhea").ok_or("missing hhea table")?;
        let num_h_metrics = u16be(data, hhea_off + 34)?;

        // hmtx advances (tables/hmtx.js)
        let (hmtx_off, _) = get(b"hmtx").ok_or("missing hmtx table")?;
        let mut advances = Vec::with_capacity(num_h_metrics as usize);
        for i in 0..num_h_metrics as usize {
            advances.push(u16be(data, hmtx_off + i * 4)?);
        }

        // cmap (tables/cmap.js:84-116)
        let (cmap_off, _) = get(b"cmap").ok_or("missing cmap table")?;
        let cmap = parse_cmap(data, cmap_off)?;

        // GSUB / GPOS
        let gsub = match get(b"GSUB") {
            Some((off, _)) => Some(parse_layout_table(data, off, parse_gsub_subtable)?),
            None => None,
        };
        let gpos = match get(b"GPOS") {
            Some((off, _)) => Some(parse_layout_table(data, off, parse_gpos_subtable)?),
            None => None,
        };

        // kern table (tables/kern.js) — parsed eagerly like opentype.js;
        // failures fail the font.
        let kerning_pairs = match get(b"kern") {
            Some((off, _)) => parse_kern(data, off)?,
            None => FxHashMap::default(),
        };

        // position.init() (position.js:22-25)
        let gpos_kern_lookups = gpos.as_ref().map(default_kerning_lookups);

        let metrics = choose_baseline_metrics(&parse_ttf_for_metrics(data));

        // glyf outlines (tables/loca.js + tables/glyf.js): loca's version
        // flag is head.indexToLocFormat (offset 50; 0 = short, ×2).
        let glyf = match (get(b"glyf"), get(b"loca")) {
            (Some((glyf_off, glyf_len)), Some((loca_off, _))) => {
                let short_version = i16be(data, head_off + 50)? == 0;
                Some(crate::vector::curves::GlyfOutlines::parse(
                    data,
                    glyf_off,
                    glyf_len,
                    loca_off,
                    num_glyphs,
                    short_version,
                )?)
            }
            _ => None,
        };

        Ok(OtjsFont {
            units_per_em,
            cmap,
            advances,
            num_glyphs,
            gsub,
            gpos,
            gpos_kern_lookups,
            kerning_pairs,
            metrics,
            glyf,
            curve_cache: std::cell::RefCell::new(FxHashMap::default()),
            contour_cache: std::cell::RefCell::new(FxHashMap::default()),
            var: None,
            vertical: crate::vertical::VerticalMetrics::parse(data).map(std::rc::Rc::new),
        })
    }

    /// `getGlyphQuadratics(fontKey, glyph)` (glyphCurves.ts:207-223): the
    /// glyph's quadratic segments in em-normalized, y-down, baseline-origin
    /// space — `quadraticsFromCommands(glyph.getPath(0, 0, 1).commands)`,
    /// cached per glyph (resolution-independent, exactly one entry).
    pub fn glyph_quadratics(
        &self,
        gid: u32,
    ) -> Result<std::rc::Rc<crate::vector::curves::GlyphQuadratics>, ShapeError> {
        if let Some(q) = self.curve_cache.borrow().get(&gid) {
            return Ok(q.clone());
        }
        let commands = self.glyph_commands(gid)?;
        let em = crate::vector::curves::em_normalize(&commands, self.units_per_em);
        let q = std::rc::Rc::new(crate::vector::curves::quadratics_from_commands(&em));
        self.curve_cache.borrow_mut().insert(gid, q.clone());
        Ok(q)
    }

    /// `getGlyphEmContours` (vectorTextExtrude.ts:49-65): flatten the glyph
    /// outline into closed contours in em-normalized, pen-relative space
    /// (y-down), cached per glyph. The browser reads `glyph.getPath(0, 0,
    /// 256)` (the same glyf path commands at 256 px); we read the same glyf
    /// commands, em-normalize, and let `em_contours_from_commands` apply the
    /// exact ×256 flatten / ÷256 round-trip.
    pub fn glyph_em_contours(
        &self,
        gid: u32,
    ) -> Result<std::rc::Rc<Vec<crate::vector::extrude::Ring>>, ShapeError> {
        if let Some(c) = self.contour_cache.borrow().get(&gid) {
            return Ok(c.clone());
        }
        let commands = self.glyph_commands(gid)?;
        let em = crate::vector::curves::em_normalize(&commands, self.units_per_em);
        let rings = std::rc::Rc::new(crate::vector::extrude::em_contours_from_commands(&em));
        self.contour_cache.borrow_mut().insert(gid, rings.clone());
        Ok(rings)
    }

    /// `Font.charToGlyphIndex` → `CmapEncoding.charToGlyphIndex`
    /// (encoding.js:165-167): `glyphIndexMap[codePointAt(0)] || 0`.
    /// The font's vertical metrics (`crate::vertical::VerticalMetrics`).
    pub fn vertical_metrics(&self) -> Option<&crate::vertical::VerticalMetrics> {
        self.vertical.as_deref()
    }

    pub fn char_to_glyph_index(&self, ch: char) -> u32 {
        self.cmap.get(&(ch as u32)).copied().unwrap_or(0)
    }

    /// True when GSUB carries a REQUIRED feature this otjs walk never applies:
    /// `rvrn` (Required Variation Alternates) or `rclt` (Required Contextual
    /// Alternates). The mirror of TS `fontHasUnsupportedRequiredFeature`
    /// (fontShapingCapability.ts): HarfBuzz and browsers always apply these, so
    /// a font carrying one renders different glyphs under the otjs path than in
    /// the browser/DOM (e.g. Fraunces italic `l` → `l.alt` via `rvrn`). Such
    /// fonts must reroute through rustybuzz to match.
    pub fn has_unsupported_required_feature(&self) -> bool {
        let Some(gsub) = &self.gsub else {
            return false;
        };
        gsub.features
            .iter()
            .any(|f| &f.tag == b"rvrn" || &f.tag == b"rclt")
    }

    /// `glyph.advanceWidth` (hmtx), consumed as `(advanceWidth || 0)`
    /// (glyphPlacement.ts:34).
    pub fn advance_width(&self, gid: u32) -> f64 {
        if let Some(var) = self.var.as_ref().filter(|v| v.instanced) {
            if let Some(a) = var.advance(gid) {
                return a;
            }
        }
        if self.advances.is_empty() {
            return 0.0;
        }
        let idx = (gid as usize).min(self.advances.len() - 1);
        self.advances[idx] as f64
    }

    /// `Font.getKerningValue` (font.js:252-261) +
    /// `Position.getKerningValue` (position.js:34-62).
    pub fn get_kerning_value(&self, left: u32, right: u32) -> Result<f64, ShapeError> {
        // gid 0 arrives as a Glyph OBJECT in JS (`leftGlyph.index || leftGlyph`,
        // font.js:253-254) and can never match a coverage table or a kern-pair
        // key — net 0 (see module header). `.notdef` never reaches kerning in
        // practice (whole-node fallback happens first).
        if left == 0 {
            return Ok(0.0);
        }
        if let (Some(lookup_idx), Some(gpos)) = (&self.gpos_kern_lookups, &self.gpos) {
            for li in lookup_idx {
                let Some(lookup) = gpos.lookups.get(*li as usize) else {
                    continue;
                };
                for st in &lookup.subtables {
                    match st {
                        GposSubtable::Pair1 {
                            coverage,
                            pair_sets,
                            pairs_ascending,
                        } => {
                            let Some(cov) = coverage.index(left) else {
                                continue;
                            };
                            // `subtable.pairSets[covIndex]` — OOB is a JS
                            // TypeError (position.js:44) → fallback-class.
                            let Some(pair_set) = pair_sets.get(cov as usize) else {
                                return Err(ShapeError::Other(format!(
                                    "pairSets[{cov}] out of bounds"
                                )));
                            };
                            let pair = if *pairs_ascending {
                                u16::try_from(right).ok().and_then(|right| {
                                    pair_set
                                        .binary_search_by_key(&right, |(second, _)| *second)
                                        .ok()
                                        .map(|i| &pair_set[i])
                                })
                            } else {
                                pair_set.iter().find(|(second, _)| *second as u32 == right)
                            };
                            if let Some((_, x_adv)) = pair {
                                return Ok(x_adv.unwrap_or(0) as f64);
                            }
                            // left found, right not — try next subtable
                            // (position.js:51).
                        }
                        GposSubtable::Pair2 {
                            coverage,
                            class_def1,
                            class_def2,
                            class1_count,
                            class2_count,
                            records,
                        } => {
                            if coverage.index(left).is_none() {
                                continue;
                            }
                            let class1 = class_def1.get(left);
                            // right gid 0 is an object in JS → getGlyphClass
                            // returns 0 (see module header).
                            let class2 = if right == 0 { 0 } else { class_def2.get(right) };
                            // `classRecords[class1][class2]` — either index
                            // OOB is a JS TypeError → fallback-class.
                            if class1 >= *class1_count as u32 || class2 >= *class2_count as u32 {
                                return Err(ShapeError::Other(format!(
                                    "classRecords[{class1}][{class2}] out of bounds"
                                )));
                            }
                            let idx = (class1 * *class2_count as u32 + class2) as usize;
                            // Format-2 coverage hit RETURNS UNCONDITIONALLY
                            // (position.js:52-57) — the early-return quirk.
                            return Ok(records.get(idx).copied().flatten().unwrap_or(0) as f64);
                        }
                        GposSubtable::Other => {}
                    }
                }
            }
            return Ok(0.0);
        }
        // No GPOS table → legacy kern table (font.js:259-260). A right-side
        // gid 0 is an object key in JS ('l,[object Object]') → never found.
        if right == 0 {
            return Ok(0.0);
        }
        Ok(self.kerning_pairs.get(&(left, right)).copied().unwrap_or(0) as f64)
    }

    /// `Font.stringToGlyphs` (font.js:178-202): tokenize per code point, map
    /// through cmap, apply latn 'liga' over latin-word ranges, drop deleted
    /// tokens. Returns glyph ids (the JS Glyph objects' `.index`).
    pub fn string_to_glyphs(&self, text: &str) -> Result<Vec<u32>, ShapeError> {
        Ok(self
            .string_to_glyphs_clustered(text)?
            .into_iter()
            .map(|(g, _)| g)
            .collect())
    }

    /// Like [`string_to_glyphs`], but also returns each surviving glyph's SOURCE
    /// CODE-POINT INDEX (its position in the input `text`'s code points). After a
    /// ligature substitution the component tokens are deleted, so the survivor's
    /// index is the ligature's FIRST code point — exactly the KineticText
    /// "ligature glyph → first code point" contract. For text with no
    /// substitutions this is just `0, 1, 2, …`. Used to map each laid-out glyph to
    /// `unitTransforms[index]`; `string_to_glyphs` is the glyph-only wrapper.
    pub fn string_to_glyphs_clustered(&self, text: &str) -> Result<Vec<(u32, usize)>, ShapeError> {
        // Arabic shaping (init/medi/fina/rlig + sentence reversal) is NOT
        // ported (Phase 3 in the browser too). The corpus can't reach it:
        // Arabic text only shapes if the run's font covers it, and coverage
        // fallback rejects it first. Scream loudly if that ever changes.
        if text
            .chars()
            .any(|c| matches!(c as u32, 0x0600..=0x065F | 0x066A..=0x06D2 | 0x06FA..=0x06FF))
        {
            return Err(ShapeError::Other(
                "arabic shaping not ported (opentype.js bidi arab features)".to_string(),
            ));
        }

        #[derive(Clone, PartialEq)]
        enum Active {
            G(u32),
            DeletedTrue,
        }

        struct Token {
            ch: char,
            active: Active,
            deleted: bool,
        }

        let mut tokens: Vec<Token> = text
            .chars()
            .map(|ch| Token {
                ch,
                active: Active::G(self.char_to_glyph_index(ch)),
                deleted: false,
            })
            .collect();

        // latn 'liga' lookup plan (registration + feature resolution).
        let liga_lookups = self.liga_lookup_indexes()?;

        if let (Some(liga), Some(gsub)) = (liga_lookups, &self.gsub) {
            // One working array per call, refilled per word (and after a substitution).
            let mut cells: Vec<Cell> = Vec::new();
            // latin word ranges: maximal runs of /[A-z]/ (char.js:33 —
            // includes `[\]^_`` — Q: latinWordCheck, contextCheck/latinWord.js).
            let is_latin = |c: char| (65..=122).contains(&(c as u32));
            let mut i = 0usize;
            while i < tokens.len() {
                if !is_latin(tokens[i].ch) {
                    i += 1;
                    continue;
                }
                let start = i;
                while i < tokens.len() && is_latin(tokens[i].ch) {
                    i += 1;
                }
                let end = i; // exclusive

                // latinLigature (features/latn/latinLigatures.js:23-38)
                let len = end - start;
                // context = range tokens' activeState values, rebuilt after
                // each substitution (a lookup that finds nothing leaves them
                // as they were).
                let fill_cells = |cells: &mut Vec<Cell>, tokens: &[Token]| {
                    cells.clear();
                    cells.extend(tokens[start..end].iter().map(|t| match t.active {
                        Active::G(g) => Cell::G(g),
                        Active::DeletedTrue => Cell::NonGlyph, // `true` — never matches coverage/components
                    }));
                };
                fill_cells(&mut cells, &tokens);
                for index in 0..len {
                    let action = lookup_feature_liga(gsub, liga, &mut cells, index)?;
                    // A match marked `cells[index]`, and the tokens change below.
                    let matched = action.is_some();
                    match action {
                        Some(LigaAction::Single(g)) => {
                            tokens[start + index].active = Active::G(g);
                        }
                        Some(LigaAction::Ligature { lig_glyph, comps }) => {
                            // applySubstitution.js:41-50
                            tokens[start + index].active = Active::G(lig_glyph as u32);
                            for k in 0..comps {
                                let t = &mut tokens[start + index + k + 1];
                                t.deleted = true;
                                t.active = Active::DeletedTrue;
                            }
                        }
                        Some(LigaAction::Chain(values)) => {
                            // applySubstitution.js:30-35
                            for (offset, v) in values.into_iter().enumerate() {
                                tokens[start + index + offset].active = Active::G(v);
                            }
                        }
                        Some(LigaAction::MultipleNoop) | None => {}
                    }
                    if matched {
                        fill_cells(&mut cells, &tokens);
                    }
                }
            }
        }

        // getTextGlyphs (bidi.js:227-238): skip deleted, take .index. The token's
        // position in `tokens` is its source code-point index (tokens are 1:1 with
        // input code points); a ligature survivor keeps its first component's index.
        Ok(tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.deleted)
            .map(|(idx, t)| match t.active {
                Active::G(g) => (g, idx),
                Active::DeletedTrue => unreachable!("deleted tokens are filtered"),
            })
            .collect())
    }

    /// The latn 'liga' feature's lookup indexes, mirroring
    /// `Bidi.registerFeatures` + `FeatureQuery.getScriptFeatures('latn')`
    /// (featureQuery.js:207-276): 'latn' defaultLangSys, else DFLT-fallback;
    /// LAST 'liga' tag wins. `None` ⇒ no liga (no GSUB / unsupported script /
    /// feature absent) — shaping is then cmap-only.
    fn liga_lookup_indexes(&self) -> Result<Option<&[u16]>, ShapeError> {
        let Some(gsub) = &self.gsub else {
            return Ok(None);
        };
        let feature_indexes: &[u16] = 'feat: {
            for s in &gsub.scripts {
                if &s.tag == b"latn" {
                    if let Some(dls) = &s.default_lang_sys {
                        break 'feat &dls.feature_indexes;
                    }
                    // latn without defaultLangSys: langSysRecords are scanned
                    // for tag === 'latn' (never a language tag) → falls
                    // through to the DFLT default (featureQuery.js:228-243).
                }
            }
            // getDefaultScriptFeaturesIndexes (featureQuery.js:207-216) —
            // note `script.script.defaultLangSys.featureIndexes` with NO null
            // check: DFLT without a defaultLangSys is a JS TypeError.
            for s in &gsub.scripts {
                if &s.tag == b"DFLT" {
                    match &s.default_lang_sys {
                        Some(dls) => break 'feat &dls.feature_indexes,
                        None => {
                            return Err(ShapeError::Other(
                                "DFLT script has no defaultLangSys".to_string(),
                            ))
                        }
                    }
                }
            }
            &[]
        };
        // mapTagsToFeatures: `tags[tag] = feature` — LAST liga wins.
        let mut liga: Option<&FeatureRecord> = None;
        for fi in feature_indexes {
            if let Some(f) = gsub.features.get(*fi as usize) {
                if &f.tag == b"liga" {
                    liga = Some(f);
                }
            }
        }
        Ok(liga.map(|f| f.lookup_list_indexes.as_slice()))
    }
}

// ─────────────────────── liga feature lookup ──────────────────────

enum LigaAction {
    Single(u32),
    Ligature {
        lig_glyph: u16,
        comps: usize,
    },
    Chain(Vec<u32>),
    /// '21' multiple substitution: matched, splices an action into the
    /// working array, but applySubstitution has no handler — a no-op.
    MultipleNoop,
}

/// Working-array cell: a glyph value, the deleted-token `true`, or an action
/// marker spliced at the current index (never matches anything afterwards).
#[derive(Clone, PartialEq)]
enum Cell {
    G(u32),
    NonGlyph, // deleted `true` boolean
    Action,   // SubstitutionAction object
}

/// `FeatureQuery.lookupFeature` for tag 'liga' (featureQuery.js:354-422):
/// walk every subtable of every liga lookup IN ORDER, `getLookupMethod`
/// throwing for unsupported type/format combos, splicing the (last) matching
/// action at `index`.
fn lookup_feature_liga(
    gsub: &LayoutTable<GsubSubtable>,
    liga_lookup_indexes: &[u16],
    cells: &mut [Cell],
    index: usize,
) -> Result<Option<LigaAction>, ShapeError> {
    let mut action: Option<LigaAction> = None;

    for li in liga_lookup_indexes {
        let Some(lookup) = gsub.lookups.get(*li as usize) else {
            // getLookupByIndex returns null → getLookupSubtables(null) is a
            // JS TypeError. Unreachable for well-formed fonts.
            return Err(ShapeError::Other(format!(
                "lookup index {li} out of bounds"
            )));
        };
        for st in &lookup.subtables {
            let current = cells[index].clone();
            let cur_gid = match &current {
                Cell::G(g) => Some(*g),
                _ => None,
            };
            match (lookup.lookup_type, st) {
                (
                    1,
                    GsubSubtable::Single1 {
                        coverage,
                        delta_glyph_id,
                    },
                ) => {
                    if let Some(g) = cur_gid {
                        if coverage.index(g).is_some() {
                            // `glyphIndex + deltaGlyphId` — unsigned, unmasked
                            let subst = g + *delta_glyph_id as u32;
                            action = Some(LigaAction::Single(subst));
                            cells[index] = Cell::Action;
                        }
                    }
                }
                (
                    1,
                    GsubSubtable::Single2 {
                        coverage,
                        substitutes,
                    },
                ) => {
                    if let Some(g) = cur_gid {
                        if let Some(ci) = coverage.index(g) {
                            // substitute[ci] — OOB → undefined → falsy → no splice
                            if let Some(s) = substitutes.get(ci as usize) {
                                if *s != 0 {
                                    action = Some(LigaAction::Single(*s as u32));
                                    cells[index] = Cell::Action;
                                }
                            }
                        }
                    }
                }
                (
                    2,
                    GsubSubtable::Multiple {
                        coverage,
                        sequences,
                    },
                ) => {
                    if let Some(g) = cur_gid {
                        if let Some(ci) = coverage.index(g) {
                            if sequences.get(ci as usize).is_some() {
                                action = Some(LigaAction::MultipleNoop);
                                cells[index] = Cell::Action;
                            }
                        }
                    }
                }
                (
                    4,
                    GsubSubtable::Ligature {
                        coverage,
                        ligature_sets,
                    },
                ) => {
                    // ligatureSubstitutionFormat1 (featureQuery.js:170-191)
                    if let Some(g) = cur_gid {
                        if let Some(ci) = coverage.index(g) {
                            let Some(lig_set) = ligature_sets.get(ci as usize) else {
                                return Err(ShapeError::Other(format!(
                                    "ligatureSets[{ci}] out of bounds"
                                )));
                            };
                            'ligs: for lig in lig_set {
                                if lig.components.is_empty() {
                                    // zero-component ligature can never
                                    // return (loop body never reaches the
                                    // last-component check) — quirk kept.
                                    continue;
                                }
                                for (l, comp) in lig.components.iter().enumerate() {
                                    let item = cells.get(index + 1 + l);
                                    let matches =
                                        matches!(item, Some(Cell::G(g2)) if *g2 == *comp as u32);
                                    if !matches {
                                        continue 'ligs;
                                    }
                                }
                                action = Some(LigaAction::Ligature {
                                    lig_glyph: lig.lig_glyph,
                                    comps: lig.components.len(),
                                });
                                cells[index] = Cell::Action;
                                break 'ligs;
                            }
                        }
                    }
                }
                (
                    6,
                    GsubSubtable::Chain3 {
                        backtrack,
                        input,
                        lookahead,
                        lookup_records,
                    },
                ) => {
                    // chainingSubstitutionFormat3 (featureQuery.js:103-160)
                    let values = chain3_lookup(
                        gsub,
                        cells,
                        index,
                        backtrack,
                        input,
                        lookahead,
                        lookup_records,
                    )?;
                    if !values.is_empty() {
                        action = Some(LigaAction::Chain(values));
                        cells[index] = Cell::Action;
                    }
                }
                (lt, GsubSubtable::Other { subst_format }) => {
                    // getLookupMethod's throw (featureQuery.js:296-318) —
                    // fired for EVERY subtable visited, even non-matching.
                    return Err(ShapeError::UnsupportedLookup {
                        lookup_type: lt,
                        subst_format: *subst_format,
                    });
                }
                (lt, _) => {
                    // Type/subtable mismatch can't happen with our parser;
                    // treat as the JS throw for safety.
                    return Err(ShapeError::UnsupportedLookup {
                        lookup_type: lt,
                        subst_format: 0,
                    });
                }
            }
        }
    }
    Ok(action)
}

/// `chainingSubstitutionFormat3` over the working array. Coverage lookups go
/// through `lookupCoverageList` (featureQuery.js:88-101) — the CURRENT value
/// is array-unwrapped there, but our cells never hold arrays (see
/// `LigaAction::MultipleNoop`); action markers and `true` never match.
fn chain3_lookup(
    gsub: &LayoutTable<GsubSubtable>,
    cells: &[Cell],
    index: usize,
    backtrack: &[Coverage],
    input: &[Coverage],
    lookahead: &[Coverage],
    lookup_records: &[(u16, u16)],
) -> Result<Vec<u32>, ShapeError> {
    let lookups_count = input.len() + lookahead.len() + backtrack.len();
    if cells.len() < lookups_count {
        return Ok(Vec::new());
    }
    let gid_at = |cell: Option<&Cell>| -> Option<u32> {
        match cell {
            Some(Cell::G(g)) => Some(*g),
            _ => None,
        }
    };

    // INPUT: every input coverage tested against the CURRENT value only
    // (featureQuery.js:88-101 — `contextParams.current` for each).
    let cur = gid_at(cells.get(index));
    let mut input_hits = 0usize;
    for cov in input {
        if cur.and_then(|g| cov.index(g)).is_some() {
            input_hits += 1;
        }
    }
    if input_hits != input.len() {
        return Ok(Vec::new());
    }

    // LOOKAHEAD from offset inputCoverage.length - 1 into the lookahead
    // slice (featureQuery.js:115-118; Tashkeel skipping is Arabic-only and
    // unreachable here).
    // Every lookahead coverage is tested against that slice's first cell.
    let lookahead_cells = cells.get(index + 1..).unwrap_or(&[]);
    if lookahead_cells.len() < lookahead.len() {
        return Ok(Vec::new());
    }
    let offset = input.len().saturating_sub(1);
    let la_cur = gid_at(lookahead_cells.get(offset));
    let la_hits = lookahead
        .iter()
        .filter(|cov| la_cur.and_then(|g| cov.index(g)).is_some())
        .count();

    // BACKTRACK reversed (featureQuery.js:126-136): its first cell is the one
    // just before the current, tested against every backtrack coverage.
    if index < backtrack.len() {
        return Ok(Vec::new());
    }
    let bt_cur = index.checked_sub(1).and_then(|i| gid_at(cells.get(i)));
    let bt_hits = backtrack
        .iter()
        .filter(|cov| bt_cur.and_then(|g| cov.index(g)).is_some())
        .count();

    if la_hits != lookahead.len() || bt_hits != backtrack.len() {
        return Ok(Vec::new());
    }

    // Context matched: run the records' inner lookups; only '12' inner
    // subtables substitute (featureQuery.js:141-158), reading `get(n)` for
    // each input position.
    let mut substitutions: Vec<u32> = Vec::new();
    for (_, lookup_list_index) in lookup_records {
        let Some(inner) = gsub.lookups.get(*lookup_list_index as usize) else {
            return Err(ShapeError::Other(format!(
                "chain inner lookup {lookup_list_index} out of bounds"
            )));
        };
        for st in &inner.subtables {
            // getLookupMethod is called per inner subtable too — unsupported
            // inner types throw.
            match (inner.lookup_type, st) {
                (
                    1,
                    GsubSubtable::Single2 {
                        coverage,
                        substitutes,
                    },
                ) => {
                    for n in 0..input.len() {
                        // contextParams.get(n): 0 → current, n>0 → lookahead[n-1]
                        let cell = if n == 0 {
                            cells.get(index)
                        } else {
                            cells.get(index + n)
                        };
                        if let Some(g) = gid_at(cell) {
                            if let Some(ci) = coverage.index(g) {
                                if let Some(s) = substitutes.get(ci as usize) {
                                    if *s != 0 {
                                        substitutions.push(*s as u32);
                                    }
                                }
                            }
                        }
                    }
                }
                (1, GsubSubtable::Single1 { .. }) => {
                    // substitutionType '11' — getLookupMethod supports it but
                    // chainingSubstitutionFormat3 only executes '12'
                    // (featureQuery.js:150-156); other supported types are
                    // skipped silently.
                }
                (2, GsubSubtable::Multiple { .. })
                | (4, GsubSubtable::Ligature { .. })
                | (6, GsubSubtable::Chain3 { .. }) => {}
                (lt, GsubSubtable::Other { subst_format }) => {
                    return Err(ShapeError::UnsupportedLookup {
                        lookup_type: lt,
                        subst_format: *subst_format,
                    });
                }
                (lt, _) => {
                    return Err(ShapeError::UnsupportedLookup {
                        lookup_type: lt,
                        subst_format: 0,
                    });
                }
            }
        }
    }
    Ok(substitutions)
}

// ──────────────────────── table parsers ───────────────────────────

/// `position.init` (position.js:22-25) + `getKerningTables` (71-75) +
/// `Layout.getDefaultScriptName`/`getScriptTable`/`getLangSysTable`/
/// `getFeatureTable`/`getLookupTables` (layout.js:128-273): the default
/// script's defaultLangSys, FIRST 'kern' feature, direct type-2 lookups.
fn default_kerning_lookups(gpos: &LayoutTable<GposSubtable>) -> Vec<u16> {
    // getDefaultScriptName: 'DFLT' if present, else 'latn', else undefined
    // (→ getScriptTable falls back to 'DFLT' and finds nothing).
    let mut script_tag: Option<&[u8; 4]> = None;
    let mut has_latn = false;
    for s in &gpos.scripts {
        if &s.tag == b"DFLT" {
            script_tag = Some(b"DFLT");
            break;
        }
        if &s.tag == b"latn" {
            has_latn = true;
        }
    }
    if script_tag.is_none() && has_latn {
        script_tag = Some(b"latn");
    }
    let wanted = script_tag.unwrap_or(b"DFLT");
    let Some(script) = gpos.scripts.iter().find(|s| &s.tag == wanted) else {
        return Vec::new();
    };
    let Some(dls) = &script.default_lang_sys else {
        return Vec::new();
    };
    // getFeatureTable: FIRST 'kern' tag among the langSys featureIndexes.
    let mut kern_feature: Option<&FeatureRecord> = None;
    for fi in &dls.feature_indexes {
        if let Some(f) = gpos.features.get(*fi as usize) {
            if &f.tag == b"kern" {
                kern_feature = Some(f);
                break;
            }
        }
    }
    let Some(feature) = kern_feature else {
        return Vec::new();
    };
    // getLookupTables(..., lookupType=2): direct type-2 only.
    feature
        .lookup_list_indexes
        .iter()
        .copied()
        .filter(|li| {
            gpos.lookups
                .get(*li as usize)
                .map(|l| l.lookup_type == 2)
                .unwrap_or(false)
        })
        .collect()
}

/// tables/cmap.js:84-116 — backward record scan, formats 4/12 only.
fn parse_cmap(data: &[u8], start: usize) -> Result<FxHashMap<u32, u32>, String> {
    let version = u16be(data, start)?;
    if version != 0 {
        return Err("cmap table version should be 0".to_string());
    }
    let num_tables = u16be(data, start + 2)?;
    let mut offset: Option<usize> = None;
    for i in (0..num_tables as usize).rev() {
        let platform_id = u16be(data, start + 4 + i * 8)?;
        let encoding_id = u16be(data, start + 4 + i * 8 + 2)?;
        if (platform_id == 3 && matches!(encoding_id, 0 | 1 | 10))
            || (platform_id == 0 && encoding_id <= 4)
        {
            offset = Some(u32be(data, start + 4 + i * 8 + 4)? as usize);
            break;
        }
    }
    let Some(offset) = offset else {
        return Err("No valid cmap sub-tables found.".to_string());
    };
    let sub = start + offset;
    let format = u16be(data, sub)?;
    let mut map = FxHashMap::default();
    if format == 12 {
        // parseCmapTableFormat12 (cmap.js:8-31) — the explicit `startGlyphId++`
        // counter mirrors the JS loop on purpose.
        #[allow(clippy::explicit_counter_loop)]
        {
            let group_count = u32be(data, sub + 12)?;
            let mut p = sub + 16;
            for _ in 0..group_count {
                let start_char = u32be(data, p)?;
                let end_char = u32be(data, p + 4)?;
                let mut glyph = u32be(data, p + 8)?;
                for c in start_char..=end_char {
                    map.insert(c, glyph);
                    glyph += 1;
                }
                p += 12;
            }
        }
    } else if format == 4 {
        // parseCmapTableFormat4 (cmap.js:32-80) — note the `segCount - 1`
        // loop: the final 0xFFFF segment is skipped.
        let seg_count = (u16be(data, sub + 6)? >> 1) as usize;
        let end_base = sub + 14;
        let start_base = sub + 16 + seg_count * 2;
        let delta_base = sub + 16 + seg_count * 4;
        let range_base = sub + 16 + seg_count * 6;
        for i in 0..seg_count.saturating_sub(1) {
            let end_count = u16be(data, end_base + i * 2)? as u32;
            let start_count = u16be(data, start_base + i * 2)? as u32;
            let id_delta = i16be(data, delta_base + i * 2)? as i32;
            let id_range_offset = u16be(data, range_base + i * 2)? as usize;
            let mut c = start_count;
            while c <= end_count {
                let glyph_index = if id_range_offset != 0 {
                    // relative to THIS entry's position in the array
                    let pos =
                        range_base + i * 2 + id_range_offset + ((c - start_count) as usize) * 2;
                    let g = u16be(data, pos)? as i32;
                    if g != 0 {
                        ((g + id_delta) & 0xFFFF) as u32
                    } else {
                        0
                    }
                } else {
                    ((c as i32 + id_delta) & 0xFFFF) as u32
                };
                map.insert(c, glyph_index);
                if c == u32::MAX {
                    break;
                }
                c += 1;
            }
        }
    } else {
        return Err(format!(
            "Only format 4 and 12 cmap tables are supported (found format {format})."
        ));
    }
    Ok(map)
}

/// tables/kern.js — version 0 (Windows, FIRST subtable only) / version 1 (Mac).
fn parse_kern(data: &[u8], start: usize) -> Result<FxHashMap<(u32, u32), i16>, String> {
    let mut pairs = FxHashMap::default();
    let version = u16be(data, start)?;
    if version == 0 {
        // parseWindowsKernTable: skip nTables, assert subtableVersion 0
        let subtable_version = u16be(data, start + 4)?;
        if subtable_version != 0 {
            return Err("Unsupported kern sub-table version.".to_string());
        }
        let n_pairs = u16be(data, start + 10)?;
        let mut p = start + 18;
        for _ in 0..n_pairs {
            let left = u16be(data, p)? as u32;
            let right = u16be(data, p + 2)? as u32;
            let value = i16be(data, p + 4)?;
            pairs.insert((left, right), value);
            p += 6;
        }
    } else if version == 1 {
        // parseMacKernTable: first subtable only, version-0 payload only
        let coverage = u16be(data, start + 12)?;
        let subtable_version = coverage & 0xFF;
        if subtable_version == 0 {
            let n_pairs = u16be(data, start + 16)?;
            let mut p = start + 24;
            for _ in 0..n_pairs {
                let left = u16be(data, p)? as u32;
                let right = u16be(data, p + 2)? as u32;
                let value = i16be(data, p + 4)?;
                pairs.insert((left, right), value);
                p += 6;
            }
        }
    } else {
        return Err(format!("Unsupported kern table version ({version})."));
    }
    Ok(pairs)
}

fn parse_coverage(data: &[u8], off: usize) -> Result<Coverage, String> {
    let format = u16be(data, off)?;
    match format {
        1 => {
            let count = u16be(data, off + 2)? as usize;
            let mut glyphs = Vec::with_capacity(count);
            for i in 0..count {
                glyphs.push(u16be(data, off + 4 + i * 2)?);
            }
            let ascending = glyphs.windows(2).all(|w| w[0] < w[1]);
            Ok(Coverage::F1 { glyphs, ascending })
        }
        2 => {
            let count = u16be(data, off + 2)? as usize;
            let mut ranges = Vec::with_capacity(count);
            for i in 0..count {
                let base = off + 4 + i * 6;
                ranges.push((
                    u16be(data, base)?,
                    u16be(data, base + 2)?,
                    u16be(data, base + 4)?,
                ));
            }
            let ascending = ranges_ascending(&ranges);
            Ok(Coverage::F2 { ranges, ascending })
        }
        f => Err(format!("unsupported coverage format {f}")),
    }
}

fn parse_class_def(data: &[u8], off: usize) -> Result<ClassDef, String> {
    let format = u16be(data, off)?;
    match format {
        1 => {
            let start_glyph = u16be(data, off + 2)?;
            let count = u16be(data, off + 4)? as usize;
            let mut classes = Vec::with_capacity(count);
            for i in 0..count {
                classes.push(u16be(data, off + 6 + i * 2)?);
            }
            Ok(ClassDef::F1 {
                start_glyph,
                classes,
            })
        }
        2 => {
            let count = u16be(data, off + 2)? as usize;
            let mut ranges = Vec::with_capacity(count);
            for i in 0..count {
                let base = off + 4 + i * 6;
                ranges.push((
                    u16be(data, base)?,
                    u16be(data, base + 2)?,
                    u16be(data, base + 4)?,
                ));
            }
            let ascending = ranges_ascending(&ranges);
            Ok(ClassDef::F2 { ranges, ascending })
        }
        f => Err(format!("unsupported classDef format {f}")),
    }
}

fn parse_lang_sys(data: &[u8], off: usize) -> Result<LangSys, String> {
    // lookupOrder (u16), reqFeatureIndex (u16), featureIndexCount, indexes
    let count = u16be(data, off + 4)? as usize;
    let mut feature_indexes = Vec::with_capacity(count);
    for i in 0..count {
        feature_indexes.push(u16be(data, off + 6 + i * 2)?);
    }
    Ok(LangSys { feature_indexes })
}

fn parse_script_list(data: &[u8], off: usize) -> Result<Vec<ScriptRecord>, String> {
    let count = u16be(data, off)? as usize;
    let mut scripts = Vec::with_capacity(count);
    for i in 0..count {
        let rec = off + 2 + i * 6;
        let tag: [u8; 4] = data
            .get(rec..rec + 4)
            .ok_or("script list truncated")?
            .try_into()
            .unwrap();
        let script_off = off + u16be(data, rec + 4)? as usize;
        let dls_off = u16be(data, script_off)? as usize;
        let default_lang_sys = if dls_off > 0 {
            Some(parse_lang_sys(data, script_off + dls_off)?)
        } else {
            None
        };
        let ls_count = u16be(data, script_off + 2)? as usize;
        let mut lang_sys_records = Vec::with_capacity(ls_count);
        for j in 0..ls_count {
            let lrec = script_off + 4 + j * 6;
            let ltag: [u8; 4] = data
                .get(lrec..lrec + 4)
                .ok_or("langSys records truncated")?
                .try_into()
                .unwrap();
            let ls_off = script_off + u16be(data, lrec + 4)? as usize;
            lang_sys_records.push((ltag, parse_lang_sys(data, ls_off)?));
        }
        scripts.push(ScriptRecord {
            tag,
            default_lang_sys,
            lang_sys_records,
        });
    }
    Ok(scripts)
}

fn parse_feature_list(data: &[u8], off: usize) -> Result<Vec<FeatureRecord>, String> {
    let count = u16be(data, off)? as usize;
    let mut features = Vec::with_capacity(count);
    for i in 0..count {
        let rec = off + 2 + i * 6;
        let tag: [u8; 4] = data
            .get(rec..rec + 4)
            .ok_or("feature list truncated")?
            .try_into()
            .unwrap();
        let feat_off = off + u16be(data, rec + 4)? as usize;
        // featureParams u16, lookupCount u16, indexes
        let lookup_count = u16be(data, feat_off + 2)? as usize;
        let mut lookup_list_indexes = Vec::with_capacity(lookup_count);
        for j in 0..lookup_count {
            lookup_list_indexes.push(u16be(data, feat_off + 4 + j * 2)?);
        }
        features.push(FeatureRecord {
            tag,
            lookup_list_indexes,
        });
    }
    Ok(features)
}

fn parse_layout_table<T>(
    data: &[u8],
    start: usize,
    parse_subtable: fn(&[u8], usize, u16) -> Result<T, String>,
) -> Result<LayoutTable<T>, String> {
    // parseVersion(1): major.minor must be 1 or 1.1 (parse.js parseVersion;
    // gsub.js:193-196 / gpos.js check.argument). With minorBase 1 the version
    // is `major + minor / 10`, so 1.1 is minor 1 (the table version that adds
    // FeatureVariations, which every variable Google font has). It was read as
    // 0x1000, which declined those fonts to the raster tier (no kinetic type).
    // The FeatureVariations offset is not read: the default instance's features.
    let major = u16be(data, start)?;
    let minor = u16be(data, start + 2)?;
    if major != 1 || minor > 1 {
        return Err(format!(
            "Unsupported GSUB/GPOS table version {major}.{minor}"
        ));
    }
    let script_list_off = start + u16be(data, start + 4)? as usize;
    let feature_list_off = start + u16be(data, start + 6)? as usize;
    let lookup_list_off = start + u16be(data, start + 8)? as usize;

    let scripts = parse_script_list(data, script_list_off)?;
    let features = parse_feature_list(data, feature_list_off)?;

    let lookup_count = u16be(data, lookup_list_off)? as usize;
    let mut lookups = Vec::with_capacity(lookup_count);
    for i in 0..lookup_count {
        let lookup_off = lookup_list_off + u16be(data, lookup_list_off + 2 + i * 2)? as usize;
        let lookup_type = u16be(data, lookup_off)?;
        // lookupFlag u16, subTableCount u16
        let subtable_count = u16be(data, lookup_off + 4)? as usize;
        let mut subtables = Vec::with_capacity(subtable_count);
        for j in 0..subtable_count {
            let st_off = lookup_off + u16be(data, lookup_off + 6 + j * 2)? as usize;
            subtables.push(parse_subtable(data, st_off, lookup_type)?);
        }
        lookups.push(Lookup {
            lookup_type,
            subtables,
        });
    }
    Ok(LayoutTable {
        scripts,
        features,
        lookups,
    })
}

/// gsub.js `subtableParsers` — incl. the assertions that fail the whole font
/// at parse time and the extension unwrap to `substFormat: 1`.
fn parse_gsub_subtable(data: &[u8], off: usize, lookup_type: u16) -> Result<GsubSubtable, String> {
    let subst_format = u16be(data, off)?;
    match lookup_type {
        1 => match subst_format {
            1 => Ok(GsubSubtable::Single1 {
                coverage: parse_coverage(data, off + u16be(data, off + 2)? as usize)?,
                delta_glyph_id: u16be(data, off + 4)?,
            }),
            2 => {
                let coverage = parse_coverage(data, off + u16be(data, off + 2)? as usize)?;
                let count = u16be(data, off + 4)? as usize;
                let mut substitutes = Vec::with_capacity(count);
                for i in 0..count {
                    substitutes.push(u16be(data, off + 6 + i * 2)?);
                }
                Ok(GsubSubtable::Single2 {
                    coverage,
                    substitutes,
                })
            }
            f => Err(format!("lookup type 1 format must be 1 or 2 (got {f})")),
        },
        2 => {
            if subst_format != 1 {
                return Err(
                    "GSUB Multiple Substitution Subtable identifier-format must be 1".to_string(),
                );
            }
            let coverage = parse_coverage(data, off + u16be(data, off + 2)? as usize)?;
            let count = u16be(data, off + 4)? as usize;
            let mut sequences = Vec::with_capacity(count);
            for i in 0..count {
                let seq_off = off + u16be(data, off + 6 + i * 2)? as usize;
                let n = u16be(data, seq_off)? as usize;
                let mut seq = Vec::with_capacity(n);
                for j in 0..n {
                    seq.push(u16be(data, seq_off + 2 + j * 2)?);
                }
                sequences.push(seq);
            }
            Ok(GsubSubtable::Multiple {
                coverage,
                sequences,
            })
        }
        4 => {
            if subst_format != 1 {
                return Err("GSUB ligature table identifier-format must be 1".to_string());
            }
            let coverage = parse_coverage(data, off + u16be(data, off + 2)? as usize)?;
            let set_count = u16be(data, off + 4)? as usize;
            let mut ligature_sets = Vec::with_capacity(set_count);
            for i in 0..set_count {
                let set_off = off + u16be(data, off + 6 + i * 2)? as usize;
                let lig_count = u16be(data, set_off)? as usize;
                let mut ligs = Vec::with_capacity(lig_count);
                for j in 0..lig_count {
                    let lig_off = set_off + u16be(data, set_off + 2 + j * 2)? as usize;
                    let lig_glyph = u16be(data, lig_off)?;
                    let comp_count = u16be(data, lig_off + 2)? as usize;
                    let n = comp_count.saturating_sub(1);
                    let mut components = Vec::with_capacity(n);
                    for k in 0..n {
                        components.push(u16be(data, lig_off + 4 + k * 2)?);
                    }
                    ligs.push(Ligature {
                        lig_glyph,
                        components,
                    });
                }
                ligature_sets.push(ligs);
            }
            Ok(GsubSubtable::Ligature {
                coverage,
                ligature_sets,
            })
        }
        6 if subst_format == 3 => {
            let mut p = off + 2;
            let read_coverages = |p: &mut usize| -> Result<Vec<Coverage>, String> {
                let count = u16be(data, *p)? as usize;
                *p += 2;
                let mut out = Vec::with_capacity(count);
                for _ in 0..count {
                    let cov_off = off + u16be(data, *p)? as usize;
                    *p += 2;
                    out.push(parse_coverage(data, cov_off)?);
                }
                Ok(out)
            };
            let backtrack = read_coverages(&mut p)?;
            let input = read_coverages(&mut p)?;
            let lookahead = read_coverages(&mut p)?;
            let rec_count = u16be(data, p)? as usize;
            p += 2;
            let mut lookup_records = Vec::with_capacity(rec_count);
            for _ in 0..rec_count {
                lookup_records.push((u16be(data, p)?, u16be(data, p + 2)?));
                p += 4;
            }
            Ok(GsubSubtable::Chain3 {
                backtrack,
                input,
                lookahead,
                lookup_records,
            })
        }
        7 => {
            // Extension wrapper: gsub.js:163-174 asserts format 1 and parses
            // the inner table eagerly (we validate the inner type has a
            // parser slot, mirroring `subtableParsers[type].call` not being
            // undefined), but featureQuery only ever sees `{substFormat: 1}`
            // on the OUTER type-7 lookup → always the unsupported-throw.
            if subst_format != 1 {
                return Err(
                    "GSUB Extension Substitution subtable identifier-format must be 1".to_string(),
                );
            }
            let inner_type = u16be(data, off + 2)?;
            if !(1..=8).contains(&inner_type) {
                return Err(format!("extension inner lookup type {inner_type} invalid"));
            }
            Ok(GsubSubtable::Other { subst_format: 1 })
        }
        // 3, 5, 6 (formats 1-2), 8: parsed by opentype.js but unsupported by
        // getLookupMethod — only the substFormat matters for the throw.
        // Their parse-time format assertions:
        3 | 8 if subst_format != 1 => Err(format!(
            "GSUB lookup type {lookup_type} identifier-format must be 1"
        )),
        5 if !(1..=3).contains(&subst_format) => {
            Err("lookup type 5 format must be 1, 2 or 3.".to_string())
        }
        6 if !(1..=3).contains(&subst_format) => {
            Err("lookup type 6 format must be 1, 2 or 3.".to_string())
        }
        _ => Ok(GsubSubtable::Other { subst_format }),
    }
}

/// gpos.js `subtableParsers` — type 2 (PairPos) fully, everything else an
/// `{error}` stub that the kerning walk never touches.
fn parse_gpos_subtable(data: &[u8], off: usize, lookup_type: u16) -> Result<GposSubtable, String> {
    if lookup_type != 2 {
        return Ok(GposSubtable::Other);
    }
    let pos_format = u16be(data, off)?;
    if pos_format != 1 && pos_format != 2 {
        return Err("GPOS lookup type 2 format must be 1 or 2.".to_string());
    }
    let coverage = parse_coverage(data, off + u16be(data, off + 2)? as usize)?;
    let value_format1 = u16be(data, off + 4)?;
    let value_format2 = u16be(data, off + 6)?;

    // parse.js parseValueRecord field order/widths (parse.js:353-377); we
    // only retain value1.xAdvance.
    let value_record_len = |fmt: u16| -> usize { (fmt.count_ones() as usize) * 2 };
    let read_value1_x_advance = |p: usize| -> Result<(Option<i16>, usize), String> {
        if value_format1 == 0 {
            return Ok((None, 0));
        }
        let mut q = p;
        let mut x_adv = None;
        if value_format1 & 0x0001 != 0 {
            q += 2;
        }
        if value_format1 & 0x0002 != 0 {
            q += 2;
        }
        if value_format1 & 0x0004 != 0 {
            x_adv = Some(i16be(data, q)?);
            q += 2;
        }
        if value_format1 & 0x0008 != 0 {
            q += 2;
        }
        for bit in [0x0010u16, 0x0020, 0x0040, 0x0080] {
            if value_format1 & bit != 0 {
                q += 2;
            }
        }
        Ok((x_adv, q - p))
    };

    if pos_format == 1 {
        let pair_set_count = u16be(data, off + 8)? as usize;
        let mut pair_sets = Vec::with_capacity(pair_set_count);
        for i in 0..pair_set_count {
            let ps_off = off + u16be(data, off + 10 + i * 2)? as usize;
            let pair_count = u16be(data, ps_off)? as usize;
            let mut pairs = Vec::with_capacity(pair_count);
            let mut p = ps_off + 2;
            for _ in 0..pair_count {
                let second_glyph = u16be(data, p)?;
                p += 2;
                let (x_adv, consumed) = read_value1_x_advance(p)?;
                p += consumed.max(value_record_len(value_format1));
                p += value_record_len(value_format2);
                pairs.push((second_glyph, x_adv));
            }
            pair_sets.push(pairs);
        }
        let pairs_ascending = pair_sets
            .iter()
            .all(|pairs| pairs.windows(2).all(|w| w[0].0 < w[1].0));
        Ok(GposSubtable::Pair1 {
            coverage,
            pair_sets,
            pairs_ascending,
        })
    } else {
        let class_def1 = parse_class_def(data, off + u16be(data, off + 8)? as usize)?;
        let class_def2 = parse_class_def(data, off + u16be(data, off + 10)? as usize)?;
        let class1_count = u16be(data, off + 12)?;
        let class2_count = u16be(data, off + 14)?;
        let mut records = Vec::with_capacity(class1_count as usize * class2_count as usize);
        let mut p = off + 16;
        for _ in 0..(class1_count as usize * class2_count as usize) {
            let (x_adv, consumed) = read_value1_x_advance(p)?;
            p += consumed.max(value_record_len(value_format1));
            p += value_record_len(value_format2);
            records.push(x_adv);
        }
        Ok(GposSubtable::Pair2 {
            coverage,
            class_def1,
            class_def2,
            class1_count,
            class2_count,
            records,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the scan answers (featureQuery.js's `indexOf` / range walk).
    fn scan(ranges: &[(u16, u16, u16)], gid: u32) -> Option<(u16, u16, u16)> {
        find_range(ranges, false, gid)
    }

    #[test]
    fn sorted_lookups_answer_what_the_scan_answers() {
        let glyphs: Vec<u16> = vec![3, 4, 9, 10, 11, 40, 41, 300, 301, 302, 999];
        let ranges: Vec<(u16, u16, u16)> = vec![
            (2, 2, 0),
            (5, 9, 1),
            (10, 10, 6),
            (12, 40, 7),
            (300, 310, 36),
        ];
        let f1 = Coverage::F1 {
            ascending: glyphs.windows(2).all(|w| w[0] < w[1]),
            glyphs: glyphs.clone(),
        };
        let f2 = Coverage::F2 {
            ascending: ranges_ascending(&ranges),
            ranges: ranges.clone(),
        };
        let classes = ClassDef::F2 {
            ascending: ranges_ascending(&ranges),
            ranges: ranges.clone(),
        };
        assert!(matches!(
            f1,
            Coverage::F1 {
                ascending: true,
                ..
            }
        ));
        assert!(matches!(
            f2,
            Coverage::F2 {
                ascending: true,
                ..
            }
        ));
        for gid in 0..1100u32 {
            let linear = (gid != 0)
                .then(|| {
                    glyphs
                        .iter()
                        .position(|g| *g as u32 == gid)
                        .map(|i| i as u32)
                })
                .flatten();
            assert_eq!(f1.index(gid), linear, "format 1, gid {gid}");
            let hit = scan(&ranges, gid);
            assert_eq!(find_range(&ranges, true, gid), hit, "gid {gid}");
            let linear = (gid != 0)
                .then(|| hit.map(|(s, _, i)| i as u32 + gid - s as u32))
                .flatten();
            assert_eq!(f2.index(gid), linear, "format 2, gid {gid}");
            assert_eq!(
                classes.get(gid),
                hit.map_or(0, |(_, _, c)| c as u32),
                "class, gid {gid}"
            );
        }
        assert_eq!(f1.index(70_000), None);
        assert_eq!(f2.index(70_000), None);
    }

    #[test]
    fn unsorted_tables_keep_the_first_match_of_the_scan() {
        // Overlapping and out of order: the scan's first match wins, as in opentype.js.
        let ranges: Vec<(u16, u16, u16)> = vec![(10, 20, 1), (5, 12, 2), (30, 29, 3)];
        assert!(!ranges_ascending(&ranges));
        let classes = ClassDef::F2 {
            ascending: ranges_ascending(&ranges),
            ranges,
        };
        assert_eq!(classes.get(11), 1);
        assert_eq!(classes.get(6), 2);
        let glyphs: Vec<u16> = vec![9, 4, 4, 7];
        let f1 = Coverage::F1 {
            ascending: glyphs.windows(2).all(|w| w[0] < w[1]),
            glyphs,
        };
        assert!(matches!(
            f1,
            Coverage::F1 {
                ascending: false,
                ..
            }
        ));
        assert_eq!(f1.index(4), Some(1));
        assert_eq!(f1.index(7), Some(3));
    }
}

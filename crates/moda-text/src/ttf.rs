//! TTF baseline metrics — byte-for-byte port of the production hand-rolled
//! parser `utils/msdf/ttfMetrics.ts` (NOT read-fonts/skrifa: the TS truth is
//! this exact parser, including its length guards — `length < 74` skips OS/2
//! entirely, `length >= 90` gates sxHeight/sCapHeight, zero-means-undefined —
//! and the `USE_TYPO_METRICS → sTypo else hhea else OS/2` choice rule at
//! ttfMetrics.ts:113-164. Matching the TS truth exactly is the DECISION §1.4
//! requirement; a real font library could disagree on malformed tables).

#[derive(Debug, Clone, PartialEq)]
pub struct Os2 {
    pub s_typo_ascender: i16,
    pub s_typo_descender: i16,
    pub s_typo_line_gap: i16,
    pub fs_selection: u16,
    pub sx_height: Option<i16>,
    pub s_cap_height: Option<i16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hhea {
    pub ascent: i16,
    pub descent: i16,
    pub line_gap: i16,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParsedTtfTables {
    pub units_per_em: Option<u16>,
    pub os2: Option<Os2>,
    pub hhea: Option<Hhea>,
}

/// `TtfMetrics` (ttfMetrics.ts:21-29).
#[derive(Debug, Clone, PartialEq)]
pub struct TtfMetrics {
    pub units_per_em: f64,
    pub ascent: f64,
    pub descent: f64,
    pub line_gap: f64,
    pub x_height: Option<f64>,
    pub cap_height: Option<f64>,
    /// 'os2' | 'hhea'
    pub source: &'static str,
}

fn u16be(data: &[u8], off: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*data.get(off)?, *data.get(off + 1)?]))
}
fn i16be(data: &[u8], off: usize) -> Option<i16> {
    Some(i16::from_be_bytes([*data.get(off)?, *data.get(off + 1)?]))
}
fn u32be(data: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_be_bytes([
        *data.get(off)?,
        *data.get(off + 1)?,
        *data.get(off + 2)?,
        *data.get(off + 3)?,
    ]))
}

/// `parseTtfForMetrics` (ttfMetrics.ts:91-111).
pub fn parse_ttf_for_metrics(data: &[u8]) -> ParsedTtfTables {
    let mut out = ParsedTtfTables::default();
    let Some(num_tables) = u16be(data, 4) else {
        return out;
    };
    let mut head: Option<(usize, usize)> = None;
    let mut os2: Option<(usize, usize)> = None;
    let mut hhea: Option<(usize, usize)> = None;
    let mut p = 12usize;
    for _ in 0..num_tables {
        let Some(tag) = data.get(p..p + 4) else { break };
        let (Some(offset), Some(length)) = (u32be(data, p + 8), u32be(data, p + 12)) else {
            break;
        };
        match tag {
            b"head" => head = Some((offset as usize, length as usize)),
            b"OS/2" => os2 = Some((offset as usize, length as usize)),
            b"hhea" => hhea = Some((offset as usize, length as usize)),
            _ => {}
        }
        p += 16;
    }
    if let Some((off, _)) = head {
        // parseHead (ttfMetrics.ts:50-56): unitsPerEm at +18, needs offset+20 bytes
        if data.len() >= off + 20 {
            if let Some(upm) = u16be(data, off + 18) {
                if upm > 0 {
                    out.units_per_em = Some(upm);
                }
            }
        }
    }
    if let Some((off, len)) = os2 {
        // parseOS2 (ttfMetrics.ts:58-80): requires table length >= 74
        if len >= 74 {
            if let (Some(fs_selection), Some(asc), Some(desc), Some(gap)) = (
                u16be(data, off + 62),
                i16be(data, off + 68),
                i16be(data, off + 70),
                i16be(data, off + 72),
            ) {
                let mut sx_height = None;
                let mut s_cap_height = None;
                if len >= 90 {
                    if let Some(x) = i16be(data, off + 86) {
                        if x > 0 {
                            sx_height = Some(x);
                        }
                    }
                    if let Some(c) = i16be(data, off + 88) {
                        if c > 0 {
                            s_cap_height = Some(c);
                        }
                    }
                }
                out.os2 = Some(Os2 {
                    fs_selection,
                    s_typo_ascender: asc,
                    s_typo_descender: desc,
                    s_typo_line_gap: gap,
                    sx_height,
                    s_cap_height,
                });
            }
        }
    }
    if let Some((off, len)) = hhea {
        // parseHhea (ttfMetrics.ts:82-89): requires length >= 10
        if len >= 10 {
            if let (Some(ascent), Some(descent), Some(line_gap)) = (
                i16be(data, off + 4),
                i16be(data, off + 6),
                i16be(data, off + 8),
            ) {
                out.hhea = Some(Hhea {
                    ascent,
                    descent,
                    line_gap,
                });
            }
        }
    }
    out
}

/// `chooseBaselineMetrics` (ttfMetrics.ts:113-164).
pub fn choose_baseline_metrics(parsed: &ParsedTtfTables) -> Option<TtfMetrics> {
    let units_per_em = parsed.units_per_em? as f64;
    if units_per_em <= 0.0 {
        return None;
    }
    const USE_TYPO_METRICS_BIT: u16 = 1 << 7;
    let use_os2_typo = parsed
        .os2
        .as_ref()
        .map(|o| (o.fs_selection & USE_TYPO_METRICS_BIT) != 0)
        .unwrap_or(false);
    let x_height = parsed
        .os2
        .as_ref()
        .and_then(|o| o.sx_height)
        .map(|v| v as f64);
    let cap_height = parsed
        .os2
        .as_ref()
        .and_then(|o| o.s_cap_height)
        .map(|v| v as f64);

    if use_os2_typo {
        let o = parsed.os2.as_ref().unwrap();
        return Some(TtfMetrics {
            units_per_em,
            ascent: o.s_typo_ascender as f64,
            descent: o.s_typo_descender as f64,
            line_gap: o.s_typo_line_gap as f64,
            x_height,
            cap_height,
            source: "os2",
        });
    }
    if let Some(h) = &parsed.hhea {
        return Some(TtfMetrics {
            units_per_em,
            ascent: h.ascent as f64,
            descent: h.descent as f64,
            line_gap: h.line_gap as f64,
            x_height,
            cap_height,
            source: "hhea",
        });
    }
    if let Some(o) = &parsed.os2 {
        return Some(TtfMetrics {
            units_per_em,
            ascent: o.s_typo_ascender as f64,
            descent: o.s_typo_descender as f64,
            line_gap: o.s_typo_line_gap as f64,
            x_height,
            cap_height,
            source: "os2",
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal sfnt with head/OS\/2/hhea carrying the given values.
    fn synth_ttf(fs_selection: u16, s_typo: (i16, i16, i16), hhea: (i16, i16, i16)) -> Vec<u8> {
        let mut data = vec![0u8; 12 + 3 * 16];
        data[4] = 0;
        data[5] = 3; // numTables
        let head_off = 12 + 3 * 16;
        let os2_off = head_off + 54;
        let hhea_off = os2_off + 96;
        let mut p = 12;
        for (tag, off, len) in [
            (b"OS/2", os2_off, 96usize),
            (b"head", head_off, 54usize),
            (b"hhea", hhea_off, 36usize),
        ] {
            data[p..p + 4].copy_from_slice(tag.as_slice());
            data[p + 8..p + 12].copy_from_slice(&(off as u32).to_be_bytes());
            data[p + 12..p + 16].copy_from_slice(&(len as u32).to_be_bytes());
            p += 16;
        }
        data.resize(hhea_off + 36, 0);
        // head.unitsPerEm @ +18
        data[head_off + 18..head_off + 20].copy_from_slice(&2048u16.to_be_bytes());
        // OS/2: fsSelection @ +62, sTypo @ +68/70/72, sxHeight @ +86
        data[os2_off + 62..os2_off + 64].copy_from_slice(&fs_selection.to_be_bytes());
        data[os2_off + 68..os2_off + 70].copy_from_slice(&s_typo.0.to_be_bytes());
        data[os2_off + 70..os2_off + 72].copy_from_slice(&s_typo.1.to_be_bytes());
        data[os2_off + 72..os2_off + 74].copy_from_slice(&s_typo.2.to_be_bytes());
        data[os2_off + 86..os2_off + 88].copy_from_slice(&1000i16.to_be_bytes());
        // hhea: ascent/descent/lineGap @ +4/6/8
        data[hhea_off + 4..hhea_off + 6].copy_from_slice(&hhea.0.to_be_bytes());
        data[hhea_off + 6..hhea_off + 8].copy_from_slice(&hhea.1.to_be_bytes());
        data[hhea_off + 8..hhea_off + 10].copy_from_slice(&hhea.2.to_be_bytes());
        data
    }

    /// Pins the `USE_TYPO_METRICS → sTypo else hhea` rule
    /// (ttfMetrics.ts:113-148) with DIVERGENT table values — the vendored
    /// corpus cannot distinguish the branch (all 19 TTFs have hhea == sTypo;
    /// verified against the production parser 2026-06-10), so the oracle gate
    /// alone would let a branch regression through.
    #[test]
    fn use_typo_metrics_bit_selects_os2_else_hhea() {
        let with_bit = synth_ttf(1 << 7, (1984, -494, 0), (2728, -680, 0));
        let parsed = parse_ttf_for_metrics(&with_bit);
        let m = choose_baseline_metrics(&parsed).unwrap();
        assert_eq!((m.source, m.ascent, m.descent), ("os2", 1984.0, -494.0));
        assert_eq!(m.x_height, Some(1000.0));

        let without_bit = synth_ttf(0, (1984, -494, 0), (2728, -680, 0));
        let parsed = parse_ttf_for_metrics(&without_bit);
        let m = choose_baseline_metrics(&parsed).unwrap();
        assert_eq!((m.source, m.ascent, m.descent), ("hhea", 2728.0, -680.0));
        // xHeight still flows from OS/2 even on the hhea branch (ttfMetrics.ts:120-122)
        assert_eq!(m.x_height, Some(1000.0));
    }

    /// OS/2 table shorter than 74 bytes is ignored entirely
    /// (ttfMetrics.ts:61) — falls to hhea.
    #[test]
    fn short_os2_table_is_skipped() {
        let mut data = synth_ttf(1 << 7, (1984, -494, 0), (2728, -680, 0));
        // shrink the recorded OS/2 length below 74
        data[12 + 12..12 + 16].copy_from_slice(&60u32.to_be_bytes());
        let parsed = parse_ttf_for_metrics(&data);
        assert!(parsed.os2.is_none());
        let m = choose_baseline_metrics(&parsed).unwrap();
        assert_eq!(m.source, "hhea");
    }
}

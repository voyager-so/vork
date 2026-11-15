//! Edit-time checks of an agent shader: the size and schema limits, the source rules (no bindings or entry points,
//! no `fract(sin(…))` hashes), naga's parse and validation of the module the placement compiles to, the `shade`
//! signature and constant loop bounds. Every error carries the line and column in the agent's own source.

use crate::{loops, module, parse_schema, Module, Placement, MAX_SOURCE_BYTES};

/// A check result: an error or warning code, a message, where it is in the agent's source (1-based), and a hint.
#[derive(Clone, Debug, PartialEq)]
pub struct Diag {
    /// `shader_invalid`, `shader_nondeterministic`, `shader_unbounded_loop` (errors); `shader_derivative` (warning).
    pub code: &'static str,
    pub message: String,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub hint: Option<String>,
}

impl Diag {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self { code: "shader_invalid", message: message.into(), line: None, column: None, hint: None }
    }

    fn at(mut self, pos: Option<(u32, u32)>) -> Self {
        if let Some((l, c)) = pos {
            self.line = Some(l);
            self.column = Some(c);
        }
        self
    }

    fn hint(mut self, h: impl Into<String>) -> Self {
        self.hint = Some(h.into());
        self
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("code".into(), self.code.into());
        m.insert("message".into(), self.message.clone().into());
        if let Some(l) = self.line {
            m.insert("line".into(), l.into());
        }
        if let Some(c) = self.column {
            m.insert("column".into(), c.into());
        }
        if let Some(h) = &self.hint {
            m.insert("hint".into(), h.clone().into());
        }
        serde_json::Value::Object(m)
    }
}

/// Everything [`check`] found.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub errors: Vec<Diag>,
    pub warnings: Vec<Diag>,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "ok": self.ok(),
            "errors": self.errors.iter().map(Diag::to_json).collect::<Vec<_>>(),
            "warnings": self.warnings.iter().map(Diag::to_json).collect::<Vec<_>>(),
            "std": crate::STD_VERSION,
        })
    }
}

/// 1-based (line, column) of byte `at` in `s`.
fn line_col(s: &str, at: usize) -> (u32, u32) {
    let at = at.min(s.len());
    let prefix = &s[..at];
    let line = prefix.matches('\n').count() as u32 + 1;
    let col = (at - prefix.rfind('\n').map_or(0, |i| i + 1)) as u32 + 1;
    (line, col)
}

/// The source with comments blanked (same length, newlines kept), so the source rules don't match inside them.
fn strip_comments(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = b.to_vec();
    let mut i = 0;
    let mut depth = 0usize;
    while i < b.len() {
        if depth == 0 && b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                out[i] = b' ';
                i += 1;
            }
            continue;
        }
        if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            depth += 1;
            out[i] = b' ';
            out[i + 1] = b' ';
            i += 2;
            continue;
        }
        if depth > 0 && b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
            depth -= 1;
            out[i] = b' ';
            out[i + 1] = b' ';
            i += 2;
            continue;
        }
        if depth > 0 && b[i] != b'\n' {
            out[i] = b' ';
        }
        i += 1;
    }
    // Only ASCII bytes were replaced, by ASCII: still UTF-8 except inside blanked multi-byte runs; map those too.
    String::from_utf8(out).unwrap_or_else(|e| {
        e.into_bytes().into_iter().map(|c| if c.is_ascii() { c as char } else { ' ' }).collect()
    })
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Byte offsets where word `w` appears as a whole token.
fn find_word(s: &str, w: &str) -> Vec<usize> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = s[from..].find(w) {
        let at = from + i;
        let end = at + w.len();
        let before = at == 0 || !is_ident(b[at - 1]);
        let after = end >= b.len() || !is_ident(b[end]);
        if before && after {
            out.push(at);
        }
        from = at + w.len().max(1);
    }
    out
}

/// After `at`, skipping whitespace, the text starts with `next`: the offset past it.
fn followed_by(s: &str, at: usize, next: &str) -> Option<usize> {
    let rest = &s[at..];
    let t = rest.trim_start();
    t.starts_with(next).then(|| at + (rest.len() - t.len()) + next.len())
}

fn source_rules(src: &str, report: &mut Report) {
    let s = strip_comments(src);
    // `fract(sin(x) * k)` and friends: a float hash whose low bits differ by GPU and compiler.
    for at in find_word(&s, "fract") {
        let Some(open) = followed_by(&s, at + "fract".len(), "(") else { continue };
        for f in ["sin", "cos"] {
            if let Some(p) = followed_by(&s, open, f) {
                if followed_by(&s, p, "(").is_some() {
                    report.errors.push(
                        Diag {
                            code: "shader_nondeterministic",
                            message: format!("`fract({f}(…))` is a float hash: its low bits differ between GPUs, so the picture would too"),
                            line: None,
                            column: None,
                            hint: None,
                        }
                        .at(Some(line_col(src, at)))
                        .hint("use the integer hashes: std::hash21(p, seed()), std::hash22, std::hash31, or the noises (std::value, std::simplex, std::worley, std::fbm)"),
                    );
                }
            }
        }
    }
    // The bind layout is fixed: a shader declares no resources, entry points or extensions.
    for (word, what) in [
        ("@group", "a binding"),
        ("@binding", "a binding"),
        ("@compute", "an entry point"),
        ("@vertex", "an entry point"),
        ("@fragment", "an entry point"),
        ("@workgroup_size", "an entry point"),
        ("enable", "an extension"),
        ("requires", "an extension"),
        ("override", "a pipeline override"),
    ] {
        let hits = if word.starts_with('@') {
            let mut v = Vec::new();
            let mut from = 0;
            while let Some(i) = s[from..].find(word) {
                let at = from + i;
                let end = at + word.len();
                if end >= s.len() || !is_ident(s.as_bytes()[end]) {
                    v.push(at);
                }
                from = end;
            }
            v
        } else {
            find_word(&s, word)
        };
        for at in hits {
            report.errors.push(
                Diag::invalid(format!("`{word}`: a shader can't declare {what}"))
                    .at(Some(line_col(src, at)))
                    .hint("write `fn shade(p: vec2<f32>) -> vec4<f32>` and helper functions; inputs come from the accessors (src, input, coverage, backdrop) and parameters from the schema"),
            );
        }
    }
    for (decl, what) in [("var<storage", "storage"), ("var<uniform", "uniform"), ("var<workgroup", "workgroup")] {
        let mut from = 0;
        while let Some(i) = s[from..].find(decl) {
            let at = from + i;
            report.errors.push(
                Diag::invalid(format!("a shader can't declare {what} variables"))
                    .at(Some(line_col(src, at)))
                    .hint("declare parameters in the schema; they are functions in the shader (`amount()`)"),
            );
            from = at + decl.len();
        }
    }
    for w in ["dpdx", "dpdy", "fwidth", "dpdxCoarse", "dpdyCoarse", "dpdxFine", "dpdyFine", "fwidthCoarse", "fwidthFine"] {
        if let Some(&at) = find_word(&s, w).first() {
            report.warnings.push(
                Diag {
                    code: "shader_derivative",
                    message: format!("`{w}` depends on how the GPU groups pixels: it may differ slightly between GPUs"),
                    line: None,
                    column: None,
                    hint: None,
                }
                .at(Some(line_col(src, at)))
                .hint("for an antialiased edge of a distance d in pixels, std::fill(d) needs no derivative"),
            );
        }
    }
}

/// The agent-source position of `span`, or None when it's outside the agent's source.
fn user_pos(m: &Module, span: naga::Span) -> Option<(u32, u32)> {
    let r = span.to_range()?;
    if r.start < m.user.start || r.start > m.user.end {
        return None;
    }
    Some(line_col(&m.wgsl[m.user.clone()], r.start - m.user.start))
}

fn clean(msg: &str) -> String {
    msg.replace("std__", "std::")
}

/// A hint for a few common naga messages.
fn hint_for(msg: &str, placement: Placement) -> Option<String> {
    if msg.contains("no definition in scope") || msg.contains("unknown") && msg.contains("identifier") {
        let acc = match placement {
            Placement::Pass => "src(p), input(i, p)",
            Placement::Fill => "nothing beyond the common accessors (the renderer clips a fill to its shape or glyphs)",
            Placement::Effect => "src(p) (the layer as drawn; backdrop(p) isn't built yet)",
            Placement::Transition => "from_scene(p), to_scene(p), progress()",
        };
        return Some(format!(
            "this placement ({}) has {acc}, and frame_size(), time(), px_scale(), uv(p), bounds(), seed(), the schema's parameters as functions, and std::* (api shader.std)",
            placement.name()
        ));
    }
    if msg.contains("automatic conversion") || msg.contains("type mismatch") || msg.contains("expected") && msg.contains("found") {
        return Some("WGSL never converts types: write f32(i), vec2<f32>(x, y), 1.0 (not 1) for floats".into());
    }
    None
}

/// Check `source` as a shader for `placement` with `schema` (a code object's schema JSON).
pub fn check(source: &str, schema: &serde_json::Value, placement: Placement) -> Report {
    let mut report = Report::default();
    if source.len() > MAX_SOURCE_BYTES {
        report.errors.push(Diag::invalid(format!(
            "the source is {} bytes; a shader is at most {} KB",
            source.len(),
            MAX_SOURCE_BYTES / 1024
        )));
        return report;
    }
    let schema = match parse_schema(schema) {
        Ok(s) => s,
        Err(e) => {
            report.errors.push(e);
            return report;
        }
    };
    if placement != Placement::Pass && !schema.images.is_empty() {
        report.errors.push(Diag::invalid(format!(
            "image parameters ({}) are inputs of a frame pass; a {} reads its own inputs",
            schema.images.join(", "),
            placement.name()
        )));
    }
    source_rules(source, &mut report);
    if !report.errors.is_empty() {
        return report;
    }
    if !find_word(&strip_comments(source), "shade").iter().any(|&at| source[..at].trim_end().ends_with("fn")) {
        report.errors.push(
            Diag::invalid("the shader has no `fn shade`")
                .hint("write `fn shade(p: vec2<f32>) -> vec4<f32> { … }` returning premultiplied linear RGBA"),
        );
        return report;
    }
    let mut modules = vec![("", module(source, &schema, placement))];
    if placement == Placement::Fill {
        modules.push((" (as a fill on text)", crate::fill_text_module(source, &schema)));
    }
    for (what, m) in modules {
        let before = report.errors.len();
        check_module(&m, placement, &mut report);
        if report.errors.len() > before {
            for e in &mut report.errors[before..] {
                e.message.push_str(what);
            }
            break;
        }
    }
    report
}

fn check_module(m: &Module, placement: Placement, report: &mut Report) {
    let parsed = match naga::front::wgsl::parse_str(&m.wgsl) {
        Ok(module) => module,
        Err(e) => {
            let pos = e.labels().find_map(|(span, _)| user_pos(m, span));
            let label = e.labels().find(|(span, _)| user_pos(m, *span).is_some()).map(|(_, l)| l.to_string());
            let mut msg = clean(e.message());
            if let Some(l) = label.filter(|l| !l.is_empty()) {
                msg = format!("{msg} ({})", clean(&l));
            }
            let hint = hint_for(&msg, placement);
            let mut d = Diag::invalid(msg).at(pos);
            d.hint = hint;
            report.errors.push(d);
            return;
        }
    };
    // `shade` must exist with the ABI's signature.
    let shade = parsed.functions.iter().find(|(_, f)| f.name.as_deref() == Some("shade"));
    match shade {
        None => {
            report.errors.push(
                Diag::invalid("the shader has no `fn shade`")
                    .hint("write `fn shade(p: vec2<f32>) -> vec4<f32> { … }` returning premultiplied linear RGBA"),
            );
            return;
        }
        Some((h, f)) => {
            let vec_of = |ty: naga::Handle<naga::Type>, size: naga::VectorSize| {
                matches!(parsed.types[ty].inner,
                    naga::TypeInner::Vector { size: s, scalar } if s == size && scalar == naga::Scalar::F32)
            };
            let ok = f.arguments.len() == 1
                && vec_of(f.arguments[0].ty, naga::VectorSize::Bi)
                && f.result.as_ref().is_some_and(|r| vec_of(r.ty, naga::VectorSize::Quad));
            if !ok {
                report.errors.push(
                    Diag::invalid("`shade` must be `fn shade(p: vec2<f32>) -> vec4<f32>`")
                        .at(user_pos(m, parsed.functions.get_span(h))),
                );
                return;
            }
        }
    }
    let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::default());
    if let Err(e) = validator.validate(&parsed) {
        // The narrowest span in the agent's source: validation errors carry the enclosing function's span too.
        let pos = e
            .spans()
            .filter(|(span, _)| user_pos(m, *span).is_some())
            .min_by_key(|(span, _)| span.to_range().map_or(usize::MAX, |r| r.len()))
            .and_then(|(span, _)| user_pos(m, *span));
        let mut msg = clean(&e.as_inner().to_string());
        // The inner error's source chain carries the specific cause.
        let mut cause: Option<&dyn std::error::Error> = std::error::Error::source(e.as_inner());
        while let Some(c) = cause {
            msg.push_str(": ");
            msg.push_str(&clean(&c.to_string()));
            cause = c.source();
        }
        let hint = hint_for(&msg, placement);
        let mut d = Diag::invalid(msg).at(pos);
        d.hint = hint;
        report.errors.push(d);
        return;
    }
    // Loops in the agent's own functions need a constant bound.
    for (h, f) in parsed.functions.iter() {
        let span = parsed.functions.get_span(h);
        let Some(r) = span.to_range() else { continue };
        if r.start < m.user.start || r.start >= m.user.end {
            continue;
        }
        for bad in loops::unbounded(&parsed, f) {
            let pos = user_pos(m, bad.span).or_else(|| user_pos(m, span));
            report.errors.push(
                Diag {
                    code: "shader_unbounded_loop",
                    message: format!(
                        "in `{}`: {}",
                        f.name.as_deref().unwrap_or("?"),
                        bad.reason
                    ),
                    line: None,
                    column: None,
                    hint: None,
                }
                .at(pos)
                .hint(format!(
                    "write `for (var i = 0; i < N; i++)` with a constant N ≤ {} (a literal or a `const`), and `break` early for a parameter-driven count",
                    crate::MAX_LOOP_ITERATIONS
                )),
            );
        }
    }
}

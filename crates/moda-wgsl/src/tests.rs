use super::*;
use serde_json::json;

const ALL: [Placement; 4] = [Placement::Pass, Placement::Fill, Placement::Effect, Placement::Transition];

fn errors(src: &str, schema: serde_json::Value, placement: Placement) -> Vec<Diag> {
    check(src, &schema, placement).errors
}

fn only_error(src: &str, schema: serde_json::Value, placement: Placement) -> Diag {
    let e = errors(src, schema, placement);
    assert_eq!(e.len(), 1, "{e:#?}");
    e.into_iter().next().unwrap()
}

#[test]
fn the_prelude_and_every_host_validate() {
    let src = "fn shade(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(0.0); }";
    for pl in ALL {
        let r = check(src, &json!({}), pl);
        assert!(r.ok(), "{pl:?}: {:#?}", r.errors);
    }
}

#[test]
fn every_std_function_is_callable() {
    let src = r#"
fn shade(p: vec2<f32>) -> vec4<f32> {
    let s = seed();
    var v = std::hash21(p, s) + std::hash22(p, s).x + std::hash31(vec3<f32>(p, time()), s);
    v += std::value(p, s) + std::simplex(p, s) + std::worley(p, s).x + std::fbm(p, 5u, s) + std::curl(p, s).y;
    v += std::sd_circle(p, 3.0) + std::sd_box(p, vec2<f32>(2.0)) + std::sd_round_box(p, vec2<f32>(2.0), 0.5);
    v += std::sd_segment(p, vec2<f32>(0.0), vec2<f32>(1.0)) + std::sd_ring(p, 4.0, 1.0);
    v += std::op_union(1.0, 2.0) + std::op_subtract(1.0, 2.0) + std::op_intersect(1.0, 2.0);
    v += std::op_smooth_union(1.0, 2.0, 0.5) + std::fill(v) + std::rotate(p, 1.0).x + std::luma(vec3<f32>(v));
    let c = std::mix_oklab(vec4<f32>(1.0, 0.0, 0.0, 1.0), vec4<f32>(0.0, 0.0, 1.0, 1.0), 0.5);
    let m = std::merge(std::premul(c), std::unpremul(vec4<f32>(v)), std::SCREEN);
    return m + std::debug(v) + vec4<f32>(std::from_oklab(std::oklab(c.rgb)), 0.0);
}
"#;
    for pl in ALL {
        let r = check(src, &json!({}), pl);
        assert!(r.ok(), "{pl:?}: {:#?}", r.errors);
    }
}

/// The guide's three worked examples (packages/sdk guide.shaders): each must pass in its placement.
#[test]
fn the_guide_examples_pass() {
    let fill = r#"
// A fill: stripes along a direction, colours from the schema.
fn shade(p: vec2<f32>) -> vec4<f32> {
    let d = std::rotate(p, angle());
    let t = 0.5 + 0.5 * sin(d.x / width() * 6.2831853);
    return std::palette(t);
}
"#;
    let r = check(fill, &json!({"angle": {"type": "number", "default": 0.4}, "width": {"type": "number", "default": 24},
        "colors": {"type": "colors", "default": ["#1b1f3a", "#f2c14e"]}}), Placement::Fill);
    assert!(r.ok(), "{:#?}", r.errors);
    let effect = r#"
// An effect: displace the layer by noise, add seeded grain.
fn shade(p: vec2<f32>) -> vec4<f32> {
    let n = std::fbm(p / 60.0, 4u, seed());
    let c = src(p + vec2<f32>(n, -n) * amount());
    let g = (std::hash21(p, seed() + 1u) - 0.5) * grain();
    return vec4<f32>(c.rgb + g * c.a, c.a);
}
"#;
    let r = check(effect, &json!({"amount": {"default": 6}, "grain": {"default": 0.08}}), Placement::Effect);
    assert!(r.ok(), "{:#?}", r.errors);
    let transition = r#"
// A transition: a soft-edged circle grows from the centre.
fn shade(p: vec2<f32>) -> vec4<f32> {
    let c = frame_size() * 0.5;
    let r = progress() * length(frame_size()) * 0.6;
    let k = std::fill((length(p - c) - r) / softness());
    return mix(from_scene(p), to_scene(p), k);
}
"#;
    let r = check(transition, &json!({"softness": {"default": 20}}), Placement::Transition);
    assert!(r.ok(), "{:#?}", r.errors);
}

#[test]
fn a_syntax_error_is_at_the_agents_line_and_column() {
    let src = "fn shade(p: vec2<f32>) -> vec4<f32> {\n    let a = 1.0\n    return vec4<f32>(a);\n}\n";
    let e = only_error(src, json!({}), Placement::Pass);
    assert_eq!(e.code, "shader_invalid");
    assert_eq!(e.line, Some(3), "{e:#?}");
}

#[test]
fn an_unknown_name_has_a_hint_naming_the_placements_accessors() {
    let src = "fn shade(p: vec2<f32>) -> vec4<f32> {\n  return backdrop(p);\n}\n";
    let e = only_error(src, json!({}), Placement::Fill);
    assert_eq!(e.code, "shader_invalid");
    assert_eq!((e.line, e.column), (Some(2), Some(10)), "{e:#?}");
    assert!(e.hint.as_deref().unwrap().contains("fill"), "{e:#?}");
    // backdrop() is planned for effects, not built: an effect that reads it is told so.
    let e = only_error(src, json!({}), Placement::Effect);
    assert!(e.hint.as_deref().unwrap().contains("backdrop(p) isn't built yet"), "{e:#?}");
}

#[test]
fn a_type_error_is_located_in_the_agents_source() {
    let src = "fn shade(p: vec2<f32>) -> vec4<f32> {\n  let x: f32 = 1u;\n  return vec4<f32>(x);\n}\n";
    let e = only_error(src, json!({}), Placement::Pass);
    assert_eq!(e.line, Some(2), "{e:#?}");
}

#[test]
fn std_names_read_as_written() {
    let src = "fn shade(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(std::nope(p)); }";
    let e = only_error(src, json!({}), Placement::Pass);
    assert!(e.message.contains("std::nope"), "{e:#?}");
    assert!(!e.message.contains("std__"), "{e:#?}");
}

#[test]
fn float_hashes_are_nondeterministic() {
    let src = "fn shade(p: vec2<f32>) -> vec4<f32> {\n  let n = fract( sin(dot(p, vec2<f32>(12.9, 78.2))) * 43758.5);\n  return vec4<f32>(n);\n}\n";
    let e = only_error(src, json!({}), Placement::Pass);
    assert_eq!(e.code, "shader_nondeterministic");
    assert_eq!((e.line, e.column), (Some(2), Some(11)));
    assert!(e.hint.unwrap().contains("std::hash21"));
    // In a comment it doesn't count.
    let ok = "// never fract(sin(x))\nfn shade(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(fract(p.x)); }";
    assert!(check(ok, &json!({}), Placement::Pass).ok());
}

#[test]
fn bindings_and_entry_points_are_refused() {
    for (src, line) in [
        ("@group(0) @binding(9) var t: texture_2d<f32>;\nfn shade(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(0.0); }", 1),
        ("fn shade(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(0.0); }\n@compute @workgroup_size(1) fn main() {}", 2),
        ("var<storage, read_write> b: array<u32>;\nfn shade(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(0.0); }", 1),
    ] {
        let e = errors(src, json!({}), Placement::Pass);
        assert!(!e.is_empty() && e.iter().all(|e| e.code == "shader_invalid"), "{e:#?}");
        assert_eq!(e[0].line, Some(line));
    }
}

#[test]
fn shade_must_have_the_abis_signature() {
    let e = only_error("fn shade(p: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.0); }", json!({}), Placement::Pass);
    assert!(e.message.contains("fn shade(p: vec2<f32>) -> vec4<f32>"), "{e:#?}");
    let e = only_error("fn main(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(0.0); }", json!({}), Placement::Pass);
    assert!(e.message.contains("no `fn shade`"), "{e:#?}");
}

#[test]
fn counted_loops_pass() {
    for body in [
        "for (var i = 0; i < 16; i++) { s += 1.0; }",
        "for (var i = 0u; i <= 1023u; i++) { s += 1.0; }",
        "for (var i = 10; i > 0; i -= 2) { s += 1.0; }",
        "for (var i = 0; i < 64; i++) { if (f32(i) >= n()) { break; } s += 1.0; }",
        "for (var i = 0; i < N; i++) { for (var j = 0; j < N; j++) { s += 1.0; } }",
        "for (var i = 0.0; i < 1.0; i += 0.125) { s += i; }",
        "for (var i = 4294967293u; i < 4294967295u; i++) { s += 1.0; }",
        "for (var i = 16777216.0; i < 16777220.0; i += 2.0) { s += i; }",
    ] {
        let src = format!("const N = 32;\nfn shade(p: vec2<f32>) -> vec4<f32> {{\n  var s = 0.0;\n  {body}\n  return vec4<f32>(s);\n}}\n");
        let r = check(&src, &json!({"n": {"default": 3}}), Placement::Pass);
        assert!(r.ok(), "{body}: {:#?}", r.errors);
    }
}

#[test]
fn unbounded_loops_are_refused_with_a_hint() {
    for (body, why) in [
        ("var i = 0; while (i < 8) { i++; }", "isn't stepped"),
        ("loop { s += 1.0; if (s > 3.0) { break; } }", "no bound"),
        ("for (var i = 0; f32(i) < n(); i++) { s += 1.0; }", "isn't a constant"),
        ("for (var i = 0; i < 5000; i++) { s += 1.0; }", "runs 5000 times"),
        ("for (var i = 0; i < 8; i--) { s += 1.0; }", "moves away"),
        ("for (var i = 0; i < 8; i++) { i = 0; }", "changed inside"),
        ("for (var i = 5u; i >= 0u; i--) { s += 1.0; }", "wraps below 0"),
        ("for (var i = 0; i != 8; i++) { s += 1.0; }", "=="),
    ] {
        let src = format!("fn shade(p: vec2<f32>) -> vec4<f32> {{\n  var s = 0.0;\n  {body}\n  return vec4<f32>(s);\n}}\n");
        let e = errors(&src, json!({"n": {"default": 3}}), Placement::Pass);
        assert!(!e.is_empty(), "{body}: no error");
        let e = &e[0];
        assert_eq!(e.code, "shader_unbounded_loop", "{body}: {e:#?}");
        assert!(e.message.contains(why), "{body}: {e:#?}");
        assert_eq!(e.line, Some(3), "{body}: {e:#?}");
        assert!(e.hint.as_deref().unwrap().contains("for (var i = 0; i < N; i++)"));
    }
}

#[test]
fn derivatives_warn() {
    let r = check("fn shade(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(fwidth(p.x)); }", &json!({}), Placement::Pass);
    assert!(r.ok());
    assert_eq!(r.warnings[0].code, "shader_derivative");
}

#[test]
fn schemas() {
    let s = parse_schema(&json!({"a": {"type": "number"}, "b": {"type": "vec2"}, "c": {"type": "color"},
        "d": {"type": "bool"}, "e": {"type": "colors", "default": ["#000", "#fff", "#f00"]}, "src": {"type": "image"}}));
    // `src` is an accessor: an image input can't use that name as a parameter … but images are inputs, not params.
    assert!(s.is_err());
    let s = parse_schema(&json!({"a": {"type": "number"}, "b": {"type": "vec2"}, "c": {"type": "color"},
        "d": {"type": "bool"}, "e": {"type": "colors", "default": ["#000", "#fff", "#f00"]}, "img": {"type": "image"}}))
    .unwrap();
    assert_eq!(s.images, vec!["img"]);
    assert_eq!(s.slots(), vec![0, 1, 2, 3, 4]);
    assert_eq!(s.slot_count(), 8);
    for bad in [
        json!({"fn": {}}),
        json!({"time": {}}),
        json!({"luminance": {}}),
        json!({"a": {"type": "mat4"}}),
        json!({"a": {"type": "colors"}}),
        json!({"a": {"type": "colors", "count": 9}}),
    ] {
        assert_eq!(parse_schema(&bad).unwrap_err().code, "shader_invalid", "{bad}");
    }
    let many: serde_json::Map<String, serde_json::Value> = (0..31).map(|i| (format!("p{i}"), json!({}))).collect();
    assert!(parse_schema(&serde_json::Value::Object(many)).is_err());
}

#[test]
fn image_inputs_belong_to_a_pass() {
    let src = "fn shade(p: vec2<f32>) -> vec4<f32> { return src(p); }";
    assert!(check(src, &json!({"img": {"type": "image"}}), Placement::Pass).ok());
    assert!(!check(src, &json!({"img": {"type": "image"}}), Placement::Effect).ok());
}

#[test]
fn parameters_and_palette_are_accessors() {
    let src = "fn shade(p: vec2<f32>) -> vec4<f32> {\n  if (on()) { return tint() * amount(); }\n  return ramp(ramp_count() - 1u) + std::palette(uv(p).x) + vec4<f32>(off(), 0.0, 1.0);\n}";
    let schema = json!({"amount": {}, "off": {"type": "vec2"}, "tint": {"type": "color"}, "on": {"type": "bool"},
        "ramp": {"type": "colors", "count": 3}});
    for pl in ALL {
        let r = check(src, &schema, pl);
        assert!(r.ok(), "{pl:?}: {:#?}", r.errors);
    }
}

#[test]
fn a_fill_is_checked_on_text_too() {
    // `curvePoints` is a text-host global: harmless on shapes, a redefinition on text. Named in the reserved list, so
    // as a parameter it's refused up front; as the agent's own global it's caught on the text module.
    let src = "var<private> curvePoints: f32;\nfn shade(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(curvePoints); }";
    let e = errors(src, json!({}), Placement::Fill);
    assert!(!e.is_empty());
    assert!(e[0].message.contains("on text"), "{e:#?}");
    assert!(check(src, &json!({}), Placement::Pass).ok());
}

#[test]
fn sizes_are_limited() {
    let big = format!("fn shade(p: vec2<f32>) -> vec4<f32> {{ return vec4<f32>(0.0); }}\n//{}", "x".repeat(MAX_SOURCE_BYTES));
    assert!(only_error(&big, json!({}), Placement::Pass).message.contains("64 KB"));
}

#[test]
fn host_slots_encode_both_directions() {
    let h = HostSlots {
        frame_size: [100.0, 50.0],
        time: 1.5,
        px_scale: 2.0,
        seed: 0xdead_beef,
        progress: 0.25,
        bounds: [0.0, 0.0, 100.0, 50.0],
        to_out: [2.0, 0.0, 0.0, 2.0, 10.0, 20.0],
    };
    let s = h.encode().unwrap();
    assert_eq!(s[1][0].to_bits(), 0xdead_beef);
    assert_eq!(s[3], [0.5, 0.0, 0.0, 0.5]);
    assert_eq!(s[4][..2], [-5.0, -10.0]);
    assert_eq!(s[5], [2.0, 0.0, 0.0, 2.0]);
    assert!(HostSlots { to_out: [0.0; 6], ..h }.encode().is_none());
}

#[test]
fn seeds_mix_the_code_hash() {
    let a = mix_seed(1, "b3:aa");
    assert_eq!(a, mix_seed(1, "b3:aa"));
    assert_ne!(a, mix_seed(2, "b3:aa"));
    assert_ne!(a, mix_seed(1, "b3:ab"));
}

#[test]
fn params_encode_in_schema_order() {
    let s = parse_schema(&json!({"a": {"default": 2}, "c": {"type": "color"}, "r": {"type": "colors", "count": 2},
        "v": {"type": "vec2"}}))
    .unwrap();
    let vals = |n: &str| match n {
        "c" => Some(json!([1.0, 0.5, 0.0, 0.5])),
        "r" => Some(json!([[1.0, 0.0, 0.0, 1.0]])),
        "v" => Some(json!([3, 4])),
        _ => None,
    };
    let color = |v: &serde_json::Value| {
        let a = v.as_array()?;
        Some([a[0].as_f64()? as f32, a[1].as_f64()? as f32, a[2].as_f64()? as f32, a[3].as_f64()? as f32])
    };
    let p = encode_params(&s, &vals, &color);
    assert_eq!(p, vec![
        [2.0, 0.0, 0.0, 0.0],
        [0.5, 0.25, 0.0, 0.5],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [3.0, 4.0, 0.0, 0.0],
    ]);
    let (sp, cd) = encode_fill(&HostSlots { frame_size: [10.0, 20.0], time: 3.0, px_scale: 1.0, seed: 7,
        progress: 0.0, bounds: [0.0; 4], to_out: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] }, &p);
    assert_eq!(sp[0][0], 3.0);
    assert_eq!(sp[1][..2], [10.0, 20.0]);
    assert_eq!(sp[2], p[0]);
    assert_eq!(cd[0], [0.0; 4]);
}

/// The fill hosts are copies of moda-shaders' production modules around their shader body: pinned here so the
/// copies can't drift from what the engine's pipelines expect.
#[test]
fn fill_hosts_match_the_production_modules() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../moda-shaders/wgsl");
    let shape = std::fs::read_to_string(dir.join("plasma-field.wgsl")).unwrap();
    let body = shape.find("// Shader ID: plasma-field").unwrap();
    assert_eq!(&shape[..body], FILL_HOST_HEAD);
    assert_eq!(&shape[shape.find("fn shadeStroke(").unwrap()..], FILL_HOST_TAIL);
    let text = std::fs::read_to_string(dir.join("plasma-field.vector-text-fill.wgsl")).unwrap();
    let inc = std::fs::read_to_string(dir.join("include/vector-text-coverage.wgsl")).unwrap();
    let text = text.replace("// #include \"vector-text-coverage.wgsl\"", inc.trim_end_matches('\n'));
    let body = text.find("// Shader ID: plasma-field").unwrap();
    assert_eq!(&text[..body], TEXT_HOST_HEAD);
    assert_eq!(&text[text.find("@fragment fn fs(").unwrap()..], TEXT_HOST_TAIL);
}

#[test]
fn reports_serialise() {
    let r = check("fn shade(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(fract(sin(p.x))); }", &json!({}), Placement::Pass);
    let j = r.to_json();
    assert_eq!(j["ok"], false);
    assert_eq!(j["errors"][0]["code"], "shader_nondeterministic");
    assert_eq!(j["errors"][0]["line"], 1);
    assert_eq!(j["std"], STD_VERSION);
}

#[test]
fn std_source_reads_as_agents_call_it() {
    let s = std_source();
    assert!(s.contains("fn std::fbm(") && !s.contains("std__"));
}

#[test]
fn a_comparison_of_mixed_types_is_located_at_the_comparison() {
    let src = "fn shade(p: vec2<f32>) -> vec4<f32> {\n  var s = 0.0;\n  let i = 3;\n  if (i < 2.5) { s = 1.0; }\n  return vec4<f32>(s);\n}\n";
    let e = only_error(src, json!({}), Placement::Pass);
    assert_eq!(e.code, "shader_invalid");
    assert_eq!(e.line, Some(4), "{e:#?}");
}

#[test]
fn machine_arithmetic_cannot_invalidate_loop_bounds() {
    for body in [
        "for (var i = 4294967294u; i <= 4294967295u; i++) {}",
        "for (var i = 16777216.0; i < 16777218.0; i += 1.0) {}",
        "for (var i = 0.0; i < 0x1p-126; i += 0x1p-127) {}",
        "for (var i = 2147483646i; i <= 2147483647i; i++) {}",
        "for (var i = -2147483647i; i >= (-2147483647i - 1i); i--) {}",
    ] {
        let src = format!("fn shade(p: vec2<f32>) -> vec4<f32> {{ {body} return vec4<f32>(0.0); }}");
        let r = check(&src, &json!({}), Placement::Pass);
        assert!(r.errors.iter().any(|e| e.code == "shader_unbounded_loop"), "{body}: {:#?}", r.errors);
    }
}

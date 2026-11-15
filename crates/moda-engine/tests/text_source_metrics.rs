use moda_engine::text_metrics::metrics_from_layout;
use serde_json::{json, Value};

fn segment(text: &str, source: &[usize], x: f64) -> Value {
    json!({"text":text,"sourceIndices":source,"x":x,"width":text.chars().count() as f64 * 10.,
        "fontSize":20.,"glyphs":text.chars().enumerate().map(|(i,_)|
            json!({"c":i,"x":x+i as f64*10.,"a":10.})).collect::<Vec<_>>()})
}
fn dump(segments: Vec<Value>) -> Value {
    json!({"paragraphs":[{"lines":[{"segments":segments,"baselineY":20.,"ascent":16.,"descent":4.}]}]})
}

#[test]
fn preserved_spaces_and_sanitized_holes_keep_the_following_metrics() {
    for (source, origin) in [("A\u{a0}\u{a0}B",3), ("A\u{2060}\u{2060}B",3)] {
        let metrics = metrics_from_layout(source, &dump(vec![segment("A",&[0],0.),segment("B",&[origin],40.)]));
        assert_eq!(metrics.chars.len(),4);
        assert_eq!(metrics.chars[origin].unwrap().x,40.);
        assert!(metrics.chars[1].is_none() && metrics.chars[2].is_none());
    }
}

#[test]
fn expansions_share_the_source_advance_and_contractions_leave_holes() {
    let expanded = metrics_from_layout("ßB", &dump(vec![segment("SSB",&[0,0,1],0.)]));
    assert_eq!(expanded.chars[0].unwrap().advance,20.);
    assert_eq!(expanded.chars[1].unwrap().x,20.);
    let contracted = metrics_from_layout("α\u{301}B", &dump(vec![segment("ΑB",&[0,2],0.)]));
    assert!(contracted.chars[1].is_none());
    assert_eq!(contracted.chars[2].unwrap().x,10.);
}

#[test]
fn explicit_origins_control_repeated_words_without_string_matching() {
    let metrics = metrics_from_layout("AB AB", &dump(vec![segment("AB",&[3,4],20.)]));
    assert!(metrics.chars[0].is_none() && metrics.chars[1].is_none());
    assert_eq!(metrics.chars[3].unwrap().x,20.);
    assert_eq!(metrics.chars[4].unwrap().x,30.);
}

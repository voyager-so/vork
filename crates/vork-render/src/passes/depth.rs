//! Depth prepass for opaque and alpha-tested geometry.
use crate::*;

pub(crate) fn encode(encoder: &mut wgpu::CommandEncoder, f: &crate::view::ViewState, r: &Renderer) {
    // Metal samples render-pass ends after fragment work. A depth-only or
    // empty bundle need not execute fragments, leaving that counter unwritten.
    // Use the existing real-work marker around the whole depth/compaction span.
    let metal = r.adapter_info.backend == wgpu::Backend::Metal;
    if metal { timing::boundary(r, encoder, &f.queries, timing::query("depth")); }
    let compacted = f.camera_cull.encode(encoder, (!metal).then_some(&f.queries));
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("depth prepass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(depth_attachment(&f.depth, false, true)),
            timestamp_writes: (!metal).then_some(wgpu::RenderPassTimestampWrites {
                query_set: &f.queries,
                beginning_of_pass_write_index: (!compacted).then_some(timing::query("depth")),
                end_of_pass_write_index: Some(timing::query("depth") + 1),
            }),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.execute_bundles([&f.depth_bundle]);
    }
    if metal { timing::boundary(r, encoder, &f.queries, timing::query("depth") + 1); }
}

#[cfg(test)]
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn diagnostics_preserve_glass_only_and_empty_depth_renders() -> Result<()> {
    let r = crate::test_renderer()?;
    let base: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/closed-glass-slab.json"))?;
    for (kind, refraction) in [("glass", "environment"), ("glass", "screen"), ("opaque", "environment"), ("culled", "environment"), ("empty", "environment")] {
        for samples in [1, 4] {
            let mut doc = base.clone();
            doc["nodes"].as_array_mut().unwrap().remove(1); // Leave only camera and glass.
            doc["materials"].as_object_mut().unwrap().remove("emitter");
            doc["render"]["resolution"] = serde_json::json!([32,32]);
            doc["render"]["settings"] = serde_json::json!({"integrator":"raster","samples":samples,"fast_gi":null,"refraction":refraction});
            if kind == "empty" {
                doc["nodes"].as_array_mut().unwrap().pop();
                // An AOV keeps this empty view on staged material evaluation.
                doc["render"]["aovs"] = serde_json::json!([{"name":"unwritten","type":"value"}]);
            }
            if matches!(kind, "opaque" | "culled") {
                doc["materials"]["glass"]["inputs"]["transmission_weight"] = serde_json::json!(0);
            }
            if kind == "culled" { doc["nodes"][1]["translation"] = serde_json::json!([1000,0,0]); }
            let mut installed = r.install(crate::test_scene(doc, std::path::Path::new("."))?, std::path::Path::new("."))?;
            if kind == "glass" {
                assert!(!installed.plan.draws.is_empty() && installed.plan.draws.iter().all(|draw| draw.transparent), "glass fixture must leave the depth bundle empty");
                assert_eq!(installed.plan.screen_refraction, refraction == "screen");
            }
            let mut frame = installed.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
            let plain = installed.render_frame(&r, &mut frame, 0.)?;
            frame.profile = true;
            // Reuse the slot so stale counters cannot hide a missing boundary.
            for _ in 0..2 {
                let measured = installed.render_frame(&r, &mut frame, 0.)?;
                crate::test_linear::assert_linear_eq(plain.linear_rgba.as_ref().expect("linear readback"), measured.linear_rgba.as_ref().expect("linear readback"));
                crate::test_linear::assert_depth_eq(plain.depth_f32.as_ref().expect("float depth readback"), measured.depth_f32.as_ref().expect("float depth readback"));
                let depth = measured.timing.gpu_spans_ms.iter().find(|(name,_)| *name == "depth").unwrap().1;
                assert!(depth.is_finite() && depth >= 0., "{kind}, refraction={refraction}, samples={samples}: {depth}");
            }
        }
    }
    Ok(())
}

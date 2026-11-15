//! Retained spatial proposal derived from the evaluated analytic light buffer.
use super::{Light, tree::{Tree, Node, analytic_sources}};

pub(crate) fn node_count(lights: usize) -> usize { lights.saturating_mul(2).saturating_sub(1).max(1) }

pub(crate) fn nodes(lights: &[Light]) -> anyhow::Result<Vec<Node>> {
    // path_sampler.sample_unit supplies 23-bit midpoint source coordinates.
    anyhow::ensure!(lights.len() <= (1 << 23), "analytic light tree exceeds the path sampler source-coordinate capacity");
    let tree = Tree::build(&analytic_sources(lights))?;
    Ok(if tree.nodes.is_empty() { vec![bytemuck::Zeroable::zeroed()] } else { tree.nodes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn changed_light_values_update_the_distribution_through_the_upload_tracker()
    -> anyhow::Result<()> {
        let r = crate::test_renderer()?;
        let buffer = |label, size| {
            crate::init(
                &r.device,
                label,
                &vec![0; size],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )
        };
        let lighting = crate::view::ViewLighting {
            active: true,
            uniform: buffer("unused uniform", 16),
            lights: buffer("shared lights", 2 * std::mem::size_of::<Light>()),
            analytic_light_tree: buffer("spatial distribution", node_count(2) * std::mem::size_of::<Node>()),
            cluster_config: buffer("unused config", 16),
            cluster_lights: buffer("unused cluster", 4),
            clusters: None,
        };
        let mut lights = vec![Light::zeroed(); 2];
        lights[0].radiance = [1.; 4];
        lights[1].radiance = [3.; 4];
        let mut changes = crate::values::ChangeTracker::default();
        lighting.upload_lights(&mut changes, &lights)?;
        assert_eq!(changes.uploads.len(), 2);
        changes.uploads.clear();
        changes.bytes.clear();
        lighting.upload_lights(&mut changes, &lights)?;
        assert!(
            changes.uploads.is_empty(),
            "unchanged energies were uploaded again"
        );
        lights.swap(0, 1);
        lighting.upload_lights(&mut changes, &lights)?;
        assert_eq!(
            changes.uploads.len(),
            2,
            "animated powers left a stale distribution"
        );
        let (_, _, offset, size) = changes.uploads.last().unwrap();
        let tree: Vec<Node> = changes.bytes[*offset as usize..(*offset + *size) as usize]
            .chunks_exact(std::mem::size_of::<Node>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let leaf = |identity| tree.iter().find(|n| n.links[0] == u32::MAX && n.source[1] == identity).unwrap();
        assert_eq!(leaf(0).lower_power[3], 1.);
        assert!((leaf(1).lower_power[3] - 1. / 3.).abs() < 1e-6);
        changes.uploads.clear(); changes.bytes.clear();
        lights[0].position[0] = 7.;
        lighting.upload_lights(&mut changes, &lights)?;
        assert_eq!(changes.uploads.len(), 2, "moving lights left stale spatial bounds");
        Ok(())
    }

    /// Splitting an emitter into coincident equal-power sources must preserve
    /// radiance on both sides of the exhaustive/sampled budget boundary. Delta
    /// sources remove emitter-direction variance, exposing normalization errors.
    #[test]
    #[ignore = "requires a ray-query GPU"]
    fn source_splitting_preserves_surface_energy() -> anyhow::Result<()> {
        let r = crate::test_renderer()?;
        use serde_json::json;
        let document = json!({
            "version":2,
            "render":{"resolution":[16,16],"background":[0,0,0]},
            "buffers":{
                "p":{"type":"vec3","data":[[-3,-3,0],[3,-3,0],[3,3,0],[-3,-3,0],[3,3,0],[-3,3,0]]},
                "n":{"type":"vec3","data":[[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]}
            },
            "geometry":{"plane":{"attributes":{"POSITION":"p","NORMAL":"n"}}},
            "materials":{"receiver":{"inputs":{"base_color":[0.8,0.5,0.2],"specular_weight":0}}},
            "nodes":[
                {"type":"camera","translation":[0,0,4]},
                {"type":"mesh","geometry":"plane","materials":["receiver"]}
            ]
        });
        for shape in ["sun", "point"] {
            let mut reference: Option<Vec<[f32;4]>> = None;
            for count in [1, 4, 5, 65] {
                let mut source = document.clone();
                for _ in 0..count {
                    let mut light = json!({
                        "type":"light","shape":shape,"translation":[0,0,2],
                        "target":[0,0,0],"energy":8. / f64::from(count)
                    });
                    light[if shape=="sun" {"angle"} else {"radius"}] = 0.into();
                    source["nodes"].as_array_mut().unwrap().push(light);
                }
                let input = vork_scene::v2::with_test_settings(&source.to_string(),
                    r#"{"samples":8,"fast_gi":null,"shadow_method":"ray"}"#)?;
                let scene = crate::test_scene(input, std::path::Path::new("."))?;
                let mut installed = r.install(scene, std::path::Path::new("."))?;
                let mut slot = installed.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
                let image = installed.render_frame(&r, &mut slot, 0.)?.linear_rgba.unwrap();
                assert!(image.iter().flatten().all(|x| x.is_finite()));
                if let Some(reference) = &reference {
                    let error = image.iter().zip(reference).flat_map(|(a,b)|
                        (0..3).map(move |c| (a[c]-b[c]).abs())).fold(0f32,f32::max);
                    assert!(error < 2e-5, "{shape}, {count} split sources: max error {error}");
                } else {
                    assert!(image.iter().any(|p| p[0] > 0.01), "positive control must be lit");
                    reference = Some(image);
                }
            }
        }
        // Coloured delta lights over an orthographic flat receiver have no
        // emitter or camera variance. Above the budget, colour variance proves
        // source selection is actually active (not an unbound-tree fallback).
        let budget = vork_slang::renderer_registry::REG_ANALYTIC_LIGHT_SAMPLES;
        let mut sampled_image: Option<Vec<[f32;4]>> = None;
        for (count, seed) in [(budget, 11), (budget+1, 11), (budget+1, 29)] {
            let mut source = document.clone();
            source["nodes"][0]["material"] = json!({"program":"std/camera",
                "inputs":{"projection":"ortho","ortho_scale":4}});
            source["materials"]["receiver"]["inputs"]["base_color"] = json!([1,1,1]);
            for i in 0..count {
                source["nodes"].as_array_mut().unwrap().push(json!({
                    "type":"light","shape":"sun","translation":[0,0,2],"target":[0,0,0],
                    "angle":0,"energy":8. / f64::from(count),
                    "color":if i%2==0 {json!([1,0,0])} else {json!([0,0,1])}
                }));
            }
            let profile = json!({"samples":8,"fast_gi":null,"shadow_method":"ray"}).to_string();
            let input = vork_scene::v2::with_test_settings(&source.to_string(), &profile)?;
            let mut scene = crate::test_scene(input, std::path::Path::new("."))?;
            scene.render.settings.seed = seed;
            scene.render.settings.disable_denoiser();
            let mut installed = r.install(scene, std::path::Path::new("."))?;
            let mut slot = installed.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
            let image = installed.render_frame(&r, &mut slot, 0.)?.linear_rgba.unwrap();
            let patch: Vec<f32> = (4..12).flat_map(|y| (4..12).map(move |x| y*16+x))
                .map(|i| image[i][0]).collect();
            let lo = patch.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = patch.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(lo.is_finite() && hi.is_finite());
            if count <= budget {
                assert!(hi-lo < 1e-5, "exact delta-light sum must be uniform: {lo}..{hi}");
            } else {
                assert!(hi-lo > 1e-3, "sampled source selection must be active: {lo}..{hi}");
                if let Some(previous) = &sampled_image {
                    assert!(image.iter().zip(previous).any(|(a,b)| (a[0]-b[0]).abs()>1e-3),
                        "render seed must scramble source selection");
                }
                sampled_image = Some(image);
            }
        }
        Ok(())
    }

    /// A 16-bit rotation selects only even ranks in a balanced 2^17-leaf
    /// hierarchy. Alternating colours must both retain positive support.
    #[test]
    #[ignore = "requires a ray-query GPU and a large analytic source set"]
    fn source_selection_retains_support_above_rotation_grid() -> anyhow::Result<()> {
        let r = crate::test_renderer()?;
        use serde_json::json;
        let mut reference = None;
        for count in [2u32, 1u32 << 17] {
            let mut nodes = vec![
                json!({"type":"camera","translation":[0,0,4],"material":{"program":"std/camera",
                    "inputs":{"projection":"ortho","ortho_scale":4}}}),
                json!({"type":"mesh","geometry":"plane","materials":["receiver"]})
            ];
            for i in 0..count {
                nodes.push(json!({"type":"light","shape":"sun","translation":[0,0,2],
                    "target":[0,0,0],"angle":0,"energy":8. / f64::from(count),
                    "color":if i%2==0 {json!([1,0,0])} else {json!([0,1,0])}}));
            }
            let source = json!({"version":2,"render":{"resolution":[16,16],"background":[0,0,0]},
                "buffers":{"p":{"type":"vec3","data":[[-3,-3,0],[3,-3,0],[3,3,0],[-3,-3,0],[3,3,0],[-3,3,0]]},
                    "n":{"type":"vec3","data":[[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]}},
                "geometry":{"plane":{"attributes":{"POSITION":"p","NORMAL":"n"}}},
                "materials":{"receiver":{"inputs":{"base_color":[1,1,1],"specular_weight":0}}},"nodes":nodes});
            let input = vork_scene::v2::with_test_settings(&source.to_string(),
                r#"{"samples":8,"fast_gi":null,"shadow_method":"ray"}"#)?;
            let mut scene = crate::test_scene(input, std::path::Path::new("."))?;
            scene.render.settings.disable_denoiser();
            let mut installed = r.install(scene, std::path::Path::new("."))?;
            let mut slot = installed.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
            let image = installed.render_frame(&r, &mut slot, 0.)?.linear_rgba.unwrap();
            let means = [0,1].map(|c| image.iter().map(|p| p[c] as f64).sum::<f64>() / image.len() as f64);
            if let Some(reference) = reference {
                let reference: [f64;2] = reference;
                for c in 0..2 {assert!(means[c] > 0.5 * reference[c],
                    "channel {c} lost source support: {} versus {}",means[c],reference[c]);}
            } else {assert!(means.iter().all(|x| *x>0.01));reference=Some(means);}
        }
        Ok(())
    }

    #[test]
    fn spatial_nodes_handle_empty_dark_and_hdr_guides() -> anyhow::Result<()> {
        assert_eq!(nodes(&[])?.len(), node_count(0));
        let mut lights = vec![Light::zeroed(); 5];
        for (i, light) in lights.iter_mut().enumerate() {
            light.position = [i as f32, 0., 0., 1.];
            light.radiance = [if i == 0 { f32::MAX } else { 0. }; 4];
        }
        let tree = nodes(&lights)?;
        assert_eq!(tree.len(), node_count(lights.len()));
        assert!(tree.iter().all(|n| n.lower_power.iter().chain(n.upper_infinite.iter()).all(|p| p.is_finite())));
        assert_eq!(tree.iter().filter(|n| n.links[0] == u32::MAX).count(), lights.len());
        Ok(())
    }
}

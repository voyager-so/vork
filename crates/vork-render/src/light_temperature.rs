//! Light temperature: final colour = color × std.color blackbody(kelvin). The one
//! blackbody is Slang (libraries/std/color.slang); an install whose lights use
//! temperature tabulates it on the GPU at every kelvin of Blender's Light.temperature
//! range, and frames interpolate linearly between neighbouring kelvins.
use crate::*;
use vork_scene::v2::{LIGHT_TEMPERATURE_MAX, LIGHT_TEMPERATURE_MIN};

const ENTRIES: usize = (LIGHT_TEMPERATURE_MAX - LIGHT_TEMPERATURE_MIN) as usize + 1;
/// Workgroup width of `tabulate`; the GPU table is padded to whole workgroups.
const GROUP: usize = 64;
const BYTES: u64 = (ENTRIES.div_ceil(GROUP) * GROUP * 16) as u64;
pub(crate) const LAYOUT: [wgpu::BindGroupLayoutEntry; 1] = [wgpu::BindGroupLayoutEntry {
    binding: 0,
    visibility: wgpu::ShaderStages::COMPUTE,
    ty: wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    },
    count: None,
}];

/// Blackbody RGB at each kelvin from LIGHT_TEMPERATURE_MIN.
pub(crate) type Table = std::sync::Arc<[[f32; 3]]>;

/// The table, when any light node uses temperature.
pub(crate) fn table(r: &Renderer, nodes: &[EvalNode]) -> Result<Option<Table>> {
    if !nodes.iter().any(|n| matches!(n.node.kind, NodeKind::Light { temperature: Some(_), .. })) {
        return Ok(None);
    }
    let d = &r.device;
    let pipeline = r.pipelines.light_temperature.get_or_init(|| {
        let module = native::shader_module(d, &native::LIGHT_TEMPERATURE_TABULATE);
        let group = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("blackbody table"), entries: &LAYOUT });
        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blackbody table"),
            bind_group_layouts: &[Some(&group)],
            immediate_size: 0,
        });
        r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("blackbody table"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(native::LIGHT_TEMPERATURE_TABULATE.entry),
            compilation_options: Default::default(),
            cache: r.cache.as_ref(),
        })
    });
    let buffer = |label, usage| crate::gpu_memory::buffer(d, &wgpu::BufferDescriptor { label: Some(label), size: BYTES, usage, mapped_at_creation: false });
    let table = buffer("blackbody table", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let readback = buffer("blackbody readback", wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
    let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blackbody table"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[entry(0, table.as_entire_binding())],
    });
    let mut encoder = d.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(ENTRIES.div_ceil(GROUP) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&table, 0, &readback, 0, BYTES);
    let submission = crate::command_finish::Segments::submit(&r.queue, encoder);
    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
    r.wait(Some(submission))?;
    rx.recv()?.context("map the blackbody table")?;
    let mapped = readback.get_mapped_range(..);
    let rows: Table = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped)[..ENTRIES].iter().map(|v| [v[0], v[1], v[2]]).collect();
    drop(mapped);
    readback.unmap();
    r.errors.check()?;
    ensure!(rows.iter().flatten().all(|v| v.is_finite() && *v >= 0.), "blackbody table is not finite");
    Ok(Some(rows))
}

/// std.color blackbody(kelvin) from the table.
pub(crate) fn blackbody(table: &[[f32; 3]], kelvin: f32) -> Result<[f32; 3]> {
    ensure!(
        (LIGHT_TEMPERATURE_MIN..=LIGHT_TEMPERATURE_MAX).contains(&kelvin),
        "light temperature {kelvin} K is outside {LIGHT_TEMPERATURE_MIN}..{LIGHT_TEMPERATURE_MAX} K"
    );
    let at = kelvin - LIGHT_TEMPERATURE_MIN;
    let i = (at as usize).min(table.len() - 2);
    let t = at - i as f32;
    Ok(std::array::from_fn(|c| table[i][c] + (table[i + 1][c] - table[i][c]) * t))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tabulated GPU blackbody matches an f64 evaluation of the same std.color
    /// formula (Planck's law over the analytic CIE 1931 fit, 380..780 nm in 5 nm steps).
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn table_is_std_color_blackbody() -> Result<()> {
        let r = crate::test_renderer()?;
        let scene: vork_scene::v2::Scene = vork_scene::v2::from_test_value(serde_json::json!({
            "version": 2,
            "nodes": [{"type": "light", "shape": "point", "temperature": 2700}],
        }))?;
        let nodes: Vec<EvalNode> = scene.nodes.iter().map(|n| EvalNode { camera: None,
            node: n.without_children(), parent: None, world: Mat4::IDENTITY, dynamic: false, embedded: None,
        }).collect();
        let table = table(&r, &nodes)?.context("a temperature light tabulates")?;
        assert_eq!(table.len(), ENTRIES);
        fn reference(t: f64) -> [f64; 3] {
            let g = |nm: f64, c: f64, l: f64, r: f64| { let t = (nm - c) * if nm < c { l } else { r }; (-0.5 * t * t).exp() };
            let mut xyz = [0f64; 3];
            for i in 0..=80 {
                let nm = 380. + 5. * f64::from(i);
                let um = nm * 0.001;
                let power = 1. / (um.powi(5) * ((14387.76877 / (um * t)).min(80.).exp() - 1.)) * if i == 0 || i == 80 { 0.5 } else { 1. };
                let c = [
                    1.056 * g(nm, 599.8, 0.0264, 0.0323) + 0.362 * g(nm, 442., 0.0624, 0.0374) - 0.065 * g(nm, 501.1, 0.049, 0.0382),
                    0.821 * g(nm, 568.8, 0.0213, 0.0247) + 0.286 * g(nm, 530.9, 0.0613, 0.0322),
                    1.217 * g(nm, 437., 0.0845, 0.0278) + 0.681 * g(nm, 459., 0.0385, 0.0725),
                ];
                for k in 0..3 { xyz[k] += c[k] * power; }
            }
            let [x, y, z] = xyz.map(|v| v / xyz[1]);
            [3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z, 0.0557 * x - 0.204 * y + 1.057 * z].map(|v| v.max(0.))
        }
        for kelvin in [800., 1234.5, 2700., 6500., 12000., 19999.5, 20000.] {
            let got = blackbody(&table, kelvin)?;
            let want = reference(f64::from(kelvin));
            for c in 0..3 {
                assert!((f64::from(got[c]) - want[c]).abs() <= 1e-4 * want[c].max(1e-2), "{kelvin} K: {got:?} vs {want:?}");
            }
        }
        // A plain light needs no table.
        let plain: Vec<EvalNode> = nodes.into_iter().map(|mut n| {
            if let NodeKind::Light { temperature, .. } = &mut n.node.kind { *temperature = None; }
            n
        }).collect();
        assert!(super::table(&r, &plain)?.is_none());
        assert!(blackbody(&table, 799.).is_err() && blackbody(&table, 20001.).is_err());
        Ok(())
    }
}

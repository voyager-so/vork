//! In-process raster host loop. Rust interfaces are unstable.
//!
//! Preparation owns installation, variant compilation and view allocation.
//! Each frame retains those owners and returns accepted GPU colour without
//! reading pixels. This is a lifecycle API, not a qualified real-time profile.
use crate::{FrameSlot, FrameTiming, InstalledScene, Renderer};
use anyhow::{Context, Result, ensure};
use std::path::Path;
mod presentation;
pub use presentation::Presentation;

/// A numeric property declared before specialization. Paths use the existing
/// animation address grammar. Values must remain inside the component bounds.
/// The host value replaces authored animation for this property.
#[derive(Clone, Debug)]
pub struct LiveInput {
    pub path: Vec<String>,
    pub initial: Vec<f32>,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

/// A prepared input handle, scoped to one session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Input { owner: u64, index: usize }

/// One value in an atomic frame publication. Duplicate handles are rejected.
pub struct Update<'a> { pub input: Input, pub value: &'a [f32] }

/// One installed scene and one retained view. Frames complete in submission
/// order; at most one is in flight. A failed GPU frame invalidates the session.
pub struct Session {
    slot: FrameSlot,
    scene: InstalledScene,
    renderer: Renderer,
    failed: bool,
    frame: u32,
    owner: u64,
    inputs: Vec<LiveInput>,
    bindings: Vec<usize>,
}

/// An accepted image. The borrow prevents another frame from overwriting it
/// until the host has submitted its GPU consumer on the session's queue.
pub struct Frame<'a> {
    pub image: &'a wgpu::Texture,
    pub timing: FrameTiming,
    pub index: u32,
}

impl Session {
    /// Prepare resolved raster settings on an existing device. The caller may
    /// perform loading and installation before opening its interactive loop.
    /// Capacity growth is permitted only during this initial readiness frame.
    pub fn prepare(renderer: Renderer, scene: vork_scene::Scene, base: &Path) -> Result<Self> {
        Self::prepare_live(renderer, scene, base, Vec::new())
    }

    pub fn prepare_live(renderer: Renderer, mut scene: vork_scene::Scene, base: &Path, inputs: Vec<LiveInput>) -> Result<Self> {
        ensure!(!scene.render.settings.integrator.is_path(),
            "interactive session requires raster settings; choose std/previs or std/draft");
        // A resident session must retain its requested quality for every frame.
        // Open before installation, whose offline policy can drop unavailable OIDN.
        if scene.render.settings.integrator.denoiser().is_some() {
            crate::denoise::ensure_runtime(&renderer)
                .context("interactive denoising unavailable; set VORK_OIDN to the bundled OIDN library or choose std/previs")?;
        }
        let targets = declare_inputs(&mut scene, &inputs)?;
        let mut scene = renderer.install_with_options(scene, base, crate::InstallOptions {
            raster_only: true,
            readback: Default::default(),
            variants: crate::pipelines::VariantCompile::Inline,
            ..Default::default()
        })?;
        if !inputs.is_empty() {
            ensure!(scene.displacement.is_none() && scene.compute.is_none() && scene.graph.views.images.is_empty(),
                "live inputs currently require triangle scenes without displacement, compute or render-image captures; bake these resources before preparation");
        }
        let bindings = scene.animation.bind_live(&targets, &inputs.iter().map(|input| input.initial.clone()).collect::<Vec<_>>())?;
        let mut slot = scene.create_frame_slot(&renderer, Default::default())?;
        slot.profile = true;
        loop {
            match slot.submit_gpu(&scene, &renderer, 0.).and_then(|()| slot.collect(&renderer)) {
                Ok(_) => break,
                Err(error) => {
                    let Some(redo) = error.downcast_ref::<crate::FrameRedo>() else { return Err(error); };
                    scene.redo_frame(&renderer, std::slice::from_mut(&mut slot), redo.0, 0)?;
                }
            }
        }
        renderer.settle_pipelines()?;
        // Readiness is not a host frame. Its cached identity (zero at t=0)
        // must not satisfy the first live batch before those values render.
        slot.completed = None;
        static NEXT_OWNER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let owner = NEXT_OWNER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Self { slot, scene, renderer, failed: false, frame: 0, owner, inputs, bindings })
    }

    pub fn renderer(&self) -> &Renderer { &self.renderer }
    /// Resolved settings after device/provider admission and device-dependent fallbacks.
    pub fn render(&self) -> &vork_scene::Render { &self.scene.render }
    pub fn resolution(&self) -> [u32; 2] { self.scene.render.resolution }
    pub fn input(&self, index: usize) -> Option<Input> {
        (index < self.inputs.len()).then_some(Input { owner: self.owner, index })
    }

    /// Evaluate animation at seconds and complete one accepted GPU image.
    /// Diagnostic counters are read; colour pixels remain GPU-resident.
    /// A capacity miss refuses the frame: reduce scene complexity/resolution
    /// and prepare again. No automatic frame-time reinstall.
    pub fn frame(&mut self, time: f32) -> Result<Frame<'_>> {
        self.update_frame(time, &[])
    }

    /// Validate the complete batch before publishing anything. Omitted values
    /// retain their preceding publication. Animation time is part of the same
    /// call. Invalid input leaves the session usable and changes no live value.
    pub fn update_frame(&mut self, time: f32, updates: &[Update<'_>]) -> Result<Frame<'_>> {
        ensure!(!self.failed, "session is invalid after a GPU frame failure; recreate the renderer and prepare the scene again");
        ensure!(time.is_finite() && time >= 0., "frame time must be finite, nonnegative seconds");
        let next = self.frame.checked_add(1).context("session frame identity exhausted; prepare a new session")?;
        validate_updates(self.owner, &self.inputs, updates)?;
        self.scene.animation.publish_live(updates.iter().map(|update| (self.bindings[update.input.index], update.value)));
        let result = self.slot.submit_indexed_outputs(&self.scene, &self.renderer, self.frame,
            vork_time::Request::Numeric(time), Default::default())
            .and_then(|()| self.slot.collect(&self.renderer));
        let output = match result {
            Ok(output) => output,
            Err(error) => {
                self.failed = true;
                return Err(error).context("interactive frame failed; reduce scene complexity/resolution for a capacity failure, or recreate the renderer after device loss, then prepare again");
            }
        };
        let index = self.frame;
        self.frame = next;
        Ok(Frame { image: &self.slot.view.output, timing: output.timing, index })
    }
}

fn validate_value(input: &LiveInput, value: &[f32]) -> Result<()> {
    ensure!(value.len() == input.min.len() && value.iter().enumerate().all(|(i, v)|
        v.is_finite() && *v >= input.min[i] && *v <= input.max[i]),
        "live input {:?} requires {} finite components inside its prepared min/max bounds", input.path, input.min.len());
    if input.path.last().is_some_and(|field| field == "rotation") {
        ensure!(value.len() == 4 && (value.iter().map(|v| v*v).sum::<f32>() - 1.).abs() < 1e-4,
            "live rotation {:?} requires a normalized [x,y,z,w] quaternion", input.path);
    }
    if input.path.last().is_some_and(|field| field == "scale") {
        ensure!(value.iter().all(|v| *v != 0.), "live scale {:?} must be nonzero on every axis", input.path);
    }
    Ok(())
}

fn validate_updates(owner: u64, inputs: &[LiveInput], updates: &[Update<'_>]) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for update in updates {
        ensure!(update.input.owner == owner && update.input.index < inputs.len(), "live input handle belongs to another session");
        ensure!(seen.insert(update.input.index), "duplicate live input in one frame batch");
        validate_value(&inputs[update.input.index], update.value)?;
    }
    Ok(())
}

fn declare_inputs(scene: &mut vork_scene::Scene, inputs: &[LiveInput]) -> Result<Vec<vork_scene::v2::TrackTarget>> {
    use vork_scene::v2::{TrackTarget as T, NodeProperty as P};
    if inputs.is_empty() { return Ok(Vec::new()); }
    ensure!(scene.geometry.values().all(|g| matches!(g.topology, vork_scene::v2::Topology::Triangles)),
        "live inputs require triangle geometry; bake curves, points and implicit surfaces to triangles before preparation");
    ensure!(scene.simulation.is_none(), "live inputs cannot rewrite simulation history; use a scene without simulation");
    ensure!(scene.render.motion_blur.as_ref().is_none_or(|blur| blur.shutter == 0.)
        && !scene.track_targets()?.iter().flatten().any(|target| matches!(target, T::Render(vork_scene::v2::RenderProperty::MotionBlurShutter))),
        "live inputs do not yet retain previous value generations for shutter blur; disable motion blur and remove shutter tracks");
    let mut paths = std::collections::BTreeSet::new();
    let mut tracks = Vec::new();
    for input in inputs {
        ensure!(!input.min.is_empty() && input.min.len() == input.max.len()
            && input.min.iter().zip(&input.max).all(|(a,b)| a.is_finite() && b.is_finite() && a <= b)
            && input.min != input.max, "live input {:?} needs finite ordered, nonconstant min/max bounds", input.path);
        validate_value(input, &input.initial)?;
        ensure!(paths.insert(input.path.clone()), "duplicate live declaration {:?}", input.path);
        ensure!(!scene.animation.iter().flat_map(|l| l.tracks.iter().flatten()).any(|t| t.path == input.path),
            "live input {:?} replaces authored animation; remove the track for this property", input.path);
        let json_value = |v: &[f32]| if v.len() == 1 { serde_json::json!(v[0]) } else { serde_json::json!(v) };
        tracks.push(serde_json::json!({"path": input.path, "keys": [[0, json_value(&input.min)], [1, json_value(&input.max)]]}));
    }
    scene.animation.push(serde_json::from_value(serde_json::json!({"tracks": tracks}))?);
    scene.validate()?;
    let targets = scene.track_targets()?.pop().context("missing live declaration layer")?;
    ensure!(targets.iter().all(|target| matches!(target,
        T::Node { property: P::Translation | P::Rotation | P::Scale, .. }
        | T::Imported { property: P::Translation | P::Rotation | P::Scale, .. }
        | T::Material { .. })),
        "live inputs currently support node transforms and numeric material/program inputs; use frame time for animation");
    Ok(targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input() -> LiveInput {
        LiveInput { path: vec!["nodes".into(), "camera".into(), "translation".into()],
            initial: vec![0.,0.,0.], min: vec![-10.;3], max: vec![10.;3] }
    }
    #[test]
    fn invalid_batch_is_rejected_before_publication() {
        let inputs = [input(), input()];
        let a = Input { owner: 7, index: 0 };
        let b = Input { owner: 7, index: 1 };
        assert!(validate_updates(7, &inputs, &[Update {input:a, value:&[1.,2.,3.]}, Update {input:b, value:&[0.,f32::NAN,0.]}]).is_err());
        assert!(validate_updates(7, &inputs, &[Update {input:a, value:&[11.,0.,0.]}]).is_err());
        assert!(validate_updates(8, &inputs, &[Update {input:a, value:&[0.;3]}]).is_err());
        assert!(validate_updates(7, &inputs, &[Update {input:a, value:&[0.;3]}, Update {input:a, value:&[1.;3]}]).is_err());
        assert!(validate_updates(7, &inputs, &[Update {input:a, value:&[1.,2.,3.]}, Update {input:b, value:&[-1.,-2.,-3.]}]).is_ok());
    }

    #[test]
    fn declarations_use_existing_track_validation_and_reject_duplicate_ownership() -> Result<()> {
        let mut scene = vork_scene::Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera","name":"camera"}]}"#)?;
        let profile = vork_scene::Profile::load("std/previs")?;
        scene.apply_settings(profile.resolve(&Default::default())?, &profile.name)?;
        let targets = declare_inputs(&mut scene, &[input()])?;
        assert!(matches!(&targets[0], vork_scene::v2::TrackTarget::Node { property: vork_scene::v2::NodeProperty::Translation, .. }));
        assert!(declare_inputs(&mut scene, &[input()]).is_err());
        Ok(())
    }

    fn pixels(r: &Renderer, image: &wgpu::Texture) -> Result<Vec<u8>> {
        let pitch = (image.width()*4).div_ceil(256)*256;
        let buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("session test readback"), size: u64::from(pitch)*u64::from(image.height()),
            usage: wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST, mapped_at_creation:false,
        });
        let mut encoder = r.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(image.as_image_copy(), wgpu::TexelCopyBufferInfo { buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset:0, bytes_per_row:Some(pitch), rows_per_image:None } }, image.size());
        let submission = r.queue.submit([encoder.finish()]);
        let (tx,rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {let _=tx.send(result);});
        r.wait(Some(submission))?; rx.recv()??;
        let data = buffer.slice(..).get_mapped_range();
        let mut rgba:Vec<u8> = data.chunks(pitch as usize).flat_map(|row| row[..image.width() as usize*4].iter().copied()).collect();
        if image.format() == wgpu::TextureFormat::Bgra8Unorm { for pixel in rgba.chunks_exact_mut(4) { pixel.swap(0,2); } }
        Ok(rgba)
    }

    #[test]
    #[ignore = "requires a GPU; scheduled explicitly by the main GPU contracts"]
    fn gpu_live_frames_match_fresh_installs_and_retain_view_and_pipelines() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut document: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/closed-glass-slab.json"))?;
        document["nodes"][2]["materials"] = serde_json::json!(["emitter"]);
        document["materials"].as_object_mut().unwrap().remove("glass");
        document["nodes"][2]["scale"] = serde_json::json!([0.5,0.5,0.5]);
        document["nodes"][1]["visible"] = false.into();
        let mut scene = crate::test_scene(document, Path::new("."))?;
        let profile = vork_scene::Profile::load("std/previs")?;
        scene.apply_settings(profile.resolve(&[("samples".into(),serde_json::json!(1))].into_iter().collect())?, &profile.name)?;
        scene.render.resolution = [32,32];
        let inputs = vec![
            LiveInput { path: vec!["nodes".into(),"slab".into(),"translation".into()], initial:vec![0.;3], min:vec![-1.;3],max:vec![1.;3] },
            LiveInput { path:vec!["materials".into(),"emitter".into(),"inputs".into(),"emission_luminance".into()],initial:vec![1.],min:vec![0.25],max:vec![2.] },
        ];
        let mut session = Session::prepare_live(r, scene.clone(), Path::new("."), inputs)?;
        let texture = session.slot.view.output.clone();
        let a = session.input(0).unwrap(); let b = session.input(1).unwrap();
        let mut previous = Vec::new();
        for (i, x, energy) in [(0,0.25,0.5),(1,-0.25,1.5)] {
            session.renderer.settle_pipelines()?;
            let pipelines = session.renderer.pipeline_timings().len();
            // Paused animation must still publish each new host generation,
            // including the first batch at the readiness frame's time.
            let image = session.update_frame(0., &[Update {input:a,value:&[x,0.,0.]},Update {input:b,value:&[energy]}])?.image.clone();
            assert_eq!(texture, image, "view output was replaced during frame");
            assert_eq!(pipelines,session.renderer.pipeline_timings().len(),"pipeline creation during live frame");
            let actual = pixels(&session.renderer, &image)?;
            let mut reference = scene.clone();
            reference.nodes[2].transform.translation = [x,0.,0.];
            reference.materials.get_mut("emitter").unwrap().inputs.insert("emission_luminance".into(), vork_scene::v2::Input::Number(energy));
            let reference = session.renderer.install(reference, Path::new("."))?;
            let mut frame = reference.create_frame_slot(&session.renderer, Default::default())?;
            frame.submit_indexed_outputs(&reference,&session.renderer,i,vork_time::Request::Numeric(0.),crate::ReadbackOutputs {color:true,..Default::default()})?;
            let expected = frame.collect(&session.renderer)?.rgba;
            assert_eq!(actual.len(), expected.len(), "fresh-install readback shape");
            let max_error = actual.iter().zip(&expected).map(|(a,b)| a.abs_diff(*b)).max().unwrap_or(0);
            assert!(max_error <= 1,"live frame {i} differs from fresh installation: max byte error {max_error}");
            if !previous.is_empty() { assert_ne!(previous,actual,"live transform/material update had no visible effect"); }
            previous = actual;
        }
        Ok(())
    }
}

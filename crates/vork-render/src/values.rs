//! Per-frame values: node and camera evaluation, uniforms, lights, deformation
//! parameters, packed into one upload per sample.
use super::*;

/// (destination, destination offset, source offset in the packed bytes, size)
pub(crate) type Upload = (wgpu::Buffer, u64, u64, u64);
/// Byte history makes overlapping dirty writes exact without scanning other uploads.
#[derive(Default)]
pub(crate) struct UploadCache {
    bytes: Vec<u8>,
    valid: Vec<bool>,
    stamp: u64,
}
impl UploadCache {
    fn changed(&mut self, offset: usize, bytes: &[u8]) -> bool {
        let end = offset + bytes.len();
        if self.bytes.len() < end {
            self.bytes.resize(end, 0);
            self.valid.resize(end, false);
        }
        if self.valid[offset..end].iter().all(|v| *v) && self.bytes[offset..end] == *bytes {
            return false;
        }
        self.bytes[offset..end].copy_from_slice(bytes);
        self.valid[offset..end].fill(true);
        true
    }
}

/// Per-slot value and upload history, keyed by the resource it affects.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum ChangeKey {
    Buffer(wgpu::Buffer),
    CameraSelection(wgpu::Buffer),
    LightTree(wgpu::Buffer),
    Subdivision(usize, usize),
    SampleTime(usize),
    ComputeTime(usize),
    DiscreteTime(usize),
    /// Inputs of an install-shared published render image, in view-table order.
    Capture(usize),
    ImageClock(String),
    /// The frame slot's acceleration structures have had their first build.
    RayStructures,
    /// Logical displacement evaluation, independent of its packed control page.
    Displacement(usize),
    DisplacedPositions(usize),
    ComputedGeometry(usize),
    ComputedPositions(usize),
    MaterialAttributes,
    /// Source publication consumed by an installed shared field index.
    FieldIndex(wgpu::Buffer),
    /// Logical named stream publication, independent of packed GPU page identity.
    Stream(String),
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum LightingKey {
    /// The irradiance volume's lighting inputs (probes::key).
    Probes,
    ProbeGeometry,
    EnvironmentInputs,
    EnvironmentRadiance,
    Fibre,
}
#[derive(Default)]
pub(crate) struct ChangeTracker {
    pub(crate) placement_revision: u64,
    /// Publication generation of render-backed images under the submission lock.
    /// Image-dependent lighting consumes seed publications from this frame.
    pub(crate) image_revision: u64,
    history: std::collections::HashMap<ChangeKey, UploadCache>,
    sample: u64,
    pub(crate) bytes: Vec<u8>,
    pub(crate) uploads: Vec<Upload>,
}
/// Submission-ordered history of retained lighting,
/// independent of upload slots.
#[derive(Default)]
pub(crate) struct LightingTracker {
    history: std::collections::HashMap<LightingKey, UploadCache>,
    sample: u64,
}
impl LightingTracker {
    pub(crate) fn invalidate_probes(&mut self) {
        self.history.remove(&LightingKey::Probes);
        self.history.remove(&LightingKey::ProbeGeometry);
    }

    /// Last input change, retained after the producer's dirty sample. Consumers
    /// may skip submissions and must still observe an intervening update.
    pub(crate) fn revision(&self, key: LightingKey) -> u64 {
        self.history.get(&key).map_or(0, |h| h.stamp)
    }
    pub(crate) fn dirty(&self, key: LightingKey) -> bool {
        self.history
            .get(&key)
            .is_some_and(|h| h.stamp == self.sample)
    }
    pub(crate) fn changed(&mut self, key: LightingKey, bytes: &[u8]) -> bool {
        let history = self.history.entry(key).or_default();
        let resized = history.bytes.len() != bytes.len();
        let changed = history.changed(0, bytes);
        history.bytes.truncate(bytes.len());
        history.valid.truncate(bytes.len());
        let changed = resized || changed;
        if changed {
            history.stamp = self.sample;
        }
        changed
    }

}
impl ChangeTracker {
    pub(crate) fn observe_stream(&mut self, resource: &str, version: u64) {
        self.changed(ChangeKey::Stream(resource.into()), &version.to_le_bytes());
    }
    pub(crate) fn stream_version(&self, resource: &str) -> u64 {
        self.history.get(&ChangeKey::Stream(resource.into())).map_or(0, |h| u64::from_le_bytes(h.bytes.as_slice().try_into().expect("stream version bytes")))
    }

    pub(crate) fn invalidate_captures(&mut self) {
        self.history.retain(|key, _| !matches!(key, ChangeKey::Capture(_)));
    }

    /// Canonical value history, independent of the frame slot's buffer identity.
    /// Validity matters: the first write of zero can replace a nonzero installed value.
    pub(crate) fn append_buffer_inputs(&self, buffer: &wgpu::Buffer, inputs: &mut Vec<u8>) {
        self.append_inputs(ChangeKey::Buffer(buffer.clone()), inputs);
    }
    pub(crate) fn append_inputs(&self, key: ChangeKey, inputs: &mut Vec<u8>) {
        let history = self.history.get(&key);
        inputs.extend_from_slice(&(history.map_or(0, |h| h.bytes.len()) as u64).to_le_bytes());
        if let Some(history) = history {
            inputs.extend_from_slice(&history.bytes);
            inputs.extend(history.valid.iter().map(|v| u8::from(*v)));
        }
    }
    /// Publish a derived resource without inventing a second dirty mechanism.
    pub(crate) fn publish(&mut self, key: ChangeKey) {
        self.history.entry(key).or_default().stamp = self.sample;
    }
    /// Record an external GPU replacement at its encoding boundary. Forget both
    /// the destination CPU byte histories and the derived input key: unchanged
    /// inputs must recompute, and equal old CPU bytes must be uploaded again.
    /// Already queued uploads are retained; callers encode them before replacement.
    pub(crate) fn replace_gpu(&mut self, buffers: &[&wgpu::Buffer], dependent: ChangeKey) {
        for buffer in buffers {
            self.history.remove(&ChangeKey::Buffer((*buffer).clone()));
        }
        self.history.remove(&dependent);
    }
    /// Last sample that published CPU bytes to this buffer. A consumer also
    /// checks dirty() when it may run again inside that same sample.
    pub(crate) fn buffer_revision(&self, buffer: &wgpu::Buffer) -> u64 {
        self.history.get(&ChangeKey::Buffer(buffer.clone())).map_or(0, |history| history.stamp)
    }
    pub(crate) fn buffer_dirty(&self, buffer: &wgpu::Buffer) -> bool {
        self.dirty(ChangeKey::Buffer(buffer.clone()))
    }
    pub(crate) fn dirty(&self, key: ChangeKey) -> bool { self.history.get(&key).is_some_and(|h|h.stamp == self.sample) }
    pub(crate) fn changed(&mut self, key: ChangeKey, bytes: &[u8]) -> bool {
        let history = self.history.entry(key).or_default();
        let resized = history.bytes.len() != bytes.len();
        let changed = history.changed(0, bytes);
        history.bytes.truncate(bytes.len());
        history.valid.truncate(bytes.len());
        let changed = resized || changed;
        if changed {
            history.stamp = self.sample;
        }
        changed
    }
    /// Recompute CPU geometry only when its deformation inputs change. The same
    /// history owns both derived bounds and the normal dirty/upload mechanism.
    pub(crate) fn derive_upload<T: Pod + Copy>(
        &mut self, key: ChangeKey, input: &[u8], buffer: &wgpu::Buffer, offset: u64,
        calculate: impl FnOnce() -> Result<(T, Vec<u8>)>,
    ) -> Result<T> {
        if let Some(history) = self.history.get(&key) {
            if history.bytes.len() == input.len() + std::mem::size_of::<T>() && history.bytes[..input.len()] == *input {
                return Ok(bytemuck::pod_read_unaligned(&history.bytes[input.len()..]));
            }
        }
        let (value, bytes) = calculate()?;
        self.upload(buffer, offset, &bytes);
        let history = self.history.entry(key).or_default();
        history.bytes.clear(); history.bytes.extend_from_slice(input); history.bytes.extend_from_slice(bytemuck::bytes_of(&value));
        Ok(value)
    }
    pub(crate) fn upload(&mut self, buffer: &wgpu::Buffer, offset: u64, bytes: &[u8]) {
        let history = self
            .history
            .entry(ChangeKey::Buffer(buffer.clone()))
            .or_default();
        if !history.changed(offset as usize, bytes) {
            return;
        }
        history.stamp = self.sample;
        self.record_upload(buffer, offset, bytes);
    }
    fn record_upload(&mut self, buffer: &wgpu::Buffer, offset: u64, bytes: &[u8]) {
        let source = self.bytes.len() as u64;
        self.bytes.extend_from_slice(bytes);
        // Preserve field/joint granularity without a separate dirty-range detector.
        if let Some((last_buffer, last_offset, last_source, size)) = self.uploads.last_mut() {
            if last_buffer == buffer
                && *last_offset + *size == offset
                && *last_source + *size == source
            {
                *size += bytes.len() as u64;
                return;
            }
        }
        self.uploads
            .push((buffer.clone(), offset, source, bytes.len() as u64));
    }
}

/// One evaluation per keyed node/time; static nodes borrow installed values.
pub(crate) struct SampledNodes<'a> {
    nodes: &'a [EvalNode],
    values: &'a [Option<EvalNode>],
}
pub(crate) fn sample_storage(nodes: &[EvalNode]) -> Vec<Option<EvalNode>> {
    nodes
        .iter()
        .map(|n| {
            n.dynamic.then(|| EvalNode { camera: n.camera.clone(),
                node: n.node.clone(),
                parent: n.parent,
                world: n.world,
                dynamic: n.dynamic,
                embedded: n.embedded,
            })
        })
        .collect()
}
pub(crate) struct OffsetValues {
    samples: animation::Samples,
    nodes: Vec<Option<EvalNode>>,
    worlds: Vec<Mat4>,
    groups: Vec<OffsetGroup>,
}
#[derive(Default)]
struct OffsetGroup {
    offset: f32,
    objects: Vec<usize>,
    deforms: Vec<usize>,
    nodes: Vec<usize>,
    world_order: Vec<usize>,
    assets: Vec<usize>,
}
pub(crate) fn offset_storage(s: &InstalledScene) -> OffsetValues {
    let mut groups = std::collections::BTreeMap::<u32, OffsetGroup>::new();
    for &object in &s.dynamic_objects {
        let root = &s.roots[object];
        let bits = if root.time_offset == 0. {
            0
        } else {
            root.time_offset.to_bits()
        };
        let group = groups.entry(bits).or_default();
        group.offset = root.time_offset;
        group.objects.push(object);
        group.nodes.push(root.node);
    }
    for (index, draw) in s.draws.iter().enumerate() {
        let Some(deform) = draw.deform.as_ref().filter(|deform| deform.leader == index) else {
            continue;
        };
        let root = &s.roots[draw.object as usize];
        let bits = if root.time_offset == 0. {
            0
        } else {
            root.time_offset.to_bits()
        };
        let group = groups.entry(bits).or_default();
        group.offset = root.time_offset;
        group.deforms.push(index);
        group.nodes.push(root.node);
        if let Some(native) = &deform.native {
            group.nodes.push(native.mesh);
            group
                .nodes
                .extend(native.joints.iter().map(|(joint, _)| *joint));
        } else {
            group
                .assets
                .push(deform.asset.expect("glTF deformation asset"));
        }
    }
    for group in groups.values_mut() {
        let mut required = std::collections::BTreeSet::new();
        let mut pending = std::mem::take(&mut group.nodes);
        while let Some(index) = pending.pop() {
            if !required.insert(index) {
                continue;
            }
            if let Some(target) = s.targets.aims[index].and_then(crate::targets::Aim::node) { pending.push(target); }
            if let Some(parent) = s.nodes[index].parent {
                pending.push(parent);
            }
            if let Some((asset, _, root, _)) = s.nodes[index].embedded {
                group.assets.push(asset);
                pending.push(root);
            }
        }
        group.nodes = required.into_iter().collect();
        group.world_order = s.targets.order.iter().copied().filter(|i| group.nodes.binary_search(i).is_ok()).collect();
        group.assets.sort_unstable();
        group.assets.dedup();
    }
    OffsetValues {
        samples: s.animation.samples(),
        nodes: sample_storage(&s.nodes),
        worlds: s.nodes.iter().map(|n| n.world).collect(),
        groups: groups.into_values().collect(),
    }
}
pub(crate) fn sample_frame(
    s: &InstalledScene,
    request: impl Into<vork_time::Request>,
    samples: &mut animation::Samples,
    values: &mut [Option<EvalNode>],
    worlds: &mut [Mat4],
) -> Result<()> {
    let request=request.into();
    let time=request.numeric()?;
    sample_pose(&s.nodes, &s.assets, &s.animation, &s.targets, &s.environment, &request, samples, values, worlds)?;
    let active_camera=s.animation.camera_at_request(&request,s.camera)?;
    for (index,node) in values.iter_mut().enumerate().filter_map(|(index,node)|node.as_mut().map(|node|(index,node))) {
        // Rule 1: an absent camera's program state is not evaluated.
        if s.animation.absent(samples,index) {continue;}
        if let Some(camera)=&mut node.camera {
            camera.time=time;
            for (binding,value) in s.animation.material_values(samples,0.) {
                if binding.slot>=camera.offset && binding.slot<camera.offset+camera.words.len() {
                    let words=programs::sample_words(value,binding);let start=binding.slot-camera.offset;
                    camera.words[start..start+words.len()].copy_from_slice(&words);
                }
            }
            let selected=std::iter::once(s.render.clone()).filter(|_|index==active_camera)
                .chain(s.scene.images.values().filter(|image|image.render.as_ref().is_some_and(|render|
                    render.camera.as_deref().map_or(index==s.camera,|name|name==node.node.name))).filter_map(|image|image.resolved_render(&s.render)));
            for render in selected {
                let calibration=camera.calibration(render.resolution[0] as f32/render.resolution[1] as f32)?;
                ensure!(render.settings.integrator.is_path() || calibration.is_some(),"raster camera must retain projective calibration at every sample");
            }
        }
    }
    Ok(())
}

/// Shared installation and frame pose evaluation; authoring defaults stay untouched.
pub(crate) fn sample_pose(
    originals: &[EvalNode],
    assets: &[Asset],
    animation: &animation::Plan,
    targets: &crate::targets::Plan,
    _environment: &environment::Environment,
    request: impl Into<vork_time::Request>,
    samples: &mut animation::Samples,
    values: &mut [Option<EvalNode>],
    worlds: &mut [Mat4],
) -> Result<()> {
    sample_pose_inputs(originals, assets, animation, targets, request, samples, values, worlds)
}

/// Geometry stages use the same authored pose evaluation at canonical tick times.
pub(crate) fn sample_pose_inputs(
    originals: &[EvalNode],
    assets: &[Asset],
    animation: &animation::Plan,
    targets: &crate::targets::Plan,
    request: impl Into<vork_time::Request>,
    samples: &mut animation::Samples,
    values: &mut [Option<EvalNode>],
    worlds: &mut [Mat4],
) -> Result<()> {
    let request=request.into();
    let time=request.numeric()?;
    animation.sample_request_into(time, &request, samples)?;
    apply_pose_samples(originals, assets, animation, targets, time, samples, values, worlds)
}

/// Apply an already sampled plan, including the frame evaluator's instance offsets.
pub(crate) fn apply_pose_samples(
    originals: &[EvalNode],
    assets: &[Asset],
    animation: &animation::Plan,
    targets: &crate::targets::Plan,
    time: f32,
    samples: &animation::Samples,
    values: &mut [Option<EvalNode>],
    worlds: &mut [Mat4],
) -> Result<()> {
    animation.apply_nodes(samples, values);
    let nodes = SampledNodes::new(originals, values);
    let mut poses = std::collections::HashMap::new();
    for position in 0..originals.len() {
        let index = targets.order.get(position).copied().unwrap_or(position);
        let original = &originals[index];
        let node = &nodes[index];
        // Rule 1: an absent node, an absent asset's imported nodes included, is not evaluated.
        if animation.absent(samples, index) { continue; }
        if let Some((asset, local, root, _)) = original.embedded {
            let pose = poses.entry(asset).or_insert_with(|| assets[asset].pose_with_plan(samples, animation));
            worlds[index] = worlds[root] * pose.worlds[local];
        } else if original.dynamic {
            let world = node_world(&node.node, node.parent.map_or(Mat4::IDENTITY, |p| worlds[p]))?;
            worlds[index] = targets.aim(index, world, worlds, &node.node, |target| animation.absent(samples, target)).map_err(|error| anyhow::anyhow!("{error} at sample time {time}"))?;
        }
    }
    for (index, original) in originals.iter().enumerate() {
        if let Some((asset, local, _, _)) = original.embedded {
            // An absent imported node keeps the visibility its absence gave it: off.
            if animation.absent(samples, index) { continue; }
            if let Some(node) = values[index].as_mut() {
                node.node.visibility = lights::visibility_from_mask(poses[&asset].visibility[local]);
            }
        }
    }
    for &(camera, target) in &targets.focus {
        if animation.absent(samples, camera) { continue; }
        ensure!(!animation.absent(samples, target), "node {:?}: focus_object {:?} is absent (active); focus on a node that exists at sample time {time}",
            originals[camera].node.name, originals[target].node.name);
        if let Some(state) = &mut values[camera].as_mut().expect("focus camera is dynamic").camera {
            state.focus = crate::targets::focus_distance(worlds[camera], worlds[target]);
        }
    }
    Ok(())
}
impl<'a> SampledNodes<'a> {
    fn new(nodes: &'a [EvalNode], values: &'a [Option<EvalNode>]) -> Self {
        Self { nodes, values }
    }
}
impl std::ops::Index<usize> for SampledNodes<'_> {
    type Output = EvalNode;
    fn index(&self, id: usize) -> &EvalNode {
        self.values[id].as_ref().unwrap_or(&self.nodes[id])
    }
}

impl FrameSlot {
    /// Evaluates this sample's time-varying values. Returns the packed upload bytes,
    /// their destinations, the camera view matrix and the frame uniform.
    pub(crate) fn prepare_values(
        &mut self,
        s: &InstalledScene,
        r: &Renderer,
        request: impl Into<vork_time::Request>,
        frame_time: f32,
        shadow_changes: &mut LightingTracker,
    ) -> Result<(Vec<u8>, Vec<Upload>, Mat4, Uniform)> {
        self.sample_request=request.into();
        let time=self.sample_request.numeric()?;
        let exact=!s.footage.is_empty() || s.compute.is_some() || s.animation.has_camera_tracks() || s.environment.has_media() || s.animation.has_discrete_tracks() || s.animation.has_clocks();
        self.previous_request=if self.exposure_cut {self.sample_request.clone()}
            else if exact {self.sample_request.previous(s.render.fps)?}
            else {vork_time::Request::Numeric((time-1./s.render.fps.as_f32()).max(0.))};
        // Queue writes would all execute before a batched submission. Pack each
        // sample's values and copy them immediately before its rendering instead.
        let changes = &mut self.changes;
        changes.sample += 1;
        shadow_changes.sample += 1;
        changes.bytes.clear();
        changes.uploads.clear();
        let camera_index = s.animation.camera_at_request(&self.sample_request,s.camera)?;
        let generation = s.animation.live_generation();
        let sample_key=[time.to_bits(),camera_index as u32,generation as u32,(generation >> 32) as u32];
        let mut discrete_key=Vec::new();
        s.animation.append_discrete_key(&self.sample_request,&mut discrete_key)?;
        if s.animation.has_discrete_tracks() {
            for offset in s.animation.material_offsets().iter().copied().chain(self.offset_values.groups.iter().map(|group|group.offset)).filter(|&offset|offset!=0.) {
                s.animation.append_discrete_offset_key(&self.sample_request,offset,&mut discrete_key)?;
            }
        }
        let discrete_changed=!discrete_key.is_empty() && changes.changed(ChangeKey::DiscreteTime(0),&discrete_key);
        let time_changed = changes.changed(ChangeKey::SampleTime(0), bytemuck::cast_slice(&sample_key)) | discrete_changed;
        // The exposure holds the submission lock for retained lighting and simulation.
        // Nominal controls may use another view.
        self.reuse_values = !time_changed && self.view.sample_index > 0 && !self.view.nominal_sample;
        if time_changed {
            sample_frame(
                s,
                &self.sample_request,
                &mut self.animation_values,
                &mut self.sampled_nodes,
                &mut self.worlds,
            )?;
            s.animation
                .apply_render(&mut self.animation_values, &s.render);
        }
        // Rule 1: an absent camera has no view to render through.
        ensure!(!s.animation.absent(&self.animation_values, camera_index),
            "render.camera: camera {:?} is absent (active); render through a camera that exists at sample time {time}", s.nodes[camera_index].node.name);
        let render = self.animation_values.render.as_ref().unwrap();
        let nodes = SampledNodes::new(&s.nodes, &self.sampled_nodes);
        let cut=self.exposure_cut;
        let previous_time=self.previous_request.numeric()?;
        let next_time=time;
        let next_changed=changes.changed(ChangeKey::SampleTime(2), bytemuck::cast_slice(&sample_key)) | discrete_changed;
        if next_changed {
            sample_frame(s,&self.sample_request,&mut self.next_animation_values,&mut self.next_nodes,&mut self.next_worlds)?;
        }
        // Spatial AA/aperture samples at one time reuse the evaluated pose and render values.
        let previous_key = [previous_time.to_bits(), u32::from(cut),s.animation.camera_at_request(&self.previous_request,s.camera)? as u32,
            generation as u32,(generation >> 32) as u32];
        discrete_key.clear();
        s.animation.append_discrete_key(&self.previous_request,&mut discrete_key)?;
        if s.animation.has_discrete_tracks() {
            for offset in s.animation.material_offsets().iter().copied().chain(self.offset_values.groups.iter().map(|group|group.offset)).filter(|&offset|offset!=0.) {
                s.animation.append_discrete_offset_key(&self.previous_request,offset,&mut discrete_key)?;
            }
        }
        let previous_discrete_changed=!discrete_key.is_empty() && changes.changed(ChangeKey::DiscreteTime(1),&discrete_key);
        let previous_changed=changes.changed(
            ChangeKey::SampleTime(1), bytemuck::cast_slice(&previous_key)) | previous_discrete_changed;
        // Clamped middle times may repeat while either endpoint still changes.
        self.reuse_values &= !next_changed && !previous_changed;
        if let Some(compute)=&s.compute {
            let current=&self.sample_request;
            for (index,request) in [current,&self.previous_request].into_iter().enumerate() {
                self.reuse_values &= !changes.changed(ChangeKey::ComputeTime(index),bytemuck::cast_slice(&compute.request_key(request)?));
            }
        }
        if previous_changed {
            if cut {
                self.previous_animation_values
                    .clone_from(&self.animation_values);
                for (previous, current) in self.previous_nodes.iter_mut().zip(&self.sampled_nodes) {
                    if let (Some(previous), Some(current)) = (previous, current) {
                        previous.node.clone_from(&current.node);
                        previous.camera.clone_from(&current.camera);
                    }
                }
                self.previous_worlds.clone_from(&self.worlds);
            } else {
                sample_frame(
                    s,
                    &self.previous_request,
                    &mut self.previous_animation_values,
                    &mut self.previous_nodes,
                    &mut self.previous_worlds,
                )?;
            }
        }
        let previous_nodes = SampledNodes::new(&s.nodes, &self.previous_nodes);
        let previous_worlds = &mut self.previous_worlds;
        let camera = &nodes[camera_index];
        let previous_camera = &previous_nodes[camera_index];
        let (near, far, ortho_height) = camera_clip(camera, self.view.width as f32 / self.view.height as f32);
        let eye = self.worlds[camera_index].transform_point3(Vec3::ZERO);
        let view = node_view(camera_index, &nodes, &self.worlds)?;
        let aspect = self.view.width as f32 / self.view.height as f32;
        let physical = physical_camera(camera, aspect);
        let projection = camera_projection(camera, aspect);
        let nominal_view = view;
        let nominal_projection = projection;
        let nominal_vp = projection * view;
        // Decoupled visibility already samples the pixel and lens within a step.
        let camera_sample_count = if self.view.nominal_sample || (s.plan.decoupled && self.view.visibility_samples > 1) {
            1
        } else {
            self.view.sample_count
        };
        let projection = crate::sampling::projection(
                projection,
                self.view.internal,
                self.view.sample_index,
                camera_sample_count,
        );
        let sampled_lens = !self.view.nominal_sample;
        let (projection, view) = crate::sampling::lens(
            projection,
            view,
            physical.as_ref().filter(|_| sampled_lens),
            self.view.sample_index,
            camera_sample_count,
        );
        let camera_sample = camera_sample(self.view.internal, physical.as_ref().filter(|_| sampled_lens),
            self.view.sample_index, camera_sample_count);
        let eye = if sampled_lens && physical.as_ref().is_some_and(|p| p.radius>0.) {
            view.inverse().transform_point3(Vec3::ZERO)
        } else {
            eye
        };
        let vp = projection * view;
        let previous_view = node_view(camera_index, &previous_nodes, previous_worlds)?;
        let previous_projection = camera_projection(previous_camera, aspect);
        let previous_projection = crate::sampling::projection(
                previous_projection,
                self.view.internal,
                self.view.sample_index,
                camera_sample_count,
        );
        let previous_physical = physical_camera(previous_camera, aspect);
        let (previous_projection, previous_view) = crate::sampling::lens(
            previous_projection,
            previous_view,
            previous_physical.as_ref().filter(|_| sampled_lens),
            self.view.sample_index,
            camera_sample_count,
        );
        let previous_vp = previous_projection * previous_view;
        let next_nodes=SampledNodes::new(&s.nodes,&self.next_nodes);
        let next_camera=&next_nodes[camera_index];
        let next_projection=crate::sampling::projection(camera_projection(next_camera,aspect),self.view.internal,self.view.sample_index,camera_sample_count);
        let next_physical=physical_camera(next_camera,aspect);
        let (next_projection,next_view)=crate::sampling::lens(next_projection,node_view(camera_index,&next_nodes,&self.next_worlds)?,next_physical.as_ref().filter(|_|sampled_lens),self.view.sample_index,camera_sample_count);
        let next_vp=next_projection*next_view;
        if self.view.nominal_sample || self.view.sample_index + 1 == self.view.sample_count {
            self.camera_data = crate::CameraRecord::new(
                nominal_view,
                nominal_projection,
                render.resolution,
                ortho_height,
                far,
            );
            if calibration(camera,aspect).is_none() {self.camera_data.k=None;self.camera_data.dist=None;self.camera_data.depth_unit_m=1.;}
        }
        let lens_height=self.view.internal[1] as f32;
        let lens_scale=|active| physical.as_ref().filter(|_| active).map_or(0., |p| p.circle_scale(lens_height));
        let environment = environment_node::sample(s.environment.node, &nodes, &self.worlds)?;
        let environment_visibility = s.environment.node.map_or(constants::RAY_VISIBILITY_ALL, |node| buffers::node_visibility_mask(&nodes, node));
        let environment_constant = environment.constant(s.environment.settings.has_radiance());
        // Camera rays see the world only when it is visible to them: a hidden
        // world leaves the camera background (constant and image) black.
        let camera_world = environment_visibility & constants::RAY_CAMERA != 0;
        let background = match &render.background {
            vork_scene::Background::Color(color) => *color,
            vork_scene::Background::Mode(vork_scene::BackgroundMode::Environment) => {
                if camera_world { environment_constant } else { [0.; 3] }
            }
            vork_scene::Background::Mode(vork_scene::BackgroundMode::Transparent) => [0.; 3],
            // A 2D canvas's background; a 3D world refuses it when the scene loads.
            vork_scene::Background::Css(_) => [0.; 3],
        };
        let environment_links = s.environment.node.map(|node| s.links.illuminates(node));
        // Jittered temporal colour never writes controls; its nominal pass does.
        let controls = true;
        let composite_has_shutter = self.view.composite_steps>1 && !self.view.nominal_sample;
        let composite_steps = if self.view.nominal_sample {1} else {self.view.composite_steps.max(1)};
        let step_iterations = self.view.sample_count as u32 / composite_steps;
        let composite_samples = self.view.visibility_samples * step_iterations;
        let (composite_optics,composite_sensor) = composite_camera(camera,aspect,sampled_lens,composite_samples);
        let uniform = Uniform {
            vp,
            previous_vp,
            next_vp,
            light_vp: Mat4::IDENTITY,
            inverse_vp: vp.inverse(),
            view,
            ao: ao_uniform(s.plan.effects.indirect),
            projection: [
                projection.y_axis.y,
                projection.x_axis.x,
                match render.settings.integrator.reflections() {
                    None => constants::REFLECTIONS_OFF,
                    Some(vork_scene::Reflections::Environment) => constants::REFLECTIONS_ENVIRONMENT,
                    // Ray hits replace the environment through the same screen-correction contract.
                    Some(vork_scene::Reflections::Auto | vork_scene::Reflections::Ray) => {
                        constants::REFLECTIONS_RAY
                    }
                },
                ((u32::from(self.view.write_color && !self.view.nominal_sample)
                    * constants::OUTPUT_GPU_COLOR)
                    // 16-bit colour also publishes the 8-bit plane that exposure metering reads.
                    | (u32::from((self.view.readback_outputs.color || self.view.readback_outputs.color16) && !self.view.nominal_sample)
                        * constants::OUTPUT_COLOR_PLANE)
                    | (u32::from(self.view.readback_outputs.color16 && !self.view.nominal_sample)
                        * vork_slang::registry::OUTPUT_COLOR_PLANE16)
                    | (u32::from(controls && self.view.readback_outputs.depth_ids)
                        * constants::OUTPUT_DEPTH_IDS)
                    | (u32::from(controls && self.view.readback_outputs.normals)
                        * constants::OUTPUT_NORMALS)
                    | (u32::from(controls && self.view.readback_outputs.motion)
                        * constants::OUTPUT_MOTION)
                    | (u32::from(
                        self.view.readback_outputs.float.color && !self.view.nominal_sample,
                    ) * constants::OUTPUT_FLOAT_COLOR)
                    | (u32::from(controls && self.view.readback_outputs.float.depth)
                        * constants::OUTPUT_FLOAT_DEPTH)
                    | (u32::from(controls && self.view.readback_outputs.float.normals)
                        * constants::OUTPUT_FLOAT_NORMALS)
                    | (u32::from(controls && self.view.readback_outputs.float.motion)
                        * constants::OUTPUT_FLOAT_MOTION)
                    | (crate::compositor::channel_flags(s, &render.compositor))) as f32,
            ],
            aperture: aperture_shape(camera, aspect),
            lens: [physical.as_ref().filter(|_| sampled_lens).map_or(0., crate::sampling::aperture_radius),
                physical.as_ref().map_or(0., |p| p.focus_distance),
                2. / self.view.internal[0] as f32, 2. / self.view.internal[1] as f32],
            temporal: if self.view.nominal_sample {
                [0., 1., self.view.sample_count as f32, if render.settings.integrator.is_path() {vork_slang::registry::DENOISE_GUIDES_PATH as f32} else {0.}]
            } else {
                [self.view.sample_index as f32, self.view.sample_count as f32,
                    self.view.sample_count as f32, if render.settings.integrator.is_path() {vork_slang::registry::DENOISE_GUIDES_PATH as f32} else {0.}]

            },
            environment: [
                if s.environment.settings.has_radiance() {
                    environment.intensity
                } else {
                    0.
                },
                s.environment.initial_filter_scale(),
                5.,
                if render.background.is_environment() && camera_world {
                    1.
                } else {
                    0.
                },
            ],
            environment_rotation: environment.rotation,
            environment_color: [
                environment.color[0],
                environment.color[1],
                environment.color[2],
                0.,
            ],
            environment_mean: [
                s.environment.initial_mean()[0],
                s.environment.initial_mean()[1],
                s.environment.initial_mean()[2],
                f32::from(s.environment.lighting_is_black),
            ],
            scene: [time, self.nominal_frame as f32, f32::from(cut), render.fps.as_f32()],
            shadow_links: [0, 1, 0, 0],
            sampling: {
                // Word meanings: Frame in vork_prelude.slang. Outside decoupled colour,
                // control resolve writes the ambient controls.
                let colour = !self.view.nominal_sample && self.view.color_enabled;
                let mut words=s.sampling.uniform(if colour {self.view.sample_count * self.view.visibility_samples as usize} else {1}, if colour {self.view.visibility_samples as usize} else {1});
                words[3]=if colour && s.plan.decoupled {self.view.visibility_samples} else {0};
                words
            },
            // The shadow and ray sequences cover the render's samples: decoupled slots take
            // map taps and emitter rays for their covered samples within one exposure sample.
            integrator: [0, 0, if self.view.nominal_sample || !self.view.color_enabled {1} else {(self.view.sample_count * self.view.visibility_samples as usize) as u32}, s.shadows().ray_sequence_word(if self.view.nominal_sample || !self.view.color_enabled {1} else {self.view.sample_count * self.view.visibility_samples as usize}) | u32::from(render.settings.integrator.subsurface_method()==Some(vork_scene::v2::SubsurfaceMethod::Thickness))*vork_slang::registry::INTEGRATOR_SUBSURFACE_THICKNESS | u32::from(render.settings.previs)*vork_slang::registry::INTEGRATOR_PREVIS_CENTER],
            environment_links: [
                environment_visibility
                    | if environment_links.is_none_or(|link| link.all) {
                        constants::LIGHT_LINK_ALL
                    } else {
                        0
                    },
                environment_links.map_or(0, |link| link.mask),
                match &render.settings.integrator {
                    vork_scene::Integrator::Raster(_) => {
                        let limit=crate::constants::RASTER_TRANSMISSION_BOUNCES;
                        #[cfg(test)]
                        let limit=r.reference_transport_bounces.unwrap_or(limit);
                        limit
                    },
                    vork_scene::Integrator::Path(_) => 0,
                },
                transmission_word(render),
            ],
            medium: [s.medium_base,s.rays.as_ref().map_or(0,|r|r.emitters),s.environment_base,u32::from(environment_visibility & constants::RAY_VOLUME!=0)],
            display: [f32::from_bits(render.settings.seed),render.resolution[0] as f32,render.resolution[1] as f32,self.view.post.display_base.unwrap_or(0) as f32],
            post: [if s.has_catchers {1.} else {post_mode(render)},
                frame_time,
                // The display stage's dither switch: zero dithers; only tests turn it off (Render::dither).
                if render.dither {0.} else {1.},
                lens_scale(sampled_lens && s.plan.decoupled && self.view.visibility_samples > 1)],
            composite_projection: projection,
            composite_optics,
            composite_sensor,
            composite_shutter: if self.view.nominal_sample {[time;4]} else {self.view.composite_shutter},
            composite_step: [if composite_has_shutter {self.view.sample_index as f32} else {0.},composite_steps as f32,render.settings.samples as f32,step_iterations as f32],
            camera_sample,
            camera_program: crate::camera_program::selection(s,camera_index,camera,aspect),
            field_geometry: {let mut fields=s.rays.as_ref().map_or([0;4],|r|r.implicit);fields[2]=u32::from(!s.implicit_fields.is_empty());fields[3]=0;fields},
            eye: eye.extend(1.).to_array(),
            background,
            light_count: 0.,
            radiance: [
                environment_constant[0],
                environment_constant[1],
                environment_constant[2],
                0.,
            ],
            params: [near, far, 0., 0.],
            flags: [
                if ortho_height.is_some() { 1. } else { 0. },
                0.,
                if render.background.transparent() {
                    1.
                } else {
                    0.
                },
                if s.plan.effects.indirect { 1. } else { 0. },
            ],
        };
        self.view.post.compositor_inputs.upload(s, frame_time, changes)?;
        if let Some(buffer) = &self.view.post.display_inputs {
            s.display_program.upload(buffer, frame_time, changes);
        }
        changes.upload(&self.view.uniform, 0, bytemuck::bytes_of(&uniform));
        if !self.reuse_values {
        // Animated material inputs write this frame's copy of their records; instances
        // with a time offset have their own record copies, sampled at time + offset.
        if let Some(frame) = &self.materials {
            let mut write = |values: &animation::Samples, offset: f32| {
                for (binding, value) in s.animation.material_values(values, offset) {
                    if binding.storage.is_some() {
                        let bytes = programs::sample_words(value, binding);
                        changes.upload(&frame.records, binding.slot as u64 * 4, bytemuck::cast_slice(&bytes));
                        continue;
                    }
                    let (crate::animation::Sample::Numeric(value), Some((at, width))) =
                        (value, materials::value_location(binding.slot, binding.parameter))
                    else {
                        continue;
                    };
                    changes.upload(&frame.records, at, bytemuck::cast_slice(&value[..width]));
                }
            };
            write(&self.animation_values, 0.);
            for &offset in s.animation.material_offsets() {
                let samples = &mut self.offset_values.samples;
                s.animation.sample_material_request(&self.sample_request,time,offset,samples)?;
                write(samples, offset);
            }
        }
        let mut local_bounds = vec![None; s.draws.len()];
        let mut displacement_spaces=if s.displacement.is_some() {vec![[Mat4::IDENTITY;3];s.draws.len()]} else {Vec::new()};
        let OffsetValues {
            samples: offset_samples,
            nodes: offset_nodes,
            worlds: offset_worlds,
            groups,
        } = &mut self.offset_values;
        for group in groups {
            for (pose,pose_time) in [time,previous_time,next_time].into_iter().enumerate() {
                let current=pose==0;
                let mut poses = std::collections::HashMap::new();
                if group.offset != 0. && (current || !cut) {
                    s.animation.sample_nodes_request(
                        if pose==1 {&self.previous_request}else{&self.sample_request},
                        pose_time, group.offset,
                        offset_samples,
                        &group.nodes,
                        &group.assets,
                    )?;
                    s.animation
                        .apply_nodes_subset(offset_samples, offset_nodes, &group.nodes);
                    let shifted_nodes = SampledNodes::new(&s.nodes, offset_nodes);
                    let order = if group.world_order.is_empty() { &group.nodes } else { &group.world_order };
                    for &index in order {
                        let node = &shifted_nodes[index];
                        if s.animation.absent(offset_samples, index) { continue; }
                        if let Some((asset, local, root, _)) = s.nodes[index].embedded {
                            let pose = poses
                                .entry(asset)
                                .or_insert_with(|| s.assets[asset].pose_sampled(offset_samples));
                            offset_worlds[index] = offset_worlds[root] * pose.worlds[local];
                        } else {
                            let world = node_world(&node.node, node.parent.map_or(Mat4::IDENTITY, |p| offset_worlds[p]))?;
                            offset_worlds[index] = s.targets.aim(index, world, offset_worlds, &node.node, |target| s.animation.absent(offset_samples, target)).map_err(|error| anyhow::anyhow!("{error} at sample time {}", pose_time + group.offset))?;
                        }
                    }
                }
                let (sampled, values, worlds) = if group.offset == 0. {
                    if current {
                        (
                            &self.animation_values,
                            self.sampled_nodes.as_slice(),
                            self.worlds.as_slice(),
                        )
                    } else if pose==1 {
                        (
                            &self.previous_animation_values,
                            self.previous_nodes.as_slice(),
                            self.previous_worlds.as_slice(),
                        )
                    } else {(&self.next_animation_values,self.next_nodes.as_slice(),self.next_worlds.as_slice())}
                } else {
                    (
                        &*offset_samples,
                        offset_nodes.as_slice(),
                        offset_worlds.as_slice(),
                    )
                };
                let sampled_nodes = SampledNodes::new(&s.nodes, values);
                for &object in &group.objects {
                    let root = &s.roots[object];
                    let mut world = root.world(worlds);
                    // Rule 1: an absent object is hidden without evaluating its asset's pose.
                    let absent = s.animation.absent(sampled, root.node);
                    let hidden_scale = absent || buffers::node_collapsed(&sampled_nodes, root.node)
                        || root.imported_pose.is_some_and(|(asset, local)| {
                            poses.entry(asset).or_insert_with(|| s.assets[asset].pose_sampled(sampled)).collapsed[local]
                        });
                    if hidden_scale {
                        world.x_axis = glam::Vec4::ZERO; world.y_axis = glam::Vec4::ZERO; world.z_axis = glam::Vec4::ZERO;
                    }
                    let value = &mut self.object_values[object];
                    if current {
                        value.model = world;
                        value.metadata[1] = if hidden_scale {0} else {
                            buffers::node_visibility_mask(&sampled_nodes, root.node)
                                & if let Some((asset, local)) = root.imported_pose {
                                    poses.entry(asset).or_insert_with(|| s.assets[asset].pose_sampled(sampled)).visibility[local]
                                } else { root.imported_visibility }
                        };
                        value.metadata[1] |= buffers::node_matte_flags(&sampled_nodes,root.node);
                        if value.metadata[1]!=0 {value.metadata[1] |= s.objects[object].metadata[1]&vork_slang::registry::OBJECT_IMPLICIT;}
                    } else if pose==1 {
                        value.previous_model = world;
                    } else {value.next_model=world;}
                }
                let offset = group.offset;
                let offset_bits = offset.to_bits();
                let mut palettes = std::collections::HashMap::new();
                for &draw_id in &group.deforms {
                    let draw = &s.draws[draw_id];
                    // Rule 1: an absent object's deformation is not evaluated; its draw is hidden.
                    if s.animation.absent(sampled, s.roots[draw.object as usize].node) { continue; }
                    let deform = draw.deform.as_ref().unwrap();
                    let key = if let Some(native) = &deform.native {
                        (true, native.mesh, 0, None, offset_bits)
                    } else {
                        (
                            false,
                            deform.asset.expect("glTF deformation asset"),
                            deform.node,
                            deform.skin,
                            offset_bits,
                        )
                    };
                    let (matrices, weights) = palettes.entry(key).or_insert_with(|| {
                        if let Some(native) = &deform.native {
                            let nodes = SampledNodes::new(&s.nodes, values);
                            let NodeKind::Mesh { morph_weights, .. } =
                                &nodes[native.mesh].node.kind
                            else {
                                unreachable!("native deformation belongs to a mesh node")
                            };
                            native.palette(worlds, morph_weights)
                        } else {
                            let asset = deform.asset.expect("glTF deformation asset");
                            let pose = poses
                                .entry(asset)
                                .or_insert_with(|| s.assets[asset].pose_sampled(sampled));
                            (
                                s.assets[asset].matrices(pose, deform.node, deform.skin),
                                pose.weights[deform.node].clone(),
                            )
                        }
                    });
                    if !displacement_spaces.is_empty() && deform.native.is_none() {
                        displacement_spaces[deform.job][pose]=poses[&deform.asset.unwrap()].worlds[deform.node];
                    }
                    let bounds = self.geometry_chain.upload_pose(s,draw_id,pose,matrices,weights,changes)?;
                    if current {local_bounds[deform.job]=bounds;}
                }
            }
        }
        let displacement_bounds = self.geometry_chain.evaluate(r, s, &self.geometry_bounds, changes,
            &self.sample_request, &self.previous_request, &mut self.simulation_checkpoint,
            self.materials.as_ref(), &self.object_values, &displacement_spaces, &self.queries,
            time, previous_time, next_time)?;
        for &object in &s.dynamic_objects {
            if let Some((compute, offset)) = self.compute.as_ref().and_then(|c| c.parent_offset(object).map(|offset| (c, offset))) {
                changes.upload(&compute.parents, offset, bytemuck::bytes_of(&self.object_values[object]));
            } else {
                changes.upload(&self.objects, (object * std::mem::size_of::<GpuObject>()) as u64,
                    bytemuck::bytes_of(&self.object_values[object].gpu()));
            }
        }
        let mut shadow_local_bounds=s.rays.as_ref().filter(|r|r.dynamic_shadow_bounds()).map(|_|Vec::with_capacity(s.draws.len()));
        for (index, draw) in s.draws.iter().enumerate() {
            let displaced=displacement_bounds.get(index).copied().flatten();
            let gpu_bounds=self.deforms[index].as_ref().is_some_and(|g|g.gpu_bounds.is_some());
            let evaluated=if gpu_bounds {None} else {displaced.or_else(||draw.deform.as_ref().and_then(|deform|local_bounds[deform.job]))};
            self.geometry_bounds[index]=displaced;
            if let Some(bounds)=&mut shadow_local_bounds {
                bounds.push(evaluated.unwrap_or_else(||if draw.evaluated() {(Vec3::splat(f32::NEG_INFINITY),Vec3::splat(f32::INFINITY))} else {draw.bounds}));
            }
            if let Some((lo,hi))=displaced {
                let bounds=[lo.x,lo.y,lo.z,hi.x,hi.y,hi.z,1.];
                let records=&self.materials.as_ref().unwrap().records;
                for object in draw.instances.clone() {
                    let at=(self.object_values[object as usize].metadata[2] as u64+6)*4;
                    changes.upload(records,at,bytemuck::cast_slice(&bounds));
                }
            } else if s.rays.is_some() {
                if let (Some((lo,hi)),Some(frame))=(evaluated,self.materials.as_ref()) {
                    // The same current geometry bounds supply ray-origin uncertainty.
                    let object=draw.instances.start as usize;
                    let offset=4*u64::from(self.object_values[object].metadata[2]+6);
                    changes.upload(&frame.records,offset,bytemuck::cast_slice(&[lo.x,lo.y,lo.z,hi.x,hi.y,hi.z]));
                }
            }

        }
        if let (Some(rays),Some(local))=(&self.rays,shadow_local_bounds) {rays.prepare_shadow_bounds(s,&local,changes);}
        {
            let mut lights = s.light_plan.evaluate(
                &nodes,
                &self.worlds,
                &s.links,
            )?;
            if s.rays.as_ref().is_some_and(|rays| rays.opaque) {
                for light in &mut lights {
                    let flags = &mut light.shadow_offsets[constants::LIGHT_RAY_WORD];
                    if *flags & constants::LIGHT_RAY_SHADOWED_BY_ALL != 0 {
                        *flags |= vork_slang::registry::LIGHT_RAY_OPAQUE;
                    }
                }
            }
            for (index,&id) in s.light_plan.nodes.iter().enumerate() {
                if let Some(&address)=s.emitter_profiles.get(&id) {
                    let source=node_view(id,&nodes,&self.worlds)?.inverse();
                    let mut rotation=source.to_scale_rotation_translation().1;
                    if rotation.w<0. {rotation=-rotation;}
                    lights[index].emission=[rotation.x,rotation.y,rotation.z,f32::from_bits(address)];
                }
            }
            self.evaluated_lights.clone_from(&lights);
            if let Some(subset) = &s.main_subset {
                for (light, included) in lights.iter_mut().zip(&subset.lights) {
                    if !included {
                        light.radiance[..3].fill(0.);
                    }
                }
            }
            let mut u = uniform;
            u.light_count = lights.len() as f32;
            u.shadow_links[2] = u32::from(lights::needs_receiver_masks(&lights));
                self.receiver_masks = u.shadow_links[2];
            u.radiance[3] = 1.;
            changes.upload(&self.view.lighting.uniform, 0, bytemuck::bytes_of(&u));
            self.view.lighting.upload_lights(changes, &lights)?;
        }
        } else {
            // Lighting is invariant, but projection/lens/phase/output fields are not.
            let mut u = uniform;
            u.light_count = self.evaluated_lights.len() as f32;
            u.shadow_links[2] = self.receiver_masks;
            u.radiance[3] = 1.;
            changes.upload(&self.view.lighting.uniform, 0, bytemuck::bytes_of(&u));
        }
        if let (Some(p),Some(slot))=(&s.fibre,&self.fibre) {
            slot.prepare(&p.storage(),s,&self.object_values,&self.evaluated_lights,uniform,self.materials.as_ref().map(|m| &m.records),changes,shadow_changes);
        }
        let spatial_camera = (!sampled_lens || physical.as_ref().is_none_or(|p| p.radius==0.))
            && (self.view.sample_count > 1 || self.view.visibility_samples > 1) && !self.view.nominal_sample
            && render.motion_blur.as_ref().is_none_or(|m|m.shutter==0.);
        let selection_vp = if spatial_camera || uniform.post[3]>0. {
            let radius=crate::sampling::lens_selection_radius(uniform.post[3], uniform.lens[1], near, far);
            crate::sampling::selection_projection(nominal_vp, self.view.internal, radius)
        } else { uniform.vp };
        self.view.camera_cull.prepare_exposure(selection_vp, &self.object_values,
            &self.geometry_bounds, changes, self.reuse_values && spatial_camera);
        if !self.view.transparent_order.is_empty() {
            let indirect = self.view.transparent_indirect.is_some();
            let sorted = passes::transparent::sort(
                s,
                &self.object_values,
                &mut self.view.transparent_order,
                view,
                indirect,
            );
            if let (Some(objects), Some(indirect), Some((values, args, runs))) = (
                &self.view.transparent_instances,
                &self.view.transparent_indirect,
                sorted,
            ) {
                // Only the draw-slot mapping changes; the shared object table
                // retains canonical identities for raster shading and ray queries.
                for (i, value) in values {
                    changes.upload(
                        objects,
                        (i * std::mem::size_of::<u32>()) as u64,
                        bytemuck::bytes_of(&value),
                    );
                }
                changes.upload(indirect, 0, bytemuck::cast_slice(&args));
                self.view.transparent_runs = runs;
            }
        }
        Ok((
            std::mem::take(&mut changes.bytes),
            std::mem::take(&mut changes.uploads),
            view,
            uniform,
        ))
    }
    /// Copies this sample's packed values into place at the start of its commands.
    pub(crate) fn copy_values(
        &mut self,
        r: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        upload_bytes: &[u8],
        uploads: &[Upload],
        batch: bool,
    ) {
        if upload_bytes.is_empty() { return; }
        let size = upload_bytes.len() as u64;
        // Batched commands must retain distinct immutable sources until publication.
        // Streaming commands submit before the next sample and can reuse their source.
        if batch || self.sample_upload.size() < size {
            self.sample_upload = crate::gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
                label: Some("sample values"), size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if batch {
            self.exposure_upload.push((self.sample_upload.clone(), upload_bytes.to_vec()));
        } else {
            r.queue.write_buffer(&self.sample_upload, 0, upload_bytes);
        }
        for (destination, offset, source, size) in uploads {
            encoder.copy_buffer_to_buffer(
                &self.sample_upload,
                *source,
                destination,
                *offset,
                *size,
            );
        }
    }
}

pub(crate) fn deformed_bounds(
    bounds: &[Vec<(Vec3, Vec3)>],
    matrices: &[Mat4],
    weights: &[f32],
) -> (Vec3, Vec3) {
    let mut result = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for (matrix, intervals) in matrices.iter().zip(bounds) {
        let (mut lo, mut hi) = intervals[0];
        if !lo.is_finite() {
            continue;
        }
        for (index, &(dlo, dhi)) in intervals[1..].iter().enumerate() {
            let weight = weights.get(index).copied().unwrap_or(0.);
            lo += (dlo * weight).min(dhi * weight);
            hi += (dlo * weight).max(dhi * weight);
        }
        let transformed = crate::shadow_envelope::transformed(*matrix, (lo, hi));
        result.0 = result.0.min(transformed.0);
        result.1 = result.1.max(transformed.1);
    }
    result
}

fn aperture_shape(node: &EvalNode, aspect: f32) -> [f32;4] {
    let Some(c)=node.camera.as_ref().and_then(|c|c.calibration(aspect).ok().flatten()) else {return [0.,0.,1.,0.]};
    [c[2][2],c[2][3],c[3][0],f32::from(c[0][2]!=0. || c[0][3]!=0.)]
}

/// The pixel jitter and lens offset `sampling::projection` and `sampling::lens`
/// give every pixel of this camera sample (`Uniform::camera_sample`).
fn camera_sample(resolution: [u32; 2], lens: Option<&crate::sampling::Lens>, index: usize, count: usize) -> [f32; 4] {
    let jitter = if count > 1 { crate::sampling::clip_offset(resolution, index, count) } else { [0.; 2] };
    let offset = crate::sampling::lens_offset(lens, index, count);
    [jitter[0], jitter[1], offset.x, offset.y]
}

fn calibration(node: &EvalNode, aspect: f32) -> Option<[[f32;4];4]> {
    node.camera.as_ref()?.calibration(aspect).expect("validated camera calibration")
}
fn composite_camera(node: &EvalNode, aspect: f32, sampled: bool, samples: u32) -> ([f32;4],[f32;4]) {
    let Some(c)=calibration(node,aspect) else {return ([0.;4],[0.;4])};
    let sampled=sampled && c[2][0]>0.;
    ([c[2][0],0.,c[2][1],1.], [c[0][0],c[0][1],f32::from(sampled),if sampled {samples.max(1) as f32} else {1.}])
}
fn physical_camera(node: &EvalNode, aspect: f32) -> Option<crate::sampling::Lens> {
    let c=calibration(node,aspect)?;
    if c[1][2]!=0. {return None;}
    Some(crate::sampling::Lens {radius:c[2][0],screen_scale:[c[0][0],c[0][1]],
        focus_distance:c[2][1],aperture_blades:c[2][2] as u32,aperture_rotation:c[2][3],aperture_ratio:c[3][0]})
}
fn camera_clip(node: &EvalNode, aspect: f32) -> (f32,f32,Option<f32>) {
    calibration(node,aspect).map_or((0.,65535.,None),|c|(c[1][0],c[1][1],(c[1][2]!=0.).then_some(2./c[0][1])))
}
pub(crate) fn camera_projection(node: &EvalNode, aspect: f32) -> Mat4 {
    let Some(c)=calibration(node,aspect) else {return Mat4::IDENTITY;};
    let [near,far,ortho,_]=c[1];
    let mut p=if ortho!=0. {Mat4::orthographic_rh(-1./c[0][0],1./c[0][0],-1./c[0][1],1./c[0][1],near,far)}
        else {Mat4::perspective_rh(2.*(1./c[0][1]).atan(),c[0][1]/c[0][0],near,far)};
    let mut shift=Mat4::IDENTITY;shift.w_axis.x=c[0][2];shift.w_axis.y=c[0][3];p=shift*p;p
}

/// Compose local TRS; the installed target plan applies aiming afterward.
pub(super) fn node_world(node: &Node, parent: Mat4) -> Result<Mat4> {
    Ok(parent * node.transform.matrix())
}

pub(super) fn node_view(
    id: usize,
    nodes: &(impl std::ops::Index<usize, Output = EvalNode> + ?Sized),
    worlds: &[Mat4],
) -> Result<Mat4> {
    let world = worlds[id];
    let eye = world.transform_point3(Vec3::ZERO);
    // A collapsed frame has no recoverable orientation. Compose authored rotations
    // without scale, as for Blender's analytic lights and cameras.
    let orientation = if buffers::node_collapsed(nodes, id) || crate::collapsed(world) {
        let mut chain = Vec::new();
        let mut current = Some(id);
        while let Some(index) = current {
            chain.push(index);
            current = nodes[index].parent;
        }
        let mut frame = Mat4::IDENTITY;
        for index in chain.into_iter().rev() {
            let node = &nodes[index].node;
            if let Some(target) = &node.target {
                let target = match target {
                    vork_scene::AimTarget::Point(point) => Vec3::from_array(*point),
                    vork_scene::AimTarget::Node(name) => {
                        let target = (0..worlds.len()).find(|&i| nodes[i].node.name == *name)
                            .context("view target lost its installed node")?;
                        worlds[target].w_axis.truncate()
                    }
                };
                let origin = worlds[index].w_axis.truncate();
                let direction = (target - origin).try_normalize().context("view target coincides with node origin")?;
                let up = if direction.dot(Vec3::Y).abs() > constants::ORIENTATION_UP_PARALLEL {Vec3::Z} else {Vec3::Y};
                frame = Mat4::look_to_rh(origin, view_axis(origin, direction), up).inverse();
            }
            frame *= Mat4::from_quat(glam::Quat::from_array(node.transform.resolved_rotation()).normalize());
        }
        frame
    } else {world};
    let direction = view_axis(eye, orientation.transform_vector3(Vec3::NEG_Z).normalize());
    let mut up = orientation.transform_vector3(Vec3::Y).normalize();
    if direction.dot(up).abs() > constants::ORIENTATION_UP_PARALLEL {
        up = orientation.transform_vector3(Vec3::Z).normalize();
        if direction.dot(up).abs() > constants::ORIENTATION_UP_PARALLEL {
            up = orientation.transform_vector3(Vec3::X).normalize();
        }
    }
    Ok(Mat4::look_to_rh(eye, direction, up))
}

/// Keep the established eye/target frame when the unit target retains its
/// direction. At large translations its addition can quantize or erase that
/// direction; construct directly once the round-trip exceeds the orientation
/// tolerance instead of accepting an inaccurate or non-finite frame.
pub(super) fn view_axis(eye: Vec3, direction: Vec3) -> Vec3 {
    let axis = (eye + direction) - eye;
    let rounded = axis.normalize_or_zero();
    if rounded.distance_squared(direction) > constants::ORIENTATION_EPSILON {
        direction
    } else {
        rounded
    }
}

#[cfg(test)]
mod orientation_tests {
    use super::*;
    #[test]
    fn change_keys_detect_length_changes_and_equal_prefixes() {
        let mut changes = super::LightingTracker::default();
        let key = super::LightingKey::Probes;
        assert!(changes.changed(key.clone(), &[1, 2, 3, 4]));
        assert!(!changes.changed(key.clone(), &[1, 2, 3, 4]));
        assert!(changes.changed(key.clone(), &[1, 2]));
        assert!(!changes.changed(key.clone(), &[1, 2]));
        assert!(changes.changed(key, &[1, 2, 3, 4]));
    }

    #[test]
    fn camera_view_accepts_zero_and_small_nonzero_scale() {
        for scale in [0.,1e-8] {
            let world=Mat4::from_scale(Vec3::new(1.,1.,scale));
            let nodes=vec![EvalNode { camera: None,node:vork_scene::v2::from_test_value(serde_json::json!({"type":"camera","scale":[1.,1.,scale]})).unwrap(),
                world,parent:None,dynamic:false,embedded:None}];
            let view=node_view(0,&nodes,&[world]).unwrap();
            assert!(view.is_finite());
            assert!(view.abs_diff_eq(Mat4::IDENTITY,1e-6));
        }
    }

    #[test]
    fn zero_scale_light_retains_target_orientation() {
        let node: Node = vork_scene::v2::from_test_value(serde_json::json!({
            "type":"light","shape":"sun","scale":[0,0,0],"target":[1,0,0]})).unwrap();
        let world = node.transform.matrix();
        let nodes = vec![EvalNode { camera: None,node,world,parent:None,dynamic:false,embedded:None}];
        let frame = node_view(0,&nodes,&[world]).unwrap().inverse();
        assert!(frame.is_finite());
        assert!(frame.transform_vector3(Vec3::NEG_Z).abs_diff_eq(Vec3::X,1e-6));
    }

    #[test]
    fn rounded_view_axis_keeps_unit_length_when_only_magnitude_is_quantized() {
        let direction = Vec3::ONE.normalize();
        let eye = Vec3::splat(1_048_576.);
        let axis = view_axis(eye, direction);
        assert!(axis.abs_diff_eq(direction, 1e-6));
        let view = Mat4::look_to_rh(eye, axis, Vec3::Y);
        for basis in [Vec3::X, Vec3::Y, Vec3::Z] {
            assert!((view.transform_vector3(basis).length() - 1.).abs() < 1e-6);
        }
    }

    #[test]
    fn ordinary_camera_view_retains_the_shadow_fit_frame() {
        let eye = Vec3::new(7., 5., 11.);
        let rotation = glam::Quat::from_xyzw(-0.14237286, 0.27649665, 0.041459076, 0.9495055);
        let world = Mat4::from_rotation_translation(rotation, eye);
        let nodes = vec![EvalNode { camera: None, node: vork_scene::v2::from_test_value(serde_json::json!({"type":"camera"})).unwrap(),
            world, parent: None, dynamic: false, embedded: None }];
        let target = eye + world.transform_vector3(Vec3::NEG_Z).normalize();
        let expected = Mat4::look_at_rh(eye, target, world.transform_vector3(Vec3::Y).normalize());
        assert_eq!(node_view(0, &nodes, &[world]).unwrap(), expected);
    }

    #[test]
    fn camera_orientation_is_independent_of_world_translation() {
        let rotation=glam::Quat::from_rotation_y(0.3)*glam::Quat::from_rotation_z(0.2);
        for offset in [0.,10_000_000.,100_000_000.] {
            let world=Mat4::from_rotation_translation(rotation,Vec3::splat(offset));
            let nodes=vec![EvalNode { camera: None,node:vork_scene::v2::from_test_value(serde_json::json!({"type":"camera"})).unwrap(),
                world,parent:None,dynamic:false,embedded:None}];
            let view=node_view(0,&nodes,&[world]).unwrap();
            for axis in [Vec3::X,Vec3::Y,Vec3::Z] {
                assert!(view.transform_vector3(rotation*axis).abs_diff_eq(axis,1e-6));
            }
        }
    }

    #[test]
    fn rotation_roll_and_parent_space_transform() {
        let node: Node = vork_scene::v2::from_test_value(serde_json::json!({
            "type": "camera", "rotation": [0.,0.,0.70710677,0.70710677]
        }))
        .unwrap();
        let world = node.transform.matrix();
        let mut nodes = vec![EvalNode { camera: None,
            node,
            world,
            parent: None,
            dynamic: false,
            embedded: None,
        }];
        let rolled = node_view(0, &nodes, &[world]).unwrap().inverse();
        assert!(rolled.transform_vector3(Vec3::X).abs_diff_eq(Vec3::Y, 1e-5));
        assert!(
            rolled
                .transform_vector3(Vec3::NEG_Z)
                .abs_diff_eq(Vec3::NEG_Z, 1e-5)
        );
        // Camera orientation composes with the parent transform.
        nodes[0].node.transform.rotation =
            Some(glam::Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2).to_array());
        nodes[0].parent = Some(1);
        let parent = Mat4::from_translation(Vec3::new(5., 2., 3.));
        nodes.push(EvalNode { camera: None,
            node: vork_scene::v2::from_test_value(serde_json::json!({"type":"node","children":[]})).unwrap(),
            world: parent,
            parent: None,
            dynamic: false,
            embedded: None,
        });
        let aimed_world = node_world(&nodes[0].node, parent).unwrap();
        let aimed = node_view(0, &nodes, &[aimed_world, parent])
            .unwrap()
            .inverse();
        assert!(
            aimed
                .transform_vector3(Vec3::NEG_Z)
                .abs_diff_eq(Vec3::X, 1e-5)
        );
    }
}

#[cfg(test)]
mod upload_cache_tests {
    use super::UploadCache;
    #[test]
    fn external_gpu_replacement_invalidates_outputs_and_derivation_together() {
        use super::{ChangeKey, ChangeTracker};
        let (device, _) = wgpu::Device::noop(&Default::default());
        let [commands, instances, other] = std::array::from_fn(|_| device.create_buffer(
            &wgpu::BufferDescriptor { label: None, size: 16,
                usage: wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
        let key = ChangeKey::CameraSelection(commands.clone());
        let mut changes = ChangeTracker::default();
        changes.sample = 1;
        changes.changed(ChangeKey::CameraSelection(other.clone()), b"other camera");
        changes.upload(&other, 0, &[7; 16]);
        // A replacement can precede the first CPU publication, occur repeatedly
        // in one sample, or follow retained values from an earlier sample.
        for advance in [false, false, true, false] {
            if advance { changes.sample += 1; }
            changes.replace_gpu(&[&commands, &instances], key.clone());
            assert!(changes.changed(key.clone(), b"same camera"));
            assert!(!changes.changed(key.clone(), b"same camera"));
            changes.bytes.clear();
            changes.uploads.clear();
            changes.upload(&commands, 0, &[0; 16]);
            changes.upload(&instances, 4, &[2; 8]);
            assert_eq!(changes.uploads.len(), 2, "equal prior CPU bytes must republish every destination");
            assert!(changes.buffer_dirty(&commands));
            assert!(changes.buffer_dirty(&instances));
            changes.bytes.clear();
            changes.uploads.clear();
            changes.upload(&commands, 4, &[0; 4]);
            changes.upload(&instances, 8, &[2; 4]);
            changes.upload(&other, 0, &[7; 16]);
            assert!(changes.uploads.is_empty(), "CPU history resumes and unrelated outputs stay retained");
            assert!(!changes.changed(ChangeKey::CameraSelection(other.clone()), b"other camera"));
        }
    }

    #[test]
    fn upload_publication_survives_merging_and_drain_until_the_next_sample() {
        let (device, _) = wgpu::Device::noop(&Default::default());
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 16, usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let other = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 16, usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut changes = super::ChangeTracker::default();
        changes.sample = 1;
        assert!(!changes.buffer_dirty(&buffer));
        changes.upload(&buffer, 4, &[0; 4]); // First zero write is a publication.
        changes.upload(&buffer, 8, &[1; 4]); // Adjacent ranges merge.
        assert_eq!(changes.uploads.len(), 1);
        assert!(changes.buffer_dirty(&buffer));
        assert!(!changes.buffer_dirty(&other));
        changes.uploads.clear(); // Submission takes the work list, not its history.
        changes.bytes.clear();
        assert!(changes.buffer_dirty(&buffer));
        changes.sample += 1;
        changes.upload(&buffer, 4, &[0; 4]);
        assert!(!changes.buffer_dirty(&buffer));
        assert!(changes.uploads.is_empty());
        changes.upload(&buffer, 6, &[2; 4]); // Partial overlap.
        changes.upload(&buffer, 6, &[2; 4]); // Unchanged write cannot clear dirty.
        assert!(changes.buffer_dirty(&buffer));
        assert_eq!(changes.uploads.len(), 1);
        changes.sample += 1;
        assert!(!changes.buffer_dirty(&buffer));
    }

    #[test]
    fn partial_overlaps_and_first_zero_writes_remain_dirty() {
        let mut cache = UploadCache::default();
        assert!(cache.changed(4, &[1, 2, 3, 4]));
        assert!(!cache.changed(4, &[1, 2, 3, 4]));
        assert!(cache.changed(6, &[0, 0]));
        assert!(!cache.changed(6, &[0, 0]));
        assert!(cache.changed(4, &[1, 2, 3, 4]));
        assert!(cache.changed(0, &[0, 0, 0, 0]));
        assert!(!cache.changed(0, &[0, 0, 0, 0, 1, 2, 3, 4]));
    }
}

#[cfg(test)]
mod deformation_bounds_tests {
    use super::*;
    #[test]
    fn two_joint_bounds_include_positive_and_negative_morphs() {
        let bounds = vec![
            vec![(Vec3::ZERO, Vec3::ONE), (Vec3::ZERO, Vec3::X)],
            vec![(Vec3::ZERO, Vec3::ONE), (Vec3::ZERO, Vec3::Y)],
        ];
        let matrices = [
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(2., 0., 0.)),
        ];
        let (lo, hi) = deformed_bounds(&bounds, &matrices, &[0.5]);
        assert_eq!(lo, Vec3::ZERO);
        assert_eq!(hi, Vec3::new(3., 1.5, 1.));
        let (lo, hi) = deformed_bounds(&bounds, &matrices, &[-0.5]);
        assert_eq!(lo, Vec3::new(-0.5, -0.5, 0.));
        assert_eq!(hi, Vec3::new(3., 1., 1.));
    }
}

/// frame.environment_links.w: environment refraction (bit 0) and the solid
/// glass thickness technique (2 approximate, 0 mesh exits; transmission.slang).
fn transmission_word(render: &vork_scene::Render) -> u32 {
    render.settings.integrator.raster().map_or(0, |r| u32::from(r.refraction == vork_scene::Refraction::Environment))
}
/// A view reads the scene values already evaluated for this shared frame sample.
/// It never samples tracks, poses, material inputs, or shadow state again.
pub(crate) fn capture_uniform(
    s: &InstalledScene,
    frame: &FrameSlot,
    image: &crate::render_images::InstalledImageRender,
    mut u: Uniform,
    sample: u32,
) -> Result<Uniform> {
    let samples = image.exposure_samples;
    let visibility_samples = image.visibility_samples;
    // Rule 1: an absent camera has no view to render through.
    ensure!(!s.animation.absent(&frame.animation_values, image.camera),
        "images.{}.render.camera: camera {:?} is absent (active); a render image renders through a camera that exists at sample time {}",
        image.name, s.nodes[image.camera].node.name, frame.sample_request.numeric()?);
    let nodes = SampledNodes::new(&s.nodes, &frame.sampled_nodes);
    let previous_nodes = SampledNodes::new(&s.nodes, &frame.previous_nodes);
    let node = &nodes[image.camera];
    let previous_node = &previous_nodes[image.camera];
    let render = &image.settings;
    let [w, h] = render.resolution;
    let aspect = w as f32 / h as f32;
    let lens = physical_camera(node, aspect);
    let projection = sampling::projection(
        camera_projection(node, aspect),
        render.resolution,
        sample as usize,
        samples as usize,
    );
    let (projection, view) = sampling::lens(
        projection,
        node_view(image.camera, &nodes, &frame.worlds)?,
        lens.as_ref(),
        sample as usize,
        samples as usize,
    );
    u.camera_program = crate::camera_program::selection(s, image.camera,node,aspect);
    u.camera_sample = camera_sample(render.resolution, lens.as_ref(), sample as usize, samples as usize);
    let previous_projection = sampling::projection(
        camera_projection(previous_node, aspect),
        render.resolution,
        sample as usize,
        samples as usize,
    );
    let previous_lens = physical_camera(previous_node, aspect);
    let (previous_projection, previous_view) = sampling::lens(
        previous_projection,
        node_view(image.camera, &previous_nodes, &frame.previous_worlds)?,
        previous_lens.as_ref(),
        sample as usize,
        samples as usize,
    );
    let (near, far, ortho) = camera_clip(node, aspect);
    u.composite_projection = projection;
    (u.composite_optics,u.composite_sensor) = composite_camera(node,aspect,true,visibility_samples * samples);
    u.composite_shutter = [u.scene[0];4];
    u.composite_step = [0.,1.,render.settings.samples as f32,samples as f32];
    u.display[1]=w as f32;u.display[2]=h as f32;
    u.vp = projection * view;
    u.previous_vp = previous_projection * previous_view;
    // The next pose endpoint is the current sample. It belongs to this capture
    // camera too; inheriting the parent matrix leaks its camera and jitter.
    u.next_vp = u.vp;
    u.inverse_vp = u.vp.inverse();
    u.view = view;
    u.eye = view
        .inverse()
        .transform_point3(Vec3::ZERO)
        .extend(1.)
        .to_array();
    u.params[0] = near;
    u.params[1] = far;
    u.projection[0] = projection.y_axis.y;
    u.projection[1] = projection.x_axis.x;
    u.projection[3] = (constants::OUTPUT_FLOAT_COLOR | (crate::compositor::channel_flags(s, &render.compositor))) as f32;
    u.sampling = s.sampling.uniform(render.settings.samples as usize, visibility_samples as usize);
    u.sampling[3] = if s.plan.decoupled { visibility_samples } else { 0 };
    u.environment_links[3] = transmission_word(render);
    u.post[2] = 0.;
    let lens_scale=|active| lens.as_ref().filter(|_| active).map_or(0., |p| p.circle_scale(h as f32));
    u.post[3] = lens_scale(s.plan.decoupled && visibility_samples > 1);
    u.integrator[0] = 0;
    u.integrator[1] = 0;
    u.integrator[2] = render.settings.samples;
    u.integrator[3] = s.shadows().ray_sequence_word(render.settings.samples as usize) | u32::from(render.settings.integrator.subsurface_method()==Some(vork_scene::v2::SubsurfaceMethod::Thickness))*vork_slang::registry::INTEGRATOR_SUBSURFACE_THICKNESS;
    u.temporal = [sample as f32, samples as f32, samples as f32, if render.settings.integrator.is_path() {vork_slang::registry::DENOISE_GUIDES_PATH as f32} else {0.}];

    let camera_world = u.environment_links[0] & constants::RAY_CAMERA != 0;
    u.environment[3] =
        f32::from(render.background.is_environment() && camera_world);
    u.ao = ao_uniform(s.plan.effects.indirect);
    u.aperture = aperture_shape(node,aspect);
    u.lens = [lens.as_ref().map_or(0., sampling::aperture_radius),
        lens.as_ref().map_or(0., |p| p.focus_distance), 2. / w as f32, 2. / h as f32];
    u.flags[0] = f32::from(ortho.is_some());
    u.flags[2] = f32::from(render.background.transparent());
    u.background = match &render.background {
        ::vork_scene::v2::Background::Color(color) => *color,
        ::vork_scene::v2::Background::Mode(::vork_scene::v2::BackgroundMode::Transparent) => {
            [0.; 3]
        }
        _ if camera_world => [u.radiance[0], u.radiance[1], u.radiance[2]],
        _ => [0.; 3],
    };
    u.shadow_links[2] = u32::from(lights::needs_receiver_masks(&frame.evaluated_lights));
    u.light_count = frame.evaluated_lights.len() as f32;
    u.post[0] = if s.has_catchers {1.} else {post_mode(render)};
    u.radiance[3] = 1.;
    Ok(u)
}

#[cfg(test)]
mod capture_lens_tests {
    use super::*;

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn imported_visible_keys_reach_objects_lights_and_shadow_reservations() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let mut gltf: serde_json::Value = vork_scene::v2::from_test_str(include_str!("../tests/goldens/skinned-box.gltf"))?;
        gltf["nodes"].as_array_mut().unwrap().push(serde_json::json!({"name":"parent","children":[0,3]}));
        gltf["nodes"].as_array_mut().unwrap().push(serde_json::json!({"name":"lamp","translation":[0,2,3],"extensions":{"KHR_lights_punctual":{"light":0}}}));
        gltf["scenes"][0]["nodes"] = serde_json::json!([1,2]);
        gltf["extensionsUsed"] = serde_json::json!(["KHR_lights_punctual"]);
        gltf["extensions"] = serde_json::json!({"KHR_lights_punctual":{"lights":[{"type":"point","intensity":100}]}});
        let file = std::env::temp_dir().join(format!("vork-visible-import-{}.gltf",std::process::id()));
        std::fs::write(&file,serde_json::to_vec(&gltf)?)?;
        let input = vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "render":{"resolution":[16,16],"settings":{"samples":1}},
            "materials":{"plain":{"inputs":{"base_color":[1,0,0],"emission_color":[1,0,0],"emission_luminance":1}}},
            "nodes":[{"type":"camera","name":"cam","translation":[0,0,4]},
                {"type":"asset","name":"actor","file":file,"materials":{"*":"plain"},"nodes":{"parent":{"visible":false}}}],
            "animation":[{"tracks":[{"path":["nodes","actor","nodes","parent","visible"],"keys":[[0,false],[1,true]]}]}]
        }))?;
        let installed = renderer.install(input,Path::new("."));
        std::fs::remove_file(&file)?;
        let mut installed = installed?;
        assert!(installed.plan.draws.iter().all(|draw| draw.casts_shadows));
        let mut frame = installed.create_frame_slot(&renderer,crate::ReadbackPrecision::Float)?;
        for (time,shown) in [(0.,false),(1.,true),(0.,false)] {
            let output = installed.render_frame(&renderer, &mut frame, time)?;
            assert_eq!(output.ids.iter().any(|id| *id != 0),shown,"imported visibility at {time}");
            assert!(!frame.evaluated_lights.is_empty());
            for light in &frame.evaluated_lights {
                assert_eq!(light.shadow_offsets[constants::LIGHT_CONTRIBUTION_WORD] & constants::RAY_VISIBILITY_ALL,
                    if shown {constants::RAY_VISIBILITY_ALL} else {0});
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn animated_visible_hides_environment_and_lights_through_parents() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let input = vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,"render":{"resolution":[16,16],"duration":2,"settings":{"samples":1}},
            "nodes":[{"type":"camera","name":"cam"},
                {"type":"node","name":"rig","visible":false,"children":[
                    {"type":"light","name":"world","shape":"environment","color":[1,1,1]},
                    {"type":"light","name":"lamp","shape":"point","shadowed_by":[]}]}],
            "animation":[{"tracks":[{"path":["nodes","rig","visible"],"keys":[[0,false],[1,true]]}]}]
        }))?;
        let installed = renderer.install(input, Path::new("."))?;
        let mut frame = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        let mut changes = LightingTracker::default();
        for (time, visible) in [(0.,false),(0.999,false),(1.,true),(0.5,false)] {
            let (_,_,_,uniform) = frame.prepare_values(&installed,&renderer,time,time,&mut changes)?;
            assert_eq!(uniform.environment_links[0] & constants::RAY_VISIBILITY_ALL,
                if visible {constants::RAY_VISIBILITY_ALL} else {0});
            assert_eq!(uniform.medium[3],u32::from(visible));
            assert!(!frame.evaluated_lights.is_empty());
            for light in &frame.evaluated_lights {
                assert_eq!(light.shadow_offsets[constants::LIGHT_CONTRIBUTION_WORD] & constants::RAY_VISIBILITY_ALL,
                    if visible {constants::RAY_VISIBILITY_ALL} else {0});
            }
        }
        Ok(())
    }

}

/// Fixed diffuse-query budgets, shared with the GI shader.
fn ao_uniform(enabled: bool) -> [f32; 4] {
    [constants::GI_RADIUS_M, if enabled {1.} else {0.}, constants::GI_RAYS_PER_SAMPLE as f32, constants::GI_RESOLUTION_DIVISOR as f32]
}

fn post_mode(render: &vork_scene::Render) -> f32 {
    if !render.compositor.is_empty() || render.settings.integrator.denoiser().is_some() || render.settings.reconstruct {1.}
    else {0.}
}

#[cfg(test)]
mod aim_tests {
    use super::*;
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn installation_uses_animated_poses_and_reports_evaluated_coincidence() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let document = serde_json::json!({"version":2,
            "render":{"resolution":[8,8]},
            "buffers":{"p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0]]}},
            "geometry":{"triangle":{"attributes":{"POSITION":"p"}}},
            "nodes":[{"type":"node","name":"rig","children":[
                {"type":"camera","name":"cam","target":"subject"},
                {"type":"mesh","name":"aimed","geometry":"triangle","target":"subject"}]},
                {"type":"node","name":"subject"}],
            "animation":[{"tracks":[
                {"path":["nodes","rig","translation"],"keys":[[0,[0,0,3]],[1,[0,0,4]]]},
                {"path":["nodes","subject","translation"],"keys":[[0,[1,0,0]],[1,[2,0,0]]]}]}]});
        let installed = renderer.install(vork_scene::v2::from_test_value(document.clone())?, Path::new("."))?;
        let mut values = offset_storage(&installed);
        sample_frame(&installed, 0., &mut values.samples, &mut values.nodes, &mut values.worlds)?;
        for (node, world) in installed.nodes.iter().zip(&values.worlds) {
            assert!(node.world.abs_diff_eq(*world, 1e-6));
        }
        for (root, object) in installed.roots.iter().zip(&installed.objects) {
            assert!(object.model.abs_diff_eq(root.world(&values.worlds), 1e-6));
        }
        let mut coincident = document;
        coincident["animation"][0]["tracks"][0]["keys"][0][1] = serde_json::json!([1,0,0]);
        let error = renderer.install(vork_scene::v2::from_test_value(coincident)?, Path::new(".")).err().expect("evaluated coincidence");
        assert!(error.to_string().contains("node \"cam\""), "{error:#}");
        assert!(error.to_string().contains("sample time 0"), "{error:#}");
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn keyed_camera_uses_requested_initial_time_and_checks_sampled_poses() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let document = serde_json::json!({"version":2,
            "render":{"resolution":[8,8],"fps":1,"duration":2},
            "nodes":[{"type":"camera","name":"cam","target":"subject"},
                {"type":"node","name":"subject"}],
            "animation":[{"tracks":[{"path":["nodes","cam","translation"],
                "keys":[[0,[0,0,0]],[1,[0,0,3]]]}]}]});
        let installed = renderer.install_with_options(vork_scene::v2::from_test_value(document.clone())?, Path::new("."),
            InstallOptions { initial_time: 1_f32.into(), ..Default::default() })?;
        assert!(installed.nodes[installed.camera].world.w_axis.truncate().abs_diff_eq(Vec3::new(0.,0.,3.), 1e-6));
        // Repeated seeking must sample authoring defaults rather than the installed pose.
        let mut values = offset_storage(&installed);
        for time in [0.5,1.,0.25] {
            sample_frame(&installed, time, &mut values.samples, &mut values.nodes, &mut values.worlds)?;
            assert!((values.worlds[installed.camera].w_axis.z - 3.*time).abs() < 1e-6);
        }
        let mut valid = document.clone();
        valid["animation"][0]["tracks"][0]["keys"][0][1] = serde_json::json!([0,0,3]);
        let mut installed = renderer.install(vork_scene::v2::from_test_value(valid.clone())?, Path::new("."))?;
        let mut slot = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        installed.render_frame(&renderer, &mut slot, 0.)?;
        valid["render"]["motion_blur"] = serde_json::json!({"shutter":1,"position":"start"});
        valid["render"]["settings"] = serde_json::json!({"samples":1});
        valid["animation"][0]["tracks"][0]["keys"][1][1] = serde_json::json!([0,0,0]);
        let mut installed = renderer.install(vork_scene::v2::from_test_value(valid)?, Path::new("."))?;
        let mut slot = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        // Sampled motion blur does not evaluate unused shutter endpoints.
        installed.render_frame(&renderer, &mut slot, 0.)?;
        let error = installed.render_frame(&renderer, &mut slot, 1.).err().expect("coincident sampled pose");
        // The frame-abort context wraps the pose error; the cause names the node
        // and the one evaluated shutter sample: the middle of [1, 2] at 1 fps.
        let chain = format!("{error:#}");
        assert!(chain.ends_with("node \"cam\": target coincides with the node origin at sample time 1.5"), "{chain}");
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn offset_instances_use_aimed_worlds_and_shifted_parents() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let mut input: serde_json::Value = vork_scene::v2::from_test_str(include_str!("../tests/fixtures/analytic.json"))?;
        input["render"] = serde_json::json!({"resolution":[8,8],"duration":3,"settings":{"samples":1}});
        input["buffers"]["translations"] = serde_json::json!({"type":"vec3","data":[[0,0,0],[0,0,0]]});
        input["buffers"]["offsets"] = serde_json::json!({"type":"f32","data":[0,1]});
        input["nodes"] = serde_json::json!([
            {"type":"camera","translation":[0,0,8]},
            {"type":"node","name":"parent","children":[
                {"type":"mesh","name":"aimed","geometry":"shape_box","target":[4,0,-2],
                 "instances":{"attributes":{"TRANSLATION":"translations","time_offset":"offsets"}}}]}]);
        input["animation"] = serde_json::json!([{"tracks":[
            {"path":["nodes","parent","translation"],"keys":[[0,[0,0,0]],[2,[2,0,0]]]}
        ]}]);
        let base=Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let mut installed=renderer.install(vork_scene::v2::from_test_value(input)?,&base)?;
        assert_eq!(installed.roots.len(),2);
        let mut frame=installed.create_frame_slot(&renderer,crate::ReadbackPrecision::Float)?;
        for time in [0.,1.,0.25] {
            installed.render_frame(&renderer, &mut frame, time)?;
            for (object,root) in installed.roots.iter().enumerate() {
                let eye=Vec3::new(time+root.time_offset,0.,0.);
                let expected=Mat4::look_to_rh(eye,(Vec3::new(4.,0.,-2.)-eye).normalize(),Vec3::Y).inverse();
                assert!(frame.object_values[object].model.abs_diff_eq(expected,1e-5),"time {time}, offset {}",root.time_offset);
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn additive_rotation_layers_compose_after_target_and_move_children() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let input=vork_scene::v2::from_test_value(serde_json::json!({"version":2,
            "render":{"resolution":[8,8],"duration":2},
            "nodes":[{"type":"camera","name":"cam","translation":[0,0,6]},
                {"type":"node","name":"aimed","target":"subject","rotation":{"euler":[0,0,0.2]},
                    "children":[{"type":"node","name":"child","translation":[0,0,-2]}]},
                {"type":"node","name":"subject","translation":[0,0,-3]}],
            "animation":[{"tracks":[{"path":["nodes","subject","translation"],"keys":[[0,[0,0,-3]],[1,[2,0,-3]]]}]},
                {"blend":"combine","tracks":[{"path":["nodes","aimed","rotation"],"keys":[[0,{"euler":[0,0,0]}],[1,{"euler":[0,0.7,0]}]]}]}]
        }))?;
        let installed=renderer.install(input,Path::new("."))?;
        let index=|name:&str| installed.nodes.iter().position(|node|node.node.name==name).unwrap();
        let mut values=offset_storage(&installed);
        for time in [0.,0.25,1.,0.5] {
            sample_frame(&installed,time,&mut values.samples,&mut values.nodes,&mut values.worlds)?;
            let target=Vec3::new(2.*time,0.,-3.);
            let expected=Mat4::look_to_rh(Vec3::ZERO,target.normalize(),Vec3::Y).inverse()
                * Mat4::from_rotation_z(0.2) * Mat4::from_rotation_y(0.7*time);
            assert!(values.worlds[index("aimed")].abs_diff_eq(expected,1e-5),"aim at {time}");
            assert!(values.worlds[index("child")].w_axis.truncate().abs_diff_eq(expected.transform_point3(Vec3::new(0.,0.,-2.)),1e-5),"child at {time}");
        }
        Ok(())
    }
    // Aim and focus follow animated targets at every evaluated time, including
    // shutter sub-frame times: keyed targets and targets under an animated parent.
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn aim_and_focus_follow_animated_targets_at_any_time() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let base = std::path::Path::new(".");
        let scene = serde_json::json!({"version":2,
            "render":{"resolution":[16,16],"fps":24,"duration":2,"motion_blur":{"shutter":0.5},"settings":{"samples":2}},
            "nodes":[
                {"type":"camera","name":"cam","target":"subject","translation":[0,1,6],"focus_object":"held","material":{"program":"std/camera","inputs":{"use_dof":true}}},
                {"type":"light","name":"spot","shape":"spot","target":"subject","translation":[3,4,2]},
                {"type":"light","name":"key","shape":"sun","target":"held"},
                {"type":"node","name":"rig","children":[{"type":"node","name":"held","translation":[0,0,-1]}]},
                {"type":"node","name":"subject"}],
            "animation":[
                {"tracks":[{"path":["nodes","subject","translation"],"keys":[[0,[-2,0,0]],[2,[2,1,-3]]]},
                    {"path":["nodes","rig","rotation"],"keys":[[0,[0,0,0,1]],[2,[0,0.7071068,0,0.7071068]]]},
                    {"path":["nodes","rig","translation"],"keys":[[0,[0,5,0]],[2,[4,2,1]]]}]}]});
        let s = renderer.install(vork_scene::v2::from_test_value(scene)?, &base)?;
        let index = |name: &str| s.nodes.iter().position(|n| n.node.name == name).unwrap();
        let (cam, subject, spot, key, held) = (index("cam"), index("subject"), index("spot"), index("key"), index("held"));
        let mut values = offset_storage(&s);
        let mut targets = Vec::new();
        // Frame and sub-frame (shutter) times.
        for time in [0., 0.3, 0.5 + 1. / 96., 1.13, 1.75] {
            sample_frame(&s, time, &mut values.samples, &mut values.nodes, &mut values.worlds)?;
            let worlds = &values.worlds;
            let aims = |from: usize, to: usize| {
                let eye = worlds[from].w_axis.truncate();
                let forward = worlds[from].transform_vector3(Vec3::NEG_Z).normalize();
                forward.dot((worlds[to].w_axis.truncate() - eye).normalize())
            };
            assert!(aims(cam, subject) > 0.99999, "camera at {time}: {}", aims(cam, subject));
            assert!(aims(spot, subject) > 0.99999, "spot at {time}: {}", aims(spot, subject));
            assert!(aims(key, held) > 0.99999, "sun at {time}: {}", aims(key, held));
            let dof = values.nodes[cam].as_ref().unwrap().camera.as_ref().unwrap();
            assert!((dof.focus - crate::targets::focus_distance(worlds[cam], worlds[held])).abs() < 1e-5, "focus at {time}");
            targets.push((worlds[subject].w_axis.truncate(), worlds[held].w_axis.truncate()));
        }
        assert!(targets.windows(2).all(|pair| pair[0].0.distance(pair[1].0) > 1e-4 && pair[0].1.distance(pair[1].1) > 1e-4), "{targets:?}");
        Ok(())
    }
}

#[cfg(test)]
mod retained_selection_tests {
    use super::*;
    use serde_json::json;

    fn scene() -> Result<Scene> {
        Ok(vork_scene::v2::from_test_value(json!({"version":2,
            "composition":{"space":"3d","view":{"program":"std/standard","inputs":{}}},"render":{"dither":false,"resolution":[16,16],"fps":2,"duration":2,"settings":{"integrator":"raster","samples":1}},
            "nodes":[{"type":"camera","name":"cam"},{"type":"mesh","geometry":"g","name":"actors","materials":["m"],"instances":{"attributes":{"TRANSLATION":"positions"}}}],
            "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[0,1,0]]},"positions":{"type":"vec3","data":[[0,0,-5],[8,0,-5],[30,0,-5],[40,0,-5]]}},
            "geometry":{"g":{"attributes":{"POSITION":"p"}}},
            "materials":{"m":{"inputs":{"base_weight":0,"specular_weight":0,"emission_color":[1,0,0],"emission_luminance":1}}},
            "animation":[{"tracks":[
                {"path":["nodes","cam","translation"],"interpolation":"STEP","keys":[[0,[0,0,0]],[1,[8,0,0]]]},
                {"path":["materials","m","inputs","emission_color"],"keys":[[0,[1,0,0]],[0.5,[0,1,0]],[1,[0,0,1]]]}
            ]}]}))?)
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn static_selection_retains_work_across_shading_slots_repeat_and_seek() -> Result<()> {
        let r = crate::test_renderer()?;
        for blur in [false,true] {
        let mut input = scene()?;
        if blur {
            input.render.motion_blur = Some(vork_scene::v2::from_test_value(json!({"shutter":0.5,"position":"center"}))?);
            input.render.settings.samples = 4;
        }
        let mut s = r.install(input, Path::new("."))?;
        let mut slots = [s.create_frame_slot(&r, crate::ReadbackPrecision::Float)?,s.create_frame_slot(&r, crate::ReadbackPrecision::Float)?];
        let mut first: [Option<Vec<[f32;4]>>;2] = [None,None];
        for (slot,time,changes_camera) in [(0,0.,true),(1,0.5,true),(0,0.5,false),(1,0.,false),(0,1.,true),(1,0.5,false),(0,0.,true)] {
            let before = slots[slot].view.camera_cull.test_visits();
            let actual = s.render_frame(&r,&mut slots[slot],time)?;
            let after = slots[slot].view.camera_cull.test_visits();
            if !blur && changes_camera { assert!(after > before,"camera must prepare at {time}"); }
            else if !blur { assert_eq!(after,before,"shading-only time {time} rescanned objects"); }
            let mut fresh = s.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
            let expected = s.render_frame(&r,&mut fresh,time)?;
            crate::test_linear::assert_linear_eq((actual).linear_rgba.as_ref().expect("linear readback"), (expected).linear_rgba.as_ref().expect("linear readback"));
            assert_eq!(actual.ids,expected.ids);
            crate::test_linear::assert_depth_eq(actual.depth_f32.as_ref().expect("float depth readback"), expected.depth_f32.as_ref().expect("float depth readback"));
            assert!(actual.ids.iter().any(|&id|id!=0),"selected actor must remain visible");
            if let Some(prior) = first[slot].as_ref() {
                if time == 0. && slot == 0 { crate::test_linear::assert_linear_eq(actual.linear_rgba.as_ref().unwrap(),prior); }
            } else { first[slot] = actual.linear_rgba; }
        }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn preparing_capture_staging_preserves_live_selection() -> Result<()> {
        let r = crate::test_renderer()?;
        let s = r.install(scene()?, Path::new("."))?;
        let frame = s.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        let cull = &frame.view.camera_cull;
        let vp = Mat4::perspective_rh(1., 1., 0.1, 100.);
        let mut changes = ChangeTracker::default();
        changes.sample = 1;
        cull.prepare_exposure(vp, &frame.object_values, &[], &mut changes, false);
        let samples = cull.sample_values(&r.device, 1).unwrap();
        cull.prepare_sample(vp * Mat4::from_translation(Vec3::new(-8., 0., 0.)),
            &frame.object_values, &[], &mut changes, &samples, 0);
        let before = cull.test_visits();
        changes.sample += 1;
        changes.bytes.clear();
        changes.uploads.clear();
        cull.prepare_exposure(vp, &frame.object_values, &[], &mut changes, false);
        assert_eq!(cull.test_visits(), before, "staging did not replace live destinations");
        assert!(changes.uploads.is_empty(), "staging must not invalidate live CPU byte history");
        r.errors.check()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn selection_observes_publications_bounds_and_capture_buffer_replacement() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut s = r.install(scene()?, Path::new("."))?;
        let frame = s.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        // Exercise the CPU receiver contract with a producer-owned local bound.
        // No modified scene is rendered: ownership is supplied to selection here.
        s.draws[0].computed = true;
        let cull = crate::passes::camera_cull::Resources::new(&s,&r,&frame.view.uniform,&frame.objects,s.material_table.records(),None,None);
        let mut objects = frame.object_values.clone();
        let vp = Mat4::perspective_rh(1.,1.,0.1,100.);
        let mut changes = ChangeTracker::default();
        changes.sample = 1;
        cull.prepare_exposure(vp,&objects,&[None],&mut changes,false);
        let count = cull.test_visits();
        assert_eq!(count,objects.len());
        let next = |changes: &mut ChangeTracker| { changes.sample+=1;changes.bytes.clear();changes.uploads.clear(); };
        next(&mut changes);
        cull.prepare_exposure(vp,&objects,&[None],&mut changes,false);
        assert_eq!(cull.test_visits(),count);
        assert!(changes.uploads.is_empty());
        let outside = Some((Vec3::splat(100.),Vec3::splat(101.)));
        cull.prepare_exposure(vp,&objects,&[outside],&mut changes,false);
        assert_eq!(cull.test_visits(),2*count,"new evaluated bounds invalidate");
        assert!(!changes.uploads.is_empty(),"changed bounds must publish selection");
        next(&mut changes);
        cull.prepare_exposure(vp,&objects,&[outside],&mut changes,false);
        assert_eq!(cull.test_visits(),2*count);
        objects[0].model = Mat4::from_translation(Vec3::new(1.,0.,-5.));
        changes.upload(&frame.objects,0,bytemuck::bytes_of(&objects[0].gpu()));
        cull.prepare_exposure(vp,&objects,&[outside],&mut changes,false);
        assert_eq!(cull.test_visits(),3*count);
        // A second publication has the same sample stamp; it must not retain
        // the first publication's selection.
        objects[0].metadata[1] = 0;
        changes.upload(&frame.objects,0,bytemuck::bytes_of(&objects[0].gpu()));
        cull.prepare_exposure(vp,&objects,&[outside],&mut changes,false);
        assert_eq!(cull.test_visits(),4*count);
        next(&mut changes);
        changes.upload(s.material_table.records(),0,&[1,2,3,4]);
        cull.prepare_exposure(vp,&objects,&[outside],&mut changes,false);
        assert_eq!(cull.test_visits(),4*count,"shading publication must retain selection");
        let samples = cull.sample_values(&r.device,2).unwrap();
        cull.prepare_sample(vp,&objects,&[None],&mut changes,&samples,0);
        cull.prepare_sample(vp*Mat4::from_translation(Vec3::new(-8.,0.,0.)),&objects,&[None],&mut changes,&samples,1);
        let mut encoder = r.device.create_command_encoder(&Default::default());
        cull.copy_sample(&mut encoder,&mut changes,&samples,0);
        for (buffer,offset,source,size) in &changes.uploads {
            r.queue.write_buffer(buffer,*offset,&changes.bytes[*source as usize..(*source+*size) as usize]);
        }
        r.queue.submit([encoder.finish()]);
        let before = cull.test_visits();
        next(&mut changes);
        cull.prepare_exposure(vp,&objects,&[outside],&mut changes,false);
        assert_eq!(cull.test_visits(),before+count,"capture restoration invalidates the live selection");
        for (buffer,offset,source,size) in &changes.uploads {
            r.queue.write_buffer(buffer,*offset,&changes.bytes[*source as usize..(*source+*size) as usize]);
        }
        let commands = cull.commands().unwrap();
        let read = r.device.create_buffer(&wgpu::BufferDescriptor { label:None,size:commands.size()+cull.instances.size(),
            usage:wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,mapped_at_creation:false });
        let mut encoder = r.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(commands,0,&read,0,commands.size());
        encoder.copy_buffer_to_buffer(&cull.instances,0,&read,commands.size(),cull.instances.size());
        let submission = r.queue.submit([encoder.finish()]);
        let (tx,rx) = std::sync::mpsc::channel();
        read.map_async(wgpu::MapMode::Read,..,move |result| {let _=tx.send(result);});
        r.device.poll(wgpu::PollType::Wait {submission_index:Some(submission),timeout:None})?;
        rx.recv()??;
        let mapped = read.get_mapped_range(..);
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        assert_eq!(words[1],0,"CPU selection must replace the capture's nonzero GPU count, even when its bytes equal the earlier CPU selection");
        assert!(words[commands.size() as usize / 4..].iter().all(|&word| word==0),
            "CPU selection must also replace the capture's instance map");
        drop(mapped);
        read.unmap();
        r.errors.check()?;
        Ok(())
    }
}

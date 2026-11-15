//! Mesh-bounded media use the same compiled point evaluator and transport.
use super::*;

use std::collections::BTreeMap;

// Small convex carriers retain their plane fast path; all others use the BVH.
#[path = "medium_convex_spike.rs"]
mod convex_spike;

pub(crate) struct Installed {
    names: BTreeMap<String, (u32, u32)>,
    programs: BTreeMap<String, (vork_slang::MediumProgram, Vec<u32>)>,
}
impl Installed {
    /// A spatially and temporally constant, resource-free world program whose
    /// complete coefficient expressions prove vacuum needs no transport pass.
    pub(super) fn world_is_vacuum(&self, scene: &v2::Scene, name: &str) -> Result<bool> {
        let (program, words) = &self.programs[name];
        if !program.homogeneous || program.time_dependent || program.inputs.fields.iter().any(|f|
            !matches!(f.ty, vork_slang::Type::Float | vork_slang::Type::Int | vork_slang::Type::Uint | vork_slang::Type::Bool | vork_slang::Type::Vector(_))) {
            return Ok(false);
        }
        let dynamic = crate::materials::animated_inputs(scene)?.remove(name).unwrap_or_default();
        let proof = program.inputs.varying_words(words, &dynamic);
        let abi = programs::abi();
        // This is an optional identity proof; unsupported interval operations
        // retain the existing transport rather than rejecting a valid medium.
        Ok(program.emission_free_for_inputs(&abi, &proof).unwrap_or(false)
            && program.extinction_majorant(&abi, &proof, [[0.,0.];3], [0.,0.]).is_ok_and(|bound| bound == 0.))
    }
}
pub(super) fn install(
    r: &Renderer,
    scene: &v2::Scene,
    world: Option<&str>,
    base: &Path,
    stages: &BTreeMap<String, programs::Stages>,
) -> Result<super::Installed> {
    let mut names = BTreeMap::new();
    for name in stages
        .iter()
        .filter(|(_, stages)| stages.medium)
        .map(|(name, _)| name.as_str())
        .chain(world)
    {
        names.insert(name.to_owned(), (0, 0));
    }
    // Header: descriptor offset/count, world program/base. All offsets are relative
    // to frame.medium.x so material records and program words stay in one table.
    let mut words = vec![0, 0, u32::MAX, 0];
    let mut input_bindings = Vec::new();
    let animated = crate::materials::animated_inputs(scene)?;
    let mut native = Native::default();
    let mut native_dispatch = String::from(
        "export MediumSample volume_coefficients(uint id,float3 p,MediumTransportContext context) {switch(id) {\n",
    );
    let mut programs = BTreeMap::new();
    // A majorant is stored beside its existing input words only when a sampled
    // consumer needs heterogeneous tracking. Homogeneous/raster installs retain
    // their original layout and allocate no tracking metadata.
    let compiled=names.keys().map(|name|Ok((name.clone(),compile(r,scene,name,base)?))).collect::<Result<BTreeMap<_,_>>>()?;
    native.world_only=stages.values().all(|s|!s.medium);
    native.homogeneous=compiled.values().all(|(p,_)|p.homogeneous);
    native.scattering_free=compiled.values().all(|(p,_)|p.scattering_free);
    native.emission_free=compiled.values().all(|(p,_)|p.emission_free);
    native.time_dependent=compiled.values().any(|(p,_)|p.time_dependent);
    let tracking=!native.homogeneous && scene.render.settings.integrator.is_path();
    native.tracking=tracking;
    native.point.push_str("import vork_medium_point;\n");
    let exponential=world.and_then(|name| {let program=&compiled[name].0;(!program.homogeneous).then_some(program.exponential).flatten()});
    let world_exponential=if let Some((a,b,g))=exponential {
        let v=|v:[f32;3]|format!("float3({:.9e},{:.9e},{:.9e})",v[0],v[1],v[2]);
        format!("a={};b={};g={};return true;",v(a),v(b),v(g))
    } else {"a=0.;b=0.;g=0.;return false;".to_owned()};
    native.point.push_str(&format!("public bool path_world_exponential(out float3 a,out float3 b,out float3 g) {{{world_exponential}}}\n"));

    let mut path_homogeneous=String::from("public bool path_medium_homogeneous(uint id) {switch(id) {\n");
    for (name, offsets) in &mut names {
        let (program,inputs)=&compiled[name];

        if !program.homogeneous {
            if Some(name.as_str())==world && program.exponential.is_none() {native.sampled_error=Some(format!("sampled heterogeneous world medium {name} needs a finite domain; bind it to a closed mesh boundary for sampled transport; unbounded varying world media are unsupported (vork help profiles)"));}
        }
        if tracking {words.push(0);}
        let base = words.len() as u32;
        words.extend_from_slice(inputs);
        // Input bases also identify instances in emitting dispatch. Empty layouts
        // must own a word so the next instance cannot reuse this identity.
        if inputs.is_empty() { words.push(0); }
        if animated.contains_key(name) {
            input_bindings.push((name.clone(), base as usize, program.inputs.clone(), 0.));
        }
        if let Some(&id) = programs.get(&program.native_point) {
            *offsets = (id, base);
            continue;
        }
        let id = programs.len() as u32;
        programs.insert(program.native_point.clone(), id);
        *offsets = (id, base);
        let point=native_coefficients(&program,id,true,&mut native);
        native.point.push_str(&point);
        path_homogeneous.push_str(&format!("case {id}u: return {};\n",program.homogeneous));
        native_dispatch.push_str(&format!(
            "case {id}u: return volume_{id}_medium_point(p,context);\n"
        ));
    }
    path_homogeneous.push_str("default: return true;}}\n");native.point.push_str(&path_homogeneous);
    if let Some(world) = world {
        (words[2], words[3]) = names[world];
    }
    native_dispatch.push_str("default: return MediumSample();}}\n");
    native.point.push_str(&native_dispatch.replace("export MediumSample volume_coefficients","public MediumSample path_medium_coefficients"));
    Ok(super::Installed {
        native,
        words,
        instances: Installed { names, programs: compiled },
        inputs: input_bindings,
    })
}

fn convex_carrier(draw: &install::Draw) -> Option<convex_spike::Convex> {
    // Bound certification cost, not scene admission. Larger carriers fall back.
    if draw.evaluated() || draw.count > 32*3 {return None;}
    let geometry=draw.exit_geometry.as_ref()?;
    let start=(draw.primitive_start as usize).checked_mul(3)?;
    let indices=geometry.indices.get(start..start.checked_add(draw.count as usize)?)?;
    let faces=indices.chunks_exact(3).map(|i|Some([
        geometry.vertices.get(i[0] as usize)?.POSITION,
        geometry.vertices.get(i[1] as usize)?.POSITION,
        geometry.vertices.get(i[2] as usize)?.POSITION,
    ])).collect::<Option<Vec<_>>>()?;
    convex_spike::Convex::certify(&faces).ok()
}

impl super::Installed {
    pub(crate) fn select_boundary_format(&mut self,draws:&[install::Draw],materials:&BTreeMap<String,usize>,force_general:bool) {
        let carriers:std::collections::BTreeSet<_>=self.instances.names.keys().filter_map(|name|materials.get(name).copied()).collect();
        self.native.convex=!force_general && draws.iter().filter(|draw|carriers.contains(&draw.material))
            .all(|draw|convex_carrier(draw).is_some());
    }
    pub(crate) fn mesh_words(
        &mut self,
        _r: &Renderer,
        scene: &v2::Scene,
        draws: &[install::Draw],
        roots: &[install::MeshRoot],
        materials: &BTreeMap<String, usize>,
    ) -> Result<()> {
        let bounded=&self.instances;
        let bindings: BTreeMap<_, _> = materials
            .iter()
            .filter_map(|(name, index)| {
                bounded.names.get(name).map(|binding| (*index, (name.as_str(), *binding)))
            })
            .collect();
        let animated = crate::materials::animated_inputs(scene)?;
        if self.native.tracking {
            for (name, (_, base)) in &bounded.names {
                let (program, inputs)=&bounded.programs[name];
                if program.homogeneous {continue;}
                let mut bounds=(Vec3::splat(f32::INFINITY),Vec3::splat(f32::NEG_INFINITY));
                for draw in draws.iter().filter(|d|materials.get(name)==Some(&d.material)) {
                    bounds.0=bounds.0.min(draw.bounds.0);bounds.1=bounds.1.max(draw.bounds.1);
                }
                if bounds.0.is_finite() {
                    let dynamic=animated.get(name).cloned().unwrap_or_default();
                    let proof=grid_proof_words(program,inputs,&dynamic).0;
                    self.words[*base as usize-1]=bound::certify(program,&proof,bounds,bound::certification_times(scene)).with_context(||format!("medium shader {:?} (material {name:?}): cannot certify extinction majorant; use supported shader operations",scene.materials[name].program))?.to_bits();
                }
            }
        }
        let mut descriptors = Vec::new();
        let mut phased = BTreeMap::new();
        for draw in draws {
            let Some(&(material_name, (program, inputs))) = bindings.get(&draw.material) else {
                continue;
            };
            ensure!(
                !draw.evaluated(),
                "stage 1 fog does not support deformed carrier boundaries; bake a closed triangle mesh first"
            );
            let geometry = draw
                .exit_geometry
                .as_ref()
                .context("unsupported stage 1 fog carrier geometry; use a closed triangle mesh, or render the volume externally")?;
            validate_boundary(geometry)?;
            ensure!(
                draw.count as usize == geometry.indices.len(),
                "stage 1 fog needs one closed material group per carrier; split carriers into separate meshes"
            );
            let start = draw.primitive_start as usize * 3;
            let root = self.words.len() as u32;
            {
                let mut triangles: Vec<_> = geometry.indices[start..start + draw.count as usize]
                    .chunks_exact(3)
                    .map(|indices| [indices[0], indices[1], indices[2]].map(|i| Vec3::from_array(geometry.vertices[i as usize].POSITION)))
                    .collect();
                if self.native.convex {
                    let faces = triangles.iter().map(|t| t.map(|v| v.to_array())).collect::<Vec<_>>();
                    let carrier = convex_spike::Convex::certify(&faces).map_err(anyhow::Error::msg).context("certified convex medium carrier changed during installation")?;
                    // One record and a bounded plane array; reuse the domain descriptor.
                    self.words.extend([0; 8]);
                    self.words[root as usize + 7] = carrier.planes.len() as u32;
                    for (normal, anchor) in carrier.planes {
                        let magnitude = normal.iter().map(|v|v*v).sum::<f64>().sqrt();
                        self.words.extend(normal.map(|v| (v / magnitude) as f32).map(f32::to_bits));
                        self.words.extend(anchor.map(|v|v as f32).map(f32::to_bits));
                    }
                } else { boundary_bvh(&mut self.words, &mut triangles); }
            }
            let end = self.words.len() as u32;
            for object in draw.instances.clone() {
                let time_offset=roots[object as usize].time_offset;
                let input=if time_offset == 0. || !animated.contains_key(material_name) {inputs} else if let Some(&base)=phased.get(&(program,inputs,time_offset.to_bits())) {base} else {
                    let (name,_,layout,_)=self.inputs.iter().find(|(name,base,_,offset)|name == material_name && *base==inputs as usize && *offset==0.).context("medium input binding")?;
                    {
                        let (name,layout)=(name.clone(),layout.clone());
                        let compiled=&bounded.programs[&name];
                        let words=self.words[inputs as usize..inputs as usize+compiled.1.len()].to_vec();
                        if self.native.tracking && !compiled.0.homogeneous {self.words.push(self.words[inputs as usize-1]);}
                        let base=self.words.len() as u32;
                        self.words.extend(words);
                        self.inputs.push((name,base as usize,layout,time_offset));
                        phased.insert((program,inputs,time_offset.to_bits()),base);
                        base
                    }
                };
                descriptors.extend([
                    object,
                    root,
                    end,
                    program,
                    input,
                ]);
            }
        }
        self.words[0] = self.words.len() as u32;
        self.words[1] = (descriptors.len() / constants::MEDIUM_BOUND_WORDS) as u32;
        self.words.extend(descriptors);
        Ok(())
    }
}

// Preorder, threaded BVH: min.xyz/escape, max.xyz/leaf count, followed by
// either two children or leaf triangles. Escape offsets make traversal stackless
// on every backend. A single small leaf avoids extra nodes for small boundaries.
fn boundary_bvh(words: &mut Vec<u32>, triangles: &mut [[Vec3; 3]]) {
    let at = words.len();
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    let mut centers_lo = lo;
    let mut centers_hi = hi;
    for triangle in triangles.iter() {
        let center = (triangle[0] + triangle[1] + triangle[2]) / 3.;
        centers_lo = centers_lo.min(center);
        centers_hi = centers_hi.max(center);
        for &p in triangle {lo = lo.min(p); hi = hi.max(p);}
    }
    words.extend(lo.to_array().map(|v| v.next_down().to_bits()));
    words.push(0);
    words.extend(hi.to_array().map(|v| v.next_up().to_bits()));
    words.push(0);
    if triangles.len() <= constants::MEDIUM_BVH_LEAF_TRIANGLES {
        words[at + 7] = triangles.len() as u32;
        for triangle in triangles.iter() {
            for p in triangle {words.extend(p.to_array().map(f32::to_bits));}
        }
    } else {
        let extent = centers_hi - centers_lo;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {0} else if extent.y >= extent.z {1} else {2};
        let mid = triangles.len() / 2;
        triangles.select_nth_unstable_by(mid, |a, b| {
            let center = |t: &[Vec3; 3]| (t[0][axis] + t[1][axis] + t[2][axis]) / 3.;
            center(a).total_cmp(&center(b))
        });
        let (left, right) = triangles.split_at_mut(mid);
        boundary_bvh(words, left);
        boundary_bvh(words, right);
    }
    words[at + 3] = words.len() as u32;
}

/// Weld attribute-split vertices by position before checking oriented edge incidence.
fn validate_boundary(geometry: &geometry::Geometry) -> Result<()> {
    let (edges, degenerate) = geometry::edge_uses(&geometry.indices, |index| geometry::weld_key(geometry.vertices[index as usize].POSITION));
    ensure!(!degenerate, "volume boundary contains a degenerate edge");
    ensure!(
        !edges.is_empty()
            && edges
                .values()
                .all(|&(count, winding)| count == 2 && winding == 0),
        "volume boundary must be a closed, consistently oriented manifold mesh"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cube() -> geometry::Geometry {
        let vertices = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
        .into_iter()
        .map(|p| Vertex {
            POSITION: p,
            ..Vertex::zeroed()
        })
        .collect();
        geometry::Geometry::new(
            vertices,
            vec![
                0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5, 0, 1, 5, 0,
                5, 4, 3, 7, 6, 3, 6, 2,
            ],
            0,
        )
    }
    #[test]
    fn threaded_boundary_bvh_preserves_triangles_and_prunes_large_mesh() {
        // An 81,920-triangle closed boundary, spread across disjoint cubes.
        // The ray only visits one cube; a triangle scan visits all of them.
        let mesh = cube();
        let original: Vec<_> = mesh.indices.chunks_exact(3).map(|i|
            [i[0], i[1], i[2]].map(|v| Vec3::from_array(mesh.vertices[v as usize].POSITION))).collect();
        let mut triangles = Vec::new();
        for i in 0..6826 {
            for t in &original {triangles.push(t.map(|p| p + Vec3::new(i as f32 * 3., 0., 0.)));}
        }
        for i in 0..2 {
            let offset = Vec3::new(21000. + i as f32 * 3., 0., 0.);
            let p = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z].map(|p| p + offset);
            for t in [[0,2,1], [0,1,3], [0,3,2], [1,2,3]] {triangles.push(t.map(|i| p[i]));}
        }
        assert_eq!(triangles.len(), 81_920);
        let mut words = vec![0; 4];
        boundary_bvh(&mut words, &mut triangles);
        let vec3 = |at| Vec3::from_array([0, 1, 2].map(|i| f32::from_bits(words[at+i])));
        let mut node = 4;
        let mut visited = 0;
        let mut leaves = 0;
        while node < words.len() {
            let escape = words[node+3] as usize;
            assert!(escape > node && escape <= words.len());
            let lo = vec3(node); let hi = vec3(node+4);
            if lo.x <= 0. && hi.x >= 0. && lo.y <= 0. && hi.y >= 0. && hi.z >= -2. {
                let count = words[node+7] as usize;
                if count == 0 {node += constants::MEDIUM_BVH_WORDS; continue;}
                visited += count;
                leaves += 1;
                for triangle in (node+constants::MEDIUM_BVH_WORDS..escape).step_by(9) {
                    for p in [vec3(triangle), vec3(triangle+3), vec3(triangle+6)] {
                        assert!(p.cmpge(lo).all() && p.cmple(hi).all());
                    }
                }
            }
            node = escape;
        }
        assert!(leaves > 0);
        assert!(visited < 100, "{visited} triangle candidates for {} triangles", triangles.len());
        // Every payload is retained exactly once, including winding.
        let mut rebuilt = Vec::new();
        let mut node = 4;
        while node < words.len() {
            let count = words[node+7] as usize;
            if count == 0 {node += constants::MEDIUM_BVH_WORDS; continue;}
            for i in 0..count {let t=node+constants::MEDIUM_BVH_WORDS+i*9;rebuilt.push([vec3(t),vec3(t+3),vec3(t+6)]);}
            node = words[node+3] as usize;
        }
        assert_eq!(rebuilt, triangles);
    }
    #[test]
    fn open_and_inconsistently_oriented_volume_boundaries_are_rejected() -> Result<()> {
        let mut mesh = cube();
        validate_boundary(&mesh)?;
        mesh.indices.swap(0, 1);
        assert!(validate_boundary(&mesh).is_err());
        mesh.indices.swap(0, 1);
        mesh.indices.truncate(mesh.indices.len() - 3);
        assert!(validate_boundary(&mesh).is_err());
        Ok(())
    }
    fn boundary_fixture(builder: &str, mut options: serde_json::Value) -> geometry::Geometry {
        static FIXTURES:std::sync::OnceLock<serde_json::Value>=std::sync::OnceLock::new();
        let cases=FIXTURES.get_or_init(||serde_json::from_str(include_str!("../tests/fixtures/boundary-meshes.json")).unwrap());
        let data=include_bytes!("../tests/fixtures/boundary-meshes.bin");
        options["name"]="test".into();
        let item=cases.as_array().unwrap().iter().find(|v|v["shape"]==builder && v["options"]==options)
            .unwrap_or_else(||panic!("missing authored boundary fixture: {builder} {options}"));
        let at=|field:&str|item[field].as_u64().unwrap() as usize;
        let start=at("offset");let count=at("count");
        let vertices=data[start..start+count*12].chunks_exact(12).map(|row|Vertex {
            POSITION:std::array::from_fn(|i|f32::from_le_bytes(row[i*4..i*4+4].try_into().unwrap())),
            ..Vertex::zeroed()
        }).collect();
        let start=at("indices_offset");let count=at("indices_count");
        let indices=data[start..start+count*4].chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect();
        geometry::Geometry::new(vertices,indices,0)
    }

    #[test]
    fn closed_authored_meshes_are_exact_volume_boundaries() {
        use serde_json::json;
        for r in [1., 6e6] {
            for segments in [3, 4, 7, 16, 31, 32] {
                for rings in [2, 3, 8] {
                    for (builder, options) in [
                        ("box", json!({"size":[r,2.*r,3.*r]})),
                        (
                            "sphere",
                            json!({"radius":r,"segments":segments,"rings":rings}),
                        ),
                        (
                            "cylinder",
                            json!({"radius":r,"height":3.*r,"segments":segments}),
                        ),
                        (
                            "cone",
                            json!({"radius":r,"height":3.*r,"segments":segments}),
                        ),
                        (
                            "torus",
                            json!({"radius":r,"tube":r/4.,"segments":segments,"rings":rings+1}),
                        ),
                        (
                            "capsule",
                            json!({"radius":r,"height":3.*r,"segments":segments,"rings":rings}),
                        ),
                        (
                            "capsule",
                            json!({"radius":r,"height":2.*r,"segments":segments,"rings":rings}),
                        ),
                        (
                            "extrude",
                            json!({"outline":[[0.,0.],[r,0.],[r,r],[0.,r]],"depth":r}),
                        ),
                    ] {
                        validate_boundary(&boundary_fixture(builder, options.clone()))
                            .unwrap_or_else(|e| panic!("{builder} {options}: {e}"));
                    }
                    for mut profile in [
                        vec![[0., r], [r, 0.], [0., -r]],
                        vec![[r, 0.], [2. * r, 0.], [2. * r, r], [r, r], [r, 0.]],
                    ] {
                        for _ in 0..2 {
                            let options = json!({"profile":profile,"segments":segments});
                            validate_boundary(&boundary_fixture("revolve", options.clone()))
                                .unwrap_or_else(|e| panic!("revolve {options}: {e}"));
                            profile.reverse();
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn authored_open_rims_and_real_cracks_still_fail() {
        use serde_json::json;
        for (builder, options) in [
            ("grid", json!({})),
            (
                "revolve",
                json!({"profile":[[1.,1.],[1.,-1.]],"segments":7}),
            ),
        ] {
            assert!(validate_boundary(&boundary_fixture(builder, options)).is_err());
        }
        let mut mesh = boundary_fixture("sphere", json!({"segments":7,"rings":3}));
        // Move just one UV seam vertex by a representable amount, leaving an authored crack.
        mesh.vertices[8].POSITION[2] += 1e-7;
        assert!(validate_boundary(&mesh).is_err());
    }
}



impl super::Installed {
    pub(crate) fn dense_bindings(&self,scene:&v2::Scene,medium_base:u32,draws:&[install::Draw],materials:&BTreeMap<String,usize>)->Result<(Vec<(String,u32)>,Vec<crate::dense::Certificate>)> {
        let mut destinations=Vec::new();let mut certificates=Vec::new();
        let animated=crate::materials::animated_inputs(scene)?;
        let copies=self.instances.names.iter().map(|(name,(_,base))|(name,*base))
            .chain(self.inputs.iter().filter(|(_,_,_,time)|*time!=0.).map(|(name,base,_,_)|(name,*base as u32)));
        for (name,base) in copies {
            let (program,inputs)=&self.instances.programs[name];
            let (proof,proof_slots)=grid_proof_words(program,inputs,&animated.get(name).cloned().unwrap_or_default());
            let mut grids=Vec::new();
            for field in &program.inputs.fields {
                if !matches!(field.ty,vork_slang::Type::FloatGrid|vork_slang::Type::VectorGrid) {continue;}
                let Some(v2::Input::Grid(binding))=scene.materials[name].inputs.get(&field.name) else {continue;};
                if scene.buffers[&binding.grid].dims.is_none() {continue;}
                let slot=inputs[field.offset] as usize;
                destinations.push((binding.grid.clone(),medium_base+base+slot as u32+4));
                grids.push((binding.grid.clone(),proof_slots[&field.name]));
            }
            if self.native.tracking && !grids.is_empty() && !program.homogeneous {
                let mut bounds=(Vec3::splat(f32::INFINITY),Vec3::splat(f32::NEG_INFINITY));
                for draw in draws.iter().filter(|d|materials.get(name)==Some(&d.material)) {bounds.0=bounds.0.min(draw.bounds.0);bounds.1=bounds.1.max(draw.bounds.1);}
                if bounds.0.is_finite() {
                    certificates.push(crate::dense::Certificate {program:program.clone(),words:proof,grids,destination:medium_base+base-1,bounds,times:super::bound::certification_times(scene)});
                }
            }
        }
        Ok((destinations,certificates))
    }
}

/// The certificate consumes grid ranges, never voxel payloads. Remap only its
/// private resource handles to compact metadata; GPU input addresses stay intact.
fn grid_proof_words(program:&vork_slang::MediumProgram,inputs:&[u32],dynamic:&std::collections::BTreeSet<String>)->(Vec<Option<u32>>,BTreeMap<String,usize>) {
    let mut words=program.inputs.varying_words(&inputs[..program.inputs.words],dynamic);
    let mut slots=BTreeMap::new();
    for field in &program.inputs.fields {
        if !matches!(field.ty,vork_slang::Type::FloatGrid|vork_slang::Type::VectorGrid) {continue;}
        let source=inputs[field.offset] as usize;
        let slot=words.len();
        words[field.offset]=Some(slot as u32);
        if inputs.get(source)==Some(&u32::MAX) {
            words.extend(inputs[source..source+6].iter().copied().map(Some));
        } else {
            // Sparse interpolation still has no enclosure. Keep the resource
            // unknown rather than substituting a synthetic empty grid.
            words.push(None);
        }
        slots.insert(field.name.clone(),slot);
    }
    (words,slots)
}

#[cfg(test)]
mod empty_input_tests {
    use super::*;
    use serde_json::json;

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn empty_medium_input_bases_remain_distinct() -> Result<()> {
        let r = crate::test_renderer()?;
        for path in [false, true] {
            for pair in [["empty_a", "empty_b"], ["empty_a", "animated"], ["animated", "empty_b"]] {
                let mut value = json!({"version":2,"render":{"resolution":[4,4],"duration":1},
                    "nodes":[{"type":"camera"}],
                    "programs":{
                        "empty_a":{"source":"struct Inputs {}; MediumCoefficients medium(MediumContext c,Inputs i){MediumCoefficients m={};m.absorption=float3(.1);return m;}"},
                        "empty_b":{"source":"struct Inputs {}; MediumCoefficients medium(MediumContext c,Inputs i){MediumCoefficients m={};m.emission=float3(.2);return m;}"},
                        "animated":{"source":"struct Inputs {float density=.3;}; MediumCoefficients medium(MediumContext c,Inputs i){MediumCoefficients m={};m.absorption=float3(i.density);return m;}"}},
                    "materials":{"a":{"program":pair[0]},"b":{"program":pair[1]}}});
                value["render"]["settings"] = json!({"integrator":if path {"path"} else {"raster"},"samples":1,"denoiser":null});
                if let Some(index) = pair.iter().position(|name| *name == "animated") {
                    value["animation"] = json!([{"tracks":[{"path":["materials",if index==0 {"a"} else {"b"},"inputs","density"],"keys":[[0,0.3],[1,0.6]]}]}]);
                }
                let scene: v2::Scene = v2::from_test_value(value)?;
                let stages = ["a", "b"].into_iter().map(|name|(name.to_owned(),programs::Stages {medium:true,..Default::default()})).collect();
                let installed = install(&r,&scene,None,Path::new("."),&stages)?;
                let a = installed.instances.names["a"].1;
                let b = installed.instances.names["b"].1;
                assert_ne!(a,b,"{pair:?}, path={path}");
                assert!((b as usize) < installed.words.len());
                for (name, base, _, _) in &installed.inputs {
                    assert_eq!(*base,installed.instances.names[name].1 as usize);
                }
            }
        }
        Ok(())
    }
}

impl super::Installed {
    pub(crate) fn certified_world_exponential(&self,world:&str)->bool {
        self.instances.programs.iter().all(|(name,(program,_))|
            if name==world {program.exponential.is_some()} else {program.homogeneous})
    }

    pub(crate) fn prepare_point_work(&mut self,scene:&v2::Scene)->Result<()> {
        let animated=crate::materials::animated_inputs(scene)?;
        let bounded=&self.instances;
            for (name,(program,inputs)) in &bounded.programs {
                let dynamic=animated.get(name).cloned().unwrap_or_default();
                let words=program.inputs.varying_words(&inputs[..program.inputs.words],&dynamic);
                let work=program.bounded_work_for_inputs(&programs::abi(),&words)
                    .with_context(||format!("medium program {:?} (material {name:?}) has no bounded callback work proof; use a bounded physical coefficient program, or render the volume externally",scene.materials[name].program))?;
                self.native.point_work=self.native.point_work.max(work);
            }
        Ok(())
    }
}

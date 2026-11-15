//! Named image dependencies are structural and resolved before GPU installation.
use super::*;
use std::collections::{BTreeSet, HashMap};

fn medium_materials(nodes: &[Node]) -> BTreeSet<&str> {
    let mut names = BTreeSet::new();
    for node in nodes {
        if let NodeKind::Light {
            medium: Some(name), ..
        } = &node.kind
        {
            names.insert(name.as_str());
        }
        names.extend(medium_materials(&node.children));
    }
    names
}
impl Scene {
    pub(super) fn validate_image_sources(&self, nodes: &BTreeMap<&str, &Node>) -> Result<()> {
        for (name, image) in &self.images {
            if image.picture_of.is_some() { continue; }
            ensure!(
                image.file.is_some() != image.render.is_some(),
                "image {name}: requires exactly one of file or render"
            );
            if let Some(render) = &image.render {
                ensure!(
                    matches!(image.color_space, None | Some(ColorSpace::LinearRec709 | ColorSpace::NonColor)),
                    "image {name}: render sources carry their scene-linear source colour space"
                );
                self.validate_render(&image.resolved_render(&self.render).expect("a rendered image"), nodes)?;
            }
            ensure!(image.composition.is_none() || image.render.is_some(), "image {name}: composition is a rendered image's image graph; give the image a render");
        }
        fn environment(scene: &Scene, nodes: &[Node]) -> Result<()> {
            for node in nodes {
                if let NodeKind::Light {
                    image: Some(name), ..
                } = &node.kind
                {
                    if let Some(image) = scene.images.get(name) {
                        ensure!(
                            image.render.is_none(),
                            "environment image {name}: render sources are not supported"
                        );
                    }
                }
                environment(scene, &node.children)?;
            }
            Ok(())
        }
        let medium_materials = medium_materials(&self.nodes);
        for (name, material) in &self.materials {
            let medium = medium_materials.contains(name.as_str());
            for (field, input) in &material.inputs {
                let image = match input {
                    Input::Texture(binding) => Some(&binding.image),
                    _ => None,
                };
                if image
                    .is_some_and(|image| self.images.get(image).is_some_and(|i| i.render.is_some()))
                {
                    ensure!(
                        !medium,
                        "medium material {name}: render image inputs are not implemented yet"
                    );
                    ensure!(
                        field != "geometry_opacity" && (field != "alpha_cutoff" || material.program != "std/openpbr") && field != "geometry_normal",
                        "material {name}: render images on {field} are not implemented yet"
                    );
                }
            }
        }
        environment(self, &self.nodes)
    }

    /// Image reads for one fixed subtree. Includes invisible meshes: visibility can animate.
    pub fn image_reads(&self, root: Option<&str>) -> BTreeSet<String> {
        fn visit(
            scene: &Scene,
            nodes: &[Node],
            root: Option<&str>,
            included: bool,
            out: &mut BTreeSet<String>,
        ) {
            for node in nodes {
                let included = included || root == Some(node.name.as_str());
                if included {
                    let materials: Vec<&String> = match &node.kind {
                        NodeKind::Mesh { materials, .. } => materials.iter().collect(),
                        NodeKind::Asset { materials, .. } => materials.values().collect(),
                        _ => Vec::new(),
                    };
                    for name in materials {
                        if let Some(material) = scene.materials.get(name) {
                            for input in material.inputs.values() {
                                let image = match input {
                                    Input::Texture(binding) => Some(&binding.image),
                                    _ => None,
                                };
                                if let Some(name) = image {
                                    out.insert(name.clone());
                                }
                            }
                        }
                    }
                }
                visit(scene, &node.children, root, included, out);
            }
        }
        let mut out = BTreeSet::new();
        visit(self, &self.nodes, root, root.is_none(), &mut out);
        out
    }

    /// Ordinary image handles read by a compositor chain.
    pub fn compositor_image_reads(&self,names:&[CompositeEntry])->BTreeSet<String> {
        names.iter().filter_map(|name|self.materials.get(name.material())).flat_map(|material|material.inputs.values())
            .filter_map(|input|if let Input::Texture(binding)=input {Some(binding.image.clone())} else {None}).collect()
    }
    /// Stable dependency-first order. Every declared target is checked, including unused targets.
    pub fn image_order(&self, root: Option<&str>) -> Result<Vec<String>> {
        fn visit(
            scene: &Scene,
            name: &str,
            states: &mut HashMap<String, u8>,
            out: &mut Vec<String>,
        ) -> Result<()> {
            let Some((image, render)) = scene.images.get(name).and_then(|i| i.render.as_ref().map(|render| (i, render))) else {
                return Ok(());
            };
            // A canvas's picture is drawn by the 2D engine, not captured by a camera.
            if render.root.as_deref().is_some_and(|root| scene.is_canvas_node(root)) {
                return Ok(());
            }
            match states.get(name) {
                Some(1) => anyhow::bail!("render image dependency cycle at {name}"),
                Some(2) => return Ok(()),
                _ => {}
            }
            states.insert(name.into(), 1);
            for dependency in scene.image_reads(render.root.as_deref()).into_iter().chain(scene.compositor_image_reads(image.compositor())) {
                visit(scene, &dependency, states, out)?;
            }
            states.insert(name.into(), 2);
            out.push(name.into());
            Ok(())
        }
        let mut states = HashMap::new();
        let mut all = Vec::new();
        for name in self.images.keys() {
            visit(self, name, &mut states, &mut all)?;
        }
        states.clear();
        let mut used = Vec::new();
        for name in self.image_reads(root).into_iter().chain(self.compositor_image_reads(self.compositor())) {
            visit(self, &name, &mut states, &mut used)?;
        }
        Ok(used)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn scene(images: Value, nodes: Value) -> Scene {
        crate::v2::from_test_value(json!({"version":2,"images":images,
            "nodes":[{"type":"camera","name":"cam","material":{"program":"std/camera","inputs":{"projection":"persp"}}},{"type":"node","name":"screen","children":nodes}],
            "materials":{"a":{"inputs":{"emission_color":{"image":"a"}}},"b":{"inputs":{"emission_color":{"image":"b"}}}}
        })).unwrap()
    }
    fn source(root: Option<&str>) -> Value {
        json!({"render":{"camera":"cam","root":root,"resolution":[64,64]}})
    }
    fn mesh(material: &str) -> Value {
        json!({"type":"mesh","geometry":"quad","materials":[material]})
    }
    fn valid() -> Value {
        json!({"version":2,"nodes":[{"type":"camera","name":"cam","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"images":{"feed":{"render":{"camera":"cam","root":"cam","resolution":[64,64]}}},"materials":{"screen":{"inputs":{"emission_color":{"image":"feed"}}}}})
    }
    #[test]
    fn image_inherits_parent_settings_without_authored_copies() {
        let mut parent: Render=crate::v2::from_test_value(json!({"fps":48,"aovs":[{"name":"mask","type":"value"}]})).unwrap();
        parent.settings=crate::v2::from_test_value(json!({"integrator":"path","samples":8})).unwrap();
        let image: Image=crate::v2::from_test_value(json!({"render":{"camera":"other","resolution":[32,16]}})).unwrap();
        let resolved=image.resolved_render(&parent).unwrap();
        assert_eq!(resolved.fps,crate::Time::integer(48));
        assert_eq!(resolved.settings.integrator,parent.settings.integrator);
        assert_eq!(resolved.aovs,parent.aovs);
        assert_eq!(resolved.settings.samples,8);
        assert_eq!(resolved.resolution,[32,16]);
        assert_eq!(resolved.camera.as_deref(),Some("other"));
        for (key, value) in [("fps",json!(48)),("aovs",json!([])),("samples",json!(2)),("settings",json!({}))] {
            let mut authored=json!({});
            authored[key]=value;
            let error=crate::v2::from_test_value::<ImageRender>(authored).unwrap_err().to_string();
            assert!(error.contains("unknown field") && error.contains(key),"{error}");
        }
        let omitted: Image=crate::v2::from_test_value(json!({"render":{}})).unwrap();

        assert_eq!(omitted.resolved_render(&parent).unwrap().settings.samples,8);
        assert_eq!(omitted.resolved_render(&Render::default()).unwrap().settings.samples,64);
    }
    #[test]
    fn image_sample_budget_follows_resolved_profile_and_overrides() {
        let capture: Image = crate::v2::from_test_value(json!({"render":{}})).unwrap();
        for profile in ["std/draft", "std/final", "std/path"] {
            let profile = Profile::load(profile).unwrap();
            let mut parent = Render::default();
            parent.settings = profile.resolve(&serde_json::Map::new()).unwrap();
            assert_eq!(capture.resolved_render(&parent).unwrap().settings.samples, parent.settings.samples);
            parent.settings = test_settings(&parent.settings, &[("samples".into(), json!(17))].into_iter().collect()).unwrap();
            assert_eq!(capture.resolved_render(&parent).unwrap().settings.samples, 17);

        }
    }
    #[test]
    fn image_settings_share_render_validation() {
        let mut v = valid();
        v["images"]["feed"]["render"]["background"] = json!([-1, 0, 0]);
        assert!(
            crate::v2::from_test_value::<Scene>(v)
                .unwrap()
                .validate()
                .unwrap_err()
                .to_string()
                .contains("background")
        );
    }
    #[test]
    fn image_display_fields_are_rejected_and_not_serialized() {
        for (key, value) in [
            ("duration", json!(1)),
            ("exposure", json!(2)),
            ("view_transform", json!("Standard")),
            ("dither_intensity", json!(0)),
        ] {
            let mut v = valid();
            v["images"]["feed"]["render"][key] = value;
            assert!(
                crate::v2::from_test_value::<Scene>(v)
                    .unwrap_err()
                    .to_string()
                    .contains("unknown field")
            );
        }
        let s: Scene = crate::v2::from_test_value(valid()).unwrap();
        let value = serde_json::to_value(s).unwrap();
        assert!(value["images"]["feed"]["render"].get("exposure").is_none());
    }
    #[test]
    fn control_and_medium_image_inputs_are_rejected() {
        for field in ["geometry_opacity", "alpha_cutoff", "geometry_normal"] {
            let mut v = valid();
            v["materials"]["screen"]["inputs"] = json!({field:{"image":"feed"}});
            assert!(
                crate::v2::from_test_value::<Scene>(v)
                    .unwrap()
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains(field)
            );
        }
        let mut v = valid();
        v["programs"]=json!({"custom":{"source":"struct Inputs { Image alpha_cutoff; }; Closure surface(ShadingContext c,Inputs i) {return unlit(sample(i.alpha_cutoff,c.TEXCOORD_0,c).rgb,1.);}"}});
        v["materials"]["screen"]=json!({"program":"custom","inputs":{"alpha_cutoff":{"image":"feed"}}});
        crate::v2::from_test_value::<Scene>(v).unwrap().validate().unwrap();
        let mut v = valid();
        v["materials"]["screen"] = json!({"program":{"source":include_str!("../../tests/fixtures/medium.slang")},"inputs":{"scattering":{"image":"feed"}}});
        v["nodes"].as_array_mut().unwrap().push(json!({"type":"light","shape":"environment","medium":"screen"}));
        assert!(
            crate::v2::from_test_value::<Scene>(v)
                .unwrap()
                .validate()
                .unwrap_err()
                .to_string()
                .contains("medium")
        );
    }
    #[test]
    fn medium_membership_follows_nested_lights_and_edits() {
        let mut v = valid();
        v["programs"] = json!({"custom":{"source":"struct Inputs { Image scattering; }; MediumCoefficients medium(MediumContext c, Inputs i) { return MediumCoefficients(); }"}});
        v["materials"]["screen"] = json!({"program":"custom","inputs":{"scattering":{"image":"feed"}}});
        v["nodes"].as_array_mut().unwrap().push(json!({"type":"node","children":[
            {"type":"node","children":[{"type":"light","shape":"environment","medium":"screen"}]}
        ]}));
        let mut scene: Scene = crate::v2::from_test_value(v).unwrap();
        // Membership, rather than the program's name, makes this a medium input.
        assert!(scene.validate().unwrap_err().to_string().contains("medium material screen"));
        // Validation must rediscover membership after an edit; no cached rejection.
        scene.nodes.pop();
        scene.validate().unwrap();
        let light: Node = crate::v2::from_test_value(json!({"type":"light","shape":"environment","medium":"screen"})).unwrap();
        scene.nodes.push(light);
        assert!(scene.validate().unwrap_err().to_string().contains("medium material screen"));
    }
    #[test]
    fn source_camera_defaults_match_primary_render() {
        let mut v = valid();
        v["images"]["feed"]["render"]
            .as_object_mut()
            .unwrap()
            .remove("camera");
        crate::v2::from_test_value::<Scene>(v)
            .unwrap()
            .validate()
            .unwrap();
    }
    #[test]
    fn shared_render_root_validation() {
        for image in [false, true] {
            let mut v = valid();
            let r = if image {
                &mut v["images"]["feed"]["render"]
            } else {
                &mut v["render"]
            };
            if !r.is_object() {
                *r = json!({});
            }
            r["root"] = json!("missing");
            assert!(
                crate::v2::from_test_value::<Scene>(v)
                    .unwrap()
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("render.root")
            );
        }
        let mut v = valid();
        v["render"] = json!({"root":"cam"});
        crate::v2::from_test_value::<Scene>(v)
            .unwrap()
            .validate()
            .unwrap();
    }
    #[test]
    fn whole_scene_self_cycle_is_rejected() {
        let s = scene(json!({"a":source(None)}), json!([mesh("a")]));
        assert!(
            s.image_order(None)
                .unwrap_err()
                .to_string()
                .contains("cycle")
        );
    }
    #[test]
    fn indirect_and_unused_cycles_are_rejected() {
        let nodes = json!([
            {"type":"node","name":"room_a","children":[mesh("b")]},
            {"type":"node","name":"room_b","children":[mesh("a")]}
        ]);
        let s = scene(
            json!({"a":source(Some("room_a")),"b":source(Some("room_b"))}),
            nodes,
        );
        assert!(
            s.image_order(Some("cam"))
                .unwrap_err()
                .to_string()
                .contains("cycle")
        );
    }
    #[test]
    fn dependencies_precede_consumers_and_are_reused() {
        let s = scene(
            json!({"a":source(Some("room_a")),"b":source(Some("cam"))}),
            json!([mesh("a"),mesh("a"),{"type":"node","name":"room_a","children":[mesh("b")]}]),
        );
        assert_eq!(s.image_order(None).unwrap(), vec!["b", "a"]);
        assert_eq!(s.image_reads(Some("room_a")), BTreeSet::from(["b".into()]));
        assert!(s.image_order(Some("cam")).unwrap().is_empty());
    }
}

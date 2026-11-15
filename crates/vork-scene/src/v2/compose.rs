//! Compositions on nodes (END-STATE-API §7): a node with `composition: {space: "3d"}` inside a canvas is a
//! closed 3D world, rendered to one picture and placed in the canvas; a node with `composition: {space: "2d"}`
//! inside a 3D world is a canvas, seen in the world through a picture of it (`images.X.render.root`). A
//! `{space: "2d"}` node inside a canvas paints its children to one picture, inline.
//!
//! Each such part is a document of its own, drawn by its own renderer: [`Scene::composition_scene`] is the
//! part (its children as the nodes, its tracks, the shared resources), and [`Scene::without_compositions`] the
//! document around it, where the part's node stays, empty, with its own fields and clock. Tracks move with the
//! nodes they key: the scene's tracks on nodes inside a part, and the part node's own layers that key its
//! children. Everything inside a part reads the part node's local time (§8 rule 7).
use super::*;
use std::collections::BTreeSet;

impl Scene {
    /// The space the document's own nodes are in.
    fn own_space(&self) -> Space {
        if self.is_canvas() { Space::Canvas } else { Space::World }
    }
    /// The nodes that start a part of another space, outermost first (not those inside another part). In a canvas
    /// these are worlds; in a world, canvases.
    pub fn composition_parts(&self) -> Vec<&Node> {
        fn visit<'a>(nodes: &'a [Node], space: Space, out: &mut Vec<&'a Node>) {
            for node in nodes {
                match &node.composition {
                    Some(c) if c.space != space => out.push(node),
                    _ => visit(&node.children, space, out),
                }
            }
        }
        let mut out = Vec::new();
        visit(&self.nodes, self.own_space(), &mut out);
        out
    }
    /// Whether some part still holds its nodes (a part emptied by `without_compositions` holds none).
    pub fn has_composition_parts(&self) -> bool {
        self.composition_parts().iter().any(|node| !node.children.is_empty())
    }
    /// The node named `name` and its ancestors, outermost first.
    fn node_with_ancestors(&self, name: &str) -> Option<(&Node, Vec<&Node>)> {
        fn visit<'a>(nodes: &'a [Node], name: &str, chain: &mut Vec<&'a Node>) -> Option<&'a Node> {
            for node in nodes {
                if node.name == name { return Some(node); }
                chain.push(node);
                if let Some(found) = visit(&node.children, name, chain) { return Some(found); }
                chain.pop();
            }
            None
        }
        let mut chain = Vec::new();
        visit(&self.nodes, name, &mut chain).map(|node| (node, chain))
    }
    /// Whether `name` is a node whose composition makes it a 2D canvas.
    pub fn is_canvas_node(&self, name: &str) -> bool {
        self.node_with_ancestors(name).is_some_and(|(node, _)| node.composition.as_ref().is_some_and(|c| c.space == Space::Canvas))
    }
    /// Whether `name` is a node whose composition makes it a closed 3D world.
    pub fn is_world_node(&self, name: &str) -> bool {
        self.node_with_ancestors(name).is_some_and(|(node, _)| node.composition.as_ref().is_some_and(|c| c.space == Space::World))
    }
    /// The images that are pictures of a canvas node (`render.root` names it), with that node's name.
    pub fn canvas_pictures(&self) -> Vec<(&str, &str)> {
        let canvases: BTreeSet<_> = self.composition_parts().into_iter()
            .filter(|node| node.composition.as_ref().is_some_and(|c| c.space == Space::Canvas))
            .map(|node| node.name.as_str()).collect();
        self.images.iter().filter_map(|(name, image)| {
            let root = image.render.as_ref()?.root.as_deref()?;
            canvases.contains(root).then_some((name.as_str(), root))
        }).collect()
    }

    /// The document without the inside of its parts: each part's node keeps its own fields, clock and the layers
    /// that key it, and loses its children and the tracks that key them. A document that has no part, and no
    /// material reading an image it does not hold, is that document already and is lent, not copied.
    pub fn without_compositions(&self) -> Result<std::borrow::Cow<'_, Scene>> {
        let reads_held_images = |material: &Material| material.inputs.values()
            .all(|input| !matches!(input, Input::Texture(binding) if !self.images.contains_key(&binding.image)));
        if self.composition_parts().is_empty() && self.materials.values().all(reads_held_images) {
            return Ok(std::borrow::Cow::Borrowed(self));
        }
        let mut out = self.clone();
        let mut inside = BTreeSet::new();
        let space = self.own_space();
        fn hollow(nodes: &mut [Node], space: Space, inside: &mut BTreeSet<String>) {
            for node in nodes {
                match &node.composition {
                    Some(c) if c.space != space => {
                        descendants(&node.children, inside);
                        node.children.clear();
                        for layer in &mut node.animation {
                            if layer.clip.is_some() { continue; }
                            if let Some(tracks) = &mut layer.tracks {
                                tracks.retain(|track| track.path.first().map(String::as_str) != Some("children"));
                            }
                        }
                        node.animation.retain(|layer| layer.clip.is_none() && layer.tracks.as_ref().is_some_and(|t| !t.is_empty()));
                    }
                    _ => hollow(&mut node.children, space, inside),
                }
            }
        }
        let canvases: Vec<String> = self.composition_parts().iter().filter(|n| n.composition.as_ref().is_some_and(|c| c.space == Space::Canvas))
            .map(|n| n.name.clone()).collect();
        hollow(&mut out.nodes, space, &mut inside);
        for image in out.images.values_mut() {
            if let Some(root) = image.render.as_ref().and_then(|r| r.root.clone()).filter(|root| canvases.contains(root)) {
                image.render = None;
                image.picture_of = Some(root);
            }
        }
        // Pictures wholly inside another composition are installed by that composition.
        out.images.retain(|_, image| image.render.as_ref().and_then(|r| r.root.as_ref()).is_none_or(|root| !inside.contains(root)));
        let images = out.images.keys().collect::<BTreeSet<_>>();
        out.materials.retain(|_, material| material.inputs.values().all(|input| !matches!(input, Input::Texture(binding) if !images.contains(&binding.image))));
        for layer in &mut out.animation {
            if let Some(tracks) = &mut layer.tracks {
                tracks.retain(|track| !keys_node_in(&track.path, &inside));
            }
        }
        out.animation.retain(|layer| layer.clip.as_ref().is_none_or(|clip| !inside.contains(&clip.target)));
        Ok(std::borrow::Cow::Owned(out))
    }

    /// The part that starts at node `name`, as a document of its own: the node's children as its nodes, drawn at
    /// `resolution` px. A canvas keeps the node's `viewBox`; a world is drawn with the document's render
    /// settings through its one camera, transparent where it shows no surface. Its tracks are the document's
    /// tracks on its nodes and the part node's own layers that key them (paths from the node become paths from
    /// the scene). A part under a node clock reads that clock, so the document's tracks on its nodes, in scene
    /// seconds, are refused there.
    pub fn composition_scene(&self, name: &str, resolution: [u32; 2]) -> Result<Scene> {
        let (node, ancestors) = self.node_with_ancestors(name).with_context(|| format!("no node {name:?}"))?;
        let composition = node.composition.as_ref().with_context(|| format!("node {name:?} has no composition"))?;
        let mut inside = BTreeSet::new();
        descendants(&node.children, &mut inside);
        let clocked = node.has_clock() || ancestors.iter().any(|a| a.has_clock());
        // The movies its nodes show, and the images its image graph reads.
        let mut media = self.media_sources(node);
        media.extend(self.compositor_image_reads(&composition.compositor));
        // What the part can read decides the material and image tracks it keeps. A world may read any material
        // or image, and so may a part with a world anywhere inside it: that world is extracted from this part
        // and finds its tracks here. A canvas itself reads the materials it binds: its view and its image
        // graph's entries.
        let reads_any = composition.space == Space::World || holds_world(&node.children);
        let own_materials: BTreeSet<&str> = composition.view.iter().filter_map(|view| match view { NameOr::Name(name) => Some(name.as_str()), _ => None })
            .chain(composition.compositor.iter().map(|entry| entry.material().as_str())).collect();
        let mut animation = Vec::new();
        for layer in &self.animation {
            let moved_clip = layer.clip.as_ref().is_some_and(|clip| inside.contains(&clip.target));
            let tracks: Vec<Track> = layer.tracks.iter().flatten().filter(|track| keys_node_in(&track.path, &inside)
                || (track.path.first().is_some_and(|field| field == "images") && track.path.get(1).is_some_and(|image| media.contains(image)))
                || (reads_any && matches!(track.path.first().map(String::as_str), Some("materials" | "images")))
                || (track.path.first().is_some_and(|field| field == "materials") && track.path.get(1).is_some_and(|material| own_materials.contains(material.as_str())))).cloned().collect();
            if tracks.is_empty() && !moved_clip { continue; }
            if clocked {
                let path = tracks.first().map_or_else(|| "a clip".to_string(), |t| display_path(&t.path));
                if tracks.first().is_some_and(|t| !keys_node_in(&t.path, &inside)) {
                    bail!("track {path}: a material or image keyed in scene seconds inside node {name:?}, which has its own clock, is not built yet");
                }
                bail!("track {path}: node {name:?} has its own clock, which everything in its composition reads; key this from the node's animation");
            }
            let mut moved = layer.clone();
            if !moved_clip { moved.tracks = Some(tracks); moved.clip = None; }
            animation.push(moved);
        }
        for layer in &node.animation {
            let clip = layer.clip.as_ref().filter(|clip| inside.contains(&clip.target)).cloned();
            let tracks = layer.tracks.iter().flatten().filter(|track| track.path.first().map(String::as_str) == Some("children"))
                .map(|track| -> Result<Track> {
                    let mut at = node;
                    let mut index = 0;
                    while track.path.get(index).map(String::as_str) == Some("children") {
                        let child = track.path.get(index + 1).with_context(|| format!("track {}: name a child node", display_path(&track.path)))?;
                        at = at.children.iter().find(|c| &c.name == child)
                            .with_context(|| format!("track {}: node {:?} has no child {child:?}", display_path(&track.path), at.name))?;
                        index += 2;
                    }
                    let mut moved = track.clone();
                    moved.path = ["nodes".to_string(), at.name.clone()].into_iter().chain(track.path[index..].iter().cloned()).collect();
                    Ok(moved)
                }).collect::<Result<Vec<_>>>()?;
            if tracks.is_empty() && clip.is_none() { continue; }
            animation.push(Layer {
                source_span: None, source_clock_span: Vec::new(), name: layer.name.clone(),
                tracks: if clip.is_some() { None } else { Some(tracks) }, clip,
                weight: layer.weight, blend: layer.blend, offset: layer.offset, speed: layer.speed, looping: layer.looping,
            });
        }
        let mut part = self.clone();
        part.nodes = node.children.clone();
        // Pictures of nodes outside the part, and what reads them, stay outside.
        part.images.retain(|_, image| image.render.as_ref().and_then(|r| r.root.as_ref()).is_none_or(|root| inside.contains(root)));
        let images = part.images.keys().cloned().collect::<BTreeSet<_>>();
        part.materials.retain(|_, material| material.inputs.values().all(|input| !matches!(input, Input::Texture(binding) if !images.contains(&binding.image))));
        part.animation = animation;
        part.render = Render {
            resolution, fps: self.render.fps, duration: self.render.duration.clone(),
            background: Background::Mode(BackgroundMode::Transparent), settings: self.render.settings.clone(), ..Render::default()
        };
        match composition.space {
            Space::Canvas => {
                part.composition = Some(composition.clone());
                part.simulation = None;
            }
            Space::World => part.composition = Some(composition.clone()),
        }
        Ok(part)
    }

    /// The checks of a document with parts: node names are unique across all of it, the document around the parts
    /// is checked as a document, and so is every part. A world in a canvas has one camera; a canvas in a world
    /// is seen through a picture of it.
    pub(super) fn validate_parts(&self, require_camera: bool) -> Result<()> {
        fn names<'a>(nodes: &'a [Node], seen: &mut BTreeSet<&'a str>) -> Result<()> {
            for node in nodes {
                ensure!(node.name.is_empty() || seen.insert(&node.name), "duplicate node name {:?}; node names must be unique", node.name);
                names(&node.children, seen)?;
            }
            Ok(())
        }
        names(&self.nodes, &mut BTreeSet::new())?;
        self.without_compositions()?.validate_content(require_camera)?;
        for node in self.composition_parts() {
            let name = &node.name;
            ensure!(!name.is_empty(), "a node with a composition needs a name");
            let space = node.composition.as_ref().expect("a part has a composition").space;
            let resolution = match space {
                Space::World => {
                    let [_, _, w, h] = self.canvas_view_box();
                    [w.round().max(1.) as u32, h.round().max(1.) as u32]
                }
                Space::Canvas => {
                    let pictures: Vec<&Image> = self.images.values().filter(|image| image.render.as_ref().and_then(|r| r.root.as_deref()) == Some(name.as_str())).collect();
                    ensure!(!pictures.is_empty(), "node {name:?}: a canvas in a 3D world is seen through a picture of it; name it in an image's render.root");
                    pictures[0].render.as_ref().expect("a picture renders").resolution
                }
            };
            let part = match space { Space::World => self.world_part(name, resolution)?, Space::Canvas => self.composition_scene(name, resolution)? };
            part.validate_content(true).with_context(|| format!("node {name:?}"))?;
        }
        Ok(())
    }
    /// The world that starts at node `name` (a node with `composition: {space: "3d"}` in a canvas), drawn at
    /// `resolution` px through its one camera.
    pub fn world_part(&self, name: &str, resolution: [u32; 2]) -> Result<Scene> {
        let part = self.composition_scene(name, resolution)?;
        fn cameras(nodes: &[Node]) -> usize {
            nodes.iter().map(|n| usize::from(matches!(n.kind, NodeKind::Camera { .. })) + cameras(&n.children)).sum()
        }
        let count = cameras(&part.without_compositions()?.nodes);
        ensure!(count == 1, "node {name:?}: a world in a canvas is drawn through its one camera; it has {count}");
        Ok(part)
    }
}

/// Every named node under `nodes`.
fn descendants(nodes: &[Node], out: &mut BTreeSet<String>) {
    for node in nodes {
        if !node.name.is_empty() { out.insert(node.name.clone()); }
        descendants(&node.children, out);
    }
}
/// Whether a world composition starts at one of `nodes` or below them.
fn holds_world(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| node.composition.as_ref().is_some_and(|composition| composition.space == Space::World) || holds_world(&node.children))
}
/// Whether a scene track path keys a node in `names`.
fn keys_node_in(path: &[String], names: &BTreeSet<String>) -> bool {
    path.first().map(String::as_str) == Some("nodes") && path.get(1).is_some_and(|name| names.contains(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scene(value: serde_json::Value) -> std::result::Result<Scene, String> {
        Scene::from_json(&serde_json::to_vec(&value).unwrap()).map_err(|e| format!("{e:#}"))
    }
    fn world() -> serde_json::Value {
        serde_json::json!({"name":"shot","composition":{"space":"3d"},"time":{"speed":2},"children":[
            {"type":"camera","name":"cam","translation":[0,0,5]},
            {"name":"spin","children":[{"type":"mesh","name":"box","geometry":"g"}]}],
            "animation":[{"tracks":[{"path":["children","spin","rotation"],"keys":[[0,{"euler":[0,0,0]}],[1,{"euler":[0,1,0]}]]},
                {"path":["opacity"],"keys":[[0,0],[1,1]]}]}]})
    }
    fn geometry() -> serde_json::Value {
        serde_json::json!({"p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0]]}})
    }
    #[test]
    fn composition_view_named_inline_and_default_on_root_and_nested_worlds() {
        for view in [serde_json::Value::Null, serde_json::json!("grade"),
            serde_json::json!({"program":"std/standard","inputs":{"exposure":1}})] {
            for nested in [false, true] {
                let mut doc = serde_json::json!({"version":2,
                    "materials":{"grade":{"program":"std/standard","inputs":{"exposure":2}}},
                    "buffers":geometry(),"geometry":{"g":{"attributes":{"POSITION":"p"}}}});
                if nested {
                    let mut node = world();
                    if !view.is_null() { node["composition"]["view"] = view.clone(); }
                    node["composition"]["color_space"] = serde_json::json!("Linear Rec.709");
                    doc["composition"] = serde_json::json!({"space":"2d"});
                    doc["nodes"] = serde_json::json!([node]);
                } else {
                    doc["composition"] = serde_json::json!({"space":"3d","color_space":"Linear Rec.709"});
                    if !view.is_null() { doc["composition"]["view"] = view.clone(); }
                    doc["nodes"] = serde_json::json!([{"type":"camera"}]);
                }
                let loaded = scene(doc).unwrap();
                loaded.validate().unwrap();
                // Serialization must retain the authored choice without exposing render.display.
                let serialized = serde_json::to_vec(&loaded.content_json().unwrap()).unwrap();
                assert!(!String::from_utf8_lossy(&serialized).contains("\"display\""));
                let loaded = if nested { loaded } else { Scene::from_json(&serialized).unwrap() };
                let part = if nested { loaded.composition_scene("shot", [320,180]).unwrap() } else { loaded };
                let part = Scene::from_json(&serde_json::to_vec(&part.content_json().unwrap()).unwrap()).unwrap();
                let (name, material) = part.view_material().unwrap();
                assert_eq!(material.program, if view.is_null() { "std/agx" } else { "std/standard" });
                if view.is_string() { assert_eq!(name.as_deref(), Some("grade")); }
            }
        }
        let default = scene(serde_json::json!({"version":2,"nodes":[{"type":"camera"}]})).unwrap();
        assert_eq!(default.view_material().unwrap().1.program, "std/agx");
        let canvas = scene(serde_json::json!({"version":2,"composition":{"space":"2d"},
            "buffers":geometry(),"geometry":{"g":{"attributes":{"POSITION":"p"}}},"nodes":[world()]})).unwrap();
        assert_eq!(canvas.view_material().unwrap().1.program, "std/standard");
        assert_eq!(canvas.composition_scene("shot", [32,32]).unwrap().view_material().unwrap().1.program, "std/agx");
    }

    #[test]
    fn a_canvas_takes_every_working_space_and_keeps_it_through_extraction() {
        for space in ["sRGB", "Display P3", "Linear Rec.709", "Linear Rec.2020", "ACEScg"] {
            let expected = serde_json::from_value::<WorkingColorSpace>(serde_json::json!(space)).unwrap();
            let root = scene(serde_json::json!({"version":2,"composition":{"space":"2d","color_space":space},
                "nodes":[{"type":"path","d":"M0 0H8V8H0Z","fill":"red"}]})).unwrap();
            assert_eq!(root.composition.as_ref().unwrap().working_space(), expected);
            // A canvas shown in a world keeps its own space through extraction.
            let world = scene(serde_json::json!({"version":2,"buffers":geometry(),"geometry":{"g":{"attributes":{"POSITION":"p"}}},
                "images":{"ui":{"render":{"root":"panel","resolution":[256,128]}}},
                "materials":{"screen":{"inputs":{"emission_color":{"image":"ui"}}}},
                "nodes":[{"type":"camera","name":"cam"},{"type":"mesh","name":"tv","geometry":"g","materials":["screen"]},
                    {"name":"panel","composition":{"space":"2d","color_space":space},"children":[
                        {"name":"label","type":"path","d":"M0 0H10V10Z","fill":"red"}]}]})).unwrap();
            assert_eq!(world.composition_scene("panel", [256, 128]).unwrap().composition.unwrap().working_space(), expected);
            // A canvas inside a canvas paints inline in its parent, in the parent's encoding.
            let inline = scene(serde_json::json!({"version":2,"composition":{"space":"2d"},
                "nodes":[{"name":"inner","composition":{"space":"2d","color_space":space}}]}))
                .and_then(|s| s.validate().map_err(|e| format!("{e:#}"))).unwrap_err();
            assert!(inline.contains("composition.color_space on a canvas inside a canvas is not built yet"), "{inline}");
        }
        // Omitted, a canvas is sRGB.
        let default = scene(serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":[]})).unwrap();
        assert_eq!(default.composition.as_ref().unwrap().working_space(), WorkingColorSpace::Srgb);
    }

    #[test]
    fn composition_view_refuses_unsupported_spaces_bindings_and_old_spellings() {
        for nested in [false, true] {
            for (field, value, expected) in [
                ("view", serde_json::json!("missing"), "unknown material missing"),
                ("color_space", serde_json::json!("ACEScg"), "this working space is not built yet"),
                ("display", serde_json::Value::Null, "use composition.view"),
                ("viewBox", serde_json::json!([0,0,1,1]), "a world is framed by its camera"),
            ] {
                let mut composition = serde_json::json!({"space":"3d"});
                composition[field] = value;
                let doc = if nested {
                    let mut node = world(); node["composition"] = composition;
                    serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":[node],
                        "buffers":geometry(),"geometry":{"g":{"attributes":{"POSITION":"p"}}}})
                } else { serde_json::json!({"version":2,"composition":composition,"nodes":[{"type":"camera"}]}) };
                let error = scene(doc).and_then(|s| s.validate().map_err(|e| format!("{e:#}"))).unwrap_err();
                assert!(error.contains(expected), "{error}");
            }
        }
        for field in ["output", "outputs"] {
            let error = scene(serde_json::json!({"version":2,field:{"display":"Rec.2100-PQ"}})).unwrap_err();
            assert!(error.contains("outputs are render-call destinations"), "{error}");
        }
        for field in ["display", "view"] {
            let error = scene(serde_json::json!({"version":2,"render":{field:null}})).unwrap_err();
            assert!(error.contains("use composition.view; composition.space is required too: write \"3d\"") && error.ends_with("; see vork help upgrade"), "{error}");
        }
        let error = scene(serde_json::json!({"version":2,"composition":{"space":"2d","view":null}})).unwrap_err();
        assert!(error.contains("null is not a view"));
    }

    #[test]
    fn a_world_in_a_canvas_is_a_document_of_its_own() {
        let doc = serde_json::json!({"version":2,"composition":{"space":"2d"},"render":{"resolution":[320,180]},
            "buffers":geometry(),"geometry":{"g":{"attributes":{"POSITION":"p"}}},"nodes":[world()]});
        let canvas = scene(doc.clone()).unwrap();
        let outer = canvas.without_compositions().unwrap();
        assert!(outer.nodes[0].children.is_empty() && outer.is_world_node("shot"));
        // The node keeps the layer keying its own opacity; the one keying its child moves into the world.
        assert_eq!(outer.nodes[0].animation[0].tracks.as_ref().unwrap().len(), 1);
        let inner = canvas.composition_scene("shot", [320, 180]).unwrap();
        assert!(!inner.is_canvas() && inner.nodes.len() == 2);
        assert_eq!(inner.animation[0].tracks.as_ref().unwrap()[0].path, ["nodes", "spin", "rotation"]);
        assert!(inner.render.background.transparent());
        // A scene track on a node inside a clocked world is refused in one line; two cameras are refused.
        let mut keyed = doc.clone();
        keyed["animation"] = serde_json::json!([{"tracks":[{"path":["nodes","box","translation"],"keys":[[0,[0,0,0]]]}]}]);
        let error = scene(keyed).unwrap_err();
        assert!(error.contains("node \"shot\" has its own clock") && error.lines().count() == 1, "{error}");
        let mut two = doc.clone();
        two["nodes"][0]["children"].as_array_mut().unwrap().push(serde_json::json!({"type":"camera","name":"cam2"}));
        assert!(scene(two).unwrap_err().contains("drawn through its one camera; it has 2"));
    }
    #[test]
    fn a_canvas_in_a_world_is_seen_through_a_picture() {
        let doc = serde_json::json!({"version":2,"buffers":geometry(),"geometry":{"g":{"attributes":{"POSITION":"p"}}},
            "images":{"ui":{"render":{"root":"panel","resolution":[256,128]}}},
            "materials":{"screen":{"inputs":{"emission_color":{"image":"ui"}}}},
            "nodes":[{"type":"camera","name":"cam"},{"type":"mesh","name":"tv","geometry":"g","materials":["screen"]},
                {"name":"panel","composition":{"space":"2d","viewBox":[0,0,512,256]},"children":[
                    {"name":"label","type":"path","d":"M0 0H10V10Z","fill":"red","active":[0,1]}]}]});
        let world = scene(doc.clone()).unwrap();
        assert_eq!(world.canvas_pictures(), vec![("ui", "panel")]);
        assert!(world.image_order(None).unwrap().is_empty(), "a canvas's picture is not a 3D capture");
        let canvas = world.composition_scene("panel", [256, 128]).unwrap();
        assert!(canvas.is_canvas() && canvas.canvas_view_box() == [0., 0., 512., 256.]);
        let mut unseen = doc.clone();
        unseen["images"] = serde_json::json!({});
        unseen["materials"] = serde_json::json!({});
        unseen["nodes"][1]["materials"] = serde_json::json!([]);
        assert!(scene(unseen).unwrap_err().contains("a canvas in a 3D world is seen through a picture of it"));
        let mut camera = doc;
        camera["images"]["ui"]["render"]["camera"] = "cam".into();
        assert!(scene(camera).unwrap_err().contains("images.ui.render.camera: a canvas's picture has no camera"));
    }
    #[test]
    fn nested_pictures_belong_to_their_own_world() {
        let doc = serde_json::json!({"version":2,
            "images":{"outer":{"render":{"root":"panel","resolution":[64,32]}},
                      "inner":{"render":{"root":"inner_panel","resolution":[64,32]}}},
            "materials":{"inner_material":{"inputs":{"emission_color":{"image":"inner"}}}},
            "nodes":[{"type":"camera","name":"camera"},
                {"name":"panel","composition":{"space":"2d"},"children":[
                    {"name":"world","composition":{"space":"3d"},"children":[
                        {"type":"camera","name":"inner_camera"},
                        {"name":"inner_panel","composition":{"space":"2d"},"children":[
                            {"name":"another_world","composition":{"space":"3d"},"children":[
                                {"type":"camera","name":"deep_camera"}]}]}]}]}]});
        let world = scene(doc).unwrap();
        assert_eq!(world.canvas_pictures(), vec![("outer", "panel")]);
        let outer = world.without_compositions().unwrap();
        assert!(!outer.images.contains_key("inner"));
        assert!(!outer.materials.contains_key("inner_material"));
        let canvas = world.composition_scene("panel", [64,32]).unwrap();
        let inner = canvas.world_part("world", [64,32]).unwrap();
        assert_eq!(inner.canvas_pictures(), vec![("inner", "inner_panel")]);
    }

    #[test]
    fn a_part_keeps_the_material_tracks_it_or_a_world_inside_it_reads() {
        // world → canvas `panel` → world → canvas `inner_panel`, whose view is the keyed material `lift`.
        let doc = serde_json::json!({"version":2,
            "images":{"outer":{"render":{"root":"panel","resolution":[64,32]}},
                      "inner":{"render":{"root":"inner_panel","resolution":[64,32]}},
                      "plain":{"render":{"root":"plain_panel","resolution":[64,32]}}},
            "materials":{"inner_material":{"inputs":{"emission_color":{"image":"inner"}}},
                         "lift":{"program":"std/standard","inputs":{"exposure":0}}},
            "nodes":[{"type":"camera","name":"camera"},
                {"name":"plain_panel","composition":{"space":"2d"},"children":[{"name":"mark","type":"path","d":"M0 0H10V10Z","fill":"red"}]},
                {"name":"panel","composition":{"space":"2d"},"children":[
                    {"name":"world","composition":{"space":"3d"},"children":[
                        {"type":"camera","name":"inner_camera"},
                        {"name":"inner_panel","composition":{"space":"2d","view":"lift"},"children":[
                            {"name":"label","type":"path","d":"M0 0H10V10Z","fill":"red"}]}]}]}],
            "animation":[{"tracks":[{"path":["materials","lift","inputs","exposure"],"keys":[[0,0],[1,1]]}]}]});
        let world = scene(doc).unwrap();
        let keyed = |part: &Scene| part.animation.iter().flat_map(|layer| layer.tracks.iter().flatten())
            .any(|track| track.path.iter().map(String::as_str).eq(["materials", "lift", "inputs", "exposure"]));
        // The outer canvas binds no view of its own, but the world inside it reads the track's material.
        let canvas = world.composition_scene("panel", [64,32]).unwrap();
        assert!(keyed(&canvas));
        let inner = canvas.world_part("world", [64,32]).unwrap();
        assert!(keyed(&inner));
        let viewed = inner.composition_scene("inner_panel", [64,32]).unwrap();
        assert!(keyed(&viewed) && viewed.view_material().unwrap().0.as_deref() == Some("lift"));
        // A canvas with no world inside and no view of that name reads no material.
        assert!(world.composition_scene("plain_panel", [64,32]).unwrap().animation.is_empty());
    }

    #[test]
    fn movie_paint_time_tracks_follow_the_canvas_resource() {
        let mut doc = serde_json::json!({"version":2,
            "images":{"outer":{"render":{"root":"panel","resolution":[64,32]}},"clip":{"file":"clip.mp4"}},
            "nodes":[{"type":"camera","name":"camera"},
                {"name":"panel","composition":{"space":"2d"},"children":[
                    {"name":"movie","type":"path","d":"M0 0H64V32H0Z","fill":{"image":"clip","fit":"fill"}}]}],
            "animation":[{"tracks":[{"path":["images","clip","time"],"keys":[[0,2],[1,0]]}]}]});
        for node in [
            serde_json::json!({"name":"movie","type":"path","d":"M0 0H64V32H0Z","fill":{"image":"clip","fit":"fill"}}),
            serde_json::json!({"name":"movie","type":"text","content":"A","font":"font.ttf","fill":{"image":"clip"}}),
            serde_json::json!({"name":"movie","type":"text","font":"font.ttf","content":[{"content":"A","stroke":{"image":"clip"}}]}),
        ] {
            doc["nodes"][1]["children"][0] = node;
            let world = scene(doc.clone()).unwrap();
            let canvas = world.composition_scene("panel", [64,32]).unwrap();
            assert_eq!(canvas.animation.len(), 1);
            assert_eq!(canvas.node_media(&canvas.nodes[0]), vec!["clip"]);
        }
    }

}

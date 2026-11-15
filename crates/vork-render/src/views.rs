//! Install-time view instances in submission order. Slots carry only their values.
use crate::*;
use std::ops::Range;

pub(crate) enum ViewKind {
    Camera(usize),
    Image {
        name: String,
        render: Box<crate::render_images::InstalledImageRender>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PassSet {
    Camera,
    Image,
}

/// A dense table index, fixed for the lifetime of an install.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ViewId(pub usize);

pub(crate) struct ViewDesc {
    pub kind: ViewKind,
    pub extent: [u32; 2],
    pub passes: PassSet,
    /// Producer views, always earlier in the table. Only image outputs cross views.
    pub reads: Vec<ViewId>,
}

#[derive(Default)]
pub(crate) struct ViewTable {
    pub entries: Vec<ViewDesc>,
    pub images: Range<usize>,
    pub camera: usize,
}

impl ViewTable {
    pub fn new(s: &InstalledScene, r: &Renderer) -> Result<Self> {
        let order = s.image_order.clone();
        let image_renders = crate::render_images::plan(&s.scene, s, r, &order)?;
        let mut entries = Vec::new();
        // Scene validation owns cycle detection and the stable topological order.
        // Resolve names once; replay uses table indices only.
        let mut producers = std::collections::HashMap::new();
        let dependencies = |root: Option<&str>, compositor: &[vork_scene::v2::CompositeEntry], d:crate::compositor::Dependencies,
                            producers: &std::collections::HashMap<String, ViewId>|
         -> Result<Vec<ViewId>> {
            let scene_reads=if d.beauty || d.controls() || d.aovs || d.catcher {s.scene.image_reads(root)} else {Default::default()};
            let image_reads=if d.color {crate::compositor::image_reads(&s.compositor,compositor)} else {Default::default()};
            scene_reads.into_iter().chain(image_reads)
                .filter(|name| {
                    s.scene
                        .images
                        .get(name)
                        .is_some_and(|image| image.render.is_some())
                })
                .map(|name| {
                    producers.get(&name).copied().with_context(|| {
                        format!("view reads render image {name} before its producer")
                    })
                })
                .collect()
        };
        for (name, render) in order.iter().zip(image_renders) {
            let settings = s.scene.images[name]
                .render
                .as_ref()
                .context("image view has no render source")?;
            let graph = s.scene.images[name].compositor();
            let reads = dependencies(settings.root.as_deref(), graph, crate::compositor::dependencies(&s.compositor,graph).0, &producers)?;
            let id = ViewId(entries.len());
            entries.push(ViewDesc {
                kind: ViewKind::Image {
                    name: name.clone(),
                    render: Box::new(render),
                },
                extent: settings.resolution,
                passes: PassSet::Image,
                reads,
            });
            producers.insert(name.clone(), id);
        }
        let images = 0..entries.len();
        let camera = entries.len();
        entries.push(ViewDesc {
            kind: ViewKind::Camera(s.camera),
            extent: s.render.resolution,
            passes: PassSet::Camera,
            reads: dependencies(s.render.root.as_deref(), &s.render.compositor, s.main_outputs, &producers)?,
        });
        let table = Self {
            entries,
            images,
            camera,
        };
        table.validate()?;
        Ok(table)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.camera + 1 == self.entries.len(),
            "main view must follow its producers"
        );
        for (id, view) in self.entries.iter().enumerate() {
            ensure!(
                view.extent.into_iter().all(|n| n > 0),
                "view {id} has an empty extent"
            );
            ensure!(
                view.reads.iter().all(|producer| producer.0 < id),
                "view {id} reads a later producer"
            );
            ensure!(
                matches!(
                    (&view.kind, view.passes),
                    (ViewKind::Camera(_), PassSet::Camera)
                        | (ViewKind::Image { .. }, PassSet::Image)
                ),
                "view {id} has an incompatible pass template"
            );
        }
        Ok(())
    }

    pub fn image_renders(
        &self,
    ) -> impl ExactSizeIterator<Item = &crate::render_images::InstalledImageRender> {
        self.entries[self.images.clone()].iter().map(|view| {
            let ViewKind::Image { render, .. } = &view.kind else {
                unreachable!("image range validated at install")
            };
            render.as_ref()
        })
    }
}

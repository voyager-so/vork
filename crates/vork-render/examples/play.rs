//! Minimal Rust host: cargo run --release -p vork-render --example play -- SCENE
//! WASD/QE translates the selected camera, arrows turn it, Space pauses time,
//! +/- adjusts a material's base weight when --material NAME is supplied.
//! --frames N exits after N presented frames; default 600 bounds test packets.
use anyhow::{Context, Result, ensure};
use std::{collections::HashSet, path::PathBuf, sync::Arc, time::{Duration, Instant}};
use vork_render::session::{LiveInput, Presentation, Session, Update};
use winit::{application::ApplicationHandler, event::{ElementState, WindowEvent}, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowId}};

struct App {
    scene: PathBuf,
    material: Option<String>,
    limit: u32,
    window: Option<Arc<Window>>,
    session: Option<Session>,
    presentation: Option<Presentation<'static>>,
    error: Option<anyhow::Error>,
    keys: HashSet<KeyCode>,
    position: glam::Vec3,
    rotation: glam::Quat,
    weight: f32,
    time: f32,
    paused: bool,
    last: Instant,
    frames: u32,
    frame_ms: Vec<f64>,
    present_intervals_ms: Vec<f64>,
    last_present: Option<Instant>,
    started: Instant,
    next_redraw: Instant,
}

fn camera<'a>(nodes: &'a mut [vork_scene::v2::Node], name: &str) -> Option<&'a mut vork_scene::v2::Node> {
    for node in nodes {
        if node.name == name { return Some(node); }
        if let Some(found) = camera(&mut node.children, name) { return Some(found); }
    }
    None
}

impl App {
    fn open(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let mut scene = vork_render::load_authored(&self.scene)?;
        let profile = vork_scene::Profile::load("std/previs")?;
        scene.apply_settings(profile.resolve(&Default::default())?, &profile.name)?;
        scene.render.resolution = [640,360];
        ensure!(scene.render.motion_blur.as_ref().is_none_or(|blur| blur.shutter == 0.), "play host needs motion blur disabled");
        let name = scene.render.camera.clone().context("play needs render.camera to name an authored camera")?;
        let node = camera(&mut scene.nodes, &name).context("play camera must be an authored node")?;
        ensure!(node.target.is_none(), "play fly camera needs target removed so host rotation controls its direction");
        self.position = glam::Vec3::from_array(node.transform.translation);
        self.rotation = glam::Quat::from_array(node.transform.rotation.unwrap_or([0.,0.,0.,1.]));
        let mut inputs = vec![
            LiveInput { path: vec!["nodes".into(),name.clone(),"translation".into()], initial: self.position.to_array().to_vec(), min: vec![-1000.;3], max: vec![1000.;3] },
            LiveInput { path: vec!["nodes".into(),name,"rotation".into()], initial: self.rotation.to_array().to_vec(), min: vec![-1.;4], max: vec![1.;4] },
        ];
        if let Some(name) = &self.material {
            inputs.push(LiveInput { path: vec!["materials".into(), name.clone(), "inputs".into(), "base_weight".into()], initial: vec![1.], min: vec![0.], max: vec![1.] });
        }
        let session = Session::prepare_live(vork_render::Renderer::new()?, scene, self.scene.parent().unwrap_or(std::path::Path::new(".")), inputs)?;
        let window = Arc::new(event_loop.create_window(Window::default_attributes()
            .with_title("vork play — WASD/QE, arrows, Space, +/-; Esc exits")
            .with_inner_size(winit::dpi::PhysicalSize::new(960,540)))?);
        let size = window.inner_size();
        self.presentation = Some(session.presentation(window.clone(), [size.width,size.height])?);
        self.session = Some(session);
        self.window = Some(window);
        self.last = Instant::now();
        self.started = self.last;
        self.next_redraw = self.last;
        Ok(())
    }

    fn draw(&mut self) -> Result<()> {
        let now = Instant::now(); let dt = (now-self.last).as_secs_f32().min(0.05); self.last = now;
        let axis = |positive, negative| self.keys.contains(&positive) as u8 as f32 - self.keys.contains(&negative) as u8 as f32;
        let local = glam::Vec3::new(axis(KeyCode::KeyD,KeyCode::KeyA),axis(KeyCode::KeyE,KeyCode::KeyQ),axis(KeyCode::KeyS,KeyCode::KeyW));
        self.position = (self.position + self.rotation * local * dt * 3.).clamp(glam::Vec3::splat(-999.), glam::Vec3::splat(999.));
        self.rotation = (glam::Quat::from_rotation_y(axis(KeyCode::ArrowLeft,KeyCode::ArrowRight)*dt) * self.rotation
            * glam::Quat::from_rotation_x(axis(KeyCode::ArrowUp,KeyCode::ArrowDown)*dt)).normalize();
        self.weight = (self.weight + axis(KeyCode::Equal,KeyCode::Minus)*dt).clamp(0.,1.);
        if !self.paused { self.time += dt; }
        let position = self.position.to_array(); let rotation = self.rotation.to_array(); let weight = [self.weight];
        let session = self.session.as_mut().unwrap();
        let mut updates = vec![Update { input: session.input(0).unwrap(), value: &position }, Update { input: session.input(1).unwrap(), value: &rotation }];
        if let Some(input) = session.input(2) { updates.push(Update { input, value: &weight }); }
        if session.present(self.presentation.as_mut().unwrap(), self.time, &updates)?.is_some() {
            let presented = Instant::now();
            self.frames += 1;
            self.frame_ms.push((presented-now).as_secs_f64()*1000.);
            if let Some(last) = self.last_present {
                self.present_intervals_ms.push((presented-last).as_secs_f64()*1000.);
            }
            self.last_present = Some(presented);
        }
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.error.is_some() { event_loop.exit(); return; }
        if self.window.is_none() {
            if let Err(error) = self.open(event_loop) { self.error = Some(error); event_loop.exit(); }
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        // Native events already queued before failed initialization must not use
        // the absent presentation or replace the original driver error.
        if self.error.is_some() { event_loop.exit(); return; }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.presentation.as_mut().unwrap().resize([size.width,size.height]),
            WindowEvent::Focused(false) => self.keys.clear(),
            WindowEvent::KeyboardInput { event, .. } => if let PhysicalKey::Code(key) = event.physical_key {
                if event.state == ElementState::Pressed {
                    self.keys.insert(key);
                    if key == KeyCode::Escape { event_loop.exit(); }
                    if key == KeyCode::Space && !event.repeat { self.paused = !self.paused; }
                } else { self.keys.remove(&key); }
            },
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.draw() { self.error = Some(error); event_loop.exit(); }
                if self.frames >= self.limit { event_loop.exit(); }
            },
            _ => {},
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.error.is_some() { event_loop.exit(); return; }
        // Bound retries even when occluded/acquire-timeout returns immediately.
        if self.started.elapsed() >= Duration::from_secs(120) { event_loop.exit(); return; }
        let now = Instant::now();
        if now >= self.next_redraw {
            self.next_redraw = now + Duration::from_millis(16);
            if let Some(window) = &self.window { window.request_redraw(); }
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_redraw));
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let scene = args.next().context("usage: play SCENE [--material NAME] [--frames N]")?.into();
    let mut material = None; let mut limit = 600;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--material" => material = Some(args.next().context("--material needs a name")?),
            "--frames" => limit = args.next().context("--frames needs a count")?.parse()?,
            _ => anyhow::bail!("unknown argument {arg}"),
        }
    }
    ensure!(limit > 0 && limit <= 36000, "--frames must be 1..36000");
    let mut app = App { scene, material, limit, window: None, session: None, presentation: None, error: None,
        keys: HashSet::new(), position: glam::Vec3::ZERO, rotation: glam::Quat::IDENTITY, weight:1., time:0., paused:false, last:Instant::now(), frames:0, frame_ms:Vec::new(), present_intervals_ms:Vec::new(), last_present:None, started:Instant::now(), next_redraw:Instant::now() };
    let result = EventLoop::new().context("play needs a real desktop display; use the session offscreen host on headless machines").and_then(|event_loop| event_loop.run_app(&mut app).map_err(Into::into));
    println!("{}", serde_json::json!({
        "presented_frames": app.frames, "requested_frames": app.limit,
        "elapsed_ms": app.started.elapsed().as_secs_f64()*1000.,
        "frame_wall_ms": app.frame_ms, "present_interval_ms": app.present_intervals_ms,
        "measurement": "host present submissions; frame wall includes live updates and present; intervals include pacing, not display scanout latency"
    }));
    result?;
    ensure!(app.frames >= app.limit || app.error.is_some(),
        "presented only {} of {} requested frames; keep the screen unlocked and window visible until completion", app.frames, app.limit);
    if let Some(error) = app.error { return Err(error); }
    Ok(())
}

//! Node sound uses the sound slice of the picture animation plan and the shared CPU samples.
use anyhow::{Context, Result, ensure};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};
use vork_audio::{Clip, Frame, SAMPLE_RATE, Source};
use vork_scene::{Node, Scene};

/// One source per file state for the process: every mix shares what has been decoded of it.
pub(crate) fn load(path: &Path) -> Result<Arc<Source>> {
    type Key = (PathBuf, u64, std::time::SystemTime);
    static CACHE: OnceLock<Mutex<BTreeMap<Key, Arc<Source>>>> = OnceLock::new();
    let path = path
        .canonicalize()
        .with_context(|| format!("audio: {}", path.display()))?;
    let stat = std::fs::metadata(&path)?;
    let key = (path.clone(), stat.len(), stat.modified()?);
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(source) = cache.get(&key) {
        return Ok(source.clone());
    }
    let source = Arc::new(Source::open(&path)?);
    // Bound process retention. Open scenes hold their own Arc.
    if cache.len() >= 16 {
        cache.clear();
    }
    cache.insert(key, source.clone());
    Ok(source)
}

pub(crate) struct Mix {
    nodes: Vec<crate::EvalNode>,
    plan: crate::animation::Plan,
    sound: crate::animation::SoundPlan,
    /// Nodes with a file and their ancestors, ascending.
    audible: Vec<usize>,
    /// Nodes with a file, ascending, and each one's index into `sources`.
    voices: Vec<usize>,
    files: Vec<usize>,
    /// The distinct files the voices play.
    sources: Vec<Arc<Source>>,
}
/// Blocks scheduled, then decoded for and mixed, at a time: the schedule's memory bound.
const BATCH: usize = 4096;
impl Mix {
    pub(crate) fn new(scene: &Scene, base: &Path) -> Result<Self> {
        fn visit(list: &[Node], parent: Option<usize>, nodes: &mut Vec<crate::EvalNode>) {
            for node in list {
                let i = nodes.len();
                nodes.push(crate::EvalNode {
                    node: node.without_children(),
                    parent,
                    world: glam::Mat4::IDENTITY,
                    dynamic: false,
                    camera: None,
                    embedded: None,
                });
                visit(&node.children, Some(i), nodes);
            }
        }
        let mut nodes = Vec::new();
        visit(&scene.nodes, None, &mut nodes);
        let mut voices = Vec::new();
        let mut files = Vec::new();
        let mut sources: Vec<Arc<Source>> = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            if let Some(file) = n.node.audio.as_ref().and_then(|a| a.file.as_ref()) {
                let source = load(&base.join(file))?;
                voices.push(i);
                files.push(
                    sources
                        .iter()
                        .position(|known| Arc::ptr_eq(known, &source))
                        .unwrap_or_else(|| {
                            sources.push(source);
                            sources.len() - 1
                        }),
                );
            }
        }
        // The plan resolves every track's target, owner and clock; the sound slice below keeps
        // the ones sound reads. Layers stay whole: a looping layer's span covers all its tracks.
        let mut plan = crate::animation::Plan::compile_with_buffers(scene, base, None)?;
        plan.mark_canvas_nodes(&mut nodes)?;
        let mut audible = std::collections::BTreeSet::new();
        for &voice in &voices {
            let mut parent = Some(voice);
            while let Some(i) = parent {
                audible.insert(i);
                parent = nodes[i].parent;
            }
        }
        let audible: Vec<usize> = audible.into_iter().collect();
        let sound = plan.sound_plan(&audible, |name| {
            nodes
                .iter()
                .position(|n| n.node.name == name)
                .with_context(|| format!("audio: unknown node {name:?}"))
        })?;
        Ok(Self {
            nodes,
            plan,
            sound,
            audible,
            voices,
            files,
            sources,
        })
    }
    pub(crate) fn node(&self, name: &str) -> Result<usize> {
        self.nodes
            .iter()
            .position(|n| n.node.name == name)
            .with_context(|| format!("audio: unknown node {name:?}"))
    }
    pub(crate) fn render_range(
        &self,
        first: usize,
        last: usize,
        fps: vork_time::Time,
        root: Option<usize>,
    ) -> Result<Vec<Frame>> {
        let start = if first == 0 {
            0
        } else {
            crate::audio_output::sample_offset(first as u64, fps)?
        };
        let count = crate::audio_output::sample_count((last + 1 - first) as u64, fps)?;
        self.render(
            start as f64 / SAMPLE_RATE as f64,
            (start + count) as f64 / SAMPLE_RATE as f64,
            root,
        )
    }
    pub(crate) fn render(&self, start: f64, end: f64, root: Option<usize>) -> Result<Vec<Frame>> {
        ensure!(
            start.is_finite() && end.is_finite() && end > start,
            "audio: window needs finite start < end"
        );
        let boundary = |seconds: f64| {
            let sample = seconds * SAMPLE_RATE as f64;
            if (sample - sample.round()).abs() < 1e-7 {
                sample.round()
            } else {
                sample.ceil()
            }
        };
        let first = boundary(start);
        let last = boundary(end);
        ensure!(
            first >= i64::MIN as f64 && last <= i64::MAX as f64 && last - first <= 48_000. * 3600.,
            "audio: window exceeds one hour; render a shorter range"
        );
        let mut output = Vec::with_capacity((last - first) as usize);
        let (first, last) = (first as i64, last as i64);
        let mut scratch = self.plan.sound_scratch();
        let mut values = vec![[0.; 2]; self.nodes.len()];
        let base_gains: Vec<f32> = self
            .nodes
            .iter()
            .map(|n| n.node.audio.as_ref().map_or(1., |a| a.gain))
            .collect();
        let mut gains = base_gains.clone();
        // How many frames from its start each file has been read so far, the frames
        // themselves, and the blocks scheduled but not yet mixed.
        let mut needs = vec![0; self.sources.len()];
        let mut clips: Vec<Option<Arc<Clip>>> = vec![None; self.sources.len()];
        let mut blocks = Vec::new();
        let mut cursor = first;
        while cursor < last {
            // Schedule: each block is one exact evaluation, and says how far it reads each file.
            let mut scheduled = cursor;
            while scheduled < last && blocks.len() < BATCH {
                let block =
                    self.plan
                        .sound_block(&self.sound, &self.voices, scheduled, last, &mut scratch)?;
                for (voice, &i) in self.voices.iter().enumerate() {
                    let Some(clock) = block.clock(voice) else {
                        continue;
                    };
                    let rate = self.sources[self.files[voice]].rate;
                    // A straight clock's extremes are its ends; a curved one is read at each sample.
                    let need = &mut needs[self.files[voice]];
                    let mut read = |sample: i64| -> Result<()> {
                        if !clock.sounds(sample) {
                            return Ok(());
                        }
                        let speed = clock.speed(sample)?;
                        ensure!(
                            speed.is_finite() && speed.abs() * rate as f64 / SAMPLE_RATE as f64 <= 256.,
                            "audio: node {:?}: resampling above 256 source samples per output sample is not built yet",
                            self.nodes[i].node.name
                        );
                        *need = (*need).max(Clip::reach(rate, clock.seconds(sample)?, speed));
                        Ok(())
                    };
                    if clock.straight() {
                        read(scheduled)?;
                        read(block.end() - 1)?;
                    } else {
                        for sample in scheduled..block.end() {
                            read(sample)?;
                        }
                    }
                }
                scheduled = block.end();
                blocks.push(block);
            }
            // Decode: each file continues from where it stopped, as far as these blocks read.
            // The previous frames are released first so they grow in place.
            clips.fill(None);
            for ((clip, source), &need) in clips.iter_mut().zip(&self.sources).zip(&needs) {
                if need > 0 {
                    *clip = Some(source.prefix(need)?);
                }
            }
            // Mix.
            for block in blocks.drain(..) {
                for sample in cursor..block.end() {
                    for &i in &self.audible {
                        values[i] = [0.; 2];
                    }
                    block.gains(
                        &self.plan,
                        &self.sound,
                        sample,
                        &mut scratch,
                        &base_gains,
                        &mut gains,
                    )?;
                    for (voice, &i) in self.voices.iter().enumerate() {
                        let (Some(clock), Some(clip)) =
                            (block.clock(voice), &clips[self.files[voice]])
                        else {
                            continue;
                        };
                        if !clock.sounds(sample) {
                            continue;
                        }
                        let local = clock.seconds(sample)?;
                        let speed = clock.speed(sample)?;
                        values[i] = clip.sample(
                            if speed < 0. {
                                local - 1. / clip.rate as f64
                            } else {
                                local
                            },
                            speed,
                        );
                    }
                    let mut mixed = [0.; 2];
                    for &i in self.audible.iter().rev() {
                        let gain = gains[i];
                        values[i] = values[i].map(|v| v * gain);
                        ensure!(
                            values[i].iter().all(|v| v.is_finite()),
                            "audio: node {:?} has a nonfinite mix",
                            self.nodes[i].node.name
                        );
                        if root == Some(i) {
                            mixed = values[i];
                            break;
                        }
                        if let Some(parent) = self.nodes[i].parent {
                            for c in 0..2 {
                                values[parent][c] += values[i][c];
                            }
                        } else if root.is_none() {
                            for c in 0..2 {
                                mixed[c] += values[i][c];
                            }
                        }
                    }
                    ensure!(
                        mixed.iter().all(|v| v.is_finite()),
                        "audio: nonfinite root mix; reduce source gains"
                    );
                    output.push(mixed);
                }
                cursor = block.end();
            }
        }
        Ok(output)
    }
}

pub(crate) fn has_audio(nodes: &[Node]) -> bool {
    nodes
        .iter()
        .any(|n| n.audio.is_some() || has_audio(&n.children))
}

pub(crate) struct MovieSound {
    wav: PathBuf,
    fps: vork_time::Time,
}
impl Drop for MovieSound {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.wav);
    }
}
impl MovieSound {
    pub(crate) fn prepare(
        scene: &Scene,
        base: &Path,
        first: usize,
        last: usize,
        destinations: &[PathBuf],
    ) -> Result<Option<Self>> {
        if !has_audio(&scene.nodes) {
            return Ok(None);
        }
        let mix = Mix::new(scene, base)?;
        let root = scene
            .render
            .root
            .as_deref()
            .map(|n| mix.node(n))
            .transpose()?;
        let frames = mix.render_range(first, last, scene.render.fps, root)?;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let wav = loop {
            let path = std::env::temp_dir().join(format!(
                "vork-audio-{}-{}.wav",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => break path,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        };
        let sound = Self {
            wav,
            fps: scene.render.fps,
        };
        vork_audio::write_wav(&sound.wav, &frames)?;
        let codec = if destinations
            .iter()
            .any(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("mp4")))
        {
            "aac"
        } else {
            "pcm_s16le"
        };
        let check = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(&sound.wav)
            .args(["-t", "0.01", "-c:a", codec, "-f", "null", "-"])
            .output()
            .context("audio: install ffmpeg to mux movie sound")?;
        ensure!(
            check.status.success(),
            "audio: {codec} encoding unavailable: {}",
            String::from_utf8_lossy(&check.stderr).trim()
        );
        Ok(Some(sound))
    }
    pub(crate) fn mux(&self, destinations: &[PathBuf], frames: u64) -> Result<()> {
        for path in destinations
            .iter()
            .filter(|p| crate::shot::video_destination(p))
        {
            crate::audio_output::mux_file(path, &self.wav, 1., 0., frames, self.fps)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scene(nodes: serde_json::Value) -> Scene {
        Scene::from_json(
            &serde_json::to_vec(
                &serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":nodes}),
            )
            .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn asset_export_never_silently_drops_sound() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-export-sound-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let path = base.join("scene.json");
        std::fs::write(
            &path,
            br#"{"version":2,"nodes":[{"name":"speaker","audio":{"gain":0.5},"type":"mesh","geometry":"g"}],"geometry":{"g":{"attributes":{"POSITION":"p"}}},"buffers":{"p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0]]}}}"#,
        )?;
        let mut export = crate::export::Export {
            scene: path,
            out: base.join("scene.glb"),
            allow_losses: vec!["environment_lighting".into()],
            ..Default::default()
        };
        assert!(export.run().is_err());
        assert!(!export.out.exists());
        export.allow_losses.push("audio".into());
        let report = export.run()?;
        assert!(
            report["issues"]
                .as_array()
                .unwrap()
                .iter()
                .any(|i| i["code"] == "audio")
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    fn wav_shot_needs_neither_camera_nor_render_device() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-wav-shot-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let source: Vec<_> = (0..4800).map(|i| [i as f32 / 4800.; 2]).collect();
        vork_audio::write_wav(&base.join("source.wav"), &source)?;
        let path = base.join("scene.json");
        std::fs::write(&path, br#"{"version":2,"render":{"duration":0.1,"fps":30},"nodes":[{"audio":{"file":"source.wav"}}]}"#)?;
        let output = base.join("result.wav");
        let shot = crate::shot::Shot {
            scene: path,
            out: vec![output.clone().into()],
            outputs: Vec::new(),
            frame_start: Some(1),
            frame_end: Some(2),
            ..Default::default()
        }
        .render()?;
        assert_eq!(shot.report["samples"], 3200);
        assert!(shot.into_retained().is_none());
        assert_eq!(Clip::load(&output)?.frames, source[1600..]);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[cfg(unix)]
    #[test]
    fn a_failed_wav_rerender_withdraws_the_previous_completion_record() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        // audit-core 9: WAV delivery takes the output run picture delivery takes.
        let base = std::env::temp_dir().join(format!("vork-wav-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base)?;
        vork_audio::write_wav(&base.join("source.wav"), &vec![[0.25; 2]; 4800])?;
        let path = base.join("scene.json");
        std::fs::write(&path, br#"{"version":2,"render":{"duration":0.1,"fps":30},"nodes":[{"audio":{"file":"source.wav"}}]}"#)?;
        let first = base.join("out").join("a.wav");
        let render = |out: Vec<std::path::PathBuf>, frame_end: usize| {
            crate::shot::Shot { scene: path.clone(), out: out.into_iter().map(Into::into).collect(), outputs: Vec::new(), frame_start: Some(0), frame_end: Some(frame_end), ..Default::default() }.render().map(|_| ())
        };
        let meta = crate::shot::sidecar(&first, "meta.json");
        let timings = crate::shot::sidecar(&first, "timings.json");
        render(vec![first.clone()], 2)?;
        assert!(meta.exists() && timings.exists());
        assert!(!crate::shot::sidecar(&first, "meta.json.pending").exists());
        assert_eq!(Clip::load(&first)?.frames.len(), 4800);
        // A rerender of different content whose second destination cannot be written: its
        // directory admits the run (the lock file is there) but no new file.
        let locked = base.join("locked");
        std::fs::create_dir_all(&locked)?;
        std::fs::write(locked.join(".vork-output.lock"), b"")?;
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555))?;
        let second = locked.join("b.wav");
        let failed = render(vec![first.clone(), second.clone()], 0);
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755))?;
        let error = failed.map_err(|e| format!("{e:#}")).unwrap_err();
        assert!(!error.contains("remove previous output") && !error.contains("in use"), "the run began: {error}");
        assert_eq!(Clip::load(&first)?.frames.len(), 1600, "the first destination was replaced by the new WAV");
        assert!(!second.exists(), "the second WAV was not written");
        assert!(!meta.exists(), "the replaced WAV carries no completion record from the earlier render");
        assert!(!crate::shot::sidecar(&second, "meta.json").exists());
        // A destination that cannot be replaced fails before any data changes, and also
        // leaves no completion record.
        render(vec![first.clone()], 2)?;
        assert!(meta.exists());
        let blocked = base.join("out").join("c.wav");
        std::fs::create_dir_all(blocked.join("kept"))?;
        assert!(render(vec![first.clone(), blocked.clone()], 0).is_err());
        assert!(!meta.exists());
        // One destination named twice is one namespace, as for pictures.
        let error = render(vec![first.clone(), first.clone()], 2).map_err(|e| format!("{e:#}")).unwrap_err();
        assert!(error.contains("overlapping output namespaces"), "{error}");
        // A retry that can finish publishes again.
        render(vec![first.clone()], 2)?;
        assert!(first.exists() && meta.exists() && timings.exists());
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&std::fs::read(&meta)?)?["kind"], "audio");
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    fn canvas_audio_readback_uses_the_audible_subtree() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-ask-sound-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        vork_audio::write_wav(&base.join("constant.wav"), &vec![[0.5; 2]; 1024])?;
        let doc = scene(
            serde_json::json!([{"name":"bus","audio":{"gain":0.5},"children":[{"audio":{"file":"constant.wav"}}]}]),
        );
        let answer = crate::ask::scene(
            doc,
            &base,
            &serde_json::json!([
                {"audio":"bus","read":"loudness","window":[0,0.01]},
                {"audio":"bus","window":[0,0.0000625]},
                {"audio":"bus","read":"spectrum","window":[0,0.021333333333333333]}
            ]),
        )?;
        assert_eq!(answer[0]["rms"], 0.25);
        assert_eq!(answer[0]["peak"], 0.25);
        assert_eq!(answer[1]["samples"].as_array().unwrap().len(), 3);
        assert!((answer[2]["magnitudes"][0].as_f64().unwrap() - 0.25).abs() < 1e-9);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    fn picture_only_clocks_stay_out_of_the_sound_plan() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-mix-plan-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let frames: Vec<_> = (0..4800).map(|i| [i as f32 / 4800.; 2]).collect();
        vork_audio::write_wav(&base.join("long.wav"), &frames)?;
        let mut nodes: Vec<_> = (0..8)
            .map(|i| serde_json::json!({"name":format!("shape{i}"),"time":{"start":0.001},"children":[{"active":[0,0.02]}]}))
            .collect();
        nodes.push(serde_json::json!({"name":"song","audio":{"file":"long.wav"}}));
        let mix = Mix::new(&scene(serde_json::Value::Array(nodes)), &base)?;
        assert_eq!(mix.sound.nodes(), [16]);
        assert!(mix.sound.spans());
        assert_eq!(mix.render(0., 0.01, None)?, frames[..480]);
        // Only what the window reads was decoded, and a later window continues from there.
        let decoded = |at: usize| Ok::<_, anyhow::Error>(load(&base.join("long.wav"))?.prefix(at)?.frames.len());
        assert!((480..600).contains(&decoded(1)?), "{}", decoded(1)?);
        assert_eq!(mix.render(0.05, 0.06, None)?, frames[2400..2880]);
        assert!((2880..3000).contains(&decoded(1)?), "{}", decoded(1)?);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    fn blocks_read_what_an_exact_evaluation_reads_at_every_sample() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-mix-blocks-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let frames: Vec<_> = (0..480)
            .map(|i| [(i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()])
            .collect();
        vork_audio::write_wav(&base.join("tone.wav"), &frames)?;
        // A window, a loop, a reversal, a resampled speed, a keyed existence, and gain keyed three
        // ways: a scene ramp, a scene step and a looping layer on the bus's own clock.
        let doc = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[
                {"name":"bus","audio":{"gain":0.5},"time":{"start":0.0003,"duration":0.0211,"loop":true},
                 "animation":[{"loop":true,"tracks":[{"path":["children","looped","audio","gain"],"keys":[[0,0.25],[0.0031,1]]}]}],
                 "children":[
                    {"name":"looped","time":{"duration":0.0041,"loop":true},"audio":{"file":"tone.wav"}},
                    {"name":"late","active":[0.0101,0.0173],"time":{"speed":-1},"audio":{"file":"tone.wav","gain":2}}]},
                {"name":"ramped","time":{"start":0.0007,"speed":0.75},"audio":{"file":"tone.wav"}},
                {"name":"stepped","active":[0.002,0.027],"audio":{"file":"tone.wav"}},
                {"name":"flag","audio":{"file":"tone.wav"}}],
            "animation":[{"tracks":[
                {"path":["nodes","ramped","audio","gain"],"keys":[[0,0],[0.02,1]]},
                {"path":["nodes","stepped","audio","gain"],"keys":[[0,1],[0.01251,0.25]],"interpolation":"STEP"},
                {"path":["nodes","flag","active"],"keys":[[0,false],[0.0052,true],[0.0199,false]],"interpolation":"STEP"}]}]
        }))?)?;
        let mix = Mix::new(&doc, &base)?;
        assert!(mix.sound.spans());
        let whole = mix.render(0., 0.03, None)?;
        assert_eq!(whole.len(), 1440);
        assert!(whole.iter().any(|frame| frame[0] != 0.));
        for (sample, frame) in whole.iter().enumerate() {
            let one = mix.render(
                sample as f64 / SAMPLE_RATE as f64,
                (sample + 1) as f64 / SAMPLE_RATE as f64,
                None,
            )?;
            assert_eq!(one.len(), 1);
            assert_eq!(one[0].map(f32::to_bits), frame.map(f32::to_bits), "sample {sample}");
        }
        // Keyed speeds: a ramp from a standing start with a clocked, windowed child and a gain
        // layer on its own clock; a step speed that stands for a while; a ramp read in scene
        // seconds under a retimed parent; a ramp keyed by the parent. Blocks carry each as the
        // polynomial through exact evaluations; one-sample renders are those evaluations.
        let keyed = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[
                {"name":"bent","time":{"duration":0.02},"audio":{"file":"tone.wav"},
                 "animation":[{"tracks":[{"path":["audio","gain"],"keys":[[0,0.5],[0.005,1]]}]}],
                 "children":[{"name":"under","active":[0.002,0.008],"time":{"start":0.001,"speed":0.5},"audio":{"file":"tone.wav"}}]},
                {"name":"stepper","time":{"duration":0.02},"audio":{"file":"tone.wav"}},
                {"name":"shot","time":{"speed":2,"duration":0.05},"children":[
                    {"name":"inner","time":{"duration":0.02},"audio":{"file":"tone.wav"}}]},
                {"name":"owner","time":{"start":0.0005,"duration":0.03},
                 "animation":[{"tracks":[
                    {"path":["children","kid","time","speed"],"keys":[[0,1],[0.01,0.25]]},
                    {"path":["children","kid","audio","gain"],"keys":[[0,0],[0.01,1]]}]}],
                 "children":[{"name":"kid","time":{"duration":0.02},"audio":{"file":"tone.wav"}}]}],
            "animation":[{"tracks":[
                {"path":["nodes","bent","time","speed"],"keys":[[0,0],[0.004,1.5],[0.011,0.5]]},
                {"path":["nodes","stepper","time","speed"],"keys":[[0,0.5],[0.003,0],[0.006,2]],"interpolation":"STEP"},
                {"path":["nodes","inner","time","speed"],"keys":[[0,1],[0.02,3]]}]}]
        }))?)?;
        let mix = Mix::new(&keyed, &base)?;
        assert!(mix.sound.spans());
        let whole = mix.render(0., 0.02, None)?;
        assert_eq!(whole.len(), 960);
        assert!(whole.iter().any(|frame| frame[0] != 0.));
        for (sample, frame) in whole.iter().enumerate() {
            let one = mix.render(
                sample as f64 / SAMPLE_RATE as f64,
                (sample + 1) as f64 / SAMPLE_RATE as f64,
                None,
            )?;
            assert_eq!(one[0].map(f32::to_bits), frame.map(f32::to_bits), "keyed sample {sample}");
        }
        assert_eq!(mix.render(0.002, 0.004, None)?, whole[96..192]);
        // A clock that stands for an instant silences what is below it at that instant, even a
        // child whose own rate is not zero there: the ramp starts from rest at sample 0.
        let rest = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"ramp","time":{"duration":0.05},
                "animation":[{"tracks":[{"path":["children","rider","time","speed"],"keys":[[0,1],[0.01,1]]}]}],
                "children":[{"name":"rider","time":{"duration":0.05},"audio":{"file":"tone.wav"}}]}],
            "animation":[{"tracks":[{"path":["nodes","ramp","time","speed"],"keys":[[0,0],[0.01,1]]}]}]
        }))?)?;
        let mix = Mix::new(&rest, &base)?;
        let whole = mix.render(0., 0.01, None)?;
        assert_eq!(whole[0], [0.; 2]);
        assert!(whole[1][1] != 0.);
        assert_eq!(mix.render(0., 1. / SAMPLE_RATE as f64, None)?, whole[..1]);
        for sample in [1, 2, 3, 100, 479] {
            let one = mix.render(sample as f64 / SAMPLE_RATE as f64, (sample + 1) as f64 / SAMPLE_RATE as f64, None)?;
            assert_eq!(one[0].map(f32::to_bits), whole[sample].map(f32::to_bits), "rider sample {sample}");
        }
        // Three ramps deep: a degree-eight clock, still carried in blocks where it fits.
        let deep = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"p","time":{"duration":1},
                "animation":[{"tracks":[{"path":["children","q","time","speed"],"keys":[[0,0.5],[1,2]]}]}],
                "children":[{"name":"q","time":{"duration":1},
                    "animation":[{"tracks":[{"path":["children","r","time","speed"],"keys":[[0,0.5],[1,2]]}]}],
                    "children":[{"name":"r","time":{"duration":1},"audio":{"file":"tone.wav"}}]}]}],
            "animation":[{"tracks":[{"path":["nodes","p","time","speed"],"keys":[[0,0.5],[1,2]]}]}]
        }))?)?;
        let mix = Mix::new(&deep, &base)?;
        let whole = mix.render(0., 0.05, None)?;
        assert!(whole.iter().any(|frame| frame[0] != 0.));
        for sample in (0..2400).step_by(7) {
            let one = mix.render(sample as f64 / SAMPLE_RATE as f64, (sample + 1) as f64 / SAMPLE_RATE as f64, None)?;
            assert_eq!(one[0].map(f32::to_bits), whole[sample].map(f32::to_bits), "deep sample {sample}");
        }
        // A cubic speed is read exactly at every sample; a window of it matches the whole.
        let cubic = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"bent","time":{"duration":0.01},"audio":{"file":"tone.wav"}}],
            "animation":[{"tracks":[{"path":["nodes","bent","time","speed"],"keys":[[0,1,1,1],[0.01,1,1,1]],"interpolation":"CUBICSPLINE"}]}]
        }))?)?;
        let mix = Mix::new(&cubic, &base)?;
        // More one-sample blocks than one batch schedules.
        assert!(mix.render(0., 0.1, None)?.len() > BATCH);
        let whole = mix.render(0., 0.01, None)?;
        assert!(whole.iter().any(|frame| frame[0] != 0.));
        assert_eq!(mix.render(0.002, 0.004, None)?, whole[96..192]);
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
    #[test]
    fn nested_clocks_gain_active_reverse_and_range() -> Result<()> {
        let base = std::env::temp_dir().join(format!("vork-mix-{}", std::process::id()));
        std::fs::create_dir_all(&base)?;
        let frames: Vec<_> = (0..480).map(|i| [i as f32 / 480.; 2]).collect();
        vork_audio::write_wav(&base.join("ramp.wav"), &frames)?;
        let doc = scene(serde_json::json!([
            {"name":"bus","audio":{"gain":0.5},"children":[
                {"name":"source","visible":false,"active":[0.01,0.03],"time":{"speed":-1},"audio":{"file":"ramp.wav","gain":2}}
            ]}
        ]));
        let mix = Mix::new(&doc, &base)?;
        let all = mix.render(0., 0.03, None)?;
        assert!(all[..480].iter().all(|x| *x == [0.; 2]));
        assert!((all[480][0] - frames[479][0]).abs() < 1e-6);
        assert!((all[481][0] - frames[478][0]).abs() < 1e-6);
        assert!(all[960..].iter().all(|x| *x == [0.; 2]));
        assert_eq!(mix.render(0.012, 0.018, None)?, all[576..864]);
        let forward = scene(
            serde_json::json!([{"time":{"duration":0.005},"children":[{"audio":{"file":"ramp.wav"}}]}]),
        );
        let forward = Mix::new(&forward, &base)?.render(0., 0.01, None)?;
        assert!(forward[240..].iter().all(|x| *x == [0.; 2]));
        let gain = scene(
            serde_json::json!([{"name":"source","audio":{"file":"ramp.wav"},"animation":[{"tracks":[{"path":["audio","gain"],"keys":[[0,0],[0.01,1]]}]}]}]),
        );
        let gain = Mix::new(&gain, &base)?.render(0., 0.01, None)?;
        assert!((gain[240][0] - 0.25).abs() < 1e-6);
        let held = scene(
            serde_json::json!([{"time":{"speed":0},"children":[{"audio":{"file":"ramp.wav"}}]}]),
        );
        assert!(
            Mix::new(&held, &base)?
                .render(0., 0.01, None)?
                .iter()
                .all(|x| *x == [0.; 2])
        );
        std::fs::remove_dir_all(base)?;
        Ok(())
    }
}

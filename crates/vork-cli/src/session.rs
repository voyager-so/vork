//! `vork session`: a kept-alive worker driven by JSON lines on stdin and stdout.
//!
//! One request per line, one reply per line, in order; renderer progress and
//! warnings keep going to stderr. The device, pipeline caches and the installed
//! scene live for the whole session; editing is `open` again with the whole
//! document. Same-build private: the SDK `Session` objects speak it; the framing
//! is not a public contract.
use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use vork_render::shot::{Output, Shot};
use vork_render::shot::worker::{REUSE_KEY, Source, Worker};

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    /// A scene file, or the whole document as JSON with the directory its relative files resolve against.
    Open {
        #[serde(default)]
        scene: Option<PathBuf>,
        #[serde(default)]
        document: Option<serde_json::Value>,
        #[serde(default)]
        base: Option<PathBuf>,
    },
    Render(RenderRequest),
    Ask { questions: serde_json::Value },
    Close,
}
/// Today's `render` options, as JSON.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RenderRequest {
    #[serde(deserialize_with = "one_or_many")]
    out: Vec<serde_json::Value>,
    #[serde(default)]
    at: Option<vork_time::Clock>,
    #[serde(default)]
    frame_start: Option<usize>,
    #[serde(default)]
    frame_end: Option<usize>,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    set: serde_json::Map<String, serde_json::Value>,
    /// Default colour; ["all"] selects controls and every declared AOV.
    #[serde(default)]
    channels: Vec<String>,
    #[serde(default)]
    exr_codec: Option<String>,
    #[serde(default)]
    png_depth: Option<String>,
    #[serde(default)]
    no_write: bool,
    #[serde(default)]
    gpu_only: bool,
    #[serde(default)]
    diagnostics: bool,
    /// The first picture available instead of the final one: one sample, no
    /// denoise, labelled `provisional: true`; render again for the final.
    #[serde(default)]
    provisional: bool,
}
fn one_or_many<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Vec<serde_json::Value>, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(match value { serde_json::Value::Array(values) => values, value => vec![value] })
}
fn usage(error: anyhow::Error) -> anyhow::Error { error.context(vork_render::failure::FailureKind::Usage) }
impl RenderRequest {
    fn shot(self) -> Result<Shot> {
        // Exactly what `--set KEY=VALUE` accepts: a string value is read as the
        // CLI reads it (resolution also as WxH or W,H), so the messages hold.
        let mut settings = serde_json::Map::new();
        let provisional = self.provisional;
        for (key, value) in self.set {
            anyhow::ensure!(vork_scene::CallOverrides::KEYS.contains(&key.as_str()),
                "{}", usage(anyhow::anyhow!("unknown render-call field {key}; fields: {}", vork_scene::CallOverrides::KEYS.join(", "))));
            let value = match value {
                serde_json::Value::String(text) => vork_scene::CallOverrides::parse(&format!("{key}={text}")).map_err(usage)?.1,
                value => value,
            };
            settings.insert(key, value);
        }
        if provisional {
            settings.insert("samples".into(), 1.into());
            settings.insert("denoise".into(), false.into());
        }
        let outputs = if self.channels.is_empty() { vec![Output::Color] }
            else if self.channels == ["all"] { Vec::new() }
            else { self.channels.iter().map(|s| s.parse()).collect::<Result<Vec<Output>, _>>().map_err(|e| usage(anyhow::anyhow!(e)))? };
        Ok(Shot {
            outputs,
            out: self.out.into_iter().map(|v| match v.as_str() { Some(s) => s.parse().map_err(anyhow::Error::msg), None => serde_json::from_value(v).map_err(|e| anyhow::anyhow!("output descriptor: {e}")) }).collect::<Result<_>>().map_err(usage)?,
            profile: Some(vork_scene::Profile::load(self.profile.as_deref().unwrap_or(vork_scene::DEFAULT_PROFILE)).map_err(usage)?),
            settings,
            at: self.at,
            frame_start: self.frame_start,
            frame_end: self.frame_end,
            exr_codec: match self.exr_codec { Some(codec) => codec.parse().map_err(|e: String| usage(anyhow::anyhow!(e)))?, None => Default::default() },
            png_depth: match self.png_depth { Some(depth) => depth.parse().map_err(|e: String| usage(anyhow::anyhow!(e)))?, None => Default::default() },
            no_write: self.no_write,
            gpu_only: self.gpu_only,
            diagnostics: self.diagnostics,
            provisional,
            ..Default::default()
        })
    }
}

fn failure(id: &serde_json::Value, error: &anyhow::Error) -> serde_json::Value {
    let kind = match vork_render::failure::exit_code(error) { 2 => "usage", 3 => "scene", 5 => "io", _ => "device" };
    serde_json::json!({"id": id, "ok": false, "kind": kind, "error": format!("{error:#}")})
}

fn open(worker: &mut Worker, scene: Option<PathBuf>, document: Option<serde_json::Value>, base: Option<PathBuf>) -> Result<serde_json::Value> {
    let opened = match (scene, document) {
        (Some(path), None) if base.is_none() => worker.open(Source::File(&path))?,
        (None, Some(document)) => worker.open(Source::Document { document, base: base.as_deref().unwrap_or(Path::new(".")) })?,
        _ => return Err(usage(anyhow::anyhow!(
            "open takes scene (a file path) or document (the whole scene as JSON, with base naming the directory for its relative files), not both"))),
    };
    Ok(serde_json::json!({"revision": opened.revision, REUSE_KEY: opened.reuse.name(), "changed": opened.changed}))
}

/// Serve requests until `close` or end of input, then exit without driver teardown.
pub fn run(gpu: Option<vork_render::GpuSelection>) -> Result<()> {
    if let Some(index) = gpu { vork_render::select_gpu(index); }
    let stdin = std::io::stdin().lock();
    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    let mut reply = |value: serde_json::Value| -> Result<()> {
        serde_json::to_writer(&mut stdout, &value).map_err(|e| crate::output::error(e.into()))?;
        stdout.write_all(b"\n").map_err(crate::output::error)?;
        stdout.flush().map_err(crate::output::error)
    };
    let mut worker = Worker::new();
    for line in stdin.lines() {
        let line = line.context("read session request")?;
        if line.trim().is_empty() { continue; }
        let mut value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                // An inline document is one level inside the request.
                let error = vork_scene::diagnostics::nesting_error(&error, "request")
                    .unwrap_or_else(|| anyhow::anyhow!("request is not a JSON object: {error}"));
                reply(failure(&serde_json::Value::Null, &usage(error)))?;
                continue;
            }
        };
        let id = value.as_object_mut().and_then(|object| object.remove("id")).unwrap_or(serde_json::Value::Null);
        let request: Request = match serde_json::from_value(value) {
            Ok(request) => request,
            Err(error) => { reply(failure(&id, &usage(anyhow::anyhow!("{error}; requests are open {{scene | document, base}}, render {{out, at, profile, set, channels, provisional}}, ask {{questions}} and close"))))?; continue; }
        };
        let result = match request {
            Request::Ask { questions } => worker.ask(&questions).map_err(crate::ask::single_line).map(|answers|serde_json::json!({
                "id":id,"ok":true,"revision":worker.revision(),"answers":answers,
            })),
            Request::Open { scene, document, base } => open(&mut worker, scene, document, base).map(|mut opened| {
                opened["id"] = id.clone();
                opened["ok"] = true.into();
                opened
            }),
            Request::Render(request) => { let provisional = request.provisional; request.shot().and_then(|shot| worker.render(shot)).map(|(report, reuse)| serde_json::json!({
                "id": id, "ok": true, "revision": worker.revision(), REUSE_KEY: reuse.name(), "installs": worker.installs(),
                "provisional": provisional,
                "device_reused": report["device_reused"], "install_reused": report["install_reused"],
                "result": {"warnings": report["warnings"], "exposure": report["exposure"], "sidecars": report["sidecars"], "frames": report["frames"]},
                "timing": {"wall_ms": report["wall_ms"], "device_ms": report["device_ms"], "install_ms": report["install_ms"],
                    "ready_ms": report["ready_ms"], "first_frame_ms": report["first_frame_ms"], "loop_wall_ms": report["loop_wall_ms"]},
                "report": if report["render_settings"].is_null() { serde_json::Value::Null } else { serde_json::json!({"render_settings": report["render_settings"]}) },
            })) }
            Request::Close => {
                worker.close();
                reply(serde_json::json!({"id": id, "ok": true}))?;
                break;
            }
        };
        match result {
            Ok(value) => reply(value)?,
            Err(error) => reply(failure(&id, &error))?,
        }
    }
    // Every output is delivered and every reply flushed; skip driver teardown like one-shot renders.
    drop(stdout);
    std::io::stderr().flush()?;
    worker.finish_pipeline_save();
    vork_render::shot::exit_process(0)
}

//! CPU read-back; loading and sampling never create a rendering device.
mod evaluation;
mod fields;
mod info;
mod spatial;

use anyhow::{Context, Result, bail, ensure};
use evaluation::{CpuScene, Pose};
use serde_json::{Value, json};
use std::path::Path;

struct Question {
    kind: String,
    name: String,
    read: Option<String>,
    times: Vec<vork_time::Clock>,
    many: bool,
    instance: Option<usize>,
    camera: Option<String>,
    range: Option<[usize; 2]>,
    window: Option<[f64; 2]>,
    file: Option<String>,
}
impl Question {
    fn parse(value: &Value) -> Result<Self> {
        let object = value
            .as_object()
            .context("ask: each question must be an address object")?;
        ensure!(
            !object.contains_key("file") || object.contains_key("audio"),
            "not built yet: writing an answer to file"
        );
        let kinds = [
            "node", "field", "buffer", "image", "geometry", "grid", "text", "path", "audio",
            "pixel", "ray",
        ];
        let found: Vec<_> = kinds
            .into_iter()
            .filter(|kind| object.contains_key(*kind))
            .collect();
        ensure!(
            found.len() == 1,
            "ask: each question needs exactly one address; see vork help ask"
        );
        let kind = found[0];
        ensure!(
            matches!(kind, "node" | "field" | "buffer" | "image" | "audio"),
            "not built yet: {kind} answers"
        );
        ensure!(
            kind != "node" || object[kind].as_str() != Some(""),
            "ask: give the node a name before asking for it"
        );
        let name = object[kind]
            .as_str()
            .filter(|n| !n.is_empty())
            .context("ask: address names must be nonempty strings")?
            .to_owned();
        let read = object
            .get("read")
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .context("ask: read must be a property name")
            })
            .transpose()?;
        match (kind, read.as_deref()) {
            ("node", Some("coverage" | "luminance")) => {
                bail!("not built yet: node {}", read.as_deref().unwrap())
            }
            ("node", Some("transform" | "bounds" | "screen"))
            | ("field", None)
            | ("buffer", None | Some("count"))
            | ("image", Some("info"))
            | ("audio", None | Some("loudness" | "spectrum")) => {}
            ("image", None) => bail!("not built yet: image pixels"),
            _ => bail!("ask: {kind} has no requested property; see vork help ask"),
        }
        let mut allowed = vec![kind, "read", "at"];
        let file = if kind == "audio" {
            allowed.push("file");
            object
                .get("file")
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .context("audio: file must be a path")
                })
                .transpose()?
        } else {
            None
        };
        let window = if kind == "audio" {
            allowed.push("window");
            let window: [f64; 2] = serde_json::from_value(
                object
                    .get("window")
                    .cloned()
                    .context("audio: supply window: [start,end) in scene seconds")?,
            )?;
            ensure!(
                window.iter().all(|v| v.is_finite()) && window[1] > window[0],
                "audio: window needs finite start < end"
            );
            Some(window)
        } else {
            None
        };
        if kind == "node" {
            allowed.push("instance");
        }
        if read.as_deref() == Some("screen") {
            allowed.push("camera");
        }
        if kind == "buffer" && read.is_none() {
            allowed.push("range");
        }
        for key in object.keys() {
            ensure!(
                allowed.contains(&key.as_str()),
                "ask: {key:?} is not valid for this address; see vork help ask"
            );
        }
        let many = object.get("at").is_some_and(Value::is_array);
        let times = match object.get("at") {
            Some(Value::Array(times)) => times
                .iter()
                .map(|v| serde_json::from_value(v.clone()))
                .collect::<Result<Vec<_>, _>>()?,
            Some(time) => vec![serde_json::from_value(time.clone())?],
            None => vec![vork_time::Clock::from(0.)],
        };
        ensure!(!times.is_empty(), "ask: at must contain a scene time");
        for time in &times {
            vork_time::Request::from(time).numeric()?;
        }
        let instance = object
            .get("instance")
            .map(|v| {
                v.as_u64()
                    .and_then(|i| usize::try_from(i).ok())
                    .context("ask: instance must be a nonnegative integer")
            })
            .transpose()?;
        let camera = object
            .get("camera")
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .context("ask: camera must name a node")
            })
            .transpose()?;
        let range = object
            .get("range")
            .map(|v| -> Result<_> {
                let a = v
                    .as_array()
                    .filter(|a| a.len() == 2)
                    .context("ask: range must be [first, count]")?;
                let first = a[0]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .context("ask: range first must be a nonnegative integer")?;
                let count = a[1]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .context("ask: range count must be a nonnegative integer")?;
                first
                    .checked_add(count)
                    .context("ask: range exceeds the buffer index range")?;
                Ok([first, count])
            })
            .transpose()?;
        Ok(Self {
            kind: kind.into(),
            name,
            read,
            times,
            many,
            instance,
            camera,
            range,
            window,
            file,
        })
    }
}

fn questions(value: &Value) -> Result<Vec<Question>> {
    value
        .as_array()
        .context("ask: questions must be a JSON list")?
        .iter()
        .map(Question::parse)
        .collect()
}

/// Answer a complete ordered batch; a refusal produces no partial answer list.
pub fn run(path: &Path, value: &Value) -> Result<Value> {
    // Questions are answered on the CPU from the document and its files: no device is opened.
    run_on_cpu(path, value).map_err(crate::failure::scene)
}
fn run_on_cpu(path: &Path, value: &Value) -> Result<Value> {
    let _warnings = crate::warnings::Collector::quiet().enter();
    let questions = questions(value)?;
    if questions.iter().all(|q| q.kind == "audio") {
        let bytes = vork_std::load_scene(path)?;
        let scene =
            vork_scene::Scene::from_json_for_export(&bytes, &std::fs::read(path)?, Some(path))?;
        return audio_answers(
            &crate::audio::Mix::new(&scene, path.parent().unwrap_or(Path::new(".")))?,
            questions,
        );
    }
    answer(&CpuScene::load(path)?, questions)
}

/// Session calls consume the last successfully opened document, never reread its JSON file.
pub fn scene(scene: vork_scene::Scene, base: &Path, value: &Value) -> Result<Value> {
    let _warnings = crate::warnings::Collector::quiet().enter();
    (|| {
        let questions = questions(value)?;
        if questions.iter().all(|q| q.kind == "audio") {
            return audio_answers(&crate::audio::Mix::new(&scene, base)?, questions);
        }
        answer(&CpuScene::new(scene, base)?, questions)
    })().map_err(crate::failure::scene)
}

/// A session retains decoded CPU inputs and camera compilation until its next open.
pub(crate) struct Reader(ReaderKind);
enum ReaderKind {
    Picture(CpuScene),
    Sound(crate::audio::Mix),
}
impl Reader {
    pub fn new(scene: vork_scene::Scene, base: &Path) -> Result<Self> {
        let _warnings = crate::warnings::Collector::quiet().enter();
        if scene.is_canvas() {
            Ok(Self(ReaderKind::Sound(crate::audio::Mix::new(
                &scene, base,
            )?)))
        } else {
            Ok(Self(ReaderKind::Picture(CpuScene::new(scene, base)?)))
        }
    }
    pub fn ask(&self, value: &Value) -> Result<Value> {
        let _warnings = crate::warnings::Collector::quiet().enter();
        match &self.0 {
            ReaderKind::Picture(scene) => answer(scene, questions(value)?),
            ReaderKind::Sound(mix) => audio_answers(mix, questions(value)?),
        }
    }
}

fn answer(scene: &CpuScene, questions: Vec<Question>) -> Result<Value> {
    let audio = questions
        .iter()
        .any(|q| q.kind == "audio")
        .then(|| crate::audio::Mix::new(&scene.scene, &scene.base))
        .transpose()?;
    let mut poses: Vec<(vork_time::Clock, Pose)> = Vec::new();
    let mut result = Vec::with_capacity(questions.len());
    for question in questions {
        if question.kind == "audio" {
            result.push(audio_answer(audio.as_ref().unwrap(), &question)?);
            continue;
        }
        let mut per_time = Vec::with_capacity(question.times.len());
        for time in &question.times {
            let request = vork_time::Request::from(time);
            let pose_index =
                if let Some(i) = poses.iter().position(|(t, _)| t.exact() == time.exact()) {
                    i
                } else {
                    poses.push((time.clone(), scene.sample(&request)?));
                    poses.len() - 1
                };
            let pose = &poses[pose_index].1;
            let value = match question.kind.as_str() {
                "node" => {
                    let node = scene.node(&question.name)?;
                    match question.read.as_deref().unwrap() {
                        "transform" => {
                            let placements = if question.instance.is_none() {
                                vec![(glam::Mat4::IDENTITY, 0.)]
                            } else {
                                scene.placements(node, pose, question.instance)?
                            };
                            let values=placements.into_iter().map(|(local,offset)| -> Result<Value> {
                                let shifted;
                                let pose=if offset==0. {pose} else {shifted=scene.sample_offset(&request,offset)?; &shifted};
                                let matrix=pose.worlds[node]*local;
                                ensure!(matrix.is_finite(),"ask: transform is not finite; correct the node or its animation");
                                Ok(json!(matrix.to_cols_array()))
                            }).collect::<Result<Vec<_>>>()?;
                            values.into_iter().next().unwrap_or(Value::Null)
                        }
                        "bounds" => scene
                            .bounds(node, pose, &request, question.instance)?
                            .map_or(Value::Null, |(lo, hi)| {
                                json!([lo.to_array(), hi.to_array()])
                            }),
                        "screen" => screen(
                            &scene,
                            node,
                            pose,
                            &request,
                            question.instance,
                            question.camera.as_deref(),
                        )?,
                        _ => unreachable!(),
                    }
                }
                "buffer" => {
                    scene.buffer(&question.name, question.read.is_some(), question.range)?
                }
                "image" => scene.image_info(&question.name, pose, &request)?,
                "field" => fields::read(scene, pose, &request, &question.name)?,
                _ => unreachable!(),
            };
            per_time.push(value);
        }
        result.push(if question.many {
            json!(per_time)
        } else {
            per_time.into_iter().next().unwrap()
        });
    }
    Ok(json!(result))
}

fn screen(
    scene: &CpuScene,
    node: usize,
    pose: &Pose,
    request: &vork_time::Request,
    instance: Option<usize>,
    camera: Option<&str>,
) -> Result<Value> {
    let Some(bounds) = scene.bounds(node, pose, request, instance)? else {
        return Ok(Value::Null);
    };
    let (view, projection, calibration) = scene.camera(pose, camera, request.numeric()?)?;
    let corners = spatial::corners(bounds).map(|p| view.transform_point3(p));
    let near = corners.iter().map(|p| -p.z).fold(f32::INFINITY, f32::min);
    let far = corners
        .iter()
        .map(|p| -p.z)
        .fold(f32::NEG_INFINITY, f32::max);
    if far < 0. {
        return Ok(Value::Null);
    }
    // Clip perspective bounds crossing either the camera or authored near plane.
    // Wholly forward boxes entirely outside the render clip range retain their
    // geometric extent; orthographic projection is linear and needs no clipping.
    let clip_start = calibration[1][0];
    let clip = (calibration[1][2] == 0.
        && (near <= 0. || (near <= clip_start && far >= clip_start)))
        .then_some(clip_start);
    let mut points = corners.to_vec();
    if let Some(depth) = clip {
        let plane = -depth;
        points.retain(|p| p.z <= plane);
        for i in 0..8 {
            for bit in [1, 2, 4] {
                let j = i ^ bit;
                if j <= i {
                    continue;
                }
                let a = corners[i];
                let b = corners[j];
                if (a.z < plane && b.z > plane) || (b.z < plane && a.z > plane) {
                    let mut intersection = a + (b - a) * ((plane - a.z) / (b.z - a.z));
                    intersection.z = plane;
                    points.push(intersection);
                }
            }
        }
        ensure!(
            !points.is_empty(),
            "ask: crossing bounds do not reach the camera near plane; decrease clip_start"
        );
    }
    let [width, height] = pose.render.resolution.map(|v| v as f64);
    let mut box_ = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for point in points {
        let p = projection * point.extend(1.);
        let x = (f64::from(p.x) / f64::from(p.w) + 1.) * width * 0.5;
        let y = (1. - f64::from(p.y) / f64::from(p.w)) * height * 0.5;
        ensure!(
            x.is_finite() && y.is_finite(),
            "ask: screen projection is not finite; move the node clear of the camera plane"
        );
        // The standard camera's physical aperture lies inside this ellipse;
        // polygonal blades only reduce its extent. Project every possible lens
        // origin, including the zero-blur plane at the evaluated focus depth.
        let aperture = f64::from(calibration[2][0]);
        let defocus = if aperture > 0. {
            (1. / f64::from(-point.z) - 1. / f64::from(calibration[2][1])).abs() * aperture
        } else {
            0.
        };
        let ratio = f64::from(calibration[3][0]);
        let dx = defocus * f64::from(calibration[0][0]) * width * 0.5 / ratio.max(1.);
        let dy = defocus * f64::from(calibration[0][1]) * height * 0.5 * ratio.min(1.);
        box_[0] = box_[0].min(x - dx);
        box_[1] = box_[1].min(y - dy);
        box_[2] = box_[2].max(x + dx);
        box_[3] = box_[3].max(y + dy);
    }
    box_ = [
        box_[0].floor(),
        box_[1].floor(),
        box_[2].ceil(),
        box_[3].ceil(),
    ];
    ensure!(
        box_.iter()
            .all(|v| *v >= i64::MIN as f64 && *v < i64::MAX as f64),
        "ask: projected bounds exceed integer pixel coordinates; move the node farther from the camera"
    );
    let area = (box_[2] - box_[0]) * (box_[3] - box_[1]);
    let inside = if area > 0. {
        ((box_[2].min(width) - box_[0].max(0.)).max(0.)
            * (box_[3].min(height) - box_[1].max(0.)).max(0.)
            / area)
            .clamp(0., 1.)
    } else {
        0.
    };
    Ok(
        json!({"box":box_.map(|v| v as i64),"inside":inside,"depth":[near.max(clip.unwrap_or(0.)),far]}),
    )
}
fn audio_answer(mix: &crate::audio::Mix, question: &Question) -> Result<Value> {
    let mut per_time = Vec::with_capacity(question.times.len());
    for time in &question.times {
        let root = mix.node(&question.name)?;
        let [start, end] = question.window.unwrap();
        if question.read.as_deref() == Some("spectrum") {
            ensure!(
                (end - start) * vork_audio::SAMPLE_RATE as f64 <= 65536.0000001,
                "audio: spectrum window must contain at most 65536 samples"
            );
        }
        let frames = mix.render(time.as_f64() + start, time.as_f64() + end, Some(root))?;
        let value = if question.read.is_none() {
            if let Some(file) = &question.file {
                ensure!(
                    question.times.len() == 1,
                    "audio: writing requires one time per file"
                );
                ensure!(
                    Path::new(file)
                        .extension()
                        .is_some_and(|x| x.eq_ignore_ascii_case("wav")),
                    "audio: samples write to a .wav file"
                );
                vork_audio::write_wav(Path::new(file), &frames)?;
                json!({"file": file})
            } else {
                json!({"sample_rate": vork_audio::SAMPLE_RATE, "channels": ["L", "R"], "samples": frames})
            }
        } else if question.read.as_deref() == Some("loudness") {
            let level = vork_audio::loudness(&frames);
            json!({"rms": level.rms, "peak": level.peak,
                "rms_dbfs": (level.rms > 0.).then(|| 20. * level.rms.log10()),
                "peak_dbfs": (level.peak > 0.).then(|| 20. * level.peak.log10())})
        } else {
            json!({"magnitudes": vork_audio::spectrum(&frames)?, "frequency_step": vork_audio::SAMPLE_RATE as f64 / frames.len() as f64})
        };
        if question.read.is_some() {
            if let Some(file) = &question.file {
                ensure!(
                    question.times.len() == 1,
                    "audio: writing requires one time per file"
                );
                std::fs::write(file, serde_json::to_vec(&value)?)?;
                per_time.push(json!({"file": file}));
            } else {
                per_time.push(value);
            }
        } else {
            per_time.push(value);
        }
    }
    Ok(if question.many {
        json!(per_time)
    } else {
        per_time.remove(0)
    })
}
fn audio_answers(mix: &crate::audio::Mix, questions: Vec<Question>) -> Result<Value> {
    ensure!(
        questions.iter().all(|q| q.kind == "audio"),
        "ask: picture answers about a 2D canvas are not built yet"
    );
    Ok(Value::Array(
        questions
            .iter()
            .map(|q| audio_answer(mix, q))
            .collect::<Result<Vec<_>>>()?,
    ))
}

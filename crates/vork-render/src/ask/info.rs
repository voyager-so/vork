use super::evaluation::{CpuScene, Pose};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use vork_scene::v2::BufferType;

impl CpuScene {
    pub fn buffer(&self, name: &str, count: bool, range: Option<[usize; 2]>) -> Result<Value> {
        ensure!(
            self.scene.buffers.contains_key(name),
            "ask: unknown buffer {name:?}"
        );
        ensure!(
            !crate::compute::writes_buffer(&self.scene, name),
            "not built yet: this answer needs the GPU program writing buffer {name:?}"
        );
        let buffer = self
            .buffers
            .data
            .get(name)
            .context("not built yet: grid buffer elements")?;
        let capacity = buffer.count();
        if count {
            return Ok(json!({"count":capacity,"capacity":capacity}));
        }
        let [first, count] = range.unwrap_or([0, capacity]);
        ensure!(
            first.checked_add(count).is_some_and(|end| end <= capacity),
            "ask: range exceeds buffer {name:?}; choose a slice within its count"
        );
        let values = buffer
            .words
            .chunks_exact(buffer.width)
            .skip(first)
            .take(count)
            .map(|words| {
                let words = words
                    .iter()
                    .map(|&word| match buffer.element_type {
                        BufferType::I32 => json!(word as i32),
                        BufferType::U32 | BufferType::Uvec4 => json!(word),
                        _ => json!(f32::from_bits(word)),
                    })
                    .collect::<Vec<_>>();
                if buffer.width == 1 {
                    words.into_iter().next().unwrap()
                } else {
                    json!(words)
                }
            })
            .collect::<Vec<_>>();
        Ok(json!(values))
    }

    pub fn image_time(
        &self,
        name: &str,
        pose: &Pose,
        request: &vork_time::Request,
    ) -> Result<vork_time::exact::Germ> {
        let image = self
            .scene
            .images
            .get(name)
            .with_context(|| format!("ask: unknown image {name:?}"))?;
        let time = request.exact()?;
        let base = image.time.as_ref().map_or_else(
            || time.clone(),
            |t| vork_time::exact::Germ::point(t.exact()),
        );
        self.animation
            .image_time_exact(&pose.samples, &time, name, base)
    }

    pub fn image_info(
        &self,
        name: &str,
        pose: &Pose,
        request: &vork_time::Request,
    ) -> Result<Value> {
        let image = self
            .scene
            .images
            .get(name)
            .with_context(|| format!("ask: unknown image {name:?}"))?;
        if let Some(render) = &image.render {
            return Ok(json!({"width":render.resolution[0],"height":render.resolution[1]}));
        }
        let mut path = self
            .base
            .join(image.file.as_ref().context("ask: image has no file")?);
        if crate::footage::is_sequence(&path) {
            let source = crate::footage::Source::open(&path)?;
            let index = source
                .frame_index_exact(&self.image_time(name, pose, request)?, pose.render.fps)?;
            path = source.files()[index].clone();
        } else if crate::footage::is_media(&path) {
            let probe=std::process::Command::new("ffprobe").args(["-v","error","-select_streams","v:0","-show_entries",
                "stream=width,height,duration,avg_frame_rate:stream_side_data=rotation:format=duration","-of","json"])
                .arg(&path).output().context("ask: movie info requires ffprobe on PATH")?;
            ensure!(
                probe.status.success(),
                "ask: cannot read movie info for {}; restore the movie or check it with ffprobe",
                path.display()
            );
            let value: Value = serde_json::from_slice(&probe.stdout)?;
            let stream = &value["streams"][0];
            let mut width = stream["width"]
                .as_u64()
                .context("ask: movie has no width")?;
            let mut height = stream["height"]
                .as_u64()
                .context("ask: movie has no height")?;
            if stream["side_data_list"]
                .as_array()
                .and_then(|a| a.iter().find_map(|v| v["rotation"].as_i64()))
                .is_some_and(|r| r.rem_euclid(180) == 90)
            {
                std::mem::swap(&mut width, &mut height);
            }
            let duration = stream["duration"]
                .as_str()
                .filter(|v| *v != "N/A")
                .or_else(|| value["format"]["duration"].as_str())
                .context("ask: movie has no duration")?
                .parse::<f64>()?;
            let fps = stream["avg_frame_rate"]
                .as_str()
                .context("ask: movie has no frame rate")?
                .parse::<vork_time::Time>()?;
            ensure!(
                duration.is_finite() && duration >= 0. && fps > vork_time::Time::ZERO,
                "ask: movie timing is invalid; check it with ffprobe"
            );
            return Ok(json!({"width":width,"height":height,"duration":duration,"fps":fps}));
        }
        let (width, height) = image::image_dimensions(&path).with_context(|| {
            format!(
                "ask: read image info {}; restore a readable image",
                path.display()
            )
        })?;
        Ok(json!({"width":width,"height":height}))
    }
}

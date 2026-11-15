//! Display-frame media selection, adapted from an earlier portable
//! decoder: timestamp/keyframe lookup, sequential raw-frame pipe, indexed backwards
//! seeks. No clip is decoded into an unbounded in-memory frame collection.
use anyhow::{Context, Result, ensure};
use vork_time::exact::{Germ, Rational, decimal};
use std::{io::Read, path::{Path, PathBuf}, process::{Child, ChildStdout, Command, Stdio}};

pub fn is_media(path: &Path) -> bool {
    is_sequence(path) || path.extension().is_some_and(|e|
        ["mov", "mp4", "mkv"].iter().any(|v| e.eq_ignore_ascii_case(v)))
}

pub(crate) fn is_sequence(path:&Path)->bool {
    path.file_name().and_then(|n|n.to_str()).is_some_and(|name| {
        name.match_indices('%').any(|(at,_)| {
            let tail=&name[at+1..];
            tail.starts_with('d') || tail.strip_prefix('0').is_some_and(|digits| {
                let count=digits.bytes().take_while(u8::is_ascii_digit).count();
                count>0 && digits[count..].starts_with('d')
            })
        })
    })
}

pub enum Source {
    Sequence { files:Vec<PathBuf>, fps:Option<vork_time::Time> },
    Video(Video),
}

/// The media sources one document open has read, by path. The open owns it: every clock
/// plan it compiles and every image it installs takes a source from here, so a source is
/// read from disk once per open (a sequence's listing and first frame's rate, a movie's
/// probe) and all of them agree on what was read. It is dropped with the open; nothing
/// about a source is remembered for the next one.
#[derive(Clone, Default)]
pub struct Opened(std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<PathBuf, Source>>>);
impl Opened {
    fn read<T>(&self, path: &Path, with: impl FnOnce(&Source) -> Result<T>) -> Result<T> {
        // One source however its path is spelled: the directory it is in, resolved (through
        // links, `..` and a relative base), and its file name, which for a sequence is the
        // pattern and names no file. A directory that cannot be resolved keeps its spelling
        // and the read below reports what is wrong with it.
        let directory = path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let directory = std::fs::canonicalize(directory).unwrap_or_else(|_| directory.to_path_buf());
        let key = path.file_name().map_or_else(|| path.to_path_buf(), |name| directory.join(name));
        let mut sources = self.0.lock().unwrap();
        if !sources.contains_key(&key) {
            sources.insert(key.clone(), Source::open(path)?);
        }
        with(&sources[&key])
    }
    /// The source at `path` as this open read it, with decoding state of its own.
    pub(crate) fn source(&self, path: &Path) -> Result<Source> {
        self.read(path, |source| Ok(source.fresh()))
    }
    /// That source's own length ([`Source::length`]).
    pub(crate) fn length(&self, path: &Path, fps: vork_time::Time) -> Result<Rational> {
        self.read(path, |source| source.length(fps))
    }
}

impl Source {
    /// What was read from disk, without this value's decoding state.
    fn fresh(&self) -> Self {
        match self {
            Self::Sequence {files,fps} => Self::Sequence {files:files.clone(),fps:*fps},
            Self::Video(video) => Self::Video(video.fresh()),
        }
    }
    /// Files owned by this opened source, including every discovered sequence frame.
    pub fn files(&self) -> &[PathBuf] {
        match self {
            Self::Sequence {files,..} => files,
            Self::Video(video) => std::slice::from_ref(&video.path),
        }
    }

    pub fn open(path: &Path) -> Result<Self> {
        if is_sequence(path) {
            let files=sequence(path)?;
            let fps=if files[0].extension().is_some_and(|e|e.eq_ignore_ascii_case("exr")) {
                exr::meta::MetaData::read_from_file(&files[0],false)?.headers[0].own_attributes.frames_per_second
                    .map(|(n,d)|vork_time::Time::new(n.into(),d.into())).transpose().context("EXR sequence framesPerSecond must be positive")?
            } else {None};
            ensure!(fps.is_none_or(|fps|fps > vork_time::Time::ZERO),"EXR sequence framesPerSecond must be positive");
            return Ok(Self::Sequence {files,fps});
        }
        Ok(Self::Video(Video::open(path).with_context(||format!("movie {}; restore a playable movie and install ffmpeg/ffprobe on PATH",path.display()))?))
    }
    /// Natural texel size, without decoding a frame before the device's limits are known.
    pub(crate) fn dimensions(&self) -> Result<[u32; 2]> {
        match self {
            Self::Video(video) => Ok(video.size),
            Self::Sequence { files, .. } => {
                let path = &files[0];
                let (width, height, _) = crate::texture_source::header(image::ImageFormat::from_path(path)?, &std::fs::read(path)?, u32::MAX)?;
                Ok([width, height])
            }
        }
    }
    pub(crate) fn prepare_gpu(&mut self, r: &crate::Renderer) {
        if let Self::Video(video)=self {
            // Use the installed NVIDIA adapter,
            // retaining the portable decoder for unsupported streams. FFmpeg
            // owns native decoder setup and format conversion; the existing
            // ordered texture-upload path remains the publication boundary.
            let hardware = r.adapter_info.vendor==crate::constants::NVIDIA_VENDOR_ID;
            if hardware != video.hardware { video.session = None; }
            video.hardware = hardware;
        }
    }
    /// The content's own length in seconds: frames over the rate for a sequence; for a
    /// movie the stream's declared duration, else the displayed span plus one mean frame.
    pub(crate) fn length(&self,fps:vork_time::Time)->Result<Rational> {
        match self {
            Self::Sequence {files,fps:source_fps} => {
                let rate=source_fps.map(|rate|decimal(&rate.to_string())).transpose()?.unwrap_or(decimal(&fps.to_string())?);
                Ok(Rational::from_integer(files.len().into())/rate)
            }
            Self::Video(video) => {
                if let Some(duration)=&video.duration {return Ok(duration.clone());}
                let n=video.pts.len();
                let span=(Rational::from_integer(video.pts[n-1].into())-Rational::from_integer(video.pts[0].into()))*&video.timebase;
                let frame=if n>1 {&span/Rational::from_integer((n as i64-1).into())} else {decimal(&fps.to_string())?.recip()};
                Ok(span+frame)
            }
        }
    }
    pub fn timeline(&self,fps:vork_time::Time)->Result<Vec<vork_time::Clock>> {
        self.exact_timeline(fps)?.iter().map(|time|time.value().to_string().parse()).collect()
    }
    /// All identity consumers, including timeline generation, use these same
    /// boundaries. The decoder receives the selected integer, never a rounded time.
    pub(crate) fn frame_index_exact(&self, time:&Germ, fps:vork_time::Time)->Result<usize> {
        match self {
            Self::Sequence {files,fps:source_fps} => {
                let rate=source_fps.map(|rate|decimal(&rate.to_string())).transpose()?
                    .unwrap_or(decimal(&fps.to_string())?);
                Ok(time.scale(&rate).floor_clamped(0,i64::try_from(files.len()-1)?)? as usize)
            }
            Self::Video(video) => Ok(video.frame_index(time)),
        }
    }
    pub(crate) fn exact_timeline(&self, fps:vork_time::Time)->Result<Vec<Germ>> {
        Ok(match self {
            Self::Sequence {files,fps:source_fps} => {
                let rate=source_fps.map(|rate|decimal(&rate.to_string())).transpose()?.unwrap_or(decimal(&fps.to_string())?);
                (0..files.len()).map(|n|Germ::point(Rational::from_integer(n.into())/&rate)).collect()
            }
            Self::Video(video) => {
                let mut times:Vec<_>=video.pts.iter().map(|&pts|Germ::point(
                    (Rational::from_integer(pts.into())-Rational::from_integer(video.pts[0].into()))*&video.timebase)).collect();
                times.dedup();times
            }
        })
    }
    pub(crate) fn frame_exact(&mut self,time:&Germ,fps:vork_time::Time,max_side:u32)->Result<image::DynamicImage> {
        let index=self.frame_index_exact(time,fps)?;
        match self {
            Self::Sequence {files,..} => {
                let path=&files[index];
                crate::texture_source::decode(path,&std::fs::read(path)?,max_side)
                    .with_context(||format!("decode sequence frame {}; restore a valid frame or re-export the sequence",path.display()))
            }
            Self::Video(video) => video.frame_at(index,max_side),
        }
    }
    /// The frame at `time` with its colour in `upload`, the encoding the caller uploads
    /// (`texture_source::decode_color`): `explicit` is the image's declared space; without it a sequence
    /// frame's own description decides, and footage's stream tags. `light`: the image is light
    /// (an environment), and a colour that conversion carries outside the upload's gamut is limited to it
    /// (`texture_source::decode_light`).
    pub(crate) fn frame_color_exact(&mut self,time:&Germ,fps:vork_time::Time,max_side:u32,explicit:Option<vork_scene::v2::ColorSpace>,upload:vork_scene::v2::WorkingColorSpace,light:bool)->Result<image::DynamicImage> {
        let index=self.frame_index_exact(time,fps)?;
        match self {
            Self::Sequence {files,..} => {
                let path=&files[index];
                let decode=if light {crate::texture_source::decode_light} else {crate::texture_source::decode_color};
                decode(path,&std::fs::read(path)?,max_side,explicit,upload)
                    .with_context(||format!("decode sequence frame {}",path.display()))
            }
            Self::Video(video) => {
                use vork_scene::v2::{ColorSpace as C,WorkingColorSpace as S};
                // The stream's own tags (ffprobe's names), when the image declares no space. HDR transfers and
                // wide primaries are not decoded yet and refuse. Every other SDR tag (BT.709, BT.601, sRGB, or
                // none) is read as sRGB, as Blender and browsers read video; linear light is read as linear.
                let source = match explicit {
                    Some(C::NonColor) => return video.frame_at(index,max_side),
                    Some(C::Srgb) => S::Srgb, Some(C::DisplayP3) => S::DisplayP3, Some(C::LinearRec709) => S::LinearRec709,
                    Some(C::LinearRec2020) => S::LinearRec2020, Some(C::ACEScg) => S::ACEScg,
                    None => {
                        let tagged=|what:&str|crate::texture_source::declared_colour(format!("footage {} is tagged {what}",video.path.display()));
                        if matches!(video.transfer.as_str(),"smpte2084"|"arib-std-b67") {return Err(tagged("PQ or HLG, which is not decoded yet"));}
                        match video.primaries.as_str() {
                            "smpte432" => return Err(tagged("Display P3 (smpte432): \"Display P3\" reads the tagged colours")),
                            "bt2020" => return Err(tagged("Rec.2020 (bt2020), which is not read yet")),
                            "smpte431" => return Err(tagged("DCI-P3 (smpte431), which is not read yet")),
                            _ => {}
                        }
                        if video.transfer=="linear" {S::LinearRec709} else {S::Srgb}
                    }
                };
                // A decoded video frame has no negative samples of its own.
                let frame=crate::texture_source::convert_image(video.frame_at(index,max_side)?,source,upload);
                Ok(if light {crate::texture_source::limit_to_gamut(frame)} else {frame})
            }
        }
    }
    pub(crate) fn identity(&self)->Result<blake3::Hash> {
        let mut hash=blake3::Hasher::new();
        let files=match self {Self::Sequence {files,..}=>files.clone(),Self::Video(video)=>vec![video.path.clone()]};
        for file in files {let mut part=blake3::Hasher::new();part.update_reader(std::fs::File::open(file)?)?;hash.update(part.finalize().as_bytes());}
        Ok(hash.finalize())
    }
    pub fn frame(&mut self,time:impl Into<vork_time::Request>,fps:vork_time::Time,max_side:u32)->Result<image::DynamicImage> {
        self.frame_exact(&time.into().exact()?,fps,max_side)
    }

}

pub(crate) fn sequence(path: &Path) -> Result<Vec<PathBuf>> {
    let name = path.file_name().context("sequence needs a filename")?.to_str().context("sequence filename must be UTF-8")?;
    let start = name.find('%').context("sequence pattern missing %")?;
    let tail = &name[start+1..];
    let end = tail.find('d').context("sequence pattern must use %d or %04d")?;
    let digits = &tail[..end];
    ensure!(digits.is_empty() || (digits.starts_with('0') && digits[1..].chars().all(|c| c.is_ascii_digit())), "sequence pattern must use %d or %04d");
    let width = if digits.is_empty() {0} else {digits.parse::<usize>()?};
    ensure!(width <= 20, "sequence field width exceeds 20 digits");
    let prefix = &name[..start];
    let suffix = &tail[end+1..];
    ensure!(!suffix.contains('%'), "sequence must have one frame field");
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let mut frames = std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(parent)? {
        let entry = entry?;
        let file = entry.file_name();
        let Some(file) = file.to_str() else {continue};
        let Some(number) = file.strip_prefix(prefix).and_then(|n| n.strip_suffix(suffix)) else {continue};
        if number.is_empty() || !number.chars().all(|c| c.is_ascii_digit()) {continue;}
        let index = number.parse::<u64>()?;
        if format!("{index:0width$}") != number {continue;}
        frames.insert(index, entry.path());
    }
    ensure!(!frames.is_empty(), "sequence {} has no frames", path.display());
    let mut missing = Vec::new();
    let mut previous = *frames.first_key_value().unwrap().0;
    for &index in frames.keys().skip(1) {
        if index > previous+1 {
            missing.push(if index == previous+2 {format!("{}",previous+1)} else {format!("{}..{}",previous+1,index-1)});
        }
        previous = index;
    }
    let mut errors = Vec::new();
    if !missing.is_empty() {
        errors.push(anyhow::anyhow!("sequence {} missing frames: {}; restore the missing frames or correct the pattern", path.display(), missing.join(", ")));
    }
    for frame in frames.values() {
        if let Err(error) = crate::source_file(frame) {
            errors.push(anyhow::Error::new(error).context(format!("sequence frame {}: restore a readable image file or correct the pattern", frame.display())));
        }
    }
    vork_scene::v2::diagnostics::joined(errors)?;
    Ok(frames.into_values().collect())
}

struct Session { child: Child, stdout: ChildStdout, next: usize }
impl Drop for Session {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); }
}
pub struct Video {
    path: PathBuf,
    transfer: String,
    primaries: String,
    #[cfg(test)]
    times: Vec<f64>,
    keyframes: Vec<(usize,i64)>,
    pts: Vec<i64>,
    timebase: Rational,
    /// The stream's declared duration in seconds, when the container states one.
    duration: Option<Rational>,
    last_request: Option<usize>,
    #[cfg(test)]
    decoded: usize,
    size: [u32;2],
    session: Option<Session>,
    hardware: bool,
    frames: std::collections::VecDeque<(usize, image::DynamicImage)>,
}
impl Video {
    /// The probed stream again, with no decoder session and nothing decoded.
    fn fresh(&self) -> Self {
        Self {path:self.path.clone(), transfer:self.transfer.clone(), primaries:self.primaries.clone(), #[cfg(test)] times:self.times.clone(), keyframes:self.keyframes.clone(), pts:self.pts.clone(),
            timebase:self.timebase.clone(), duration:self.duration.clone(), last_request:None, #[cfg(test)] decoded:0, size:self.size,
            session:None, hardware:false, frames:Default::default()}
    }
    fn frame_index(&self,time:&Germ)->usize {
        self.pts.partition_point(|&pts| {
            let delta=Rational::from_integer(pts.into())-Rational::from_integer(self.pts[0].into());
            time.compare_point(&(delta*&self.timebase)).is_ge()
        }).saturating_sub(1)
    }
    fn open(path: &Path) -> Result<Self> {
        // ffprobe and ffmpeg both apply container edit lists. Presentation times
        // are relative to the first displayed frame, never packet/decode order.
        let probe = Command::new("ffprobe").args(["-v","error","-select_streams","v:0","-show_frames","-show_streams","-show_entries",
            "frame=best_effort_timestamp,key_frame:stream=width,height,time_base,duration,color_transfer,color_primaries:stream_side_data=rotation","-of","json"])
            .arg(path).output().context("video inputs require ffprobe and ffmpeg on PATH")?;
        ensure!(probe.status.success(), "probe video {}: {}", path.display(), String::from_utf8_lossy(&probe.stderr));
        let value: serde_json::Value = serde_json::from_slice(&probe.stdout)?;
        // The frame probe selects only v:0. Inspect audio streams separately,
        // without decoding them, so silent plates do not report discarded audio.
        let audio_probe = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json"])
            .arg(path).output().context("probe movie audio streams")?;
        ensure!(audio_probe.status.success(), "probe movie audio {}: {}", path.display(), String::from_utf8_lossy(&audio_probe.stderr));
        let audio: serde_json::Value = serde_json::from_slice(&audio_probe.stdout)?;
        if audio["streams"].as_array().is_some_and(|streams| !streams.is_empty()) {
            crate::warnings::dropped("video_only", serde_json::json!({"file":path}),
                "movie image inputs decode video only. Attach audio: {file: ...} to a node to include the soundtrack.".into());
        }
        let stream = &value["streams"][0];
        let mut size = [stream["width"].as_u64().context("video width missing")? as u32,
                        stream["height"].as_u64().context("video height missing")? as u32];
        if let Some(rotation) = stream["side_data_list"].as_array().and_then(|a|a.iter().find_map(|s|s["rotation"].as_i64())) {
            if rotation.rem_euclid(180) == 90 {size.swap(0,1);}
        }
        let (numerator,denominator)=stream["time_base"].as_str().context("video time base missing")?.split_once('/').context("invalid video time base")?;
        let timebase=decimal(&format!("{numerator}/{denominator}"))?;
        ensure!(timebase > Rational::from_integer(0.into()), "video time base must be positive");
        let duration=stream["duration"].as_str().and_then(|text|decimal(text).ok()).filter(|d|d>&Rational::from_integer(0.into()));
        let pts:Vec<i64> = value["frames"].as_array().context("video has no displayed frames")?.iter()
            .map(|f|f["best_effort_timestamp"].as_i64().context("video frame has no presentation timestamp")).collect::<Result<_>>()?;
        ensure!(pts.windows(2).all(|p|p[0]<=p[1]), "video presentation timestamps must be ordered");
        #[cfg(test)]
        let tick=numerator.parse::<f64>()?/denominator.parse::<f64>()?;
        #[cfg(test)]
        let mut times: Vec<f64> = value["frames"].as_array().context("video has no displayed frames")?.iter()
            .map(|f| f["best_effort_timestamp"].as_i64().map(|pts|pts as f64*tick).context("video frame has no presentation timestamp"))
            .collect::<Result<_>>()?;
        ensure!(!pts.is_empty() && size.iter().all(|&n|n>0), "video has no displayed frames");
        // Only unique presentation times are seek anchors: a later keyframe
        // sharing a timestamp cannot identify which duplicate FFmpeg will emit.
        let keyframes=pts.iter().enumerate().filter_map(|(i,&time)| {
            (i>0 && value["frames"][i]["key_frame"].as_u64()==Some(1)
                && pts[i-1]<time && pts.get(i+1).is_none_or(|&t|t>time))
                .then_some((i,time))
        }).collect();
        #[cfg(test)] {
            let first = times[0];
            for time in &mut times {*time -= first;}
        }
        Ok(Self {path:path.into(), transfer:stream["color_transfer"].as_str().unwrap_or("").into(), primaries:stream["color_primaries"].as_str().unwrap_or("").into(), #[cfg(test)] times, pts, timebase, duration, keyframes, last_request:None, #[cfg(test)] decoded:0, size, session:None, hardware:false, frames:Default::default()})
    }
    #[cfg(test)]
    fn frame(&mut self, time: f64, max_side:u32) -> Result<image::DynamicImage> {
        ensure!(self.size.iter().all(|&n|n<=max_side), crate::gpu_memory::device_limit(format_args!("video {}", self.path.display()), format_args!("{}x{}", self.size[0], self.size[1]), format_args!("max_texture_dimension_2d={max_side}"), "resize the source video or use a GPU with larger texture limits"));
        let index = self.frame_index(&Germ::point(vork_time::Clock::from(time).exact()));
        self.frame_at(index,max_side)
    }
    fn frame_at(&mut self,index:usize,max_side:u32)->Result<image::DynamicImage> {
        ensure!(self.size.iter().all(|&n|n<=max_side), crate::gpu_memory::device_limit(format_args!("video {}", self.path.display()), format_args!("{}x{}", self.size[0], self.size[1]), format_args!("max_texture_dimension_2d={max_side}"), "resize the source video or use a GPU with larger texture limits"));
        let previous=self.last_request.replace(index);
        if let Some((_,frame)) = self.frames.iter().find(|(cached,_)|*cached==index) {return Ok(frame.clone());}
        // For a reverse walk retain the next predicted request too. The existing
        // bounded frame owner holds those two frames, not the whole decoded GOP.
        let preceding=previous.filter(|&p|p>index).map(|p|index.saturating_sub(p-index));
        let anchor=self.keyframes.iter().rev().find(|&&(i,_)|i<=index).copied();
        let start=anchor.map_or(0,|(i,_)|i);
        // A forward jump also seeks when its keyframe is beyond the live pipe.
        if self.session.as_ref().is_none_or(|s|s.next>index || s.next<start) {
            self.session = None;
            let mut command=Command::new("ffmpeg");
            command.args(["-v","error","-nostdin"]);
            if self.hardware {command.args(["-hwaccel","auto"]);}
            if let Some((_,time))=anchor {
                // Absolute container timestamps avoid assuming stream/container
                // start times agree. Round down at FFmpeg's microsecond boundary
                // so a fractional tick never discards the anchor itself.
                let micros=(Rational::from_integer(time.into())*&self.timebase*Rational::from_integer(1_000_000.into())).floor().to_integer().to_string();
                let (sign,digits)=micros.strip_prefix('-').map_or(("",micros.as_str()),|s|("-",s));
                let digits=format!("{digits:0>7}");
                let seek=format!("{sign}{}.{}",&digits[..digits.len()-6],&digits[digits.len()-6..]);
                command.args(["-seek_timestamp","1","-ss",&seek]);
            }
            let mut child = command.arg("-i").arg(&self.path)
                .args(["-map","0:v:0","-an","-sn","-fps_mode","passthrough","-f","rawvideo","-pix_fmt","rgba64le","pipe:1"])
                .stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().context("start video decoder")?;
            let stdout = child.stdout.take().unwrap();
            self.session = Some(Session {child,stdout,next:start});
        }
        // A read can consume a partial frame before failing. Retain the pipe
        // only after a complete request; every error drops and reaps it so the
        // next uncached request restarts at its keyframe with aligned pixels.
        let mut session = self.session.take().unwrap();
        let mut bytes = vec![0; self.size[0] as usize * self.size[1] as usize * 8];
        let mut result=None;
        while session.next<=index {
            session.stdout.read_exact(&mut bytes).with_context(||format!("decode video {} frame {}; restore or re-encode the movie",self.path.display(),session.next))?;
            #[cfg(test)] {self.decoded+=1;}
            if session.next==index || preceding==Some(session.next) {
                let pixels=bytes.chunks_exact(2).map(|b|u16::from_le_bytes([b[0],b[1]])).collect();
                let image=image::DynamicImage::ImageRgba16(image::ImageBuffer::from_raw(self.size[0],self.size[1],pixels).context("video frame size")?);
                if session.next==index {result=Some(image.clone());}
                self.frames.push_back((session.next,image));
                while self.frames.len()>crate::constants::FOOTAGE_DECODE_FRAMES {self.frames.pop_front();}
            }
            session.next+=1;
        }
        let image=result.context("video frame missing")?;
        self.session=Some(session);
        Ok(image)
    }
}

#[cfg(test)]
#[test]
fn movie_inputs_warn_only_when_audio_is_discarded() -> Result<()> {
    for has_audio in [false, true] {
        let path = std::env::temp_dir().join(format!("vork-audio-notice-{}-{has_audio}.mp4", std::process::id()));
        let mut command = Command::new("ffmpeg");
        command.args(["-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=16x16:d=0.1"]);
        if has_audio {
            command.args(["-f", "lavfi", "-i", "sine=frequency=440:duration=0.1", "-shortest", "-c:a", "aac"]);
        }
        let status = command.args(["-c:v", "libx264"]).arg(&path).status()?;
        ensure!(status.success(), "make a movie (audio={has_audio})");
        let collector = crate::warnings::Collector::default();
        let _scope = collector.enter();
        Source::open(&path)?;
        let warnings = collector.snapshot();
        assert_eq!(warnings.len(), usize::from(has_audio), "audio={has_audio}: {warnings:?}");
        if has_audio {
            assert_eq!(warnings[0].code, "video_only");
            assert!(warnings[0].dropped.contains("audio: {file: ...}"));
        }
        std::fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn oversized_video_reports_dimensions_limit_and_remedy_before_decode() {
        let mut video = Video {
            path: "source.mp4".into(), transfer:String::new(), primaries:String::new(), times: vec![0.], keyframes: vec![], pts: vec![0], duration: None,
            timebase: Rational::from_integer(1.into()), last_request: None, decoded: 0,
            size: [32, 16], session: None, hardware: false, frames: Default::default(),
        };
        for error in [video.frame(0., 8).unwrap_err(), video.frame_at(0, 8).unwrap_err()] {
            let message = error.to_string();
            for expected in ["source.mp4", "32x16", "max_texture_dimension_2d=8", "resize the source video", "vork help limits"] {
                assert!(message.contains(expected), "{message}");
            }
        }
        assert!(video.session.is_none());
        assert!(video.last_request.is_none());
    }

    #[test]
    fn exact_nominal_playback_and_timeline_share_boundaries()->Result<()> {
        for rate in ["24", "30000/1001", "60000/1001"] {
            let fps:vork_time::Time=rate.parse()?;
            let source=Source::Sequence {files:vec![PathBuf::new();107893],fps:None};
            let timeline=source.exact_timeline(fps)?;
            for (n,time) in timeline.iter().enumerate() {
                let nominal=vork_time::Request::Frame(n as u32,fps).exact()?;
                assert_eq!(time,&nominal);
                assert_eq!(source.frame_index_exact(&nominal,fps)?,n);
                if n>0 {
                    assert_eq!(source.frame_index_exact(&Germ::left(time.value().clone()),fps)?,n-1);
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore="requires a real GPU and pinned Slang"]
    fn sequence_compositor_samples_its_own_extent_and_updates_with_the_clock()->Result<()> {
        let r = crate::test_renderer()?;
        let root=std::env::temp_dir().join(format!("vork-footage-compositor-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let first=image::RgbaImage::from_fn(8,4,|x,_|image::Rgba(if x<4 {[255,0,0,255]} else {[0,255,0,255]}));
        first.save(root.join("1001.png"))?;
        image::RgbaImage::from_pixel(8,4,image::Rgba([0,0,255,255])).save(root.join("1002.png"))?;
        let input:crate::Scene=serde_json::from_value(serde_json::json!({"version":2,
            "composition":{"space":"3d","compositor":["comp"]},"render":{"resolution":[4,2],"fps":24,"motion_blur":{"shutter":1,"position":"end"},"settings":{"samples":4,"integrator":"path","denoiser":null}},
            "images":{"plate":{"file":"%04d.png"}},
            "programs":{"plate":{"source":"import vork; struct Inputs {Image plate;}; float4 composite(CompositeContext c,Inputs i) {return compositor_sample(i.plate,c.uv);}"}},
            "materials":{"comp":{"program":"plate","inputs":{"plate":{"image":"plate"}}}},
            "nodes":[{"type":"camera"}]}))?;
        let scene=r.install(input,&root)?;
        let mut slot=scene.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
        let outputs=crate::ReadbackOutputs {float:crate::FloatOutputs {color:true,..Default::default()},..Default::default()};
        slot.submit_outputs(&scene,&r,0.,outputs)?;
        let first=slot.collect(&r)?.linear_rgba.unwrap();
        assert_eq!(first[0],[1.,0.,0.,1.]);assert_eq!(first[3],[0.,1.,0.,1.]);
        slot.submit_outputs(&scene,&r,1./24.,outputs)?;
        assert!(slot.collect(&r)?.linear_rgba.unwrap().iter().all(|p|*p==[0.,0.,1.,1.]));
        // The same media clock also drives the World, without a compositor.
        let environment:crate::Scene=serde_json::from_value(serde_json::json!({"version":2,
            "render":{"resolution":[4,2],"fps":24,"settings":{"samples":1}},
            "images":{"plate":{"file":"%04d.png"}},
            "materials":{"environment_image":{"program":"std/environment_image","inputs":{"image":{"image":"plate"}}}},
            "nodes":[{"type":"camera"},{"type":"light","shape":"environment","material":"environment_image","energy":1}]}))?;
        let environment=r.install(environment,&root)?;
        let mut world=environment.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
        world.submit_outputs(&environment,&r,0.,outputs)?;
        let first=world.collect(&r)?.linear_rgba.unwrap();
        assert!(first.iter().any(|p|p[0]>0. || p[1]>0.));
        world.submit_outputs(&environment,&r,1./24.,outputs)?;
        for p in world.collect(&r)?.linear_rgba.unwrap() {assert!(p[0].abs()<0.001 && p[1].abs()<0.001 && (p[2]-1.).abs()<0.001,"retimed World: {p:?}");}
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a real GPU and pinned Slang"]
    fn aborted_recording_rejects_all_slots_until_reinstall() -> Result<()> {
        // Both early per-sample uploads and late compositor reads can abort an
        // exposure after retained keys advance. Exercise streaming and batching.
        let r = crate::Renderer::new()?;
        let root = std::env::temp_dir().join(format!("vork-aborted-recording-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        for name in ["a", "b"] {
            for (frame, color) in [(0, [255, 0, 0, 255]), (1, [0, 0, 255, 255])] {
                image::RgbaImage::from_pixel(4, 4, image::Rgba(color))
                    .save(root.join(format!("{name}{frame}.png")))?;
            }
        }
        for early in [false, true] {
            for samples in [1, 2] {
                let mut input = serde_json::json!({"version":2,
                    "composition":{"space":"3d","compositor":["comp"]},"render":{"resolution":[4,4],"fps":24,"settings":{"samples":samples,"integrator":"path","denoiser":null}},
                    "images":{"a":{"file":"a%d.png"},"b":{"file":"b%d.png"}},
                    "programs":{"plate":{"source":"import vork; struct Inputs {Image a; Image b;}; float4 composite(CompositeContext c, Inputs i) {return 0.5*(compositor_sample(i.a,c.uv)+compositor_sample(i.b,c.uv));}"}},
                    "materials":{"comp":{"program":"plate","inputs":{"a":{"image":"a"},"b":{"image":"b"}}}},
                    "nodes":[{"type":"camera","name":"camera"}]});
                if early {
                    if samples == 2 {
                        input["render"]["motion_blur"] = serde_json::json!({"shutter":1});
                    }
                    input["images"]["capture"] = serde_json::json!({"composition":{"compositor":["capture_mix"]},"render":{
                        "camera":"camera","resolution":[4,4]
                    }});
                    input["materials"]["capture_mix"] = input["materials"]["comp"].clone();
                    input["programs"]["capture_copy"] = serde_json::json!({"source":"import vork; struct Inputs {Image capture;}; float4 composite(CompositeContext c, Inputs i) {return compositor_sample(i.capture,c.uv);}"});
                    input["materials"]["comp"] = serde_json::json!({"program":"capture_copy","inputs":{"capture":{"image":"capture"}}});
                }
                let input: crate::Scene = serde_json::from_value(input)?;
                let scene = r.install(input.clone(), &root)?;
                assert!(scene.footage.values().all(|source| source.per_sample == early));
                let mut slot = scene.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
                let mut other = scene.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
                assert_eq!(!slot.image_views.is_empty(), early, "early abort covers moved capture owners");
                let outputs = crate::ReadbackOutputs { float: crate::FloatOutputs {color:true,..Default::default()},..Default::default() };
                slot.submit_outputs(&scene, &r, 0., outputs)?;
                assert!(slot.collect(&r)?.linear_rgba.unwrap().iter().all(|p| *p == [1.,0.,0.,1.]));
                other.submit_outputs(&scene, &r, 0., outputs)?;
                std::fs::rename(root.join("b1.png"), root.join("b1.hidden"))?;
                let failure = slot.submit_outputs(&scene, &r, 1./24., outputs).unwrap_err();
                std::fs::rename(root.join("b1.hidden"), root.join("b1.png"))?;
                assert_eq!(failure.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::NotFound);
                if early && samples == 2 {
                    assert_eq!(slot.view.sample_index, 1, "one streaming sample submitted before the abort");
                }
                let queued = other.collect(&r).err().expect("queued publication must be refused");
                assert!(format!("{queued:#}").contains("publication invalidated"),
                    "aborted recording must discard queued images: {queued:#}");
                for target in [&mut slot, &mut other] {
                    let retry = target.submit_outputs(&scene, &r, 1./24., outputs);
                    assert!(retry.is_err(), "aborted recording admitted another submission (early={early}, samples={samples})");
                    let error = format!("{:#}", retry.unwrap_err());
                    assert!(error.contains("reinstall") && error.contains(&failure.root_cause().to_string()), "{error}");
                }
                let mut fresh = scene.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
                assert!(fresh.submit_outputs(&scene, &r, 1./24., outputs).is_err());
                // Reinstallation is the explicit recovery boundary; the renderer
                // remains usable and restored sources must publish the new frame.
                let clean = r.install(input, &root)?;
                let mut recovered = clean.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
                recovered.submit_outputs(&clean, &r, 2./24., outputs)?;
                assert!(recovered.collect(&r)?.linear_rgba.unwrap().iter().all(|p| *p == [0.,0.,1.,1.]));
            }
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a real GPU and pinned Slang"]
    fn aborted_first_tlas_recording_cannot_be_reused() -> Result<()> {
        let r = crate::Renderer::new()?;
        let root = std::env::temp_dir().join(format!("vork-aborted-tlas-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        std::fs::write(root.join("geometry.bin"), include_bytes!("../tests/fixtures/geometry.bin"))?;
        image::RgbaImage::from_pixel(4, 4, image::Rgba([0u8,0,255,255])).save(root.join("plate0.png"))?;
        let mut input: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/analytic.json"))?;
        input["render"]["resolution"] = serde_json::json!([32,24]);
        input["render"]["settings"] = serde_json::json!({"samples":1,"integrator":"raster","denoiser":null});
        input["composition"]["space"] = "3d".into(); input["composition"]["compositor"] = serde_json::json!(["comp"]);
        input["images"] = serde_json::json!({"plate":{"file":"plate%d.png"}});
        input["programs"] = serde_json::json!({"plate":{"source":"import vork; struct Inputs {Image plate;}; float4 composite(CompositeContext c, Inputs i) {return 0.5*(compositor_sample(c.uv)+compositor_sample(i.plate,c.uv));}"}});
        input["materials"]["comp"] = serde_json::json!({"program":"plate","inputs":{"plate":{"image":"plate"}}});
        let input = vork_scene::v2::from_test_value(input)?;
        let scene = r.install(input, &root)?;
        assert!(!scene.footage["plate"].per_sample);
        let mut slot = scene.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        assert_eq!(slot.rays.as_ref().expect("ray-shadow scene").builds, 0);
        std::fs::rename(root.join("plate0.png"), root.join("plate.hidden"))?;
        let error = slot.submit(&scene, &r, 0.).unwrap_err();
        std::fs::rename(root.join("plate.hidden"), root.join("plate0.png"))?;
        assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::NotFound);
        assert_eq!(slot.rays.as_ref().unwrap().builds, 1, "the first TLAS build was recorded before the abort");
        let retry = slot.submit(&scene, &r, 0.);
        assert!(retry.is_err(), "retry admitted a recorded-but-unsubmitted TLAS");
        let error = format!("{:#}", retry.unwrap_err());
        assert!(error.contains("reinstall"), "{error}");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn image_clock_uses_track_layers_for_reverse_hold_and_loop() -> Result<()> {
        let scene:crate::Scene=serde_json::from_value(serde_json::json!({"version":2,"images":{"plate":{"file":"plate.mp4"}},"nodes":[{"type":"camera"}],
            "animation":[{"loop":true,"tracks":[{"path":["images","plate","time"],"keys":[[0,2],[1,1],[2,1],[3,0]]}]}]}))?;
        let plan=crate::animation::Plan::compile(&scene,Path::new("."))?;
        for (time,expected) in [(0.,2.),(0.5,1.5),(1.5,1.),(2.5,0.5),(3.5,1.5)] {
            let mut samples=plan.samples();plan.sample_into(time,&mut samples);
            let request=vork_time::Request::Numeric(time).exact()?;
            assert!((plan.image_time_exact(&samples,&request,"plate",request.clone())?.numeric()?-expected).abs()<1e-5);
        }
        Ok(())
    }
    #[test]
    fn failed_video_reads_restart_at_the_same_request() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-footage-recovery-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let path=root.join("source.mp4");
        let status=Command::new("ffmpeg").args(["-v","error","-y","-f","lavfi","-i","testsrc2=size=16x16:rate=6:duration=2",
            "-c:v","libx264","-g","6","-keyint_min","6","-sc_threshold","0"]).arg(&path).status()?;
        ensure!(status.success(),"make recovery video fixture");
        let expected=Command::new("ffmpeg").args(["-v","error","-i"]).arg(&path)
            .args(["-fps_mode","passthrough","-f","rawvideo","-pix_fmt","rgba64le","pipe:1"]).output()?;
        ensure!(expected.status.success(),"decode recovery reference");
        let frame_bytes=16*16*8;
        // Model a decoder ending before a frame, inside a frame, or after
        // complete frames, on both forward and reverse requests. The data
        // demuxer copies the exact prefix into a real child stdout pipe.
        for (request,previous) in [(0,None),(4,None),(4,Some(5)),(10,None),(10,Some(11))] {
            for length in [0,frame_bytes/2,2*frame_bytes,2*frame_bytes+frame_bytes/2,4*frame_bytes,4*frame_bytes+frame_bytes/2] {
                if length/frame_bytes>request {continue;}
                let mut source=Video::open(&path)?;
                source.last_request=previous;
                let start=source.keyframes.iter().rev().find(|&&(i,_)|i<=request).map_or(0,|&(i,_)|i);
                assert_eq!(start,if request>=6 {6} else {0});
                let truncated=root.join("truncated.raw");
                std::fs::write(&truncated,&expected.stdout[start*frame_bytes..start*frame_bytes+length])?;
                let mut child=Command::new("ffmpeg").args(["-v","error","-f","data","-i"]).arg(&truncated)
                    .args(["-map","0:0","-c","copy","-f","data","pipe:1"])
                    .stdout(Stdio::piped()).spawn()?;
                let stdout=child.stdout.take().unwrap();
                source.session=Some(Session {child,stdout,next:start});
                let error=source.frame_at(request,32).unwrap_err();
                let failed_frame=start+length/frame_bytes;
                assert!(error.to_string().contains(&format!("frame {failed_frame}")),"{error:#}");
                assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(),std::io::ErrorKind::UnexpectedEof);
                assert!(source.session.is_none(),"failed decoder must be closed");
                assert!(source.frames.iter().all(|(index,_)|*index<failed_frame),"never cache a partial frame");
                // Retry the identical position without clearing state or seeking.
                let actual=source.frame_at(request,32)?.to_rgba16();
                let bytes:Vec<_>=actual.as_raw().iter().flat_map(|v|v.to_le_bytes()).collect();
                assert_eq!(bytes,expected.stdout[request*frame_bytes..(request+1)*frame_bytes]);
                assert_eq!(source.session.as_ref().unwrap().next,request+1);
            }
        }
        // Metadata survives a source disappearing before decoder startup.
        // Repeated failures must remain visible, then recover when it returns.
        let mut source=Video::open(&path)?;
        let saved=root.join("saved.mp4");
        std::fs::rename(&path,&saved)?;
        for _ in 0..2 {
            assert!(source.frame_at(0,32).is_err());
            assert!(source.session.is_none());
        }
        std::fs::rename(&saved,&path)?;
        let actual=source.frame_at(0,32)?.to_rgba16();
        let bytes:Vec<_>=actual.as_raw().iter().flat_map(|v|v.to_le_bytes()).collect();
        assert_eq!(bytes,expected.stdout[..frame_bytes]);
        drop(source);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn indexed_video_seeks_preserve_frames_and_bound_reverse_work() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-footage-index-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let path=root.join("long.mp4");
        let status=Command::new("ffmpeg").args(["-v","error","-y","-f","lavfi","-i","testsrc2=size=16x16:rate=30:duration=20",
            "-c:v","libx264","-g","30","-keyint_min","30","-sc_threshold","0","-bf","3","-output_ts_offset","7"]).arg(&path).status()?;
        ensure!(status.success(),"make long-GOP video fixture");
        let expected=Command::new("ffmpeg").args(["-v","error","-i"]).arg(&path)
            .args(["-fps_mode","passthrough","-f","rawvideo","-pix_fmt","rgba64le","pipe:1"]).output()?;
        ensure!(expected.status.success(),"decode video reference");
        let mut source=Video::open(&path)?;
        let mut requests:Vec<_>=(500..600).rev().collect();
        requests.extend([5,590,60,450,300,0,599,599,10]);
        for index in requests {
            let before=source.decoded;
            let request=Germ::point((Rational::from_integer(source.pts[index].into())-Rational::from_integer(source.pts[0].into()))*&source.timebase);
            assert_eq!(source.frame_index(&request),index);
            let actual=source.frame_at(source.frame_index(&request),32)?.to_rgba16();
            let bytes:Vec<_>=actual.as_raw().iter().flat_map(|v|v.to_le_bytes()).collect();
            assert_eq!(bytes,&expected.stdout[index*16*16*8..(index+1)*16*16*8],"seek frame {index}");
            assert!(source.decoded-before<=30,"seek {index} decoded {} frames",source.decoded-before);
            assert!(source.frames.len()<=crate::constants::FOOTAGE_DECODE_FRAMES);
        }
        drop(source);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn video_timestamps_hold_reverse_and_autorotate() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-footage-video-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        for (name,color) in [("red",[255,0,0]),("green",[0,255,0]),("blue",[0,0,255])] {
            image::RgbImage::from_pixel(8,4,image::Rgb(color)).save(root.join(format!("{name}.png")))?;
        }
        std::fs::write(root.join("frames.txt"),"file 'red.png'\nduration 0.04\nfile 'green.png'\nduration 0.16\nfile 'blue.png'\nduration 0.08\nfile 'blue.png'\n")?;
        let video=root.join("vfr.mp4");
        let status=Command::new("ffmpeg").current_dir(&root).args(["-v","error","-y","-f","concat","-safe","0","-i","frames.txt","-fps_mode","vfr","-c:v","libx264","-pix_fmt","yuv444p"]).arg(&video).status()?;
        ensure!(status.success(),"make VFR video fixture");
        let mut source=Video::open(&video)?;
        source.frame(0.,16)?;source.frame(0.04,16)?;
        let next=source.session.as_ref().unwrap().next;
        source.frame(0.,16)?;source.frame(0.04,16)?;
        assert_eq!(source.session.as_ref().unwrap().next,next,"shuffled shutter samples retain the decoder session");
        assert!(source.times.windows(3).any(|t| (t[1]-t[0]-(t[2]-t[1])).abs()>0.01));
        for (time,channel) in [(0.,0),(0.039,0),(0.04,1),(0.08,1),(0.19,1),(10.,2),(0.,0)] {
            let pixel=source.frame(time,16)?.to_rgb8()[(0,0)];
            assert!(pixel[channel]>240,"at {time}: {pixel:?}");
        }
        let offset=root.join("offset.mp4");
        let status=Command::new("ffmpeg").args(["-v","error","-y","-itsoffset","3","-i"]).arg(&video).args(["-c","copy"]).arg(&offset).status()?;
        ensure!(status.success(),"make edit-list video fixture");
        let mut shifted=Video::open(&offset)?;
        assert_eq!(shifted.times[0],0.);
        assert!(shifted.frame(0.08,16)?.to_rgb8()[(0,0)][1]>240);
        drop(shifted);
        let rotated=root.join("rotated.mp4");
        let status=Command::new("ffmpeg").args(["-v","error","-y","-display_rotation","90","-i"]).arg(&video).args(["-c","copy"]).arg(&rotated).status()?;
        ensure!(status.success(),"make rotated video fixture");
        let mut rotated=Video::open(&rotated)?;
        let frame=rotated.frame(0.,16)?;assert_eq!((frame.width(),frame.height()),(4,8));
        drop(rotated);drop(source);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn exr_sequence_clock_uses_first_header_and_rejects_invalid_rates() -> Result<()> {
        use exr::prelude::*;
        let root=std::env::temp_dir().join(format!("vork-exr-clock-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let write=|index,rate| -> Result<()> {
            let attributes=LayerAttributes {frames_per_second:rate,..Default::default()};
            let channels=SpecificChannels::rgb(|_:Vec2<usize>|(index as f32,0f32,0f32));
            Image::from_layer(Layer::new((1,1),attributes,Encoding::SMALL_LOSSLESS,channels)).write().to_file(root.join(format!("{index:04}.exr")))?;
            Ok(())
        };
        // A later header cannot change the sequence's clock.
        write(2,Some((120,1)))?;
        for rate in [Some((2,1)),None] {
            write(1,rate)?;
            let mut source=Source::open(&root.join("%04d.exr"))?;
            let fps=if rate.is_some() {2.} else {24.};
            assert_eq!(source.timeline(vork_time::Time::integer(24))?.iter().map(|t|t.as_f32()).collect::<Vec<_>>(),vec![0.,1./fps]);
            assert_eq!(source.frame_index_exact(&vork_time::Request::Numeric(0.04).exact()?,vork_time::Time::integer(24))?,0);
            assert_eq!(source.frame_index_exact(&vork_time::Request::Numeric(0.04).exact()?,vork_time::Time::integer(48))?,usize::from(rate.is_none()));
            for render_fps in [vork_time::Time::integer(24),vork_time::Time::integer(48)] {
                for time in [-1.,0.,0.04,0.5,100.,0.] {
                    let expected=1.+source.frame_index_exact(&vork_time::Request::Numeric(time).exact()?,render_fps)? as f32;
                    assert_eq!(source.frame(time,render_fps,16)?.to_rgb32f()[(0,0)][0],expected);
                }
                for (index,time) in source.timeline(render_fps)?.into_iter().enumerate() {assert_eq!(source.frame_index_exact(&vork_time::Request::from(&time).exact()?,render_fps)?,index);}
            }
        }
        for rate in [(0,1),(1,0),(-1,1)] {
            write(1,Some(rate))?;
            let error=Source::open(&root.join("%04d.exr")).err().context("invalid EXR rate accepted")?;
            assert!(format!("{error:#}").contains("framesPerSecond must be positive"),"{error:#}");
        }
        std::fs::remove_dir_all(root)?;Ok(())
    }

    #[test]
    fn sequences_start_at_first_disk_frame_hold_ends_and_reject_holes() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-footage-sequence-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let dir = root.as_path();
        for n in [1001,1002] {
            image::RgbaImage::from_pixel(1,1,image::Rgba([(n-1000) as u8,0,0,255])).save(dir.join(format!("{n:04}.png")))?;
        }
        let mut source = Source::open(&dir.join("%04d.png"))?;
        for (time,expected) in [(-1.,1),(0.,1),(0.041,1),(1./24.,2),(100.,2)] {
            assert_eq!(source.frame(time,vork_time::Time::integer(24),16)?.to_rgba8()[ (0,0) ][0],expected);
        }
        std::fs::rename(dir.join("1002.png"),dir.join("1004.png"))?;
        let error = Source::open(&dir.join("%04d.png")).err().unwrap().to_string();
        assert!(error.contains("1002..1003"),"{error}");
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }
}

/// One resident texture, updated by ordered encoder copies rather than queue
/// writes so several shutter samples can coexist in one command submission.
pub(crate) struct Installed {
    pub name: String,
    pub time: Option<vork_time::Clock>,
    pub per_sample: bool,
    pub final_compositor: bool,
    pub view: wgpu::TextureView,
    source: Option<std::sync::Mutex<Source>>,
    /// A picture of a canvas node (`images.X.render.root` names a 2D composition): drawn by Moda's engine at the
    /// node's local time instead of read from a file.
    pub canvas: Option<crate::canvas_world::Picture>,
    texture: wgpu::Texture,
    mips: crate::materials::MipChain,
    /// The encoding the colour decoder is asked for.
    upload: vork_scene::v2::WorkingColorSpace,
    /// Whether an sRGB-encoded upload is decoded to linear light for the texture: a world reads scene-linear
    /// texels. A canvas's image graph reads the encoding the canvas blends in, which the decoder delivers as it
    /// is.
    linear: bool,
    color_space: Option<vork_scene::v2::ColorSpace>,
}
impl Installed {
    /// End external input work while the completed shot still owns GPU resources.
    pub(crate) fn finish(&mut self) {
        if let Some(Source::Video(video)) = self.source.as_mut().map(|source| source.get_mut().unwrap()) {
            video.session = None; // Session kills and reaps before closing its reader.
        }
    }
    /// The texture a canvas's picture is drawn into each frame: its size, scene-linear with associated alpha.
    pub(crate) fn canvas(r:&crate::Renderer,name:&str,picture:crate::canvas_world::Picture)->Result<Self> {
        let [width,height]=picture.size();
        let size=wgpu::Extent3d {width,height,depth_or_array_layers:1};
        let texture=crate::gpu_memory::texture(&r.device,&wgpu::TextureDescriptor {
            label:Some("canvas picture"),size,mip_level_count:32-width.max(height).leading_zeros(),sample_count:1,
            dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba16Float,
            usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::COPY_DST|wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats:&[],
        });
        let (pipelines,sampler)=crate::materials::mip_resources(r);
        let mips=crate::materials::MipChain::new(&r.device,&texture,None,&pipelines[2],sampler);
        Ok(Self {name:name.into(),time:None,per_sample:true,final_compositor:false,view:texture.create_view(&Default::default()),texture,mips,source:None,canvas:Some(picture),upload:vork_scene::v2::WorkingColorSpace::LinearRec709,linear:true,color_space:None})
    }
    /// A world's media image: scene-linear texels.
    pub(crate) fn new(r:&crate::Renderer,name:&str,path:&Path,time:Option<vork_time::Clock>,srgb:bool,color_space:Option<vork_scene::v2::ColorSpace>,fps:vork_time::Time,opened:&Opened)->Result<Self> {
        Self::from_source(r,name,opened.source(path)?,time,crate::texture_source::upload(srgb),true,color_space,fps)
    }
    /// The resident texture of `source`. `upload`: the encoding the colour decoder is asked for. `linear`: an
    /// sRGB-encoded upload is decoded to linear light (`Installed::linear`).
    pub(crate) fn from_source(r:&crate::Renderer,name:&str,mut source:Source,time:Option<vork_time::Clock>,upload:vork_scene::v2::WorkingColorSpace,linear:bool,color_space:Option<vork_scene::v2::ColorSpace>,fps:vork_time::Time)->Result<Self> {
        source.prepare_gpu(r);
        let instant=time.as_ref().map(vork_time::Request::from).unwrap_or(vork_time::Request::Numeric(0.)).exact()?;
        let frame=source.frame_color_exact(&instant,fps,r.device.limits().max_texture_dimension_2d,color_space,upload,false)
            .map_err(|error|crate::texture_source::for_image(error,name))?;
        let size=wgpu::Extent3d {width:frame.width(),height:frame.height(),depth_or_array_layers:1};
        let texture=crate::gpu_memory::texture(&r.device,&wgpu::TextureDescriptor {
            label:Some("footage image"),size,mip_level_count:32-size.width.max(size.height).leading_zeros(),sample_count:1,
            dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Rgba16Float,
            usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::COPY_DST|wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats:&[],
        });
        let (pipelines,sampler)=crate::materials::mip_resources(r);
        let mips=crate::materials::MipChain::new(&r.device,&texture,None,&pipelines[2],sampler);
        Ok(Self {name:name.into(),time,per_sample:true,final_compositor:false,view:texture.create_view(&Default::default()),texture,mips,source:Some(std::sync::Mutex::new(source)),canvas:None,upload,linear,color_space})
    }
    /// Resolve a canvas's owning node before either sample or compositor publication.
    pub(crate) fn canvas_instant(&self,s:&crate::InstalledScene,samples:&crate::animation::Samples,request:&Germ)->Result<Option<Germ>> {
        let picture=self.canvas.as_ref().expect("canvas picture");
        let node=s.nodes.iter().position(|n|n.node.name==picture.node())
            .with_context(||format!("image {}: canvas node {:?} is not installed",self.name,picture.node()))?;
        s.animation.local_instant(samples,node,request)
            .map(|base|s.animation.image_time_exact(samples,request,&self.name,base)).transpose()
    }
    /// Decoded media has source frames; a canvas has a continuous, sided local instant.
    /// Absence is a publication too, distinct from both a picture and an invalidated key.
    pub(crate) fn sample_key(&self,time:Option<&Germ>,fps:vork_time::Time)->Result<Vec<u8>> {
        let Some(time)=time else {return Ok(vec![0])};
        let mut key=Vec::new();
        match &self.source {
            Some(source)=>{
                key.push(1);
                key.extend_from_slice(&(source.lock().unwrap().frame_index_exact(time,fps)? as u64).to_le_bytes());
            }
            None=>{key.push(2);time.append_key(&mut key);}
        }
        Ok(key)
    }
    /// `errors` is the owning frame's shader diagnostics: a canvas picture's view reports a non-finite result
    /// there, so the frame refuses it where it is collected and the picture needs no read-back of its own.
    pub(crate) fn encode(&self,r:&crate::Renderer,encoder:&mut wgpu::CommandEncoder,time:Option<&Germ>,fps:vork_time::Time,errors:Option<&wgpu::Buffer>)->Result<()> {
        let Some(time)=time else {
            for mip in 0..self.texture.mip_level_count() {
                let view=self.texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level:mip,mip_level_count:Some(1),..Default::default()
                });
                let _pass=encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:Some("absent canvas picture"),
                    color_attachments:&[Some(wgpu::RenderPassColorAttachment {
                        view:&view,resolve_target:None,depth_slice:None,
                        ops:wgpu::Operations {load:wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),store:wgpu::StoreOp::Store},
                    })],..Default::default()
                });
            }
            return Ok(());
        };
        if let Some(picture)=&self.canvas {
            let level_zero=self.texture.create_view(&wgpu::TextureViewDescriptor {mip_level_count:Some(1),..Default::default()});
            picture.encode(r,encoder,&vork_time::Request::Exact(time.clone()),&level_zero,errors)?;
            self.mips.encode(encoder,None);
            return Ok(());
        }
        let source=self.source.as_ref().expect("a file source");
        let frame=source.lock().unwrap().frame_color_exact(time,fps,r.device.limits().max_texture_dimension_2d,self.color_space,self.upload,false)
            .map_err(|error|crate::texture_source::for_image(error,&self.name))?;
        ensure!(frame.width()==self.texture.width() && frame.height()==self.texture.height(),"image {}: media frame dimensions changed; re-export every frame at the same resolution",self.name);
        let rgba=frame.into_rgba32f();
        let pitch=(self.texture.width()*8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)*wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let mut data=vec![0u8;pitch as usize*self.texture.height() as usize];
        for (row,pixels) in rgba.as_raw().chunks_exact(self.texture.width() as usize*4).enumerate() {
            for (i,&value) in pixels.iter().enumerate() {
                let value=if self.linear && !self.upload.is_linear() && i%4!=3 {crate::texture_source::decode_srgb(value)} else {value};
                let at=row*pitch as usize+i*2;
                data[at..at+2].copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
            }
        }
        use wgpu::util::DeviceExt;
        let staging=r.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {label:Some("footage upload"),contents:&data,usage:wgpu::BufferUsages::COPY_SRC});
        encoder.copy_buffer_to_texture(wgpu::TexelCopyBufferInfo {buffer:&staging,layout:wgpu::TexelCopyBufferLayout {offset:0,bytes_per_row:Some(pitch),rows_per_image:Some(self.texture.height())}},self.texture.as_image_copy(),self.texture.size());
        self.mips.encode(encoder,None);
        Ok(())
    }
}

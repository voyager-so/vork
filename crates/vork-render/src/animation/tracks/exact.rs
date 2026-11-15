//! The consumed image-time slice of the shared animation plan.
use super::*;
use vork_time::exact::{Germ, Rational, decimal, ieee};

pub(super) struct DeferredTimeCurve {
    source: v2::Track,
    decoded: std::sync::Arc<Curve>,
    value: std::sync::OnceLock<std::result::Result<TimeCurve,String>>,
}
impl DeferredTimeCurve {
    pub(super) fn new(source:&v2::Track,decoded:std::sync::Arc<Curve>)->Self {
        Self {source:source.clone(),decoded,value:Default::default()}
    }
    pub(super) fn get(&self)->Result<&TimeCurve> {
        self.value.get_or_init(||TimeCurve::compile(&self.source,&self.decoded).map_err(|e|format!("{e:#}")))
            .as_ref().map_err(|error|anyhow::anyhow!("{error}"))
    }
}
pub(super) struct TimeSpan {
    pub(super) endpoints:Vec<vork_time::Clock>,
    value:std::sync::OnceLock<(Rational,Rational)>,
}
impl TimeSpan {
    pub(super) fn new(endpoints:Vec<vork_time::Clock>)->Self {Self {endpoints,value:Default::default()}}
    fn get(&self)->&(Rational,Rational) {
        self.value.get_or_init(|| {
            let mut times=self.endpoints.iter().map(vork_time::Clock::exact);
            let first=times.next().expect("time slice has keys");
            times.fold((first.clone(),first),|(lo,hi),t|(lo.min(t.clone()),hi.max(t)))
        })
    }
}
pub(super) fn raw_times(track:&v2::Track,decoded:&[f32])->Result<Vec<vork_time::Clock>> {
    match &track.keys {
        v2::Keys::Inline(keys)=>keys.iter().map(|key|serde_json::from_value(key[0].clone()).map_err(Into::into)).collect(),
        v2::Keys::Buffers {..}=>Ok(decoded.iter().map(|&t|vork_time::Clock::binary(t)).collect()),
    }
}

pub(super) struct TimeCurve {
    times: Vec<Rational>,
    values: Vec<Rational>,
    interpolation: Interpolation,
}

fn scalar(value: &serde_json::Value) -> Result<Rational> {
    let value = if let Some(array) = value.as_array() {
        ensure!(array.len() == 1, "time track must be scalar or width one");
        &array[0]
    } else { value };
    match value {
        serde_json::Value::Number(number) => decimal(&number.to_string()),
        serde_json::Value::String(text) if text.contains('/') => decimal(text),
        _ => anyhow::bail!("time track needs numeric seconds or a rational clock"),
    }
}

pub(super) fn times(track: &v2::Track, decoded: &[f32]) -> Result<Vec<Rational>> {
    let times=raw_times(track,decoded)?.iter().map(vork_time::Clock::exact).collect::<Vec<_>>();
    ensure!(!times.is_empty() && times.iter().all(|t|*t>=Rational::from_integer(0.into())) && times.windows(2).all(|p|p[0]<p[1]),"clock keys must be nonnegative and strictly increasing");
    Ok(times)
}

impl TimeCurve {
    pub(super) fn compile(track: &v2::Track, decoded: &Curve) -> Result<Self> {
        ensure!(decoded.width == 1, "time track must be scalar or width one");
        let times = times(track, &decoded.times)?;
        let values = match &track.values {
            v2::Values::Inline(values) => values.iter().map(scalar).collect::<Result<Vec<_>>>()?,
            v2::Values::Buffer(_) => decoded.values.iter().map(|&value| ieee(value)).collect::<Result<Vec<_>>>()?,
        };
        ensure!(times.windows(2).all(|p|p[0] < p[1]), "time track keys must increase");
        Ok(Self { times, values, interpolation: decoded.interpolation })
    }
    pub(super) fn key_tables(&self) -> (&[Rational], &[Rational]) { (&self.times, &self.values) }
    pub(super) fn is_cubic(&self) -> bool { self.interpolation == Interpolation::Cubic }
    pub(super) fn is_step(&self) -> bool { self.interpolation == Interpolation::Step }
    pub(super) fn sample(&self, time: &Germ) -> Germ {
        let upper = self.times.partition_point(|key| time.compare_point(key).is_ge());
        let a = upper.saturating_sub(1);
        let b = upper.min(self.times.len()-1);
        let cubic = self.interpolation == Interpolation::Cubic;
        let value = |key: usize, part: usize| Germ::point(self.values[key * if cubic {3} else {1} + part].clone());
        if a == b || self.interpolation == Interpolation::Step {
            return value(a, usize::from(cubic));
        }
        let dt = &self.times[b] - &self.times[a];
        let u = time.sub(&Germ::point(self.times[a].clone())).scale(&dt.recip());
        let constant = |n: i64| Germ::point(Rational::from_integer(n.into()));
        if cubic {
            let [h00,h10,h01,h11] = hermite_basis(u,constant(1),constant(2),constant(3));
            h00 * value(a,1) + h10.scale(&dt) * value(a,2)
                + h01 * value(b,1) + h11.scale(&dt) * value(b,0)
        } else {
            value(a,0) * (constant(1)-u.clone()) + value(b,0) * u
        }
    }
}

impl Plan {
    /// Weights have already been evaluated by the numeric plan. No numeric
    /// weight is decoded as a timestamp or evaluated a second time here.
    pub fn image_time_exact(&self, samples: &Samples, time: &Germ, name: &str, base: Germ) -> Result<Germ> {
        let mut result = base;
        let mut index = 0;
        for layer in &self.layers {
            let mut local = None;
            for track in &layer.tracks {
                if matches!(&track.target, Some(v2::TrackTarget::ImageTime {image}) if image == name)
                    && samples.weights[index] != 0.
                {
                    let curve = track.time_curve.as_ref().context("image time curve is installed")?.get()?;
                    if local.is_none() {
                        let mut mapped = time.scale(&ieee(layer.source.speed)?)
                            .add(&Germ::point(ieee(layer.source.offset)?));
                        if layer.source.looping {
                            if let Some(span) = &layer.time_span {
                                let (a,b)=span.get();
                                if b > a { mapped = mapped.euclidean_loop(a,b)?; }
                            }
                        }
                        local = Some(mapped);
                    }
                    let value = curve.sample(local.as_ref().unwrap());
                    let weight = ieee(samples.weights[index])?;
                    result = if matches!(layer.source.blend, v2::Blend::Add) {
                        result.add(&value.scale(&weight))
                    } else {
                        result.add(&value.sub(&result).scale(&weight))
                    };
                }
                index += 1;
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn q(text: &str) -> Rational { decimal(text).unwrap() }
    fn plan(animation: serde_json::Value, buffers: serde_json::Value) -> Result<Plan> {
        let scene = v2::Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"render":{"duration":3600,"fps":"30000/1001","background":[0,0,0]},
            "nodes":[{"type":"camera","material":{"program":"std/camera"}}],
            "images":{"plate":{"file":"plate.mp4"}}, "animation":animation,"buffers":buffers
        }))?)?;
        scene.validate()?;
        Plan::compile(&scene,std::path::Path::new("."))
    }
    fn evaluate(plan: &Plan, time: &Germ) -> Result<Germ> {
        let mut samples = plan.samples();
        plan.sample_into(time.numeric()?, &mut samples);
        plan.image_time_exact(&samples,time,"plate",time.clone())
    }
    #[test]
    fn linked_hour_identity_has_no_frame_selection_errors() -> Result<()> {
        for (interpolation,keys) in [
            ("LINEAR", serde_json::json!([[0,0],[3600,3600]])),
            ("CUBICSPLINE",serde_json::json!([[0,0,1,1],[3600,3600,1,1]])),
        ] {
            let plan = plan(serde_json::json!([{"tracks":[{"path":["images","plate","time"],"keys":keys,"interpolation":interpolation}]}]),serde_json::json!({}))?;
            let fps = q("30000/1001");
            for n in 0..107893_i64 {
                let time = Germ::point(Rational::from_integer(n.into()) / &fps);
                let result = evaluate(&plan,&time)?;
                assert_eq!(result,time,"{interpolation} frame {n}");
                assert_eq!(result.scale(&fps).floor_clamped(0,107892)?,n);
            }
        }
        Ok(())
    }
    #[test]
    fn exact_clock_keys_can_share_numeric_lowering_and_media_clamps_large_values()->Result<()> {
        let exact_plan=plan(serde_json::json!([{"tracks":[{"path":["images","plate","time"],"keys":[["1/1",0],["100000000000000000001/100000000000000000000",1]],"interpolation":"STEP"}]}]),serde_json::json!({}))?;
        let boundary=q("100000000000000000001/100000000000000000000");
        assert_eq!(evaluate(&exact_plan,&Germ::point(boundary.clone()))?.floor_clamped(0,1)?,1);
        assert_eq!(evaluate(&exact_plan,&Germ::left(boundary))?.floor_clamped(0,1)?,0);
        let large=format!("1{}/1","0".repeat(400));
        let exact_plan=plan(serde_json::json!([{"tracks":[{"path":["images","plate","time"],"keys":[[0,large]]}]}]),serde_json::json!({}))?;
        assert_eq!(evaluate(&exact_plan,&Germ::point(q("0")))?.floor_clamped(0,99)?,99);
        Ok(())
    }
    #[test]
    fn linked_buffer_clocks_and_numeric_width_one_weights() -> Result<()> {
        let plan = plan(serde_json::json!([
            {"tracks":[{"path":["animation","retime","weight"],"keys":[[0,[0.5]],[1,[0.5]]]}]},
            {"name":"retime","tracks":[{"path":["images","plate","time"],"keys":{"times":"t","values":"v"}}]}
        ]),serde_json::json!({"t":{"type":"f32","data":[0,1]},"v":{"type":"f32","data":[0.1,0.1]}}))?;
        let time = Germ::point(q("1/2"));
        let expected = time.add(&Germ::point(ieee(0.1)?).sub(&time).scale(&ieee(0.5)?));
        assert_eq!(evaluate(&plan,&time)?,expected);
        Ok(())
    }
    #[test]
    fn linked_reverse_composition_preserves_left_limit() -> Result<()> {
        let plan = plan(serde_json::json!([{"tracks":[{
            "path":["images","plate","time"],"keys":[[0,1],[1,0]]
        }]}]),serde_json::json!({}))?;
        let left = Germ::left(q("1/2"));
        let right = evaluate(&plan,&left)?;
        assert_eq!(right,Germ::right(q("1/2")));
        assert_eq!(evaluate(&plan,&right)?,left);
        Ok(())
    }
    #[test]
    fn unconsumed_image_curves_and_spans_remain_uncompiled()->Result<()> {
        let plan=plan(serde_json::json!([{"loop":true,"tracks":[{"path":["images","plate","time"],"keys":[[0,0],[1,1]]}]}]),serde_json::json!({}))?;
        assert!(plan.layers[0].time_span.as_ref().unwrap().value.get().is_none());
        assert!(plan.layers[0].tracks[0].time_curve.as_ref().unwrap().value.get().is_none());
        let time=Germ::point(q("1/2"));
        assert_eq!(evaluate(&plan,&time)?,time);
        assert!(plan.layers[0].time_span.as_ref().unwrap().value.get().is_some());
        assert!(plan.layers[0].tracks[0].time_curve.as_ref().unwrap().value.get().is_some());
        Ok(())
    }
    #[test]
    fn ordinary_tracks_compile_no_exact_curve_or_span() -> Result<()> {
        let plan = plan(serde_json::json!([{"tracks":[{
            "path":["render","background"],"keys":[[0,[0,0,0]],[1,[1,1,1]]]
        }]}]),serde_json::json!({}))?;
        assert!(plan.layers.iter().all(|layer|layer.time_span.is_none() && layer.tracks.iter().all(|track|track.time_curve.is_none())));
        Ok(())
    }
    #[test]
    fn linked_extremum_and_ordered_blend_cancellation() -> Result<()> {
        let maximum = plan(serde_json::json!([{"tracks":[{
            "path":["images","plate","time"],"interpolation":"CUBICSPLINE",
            "keys":[[0,0,0,2],[1,0,-2,0]]
        }]}]),serde_json::json!({}))?;
        let left = Germ::left(q("1/2"));
        let value = evaluate(&maximum,&left)?;
        assert_eq!(value.value(),&q("1/2"));
        assert!(value.compare_point(&q("1/2")).is_lt());
        assert_eq!(value.scale(&q("24")).floor_clamped(0,24)?,11);
        let cancellation = plan(serde_json::json!([
            {"tracks":[{"path":["images","plate","time"],"keys":[[0,-0.5],[1,0.5]]}]},
            {"blend":"combine","tracks":[{"path":["images","plate","time"],"interpolation":"CUBICSPLINE","keys":[[0,0.75,-2,-2],[1,-0.25,0,0]]}]}
        ]),serde_json::json!({}))?;
        let value = evaluate(&cancellation,&left)?;
        assert_eq!(value.value(),&q("0"));
        assert!(value.compare_point(&q("0")).is_gt());
        Ok(())
    }
    #[test]
    fn tiny_numeric_controls_are_not_clock_denominator_errors() -> Result<()> {
        for weight in [serde_json::json!(1e-20),serde_json::json!([1e-20])] {
            let plan = plan(serde_json::json!([
                {"tracks":[{"path":["animation","retime","weight"],"keys":[[0,weight]]}]},
                {"name":"retime","weight":0,"speed":1e-20,"tracks":[{"path":["images","plate","time"],"keys":[[0,0],[1,1]]}]}
            ]),serde_json::json!({}))?;
            let time = Germ::point(q("1"));
            let control = ieee(1e-20)?;
            let expected = time.add(&time.scale(&control).sub(&time).scale(&control));
            assert_eq!(evaluate(&plan,&time)?,expected);
        }
        Ok(())
    }
}

impl CompiledLayer {
    /// The exact span a looping layer's clock wraps over.
    pub(super) fn loop_span(&self)->Option<&(Rational,Rational)> {
        if !self.source.looping {return None}
        self.time_span.as_ref().map(|span|span.get()).filter(|span|span.1>span.0)
    }
    pub(super) fn exact_local(&self,time:&Germ)->Result<Germ> {
        let mut local=time.scale(&ieee(self.source.speed)?).add(&Germ::point(ieee(self.source.offset)?));
        if let Some((start,end))=self.loop_span() {local=local.euclidean_loop(start,end)?;}
        Ok(local)
    }
}
impl Plan {
    pub fn has_camera_tracks(&self)->bool {self.camera_tracks}
    pub fn camera_at_request(&self,time:&vork_time::Request,fallback:usize)->Result<usize> {
        if !self.has_camera_tracks() {return Ok(fallback)}
        let mut samples=self.samples();
        let numeric=match time {vork_time::Request::Exact(t)=>t.numeric_side()?,_=>time.numeric()?};
        self.sample_where_request(numeric,Some(time),&mut samples,|track|track.camera_keys.is_some())?;
        self.camera_for_request(&samples,time,fallback)
    }
    pub fn camera_for_request(&self,samples:&Samples,time:&vork_time::Request,fallback:usize)->Result<usize> {
        self.camera_selection(samples,time)?.map(|name|self.cameras.get(name).copied().context("camera track must refer to installed camera")).unwrap_or(Ok(fallback))
    }
    pub(super) fn camera_selection<'a>(&'a self,samples:&Samples,time:&vork_time::Request)->Result<Option<&'a str>> {
        let mut result=None;
        let mut index=0;
        let mut request=None;
        for layer in &self.layers {
            let mut local=None;
            for track in &layer.tracks {
                if let Some(keys)=&track.camera_keys {
                    if samples.weights[index]>=0.5 && !matches!(layer.source.blend,v2::Blend::Add) {
                        if request.is_none() {request=Some(time.exact()?);}
                        if local.is_none() {local=Some(layer.exact_local(request.as_ref().unwrap())?);}
                        let key=keys.partition_point(|key|local.as_ref().unwrap().compare_point(key).is_ge()).saturating_sub(1);
                        let Channel::Discrete {values,..}=&track.channel else {unreachable!()};
                        let Sample::String(name)=&values[key] else {unreachable!()};
                        result=Some(name.as_ref());
                    }
                }
                index+=1;
            }
        }
        Ok(result)
    }
}

// One affine key's occurrences in traversal order. No cycle list is built.
struct CameraEvent { next:Option<Rational>, period:Option<Rational> }
impl CameraEvent {
    fn new(anchor:Rational,period:Option<Rational>,lo:&Rational,hi:&Rational,forward:bool)->Self {
        let next=if let Some(period)=&period {
            let count=if forward {((lo-&anchor)/period).ceil()}else{((hi-&anchor)/period).floor()};
            anchor+count*period
        }else{anchor};
        Self {next:(&next>=lo && &next<=hi).then_some(next),period}
    }
    fn nearest(events:&[Self],forward:bool)->Option<Rational> {
        events.iter().filter_map(|event|event.next.as_ref())
            .min_by(|a,b|if forward {a.cmp(b)}else{b.cmp(a)}).cloned()
    }
    fn advance(&mut self,lo:&Rational,hi:&Rational,forward:bool) {
        self.next=self.next.take().and_then(|next|self.period.as_ref().map(|period|if forward {next+period}else{next-period}))
            .filter(|next|next>=lo && next<=hi);
    }
}

impl CompiledLayer {
    fn camera_events(&self, track:&PropertyTrack, lo:&Rational, hi:&Rational, forward:bool)->Result<Vec<CameraEvent>> {
        let speed=ieee(self.source.speed)?;
        if speed==Rational::from_integer(0.into()) {return Ok(Vec::new())}
        let offset=ieee(self.source.offset)?;
        let times=if let Some(keys)=&track.camera_keys {keys.as_slice()}
            else {track.discrete_keys.as_ref().unwrap().exact_keys()};
        let equal=|a:usize,b:usize|match &track.channel {
            Channel::Style(track)=>track.values[a]==track.values[b],
            Channel::Discrete {values,..}=>values[a]==values[b],
            Channel::Numeric(curve)=>curve.values[a]==curve.values[b],
        };
        let span=self.time_span.as_ref().map(|span|span.get()).filter(|(a,b)|self.source.looping && b>a);
        let mut keys=Vec::new();
        for i in 1..times.len() {
            if !equal(i-1,i) && span.is_none_or(|(start,end)|&times[i]>start && &times[i]<end) {
                keys.push(times[i].clone());
            }
        }
        if let Some((start,end))=span {
            let first=times.partition_point(|key|key<=start).saturating_sub(1);
            let last=times.partition_point(|key|key<end).saturating_sub(1);
            if !equal(first,last) {keys.push(start.clone());}
        }
        let period=span.map(|(start,end)|(end-start)/&speed)
            .map(|p|if p<Rational::from_integer(0.into()) {-p}else{p});
        Ok(keys.into_iter().map(|key|CameraEvent::new((key-&offset)/&speed,period.clone(),lo,hi,forward)).collect())
    }
}
impl Plan {
    /// Clamp to the connected camera ownership interval containing nominal.
    /// STEP controls and camera keys are lazy affine event streams. Numeric
    /// activation runs partition each STEP interval before camera events are
    /// considered, so inactive or overwritten tracks do no repeated-loop work.
    pub fn clamp_camera_request(&self,nominal:&Germ,target:&Germ,fallback:usize)->Result<Germ> {
        if target==nominal || !self.has_camera_tracks() {return Ok(target.clone())}
        let forward=target.compare(nominal).is_ge();
        let (lo,hi)=if forward {(nominal.value(),target.value())}else{(target.value(),nominal.value())};
        let low=Germ::point(lo.clone()).numeric()?.max(0.).next_down().max(0.);
        let high=Germ::point(hi.clone()).numeric()?.next_up().min(f32::MAX);
        let activity=self.camera_event_activity(low,high);
        let camera=|request:Germ|self.camera_at_request(&vork_time::Request::Exact(request),fallback);
        let owner=camera(nominal.clone())?;
        let dominant=self.layers.iter().enumerate().rfind(|(i,l)|activity[*i]==Some(true)
            && !matches!(l.source.blend,v2::Blend::Add) && l.tracks.iter().any(|t|t.camera_keys.is_some())).map(|(i,_)|i);
        let mut needed:std::collections::BTreeSet<_>=self.layers.iter().enumerate()
            .filter(|(i,layer)|activity[*i].is_none() && dominant.is_none_or(|last|*i>=last)
                && !matches!(layer.source.blend,v2::Blend::Add) && layer.tracks.iter().any(|t|t.camera_keys.is_some()))
            .map(|(i,_)|i).collect();
        let fixed_activity=needed.is_empty();
        for (i,layer) in self.layers.iter().enumerate().rev() {
            if layer.tracks.iter().any(|track|track.weight_target.is_some_and(|target|needed.contains(&target))) {needed.insert(i);}
        }
        let invariant=(dominant.is_some() || fallback==owner) && self.layers.iter().enumerate()
            .filter(|(i,layer)|activity[*i]!=Some(false) && dominant.is_none_or(|last|*i>=last) && !matches!(layer.source.blend,v2::Blend::Add))
            .filter_map(|(_,layer)|layer.tracks.iter().rev().find(|track|track.camera_keys.is_some()).map(|track|(layer,track)))
            .all(|(layer,track)| {
                let Channel::Discrete {values,..}=&track.channel else {unreachable!()};
                let times=track.camera_keys.as_ref().unwrap();
                let values=if let Some((start,end))=layer.time_span.as_ref().map(|span|span.get()).filter(|(a,b)|layer.source.looping && b>a) {
                    let first=times.partition_point(|key|key<=start).saturating_sub(1);
                    let last=times.partition_point(|key|key<end).saturating_sub(1);
                    &values[first..=last]
                }else{values.as_slice()};
                values.iter().all(|value|matches!(value,Sample::String(name) if self.cameras.get(name.as_ref())==Some(&owner)))
            });
        if invariant {return Ok(target.clone())}
        let check=|boundary:&Rational|->Result<Option<Germ>> {
            let point=Germ::point(boundary.clone());
            if (forward && point.compare(nominal).is_lt()) || (!forward && point.compare(nominal).is_gt()) {return Ok(None)}
            let incoming=if forward {Germ::left(boundary.clone())}else{Germ::right(boundary.clone())};
            let outgoing=if forward {Germ::right(boundary.clone())}else{Germ::left(boundary.clone())};
            Ok(if point.compare(nominal)==std::cmp::Ordering::Equal {
                (camera(outgoing)?!=owner).then(||nominal.clone())
            }else if camera(point.clone())?!=owner {Some(incoming)}
            else if camera(outgoing)?!=owner {Some(point)}else{None})
        };
        let mut controls=Vec::new();
        if !fixed_activity {
            for layer in &self.layers {
                for track in &layer.tracks {
                    if track.discrete_keys.is_some() && track.weight_target.is_some_and(|target|needed.contains(&target)) {controls.extend(layer.camera_events(track,lo,hi,forward)?);}
                }
            }
        }
        let mut cursor=nominal.value().clone();
        loop {
            let next=CameraEvent::nearest(&controls,forward);
            let end=next.as_ref().unwrap_or(if forward {hi}else{lo});
            let (a,b)=if forward {(&cursor,end)}else{(end,&cursor)};
            let mut boundaries=vec![cursor.clone(),end.clone()];
            if self.camera_weight_animation && !fixed_activity && a<b {
                let request=vork_time::Request::Exact(Germ::point((a+b)/Rational::from_integer(2.into())));
                let low=Germ::point(a.clone()).numeric()?.max(0.).next_down().max(0.);
                let high=Germ::point(b.clone()).numeric()?.next_up().min(f32::MAX);
                boundaries.extend(self.camera_weight_boundaries(low,high,Some(&request))?.into_iter()
                    .filter(|boundary|boundary>a && boundary<b));
            }
            boundaries.sort();boundaries.dedup();if !forward {boundaries.reverse();}
            if let Some(clamped)=check(&cursor)? {return Ok(clamped)}
            for span in boundaries.windows(2) {
                let (a,b)=if forward {(&span[0],&span[1])}else{(&span[1],&span[0])};
                let request=vork_time::Request::Exact(Germ::point((a+b)/Rational::from_integer(2.into())));
                let mut samples=self.samples();
                self.sample_where_request(request.numeric()?,Some(&request),&mut samples,|_|false)?;
                // Last active replace track owns the whole open activation run.
                let active=self.layers.iter().enumerate().rev().find_map(|(i,layer)| {
                    (samples.layer_weights[i]>=0.5 && !matches!(layer.source.blend,v2::Blend::Add))
                        .then(||layer.tracks.iter().rev().find(|t|t.camera_keys.is_some()).map(|track|(layer,track))).flatten()
                });
                if let Some((layer,track))=active {
                    let mut events=layer.camera_events(track,a,b,forward)?;
                    while let Some(event)=CameraEvent::nearest(&events,forward) {
                        if let Some(clamped)=check(&event)? {return Ok(clamped)}
                        for stream in &mut events {if stream.next.as_ref()==Some(&event) {stream.advance(a,b,forward);}}
                    }
                }
                if let Some(clamped)=check(&span[1])? {return Ok(clamped)}
            }
            let Some(next)=next else {break};
            for event in &mut controls {if event.next.as_ref()==Some(&next) {event.advance(lo,hi,forward);}}
            cursor=next;
        }
        Ok(target.clone())
    }
}

/// Only discrete consumers request rational keys; continuous numeric tracks
/// neither decode nor evaluate an exact curve.
pub(super) struct DiscreteKeys {
    source: Vec<vork_time::Clock>,
    keys: std::sync::OnceLock<Vec<Rational>>,
}
impl DiscreteKeys {
    pub(super) fn new(source:Vec<vork_time::Clock>)->Self {Self {source,keys:Default::default()}}
    fn exact_keys(&self)->&[Rational] {self.keys.get_or_init(||self.source.iter().map(vork_time::Clock::exact).collect())}
    pub(super) fn upper(&self,time:&Germ)->usize {
        self.exact_keys().partition_point(|key|time.compare_point(key).is_ge())
    }
    /// The key at which `upper` next changes for a clock now at `time`: the first key above it
    /// when the clock rises, the last key at or below it when it falls.
    pub(super) fn next(&self,time:&Germ,rising:bool)->Option<&Rational> {
        let upper=self.upper(time);
        if rising {self.exact_keys().get(upper)} else {upper.checked_sub(1).map(|key|&self.exact_keys()[key])}
    }
}

impl Plan {
    /// Include discrete ownership in pose reuse without keying on raw exact
    /// time: continuous tracks retain their numeric identity and reuse cost.
    pub fn append_discrete_key(&self,request:&vork_time::Request,out:&mut Vec<u8>)->Result<()> {
        if self.has_clocks() {self.append_clock_key(&request.exact()?,out)?;}
        if !self.discrete_tracks {return Ok(());}
        let mut exact=None;
        for layer in &self.layers {
            if layer.owner_node.is_some() {continue;}
            let mut local=None;
            for track in &layer.tracks {
                if let Some(keys)=&track.discrete_keys {
                    if exact.is_none() {exact=Some(request.exact()?);}
                    if local.is_none() {local=Some(layer.exact_local(exact.as_ref().unwrap())?);}
                    out.extend_from_slice(&(keys.upper(local.as_ref().unwrap()) as u64).to_le_bytes());
                    if let Channel::Style(style)=&track.channel {
                        if let Some(clocks)=&style.clocks {
                            for pair in clocks.windows(2) {
                                let midpoint=(pair[0].exact()+pair[1].exact())/Rational::from_integer(2.into());
                                out.push(u8::from(local.as_ref().unwrap().compare_point(&midpoint).is_ge()));
                            }
                        }
                    }

                }
            }
        }
        Ok(())
    }
}

impl Plan {
    fn offset_request(&self,request:&vork_time::Request,offset:f32)->Result<Option<vork_time::Request>> {
        Ok(if self.discrete_tracks && offset!=0. {
            Some(vork_time::Request::Exact(request.exact()?.add(&Germ::point(ieee(offset)?))))
        }else{None})
    }
    pub fn append_discrete_offset_key(&self,request:&vork_time::Request,offset:f32,out:&mut Vec<u8>)->Result<()> {
        let shifted=self.offset_request(request,offset)?;
        self.append_discrete_key(shifted.as_ref().unwrap_or(request),out)
    }
    pub fn sample_material_request(&self,request:&vork_time::Request,time:f32,offset:f32,samples:&mut Samples)->Result<()> {
        let shifted=self.offset_request(request,offset)?;
        self.sample_where_request(time+offset,Some(shifted.as_ref().unwrap_or(request)),samples,|track| {
            track.material_targets.iter().flatten().any(|&t|self.material_bindings[t].offset==offset)
        })
    }
    pub fn sample_nodes_request(&self,request:&vork_time::Request,time:f32,offset:f32,samples:&mut Samples,nodes:&[usize],assets:&[usize])->Result<()> {
        let shifted=self.offset_request(request,offset)?;
        self.sample_where_request(time+offset,Some(shifted.as_ref().unwrap_or(request)),samples,|track| {
            track.node_targets.iter().any(|binding|nodes.binary_search(&binding.node).is_ok())
                || track.asset_visibility.is_some_and(|(asset,_)|assets.binary_search(&asset).is_ok())
                || track.asset_targets.iter().any(|target|assets.binary_search(&target.asset).is_ok())
        })
    }
}

#[cfg(test)]
mod discrete_tests {
    use super::*;
    fn plan(layer:serde_json::Value)->Result<Plan> {
        let scene:v2::Scene=vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,"render":{"background":[0,0,0]},"nodes":[{"type":"camera","name":"cam"}],
            "programs":{"choices":{"source":"struct Inputs {float mode;};"}},
            "materials":{"m":{"program":"choices","inputs":{"mode":"old"}}},
            "animation":[layer]
        }))?;
        Plan::compile(&scene,std::path::Path::new("."))
    }
    fn tracks()->serde_json::Value {serde_json::json!([
        {"path":["nodes","cam","visible"],"keys":[[0,true],[0.5,false]]},
        {"path":["materials","m","inputs","mode"],"keys":[[0,"old"],[0.5,"new"]]},
        {"path":["render","background"],"interpolation":"STEP","keys":[[0,[1,0,0]],[0.5,[0,0,1]]]},
        {"path":["nodes","cam","translation"],"keys":[[0,[0,0,0]],[1,[2,0,0]]]}
    ])}
    #[test]
    fn one_sided_discrete_keys_leave_continuous_evaluation_at_endpoint()->Result<()> {
        let plan=plan(serde_json::json!({"tracks":tracks()}))?;
        let mut samples=plan.samples();
        let mut identities=Vec::new();
        for (time,old) in [(Germ::left(decimal("1/2")?),true),(Germ::point(decimal("1/2")?),false),(Germ::right(decimal("1/2")?),false)] {
            let request=vork_time::Request::Exact(time);
            plan.sample_request_into(request.numeric()?,&request,&mut samples)?;
            assert_eq!(samples.values[0],Sample::Bool(old));
            assert_eq!(samples.values[1],Sample::String(if old {"old"}else{"new"}.into()));
            assert_eq!(samples.values[2],Sample::Numeric(if old {vec![1.,0.,0.]}else{vec![0.,0.,1.]}));
            assert_eq!(samples.values[3],Sample::Numeric(vec![1.,0.,0.]));
            let mut identity=Vec::new();plan.append_discrete_key(&request,&mut identity)?;identities.push(identity);
        }
        assert_ne!(identities[0],identities[1]);assert_eq!(identities[1],identities[2]);
        Ok(())
    }
    #[test]
    fn discrete_reverse_loop_constant_and_offset_own_the_incoming_side()->Result<()> {
        let half=decimal("1/2")?;
        for (layer,time,old) in [
            (serde_json::json!({"speed":-1,"offset":1,"tracks":tracks()}),Germ::left(half.clone()),false),
            (serde_json::json!({"loop":true,"tracks":tracks()}),Germ::left(decimal("1")?),false),
            (serde_json::json!({"loop":true,"tracks":tracks()}),Germ::point(decimal("1")?),true),
            (serde_json::json!({"speed":0,"offset":0.5,"tracks":tracks()}),Germ::left(half.clone()),false),
        ] {
            let plan=plan(layer)?;let request=vork_time::Request::Exact(time);let mut samples=plan.samples();
            plan.sample_request_into(request.numeric()?,&request,&mut samples)?;
            assert_eq!(samples.values[0],Sample::Bool(old));
        }
        let plan=plan(serde_json::json!({"tracks":tracks()}))?;
        let request=vork_time::Request::Exact(Germ::left(decimal("1/4")?));
        let shifted=plan.offset_request(&request,0.25)?.unwrap();
        let mut samples=plan.samples();plan.sample_request_into(0.5,&shifted,&mut samples)?;
        assert_eq!(samples.values[0],Sample::Bool(true));
        assert_eq!(shifted.previous(vork_time::Time::integer(24))?.exact()?,Germ::left(decimal("11/24")?));
        Ok(())
    }
    #[test]
    fn distinct_discrete_keys_can_share_numeric_time()->Result<()> {
        let end="500000000000000000001/1000000000000000000000";
        let plan=plan(serde_json::json!({"tracks":[
            {"path":["nodes","cam","visible"],"keys":[["1/2",true],[end,false]]},
            {"path":["render","background"],"interpolation":"STEP","keys":[["1/2",[1,0,0]],[end,[0,0,1]]]}
        ]}))?;
        let mut samples=plan.samples();
        for (time,old) in [(Germ::left(decimal(end)?),true),(Germ::point(decimal(end)?),false)] {
            let request=vork_time::Request::Exact(time);
            assert_eq!(request.numeric()?,0.5);
            plan.sample_request_into(0.5,&request,&mut samples)?;
            assert_eq!(samples.values[0],Sample::Bool(old));
            assert_eq!(samples.values[1],Sample::Numeric(if old {vec![1.,0.,0.]}else{vec![0.,0.,1.]}));
        }
        Ok(())
    }
    #[test]
    fn huge_camera_loops_skip_invariant_inactive_and_overwritten_ownership()->Result<()> {
        let camera=|keys:serde_json::Value|serde_json::json!({"path":["render","camera"],"keys":keys});
        for animation in [
            serde_json::json!([{"loop":true,"speed":1e12,"tracks":[camera(serde_json::json!([[0,"b"],[1,"b"]]))]}]),
            // The different key at the excluded loop end is never selected.
            serde_json::json!([{"loop":true,"speed":1e12,"tracks":[camera(serde_json::json!([[0,"b"],[1,"a"]]))]}]),
            serde_json::json!([{"loop":true,"speed":1e12,"weight":0,"tracks":[camera(serde_json::json!([[0,"a"],[0.5,"b"],[1,"a"]]))]}]),
            serde_json::json!([
                {"loop":true,"speed":1e12,"tracks":[camera(serde_json::json!([[0,"a"],[0.5,"b"],[1,"a"]]))]},
                {"tracks":[camera(serde_json::json!([[0,"b"]]))]}
            ]),
            // Repeated activation cannot expose the excluded loop-end key.
            serde_json::json!([
                {"loop":true,"speed":1e12,"tracks":[{"path":["animation","cut","weight"],"interpolation":"STEP","keys":[[0,0],[0.5,1],[1,0]]}]},
                {"name":"cut","loop":true,"speed":1e12,"tracks":[camera(serde_json::json!([[0,"a"],[1,"b"]]))]}
            ]),
            // Camera activation changes, but both possible owners select a.
            serde_json::json!([
                {"loop":true,"speed":1e12,"tracks":[{"path":["animation","cut","weight"],"interpolation":"STEP","keys":[[0,0],[0.5,1],[1,0]]}]},
                {"name":"cut","tracks":[camera(serde_json::json!([[0,"a"]]))]}
            ]),
        ] {
            let scene:v2::Scene=vork_scene::v2::from_test_value(serde_json::json!({
                "version":2,"render":{"camera":"a"},"nodes":[{"type":"camera","name":"a"},{"type":"camera","name":"b"}],"animation":animation
            }))?;
            let mut plan=Plan::compile(&scene,std::path::Path::new("."))?;
            plan.cameras=[("a".into(),0),("b".into(),1)].into_iter().collect();
            let a=Germ::point(decimal("0")?);let b=Germ::point(decimal("1/24")?);
            assert_eq!(plan.clamp_camera_request(&a,&b,0)?,b);
            assert_eq!(plan.clamp_camera_request(&b,&a,0)?,a);
        }
        Ok(())
    }
    #[test]
    fn fast_camera_keys_are_skipped_until_animated_activation()->Result<()> {
        let scene:v2::Scene=vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,"render":{"camera":"a"},"nodes":[{"type":"camera","name":"a"},{"type":"camera","name":"b"}],
            "animation":[
                {"tracks":[{"path":["animation","cut","weight"],"keys":[[0,0],[1,1]]}]},
                {"name":"cut","loop":true,"speed":1e12,"tracks":[{"path":["render","camera"],"keys":[[0,"a"],[0.5,"b"],[1,"a"]]}]}
            ]
        }))?;
        let mut plan=Plan::compile(&scene,std::path::Path::new("."))?;
        plan.cameras=[("a".into(),0),("b".into(),1)].into_iter().collect();
        let cut=plan.clamp_camera_request(&Germ::point(decimal("0")?),&Germ::point(decimal("1")?),0)?;
        assert!(cut.compare_point(&decimal("0.49")?).is_gt());
        assert!(cut.compare_point(&decimal("0.51")?).is_lt());
        Ok(())
    }

    #[test]
    fn huge_camera_loops_stop_at_the_nearest_exact_cut()->Result<()> {
        for speed in [1e12f32,-1e12f32] {
            let scene:v2::Scene=vork_scene::v2::from_test_value(serde_json::json!({
                "version":2,"render":{"camera":"a"},"nodes":[{"type":"camera","name":"a"},{"type":"camera","name":"b"}],
                "animation":[{"loop":true,"speed":speed,"tracks":[{"path":["render","camera"],"keys":[[0,"a"],[0.5,"b"],[1,"a"]]}]}]
            }))?;
            let mut plan=Plan::compile(&scene,std::path::Path::new("."))?;
            plan.cameras=[("a".into(),0),("b".into(),1)].into_iter().collect();
            let speed=ieee(speed.abs())?;
            let quarter=decimal("1/4")?/&speed;let half=decimal("1/2")?/&speed;
            let start=Germ::point(quarter);
            let far=Germ::point(decimal("1/24")?);
            // Reverse affine time selects b on the incoming side of the same cut.
            let expected=if plan.layers[0].source.speed>0. {Germ::left(half)}else{Germ::point(half)};
            assert_eq!(plan.clamp_camera_request(&start,&far,0)?,expected);
            let reverse_start=Germ::point(decimal("3/4")?/&speed);
            let expected=if plan.layers[0].source.speed>0. {Germ::point(decimal("1/2")?/&speed)}else{Germ::right(decimal("1/2")?/&speed)};
            assert_eq!(plan.clamp_camera_request(&reverse_start,&Germ::point(decimal("0")?),0)?,expected);
        }
        Ok(())
    }

    #[test]
    fn collapsed_numeric_step_loop_still_changes_exact_camera_ownership()->Result<()> {
        let scene:v2::Scene=vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,"render":{"camera":"a"},"nodes":[{"type":"camera","name":"a"},{"type":"camera","name":"b"}],
            "animation":[
                {"loop":true,"tracks":[{"path":["animation","cut","weight"],"interpolation":"STEP","keys":[["1/2",0],["1000000001/2000000000",1],["500000001/1000000000",0]]}]},
                {"name":"cut","weight":0,"tracks":[{"path":["render","camera"],"keys":[[0,"b"]]}]}
            ]
        }))?;
        let mut plan=Plan::compile(&scene,std::path::Path::new("."))?;
        plan.cameras=[("a".into(),0),("b".into(),1)].into_iter().collect();
        assert_eq!(plan.layers[0].span,Some((0.5,0.5)));
        assert_eq!(plan.clamp_camera_request(&Germ::point(decimal("0")?),&Germ::point(decimal("1/24")?),0)?,
            Germ::left(decimal("1/2000000000")?));
        Ok(())
    }

    #[test]
    fn step_weight_ownership_partitions_numeric_camera_crossings()->Result<()> {
        let end="500000001/1000000000";
        let scene:v2::Scene=vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,"render":{"camera":"a"},"nodes":[{"type":"camera","name":"a"},{"type":"camera","name":"b"}],
            "animation":[
                {"tracks":[{"path":["animation","driver","weight"],"interpolation":"STEP","keys":[[0,1],[end,0]]}]},
                {"name":"driver","weight":0,"tracks":[{"path":["animation","cut","weight"],"keys":[[0,0],[1,1]]}]},
                {"name":"cut","weight":0,"tracks":[{"path":["render","camera"],"keys":[[0,"b"]]}]}
            ]
        }))?;
        let mut plan=Plan::compile(&scene,std::path::Path::new("."))?;
        plan.cameras=[("a".into(),0),("b".into(),1)].into_iter().collect();
        let before=Germ::point(decimal("0.49")?);let after=Germ::point(decimal("0.51")?);
        let midpoint=(ieee(0.5f32.next_down())?+ieee(0.5)?)/Rational::from_integer(2.into());
        assert_eq!(plan.clamp_camera_request(&before,&after,0)?,Germ::left(midpoint));
        assert_eq!(plan.camera_at_request(&vork_time::Request::Exact(Germ::point(decimal("1/2")?)),0)?,1);
        assert_eq!(plan.camera_at_request(&vork_time::Request::Exact(Germ::left(decimal(end)?)),0)?,1);
        assert_eq!(plan.camera_at_request(&vork_time::Request::Exact(Germ::point(decimal(end)?)),0)?,0);
        Ok(())
    }

}

impl Plan {
    pub(super) fn weight_step_keys(&self,request:&vork_time::Request)->Result<Vec<Option<usize>>> {
        let mut keys=Vec::new();let mut exact=None;
        for layer in &self.layers {
            let mut local=None;
            for track in &layer.tracks {
                keys.push(if let Some(keys)=track.discrete_keys.as_ref().filter(|_|track.weight_target.is_some()) {
                    if exact.is_none() {exact=Some(request.exact()?);}
                    if local.is_none() {local=Some(layer.exact_local(exact.as_ref().unwrap())?);}
                    Some(keys.upper(local.as_ref().unwrap()).saturating_sub(1))
                }else{None});
            }
        }
        Ok(keys)
    }
}

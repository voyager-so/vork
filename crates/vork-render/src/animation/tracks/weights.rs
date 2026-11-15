//! Bounds on the existing scalar IEEE evaluator, used only for camera weights.
//! The operations round exactly where numeric animation rounds. A whole input
//! range with a fixed activation can be skipped; ambiguous ranges are split.
use super::*;
#[derive(Clone, Copy, Debug)]
struct Range(f32, f32);
impl Range {
    fn point(x:f32)->Self {Self(x,x)}
    fn union(self,b:Self)->Self {Self(self.0.min(b.0),self.1.max(b.1))}
    fn map(self,f:impl Fn(f32)->f32)->Self {Self(f(self.0),f(self.1))}
    fn full()->Self {Self(f32::NEG_INFINITY,f32::INFINITY)}
    fn corners(self,b:Self,f:impl Fn(f32,f32)->f32)->Self {
        let values=[f(self.0,b.0),f(self.0,b.1),f(self.1,b.0),f(self.1,b.1)];
        if values.iter().any(|x|x.is_nan()) {return Self::full()}
        Self(values.into_iter().fold(f32::INFINITY,f32::min),values.into_iter().fold(f32::NEG_INFINITY,f32::max))
    }
}
impl std::ops::Add for Range {type Output=Self;fn add(self,b:Self)->Self {self.corners(b,|a,b|a+b)}}
impl std::ops::Sub for Range {type Output=Self;fn sub(self,b:Self)->Self {self.corners(b,|a,b|a-b)}}
impl std::ops::Mul for Range {type Output=Self;fn mul(self,b:Self)->Self {self.corners(b,|a,b|a*b)}}
impl Curve {
    fn scalar_range(&self,time:Range)->Range {
        if time.0==time.1 {
            let mut value=Vec::new();self.sample(time.0,&mut value);
            return Range::point(value[0]);
        }
        let first=self.times.partition_point(|&t|t<=time.0);
        let last=self.times.partition_point(|&t|t<=time.1);
        let mut result=None;
        for upper in first..=last {
            let a=upper.saturating_sub(1);let b=upper.min(self.times.len()-1);
            let cubic=self.interpolation==Interpolation::Cubic;
            let value=|i:usize,k:usize|Range::point(self.values[i*if cubic{3}else{1}+k]);
            let part=if self.interpolation==Interpolation::Step {value(a,0)} else {
                let dt=self.times[b]-self.times[a];
                let local=Range(time.0.max(self.times[a]),time.1.min(self.times[b]));
                let u=if dt>0. {(local-Range::point(self.times[a])).map(|v|(v/dt).clamp(0.,1.))}else{Range::point(0.)};
                if cubic && value(a,1).0==value(b,1).0
                    && value(a,2).0==0. && value(b,0).0==0.
                    && value(a,1).0.is_finite() && value(a,1).0.to_bits()&0x007f_ffff==0 {
                    // h00 and h01 use opposite rounded differences. Their
                    // sum (and power-of-two-scaled sum) rounds exactly to one.
                    value(a,1)
                } else if cubic {
                    let [h00,h10,h01,h11]=hermite_basis(u,Range::point(1.),Range::point(2.),Range::point(3.));
                    h00*value(a,1)+h10*Range::point(dt)*value(a,2)+h01*value(b,1)+h11*Range::point(dt)*value(b,0)
                }else{
                    let x=value(a,0);let y=value(b,0);
                    // Equal power-of-two endpoints sum exactly under IEEE
                    // rounding: scaling (1-u) and u preserves complementarity.
                    if x.0==y.0 && x.0.is_finite() && x.0.to_bits()&0x007f_ffff==0 {x}
                    else {
                        let mut mixed=x*(Range::point(1.)-u)+y*u;
                        // For fixed u in [0,1], this expression is monotonic in
                        // both endpoints. Equal 0.5 endpoints give exactly 0.5;
                        // equal predecessor endpoints stay strictly below it.
                        // Thus a segment wholly on either side of the camera
                        // threshold cannot cross it through interpolation.
                        if x.0>=0.5 && y.0>=0.5 {mixed.0=mixed.0.max(0.5);}
                        if x.0<0.5 && y.0<0.5 {mixed.1=mixed.1.min(0.5f32.next_down());}
                        mixed
                    }
                }
            };
            result=Some(result.map_or(part,|r:Range|r.union(part)));
        }
        result.unwrap()
    }
}
impl Plan {
    fn layer_weight_ranges(&self,time:Range,step_keys:Option<&[Option<usize>]>)->Vec<Range> {
        let mut weights:Vec<_>=self.layers.iter().map(|l|Range::point(l.source.weight)).collect();
        let mut first=0;
        for (i,layer) in self.layers.iter().enumerate() {
            let keys=step_keys.map(|keys|&keys[first..first+layer.tracks.len()]);
            first+=layer.tracks.len();
            let weight=weights[i];
            if weight.0==0. && weight.1==0. {continue}
            let mut local=Range::point(layer.source.offset)+time*Range::point(layer.source.speed);
            if layer.source.looping {
                if let Some((a,b))=layer.span.filter(|(a,b)|b>a) {
                    // Remainder is monotonic between wraps. Compare the rounded
                    // subtraction actually used by the numeric evaluator.
                    let x=local-Range::point(a);let period=b-a;
                    let low=x.0.rem_euclid(period);let high=x.1.rem_euclid(period);
                    let width=(x.1 as f64-x.0 as f64).next_up();
                    local=Range::point(a)+if width<period as f64 && low<=high {
                        Range(low,high)
                    }else{Range(0.,period)};
                }
            }
            for (index,track) in layer.tracks.iter().enumerate() {
                if let Some(target)=track.weight_target {
                    let Channel::Numeric(curve)=&track.channel else {unreachable!()};
                    let value=keys.and_then(|keys|keys[index]).map_or_else(|| {
                        // Exact STEP loops can have distinct keys/spans that
                        // collapse in the numeric evaluator. Until an exact
                        // interval fixes their selection, bound every value.
                        if track.discrete_keys.is_some() {
                            curve.values.iter().map(|&value|Range::point(value)).reduce(Range::union).unwrap()
                        }else{curve.scalar_range(local)}
                    },|key|Range::point(curve.values[key]));
                    weights[target]=weights[target]+weight*if matches!(layer.source.blend,v2::Blend::Add) {value}else{value-weights[target]};
                }
            }
        }
        weights
    }
    fn camera_weight_ranges(&self,time:Range,step_keys:Option<&[Option<usize>]>)->Vec<Range> {
        self.layers.iter().zip(self.layer_weight_ranges(time,step_keys)).filter(|(l,_)|l.tracks.iter().any(|t|t.camera_keys.is_some())).map(|(_,w)|w).collect()
    }
    pub(super) fn camera_event_activity(&self, low:f32, high:f32)->Vec<Option<bool>> {
        self.layer_weight_ranges(Range(low,high),None).into_iter().map(|w| {
            if !w.0.is_finite() || !w.1.is_finite() {None}
            else if w.0>=0.5 {Some(true)} else if w.1<0.5 {Some(false)} else {None}
        }).collect()
    }
}
impl Plan {
    /// Compress the finite numeric input domain into activation runs. Each
    /// accepted range proves the same threshold result for every IEEE input;
    /// subdivision reaches the original evaluator at an individual input.
    pub(super) fn camera_weight_boundaries(&self,low:f32,high:f32,step_request:Option<&vork_time::Request>)->Result<Vec<vork_time::exact::Rational>> {
        use vork_time::exact::{Rational,ieee};
        let step_keys=step_request.map(|request|self.weight_step_keys(request)).transpose()?;
        let mut pending=vec![(low.to_bits(),high.min(f32::MAX).to_bits())];
        let mut previous:Option<Vec<bool>>=None;
        let mut result=Vec::new();
        let mut samples=self.samples();
        while let Some((a,b))=pending.pop() {
            let low=f32::from_bits(a);let high=f32::from_bits(b);
            let signature=if a==b {
                self.sample_where_request(low,step_request,&mut samples,|_|false)?;
                ensure!(samples.layer_weights.iter().all(|w|w.is_finite()),"numeric animation must be finite");
                Some(self.layers.iter().enumerate().filter(|(_,l)|l.tracks.iter().any(|t|t.camera_keys.is_some())).map(|(i,_)|samples.layer_weights[i]>=0.5).collect::<Vec<_>>())
            }else{
                self.camera_weight_ranges(Range(low,high),step_keys.as_deref()).into_iter().map(|w| {
                    if !w.0.is_finite() || !w.1.is_finite() {None}
                    else if w.0>=0.5 {Some(true)}else if w.1<0.5 {Some(false)}else{None}
                }).collect::<Option<Vec<_>>>()
            };
            if let Some(signature)=signature {
                if previous.as_ref().is_some_and(|p|*p!=signature) {
                    result.push((ieee(low.next_down())?+ieee(low)?)/Rational::from_integer(2.into()));
                }
                previous=Some(signature);
            }else{
                let mid=a+(b-a)/2;
                pending.push((mid+1,b));pending.push((a,mid));
            }
        }
        Ok(result)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use vork_time::{Request,exact::{Germ,decimal,ieee,Rational}};
    fn plan(keys:serde_json::Value, interpolation:&str)->Result<Plan> {
        let scene:v2::Scene=vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,"render":{"camera":"a"},
            "nodes":[{"type":"camera","name":"a"},{"type":"camera","name":"b"}],
            "animation":[{"tracks":[{"path":["animation","cut","weight"],"keys":keys,"interpolation":interpolation}]},
                {"name":"cut","weight":0,"tracks":[{"path":["render","camera"],"keys":[[0,"b"]]}]}]
        }))?;
        let mut plan=Plan::compile(&scene,std::path::Path::new("."))?;
        plan.cameras=[("a".into(),0),("b".into(),1)].into_iter().collect();
        Ok(plan)
    }
    #[test]
    fn first_frame_weight_interval_skips_a_billion_input_cells()->Result<()> {
        for (value,interpolation) in [(0.,"LINEAR"),(0.5f32.next_down(),"LINEAR"),(0.5,"LINEAR"),(1.,"LINEAR"),(0.5,"CUBICSPLINE")] {
            let plan=plan(serde_json::json!([[0,value],[1,value]]),interpolation)?;
            let zero=Germ::point(decimal("0")?);let frame=Germ::point(decimal("1/24")?);
            assert!(frame.numeric()?.to_bits()>1_000_000_000);
            let bounds=plan.camera_weight_ranges(Range(0.,frame.numeric()?.next_up()),None);
            assert!(bounds.iter().all(|w|w.0>=0.5 || w.1<0.5),"stable interval must prune before subdivision: {bounds:?}");
            assert!(plan.camera_weight_boundaries(0.,frame.numeric()?.next_up(),None)?.is_empty());
            assert_eq!(plan.clamp_camera_request(&frame,&zero,0)?,zero);
            assert_eq!(plan.clamp_camera_request(&zero,&frame,0)?,frame);
        }
        Ok(())
    }
    #[test]
    fn full_interval_cubic_and_linear_find_interior_camera_cuts()->Result<()> {
        for (keys,interpolation) in [(serde_json::json!([[0,0],[1,1]]),"LINEAR"),
            (serde_json::json!([[0,0,0,4],[1,0,-4,0]]),"CUBICSPLINE")] {
            let plan=plan(keys,interpolation)?;
            for (a,b) in [("0","1"),("1","0")] {
                let a=Germ::point(decimal(a)?);let b=Germ::point(decimal(b)?);
                let clamped=plan.clamp_camera_request(&a,&b,0)?;
                assert_ne!(clamped,b);
                assert_eq!(plan.camera_at_request(&Request::Exact(a),0)?,plan.camera_at_request(&Request::Exact(clamped),0)?);
            }
        }
        Ok(())
    }
    #[test]
    fn range_search_matches_exhaustive_numeric_cells_near_crossings()->Result<()> {
        for (keys,interpolation,center) in [
            (serde_json::json!([[0,0],[1,1]]),"LINEAR",0.5f32),
            (serde_json::json!([[0,0,0,4],[1,0,-4,0]]),"CUBICSPLINE",0.1464466),
            (serde_json::json!([[0,0,0,4],[1,0,-4,0]]),"CUBICSPLINE",0.8535534),
        ] {
            let plan=plan(keys,interpolation)?;
            let lo=center.to_bits()-2048;let hi=center.to_bits()+2048;
            let mut samples=plan.samples();let mut previous=None;let mut expected=Vec::new();
            for bits in lo..=hi {
                let t=f32::from_bits(bits);plan.sample_into_where(t,&mut samples,|_|false);
                let active=samples.layer_weights[1]>=0.5;
                if previous.is_some_and(|p|p!=active) {
                    expected.push((ieee(t.next_down())?+ieee(t)?)/Rational::from_integer(2.into()));
                }
                previous=Some(active);
            }
            assert_eq!(plan.camera_weight_boundaries(f32::from_bits(lo),f32::from_bits(hi),None)?,expected);
        }
        Ok(())
    }
    #[test]
    fn numeric_bounds_enclose_reverse_loops_and_ordered_weight_blends()->Result<()> {
        let scene:v2::Scene=vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,"render":{"camera":"a"},"nodes":[{"type":"camera","name":"a"},{"type":"camera","name":"b"}],
            "animation":[
                {"loop":true,"offset":1,"speed":-2,"tracks":[{"path":["animation","driver","weight"],"keys":[[0,0.2],[0.4,0.8],[1,0.1]]}]},
                {"name":"driver","weight":0,"tracks":[{"path":["animation","cut","weight"],"keys":[[0,0,0,4],[1,0,-4,0]],"interpolation":"CUBICSPLINE"}]},
                {"blend":"combine","weight":0.25,"tracks":[{"path":["animation","cut","weight"],"keys":[[0,0.1],[1,0.7]]}]},
                {"name":"cut","weight":0,"tracks":[{"path":["render","camera"],"keys":[[0,"b"]]}]}
            ]
        }))?;
        let plan=Plan::compile(&scene,std::path::Path::new("."))?;
        let mut samples=plan.samples();
        for i in 0..200 {
            let low=i as f32/100.;let high=low+0.02;
            let bound=plan.camera_weight_ranges(Range(low,high),None)[0];
            for j in 0..=100 {
                let t=low+(high-low)*(j as f32/100.);
                plan.sample_into_where(t,&mut samples,|_|false);
                let value=samples.layer_weights[3];
                assert!(value>=bound.0 && value<=bound.1,"{t}: {value} outside {bound:?}");
            }
        }
        Ok(())
    }

}

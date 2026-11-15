//! Finite f32 enclosures for medium certification, using the effect interpreter's
//! resolved calls, stores, immutable words, branch joins and fixed-loop execution.
//! Unsupported operations stay unknown; resource validation stubs are never proofs.
use super::*;

fn bounds(v: &Value) -> Option<(f64, f64)> {
    let (mut a,mut b)=match *v {
        Value::Integer(n) => (n as f64,n as f64),
        Value::IntegerRange(a,b) if a<=b => (a as f64,b as f64),
        Value::Number(n) if n.is_finite() => (n,n),
        Value::NumberRange(a,b) if a.is_finite() && b.is_finite() && a<=b => (a,b),
        _ => return None,
    };
    // Shader inputs and literal operands can flush too, before an operation.
    if a>0. && a<f64::from(f32::MIN_POSITIVE) {a=0.;}
    if b<0. && b> -f64::from(f32::MIN_POSITIVE) {b=0.;}
    Some((a,b))
}

fn enclosure(lo:f64,hi:f64)->Value {enclosure_ulps(lo,hi,1)}
fn enclosure_ulps(lo: f64, hi: f64, ulps:usize) -> Value {
    if !lo.is_finite() || !hi.is_finite() || lo>hi {return Value::Unknown;}
    // Round outward at every operation, including possible flush-to-zero. The
    // additional representable neighbour also covers f64 endpoint rounding.
    let mut a=lo as f32;let mut b=hi as f32;
    for _ in 0..ulps {a=a.next_down();b=b.next_up();}
    if !a.is_finite() || !b.is_finite() {return Value::Unknown;}
    let lo=f64::from(a);let hi=f64::from(b);
    Value::NumberRange(if lo>0. && lo<f64::from(f32::MIN_POSITIVE) {0.}else{lo},
        if hi<0. && hi> -f64::from(f32::MIN_POSITIVE) {0.}else{hi})
}
pub(super) fn binary(op:naga::BinaryOperator,a:Value,b:Value) -> Value {
    use naga::BinaryOperator as B;
    match (a,b) {
        (Value::Parts(a),Value::Parts(b)) if a.len()==b.len()=>Value::Parts(a.into_iter().zip(b).map(|(a,b)|binary(op,a,b)).collect()),
        (Value::Parts(a),b)=>Value::Parts(a.into_iter().map(|a|binary(op,a,b.clone())).collect()),
        (a,Value::Parts(b))=>Value::Parts(b.into_iter().map(|b|binary(op,a.clone(),b)).collect()),
        (a,b)=> {
            let (Some((a,b)),Some((c,d)))=(bounds(&a),bounds(&b)) else {return Value::Unknown;};
            if a==b && c==d {
                if let value @ Value::Number(_)=exact_f32(op,a,c) {return value;}
            }
            if op==B::Multiply && ((a==0. && b==0.) || (c==0. && d==0.)) {return Value::Number(0.);}
            match op {
                B::Add=>enclosure(a+c,b+d), B::Subtract=>enclosure(a-d,b-c),
                B::Multiply | B::Divide=> {
                    if op==B::Divide && (c<=0. && d>=0. || c.abs().min(d.abs())<f64::from(f32::MIN_POSITIVE) || c.abs().max(d.abs())>2f64.powi(126)) {return Value::Unknown;}
                    let values=if op==B::Multiply {[a*c,a*d,b*c,b*d]}else{[a/c,a/d,b/c,b/d]};
                    // Vulkan OpFDiv permits 2.5 ULP, not correctly rounded division.
                    enclosure_ulps(values.into_iter().fold(f64::INFINITY,f64::min),values.into_iter().fold(f64::NEG_INFINITY,f64::max),if op==B::Divide {4}else{1})
                },
                B::Equal if a==b && c==d && a==c=>Value::Bool(true),
                B::Equal if b<c || d<a=>Value::Bool(false),
                B::NotEqual if a==b && c==d && a==c=>Value::Bool(false),
                B::NotEqual if b<c || d<a=>Value::Bool(true),
                B::LessEqual if b<=c=>Value::Bool(true), B::LessEqual if a>d=>Value::Bool(false),
                B::GreaterEqual if a>=d=>Value::Bool(true), B::GreaterEqual if b<c=>Value::Bool(false),
                B::Less if b<c=>Value::Bool(true), B::Less if a>=d=>Value::Bool(false),
                B::Greater if a>d=>Value::Bool(true),B::Greater if b<=c=>Value::Bool(false),
                _=>Value::Unknown,
            }
        }
    }
}
fn extrema(a:Value,b:Value,maximum:bool)->Value {
    match (a,b) {
        (Value::Parts(a),Value::Parts(b)) if a.len()==b.len()=>Value::Parts(a.into_iter().zip(b).map(|(a,b)|extrema(a,b,maximum)).collect()),
        (Value::Parts(a),b)=>Value::Parts(a.into_iter().map(|a|extrema(a,b.clone(),maximum)).collect()),
        (a,Value::Parts(b))=>Value::Parts(b.into_iter().map(|b|extrema(a.clone(),b,maximum)).collect()),
        (a,b)=> {
            let integers=matches!(a,Value::Integer(_)|Value::IntegerRange(..)) && matches!(b,Value::Integer(_)|Value::IntegerRange(..));
            let (Some((a,b)),Some((c,d)))=(bounds(&a),bounds(&b)) else {return Value::Unknown;};
            let (lo,hi)=if maximum {(a.max(c),b.max(d))}else{(a.min(c),b.min(d))};
            if integers {
                if lo==hi {Value::Integer(lo as i64)}else{Value::IntegerRange(lo as i64,hi as i64)}
            }else if lo==hi {Value::Number(lo)}else{Value::NumberRange(lo,hi)}
        }
    }
}
fn clamp(value:Value,lo:Value,hi:Value)->Value {
    match (&lo,&hi) {
        (Value::Parts(a),Value::Parts(b)) if a.len()==b.len()=> {
            let Value::Parts(v)=value else {return Value::Unknown;};
            if v.len()!=a.len() {return Value::Unknown;}
            Value::Parts(v.into_iter().zip(a.iter().zip(b)).map(|(v,(a,b))|clamp(v,a.clone(),b.clone())).collect())
        },
        _=> {
            if !matches!((bounds(&lo),bounds(&hi)),(Some((_,a)),Some((b,_))) if a<=b) {return Value::Unknown;}
            extrema(extrema(value,lo,true),hi,false)
        }
    }
}
fn floor(v:Value)->Value {
    match v {
        Value::Parts(v)=>Value::Parts(v.into_iter().map(floor).collect()),
        v=>bounds(&v).map_or(Value::Unknown,|(a,b)| {
            let (a,b)=(a.floor(),b.floor());
            if a==b {Value::Number(a)}else{Value::NumberRange(a,b)}
        }),
    }
}
fn negate(v:Value)->Value {
    match v {
        Value::Parts(v)=>Value::Parts(v.into_iter().map(negate).collect()),
        v=>bounds(&v).map_or(Value::Unknown,|(a,b)|Value::NumberRange(-b,-a)),
    }
}
// Vulkan GLSL.std.450 sqrt inherits 1/inversesqrt: inverse sqrt has
// 2 ULP error, division 2.5 ULP. Enclose both operations, rather than
// adding an empirical percentage. This also encloses Metal fast sqrt's
// x*rsqrt(x) implementation (rsqrt <=2 ULP; multiplication rounded).
// https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation
fn square_root(v:Value)->Value {
    match v {
        Value::Parts(v)=>Value::Parts(v.into_iter().map(square_root).collect()),
        v=> {
            let Some((a,b))=bounds(&v) else {return Value::Unknown;};
            if a<0. {return Value::Unknown;}
            if b==0. {return Value::NumberRange(0.,f64::from(f32::MIN_POSITIVE));}
            let endpoint=|x:f64| {
                let inverse=1./x.sqrt();
                let inv=enclosure_ulps(inverse,inverse,3);
                binary(naga::BinaryOperator::Divide,Value::Number(1.),inv)
            };
            let Some((_,hi))=bounds(&endpoint(b)) else {return Value::Unknown;};
            let lo=if a==0. {0.}else {
                let Some((lo,_))=bounds(&endpoint(a)) else {return Value::Unknown;};lo
            };
            Value::NumberRange(lo.max(0.),hi)
        }
    }
}
// Vulkan exp/exp2 permit (3 + 2*abs(x)) ULP; precise Metal permits
// 4 ULP. One absolute error covers the entire operand interval, including
// changes in output ULP size at powers of two. Endpoint-only ULP stepping
// would not establish that property for the lower endpoint.
// https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation
// https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
fn exponential(v:Value)->Value {exponential_base(v,false)}
fn exponential_base(v:Value,base_two:bool)->Value {
    match v {
        Value::Parts(v)=>Value::Parts(v.into_iter().map(|v|exponential_base(v,base_two)).collect()),
        v=> {
            let Some((a,b))=bounds(&v) else {return Value::Unknown;};
            let minimum=if base_two {-126.}else{f64::from(f32::MIN_POSITIVE).ln()};
            let maximum=if base_two {f64::from(f32::MAX).log2()}else{f64::from(f32::MAX).ln()};
            if a<minimum || b>maximum {return Value::Unknown;}
            let lo=if base_two {a.exp2()}else{a.exp()};
            let hi=if base_two {b.exp2()}else{b.exp()};
            let rounded=hi as f32;
            if !rounded.is_finite() {return Value::Unknown;}
            let ulp=f64::from(rounded.next_up())-f64::from(rounded);
            let error=((3.+2.*a.abs().max(b.abs())).ceil().max(4.)+1.)*ulp;
            enclosure(lo-error,hi+error)
        }
    }
}
fn absolute(v:Value)->Value {
    match v {
        Value::Integer(n)=>integer_result(n.checked_abs()),
        Value::IntegerRange(a,b)=> {
            let (Some(a_abs),Some(b_abs))=(a.checked_abs(),b.checked_abs()) else {return Value::Unknown;};
            let lo=if a<=0 && b>=0 {0}else{a_abs.min(b_abs)};
            let hi=a_abs.max(b_abs);
            if hi<=i32::MAX as i64 {Value::IntegerRange(lo,hi)}else{Value::Unknown}
        },
        Value::Parts(v)=>Value::Parts(v.into_iter().map(absolute).collect()),
        v=>bounds(&v).map_or(Value::Unknown,|(a,b)|Value::NumberRange(if a<=0. && b>=0. {0.}else{a.abs().min(b.abs())},a.abs().max(b.abs()))),
    }
}
// Integer-to-float conversions have implementation-defined rounding unless a
// rounding execution mode is declared. Exact conversions stay exact; otherwise
// enclose both adjacent f32 values rather than selecting the host's rounding.
fn integer_float(lo:i64,hi:i64)->Value {
    let a=lo as f32;let b=hi as f32;
    if f64::from(a)==lo as f64 && f64::from(b)==hi as f64 {
        if lo==hi {Value::Number(f64::from(a))}else{Value::NumberRange(f64::from(a),f64::from(b))}
    }else{enclosure(lo as f64,hi as f64)}
}
fn floating(v:Value)->bool {match v {Value::Number(_)|Value::NumberRange(..)=>true,Value::Parts(v)=>v.into_iter().any(floating),_=>false}}
pub(super) fn operator(e:&E,get:&impl Fn(naga::Handle<E>)->Value)->Option<Value> {
    use naga::MathFunction as M;
    Some(match *e {
        E::As {expr,kind:naga::ScalarKind::Float,convert:Some(4)}=>match get(expr) {
            Value::Integer(n)=>integer_float(n,n),
            Value::IntegerRange(a,b)=>integer_float(a,b),
            _=>return None,
        },
        E::Binary {op,left,right} if floating(get(left)) || floating(get(right))=>binary(op,get(left),get(right)),
        E::Unary {op:naga::UnaryOperator::Negate,expr} if floating(get(expr))=>negate(get(expr)),
        E::Swizzle {size,vector,pattern}=>Value::Parts(pattern[..size as usize].iter().map(|c|get(vector).part(*c as usize)).collect()),
        E::Math {fun:M::Floor,arg,..}=>floor(get(arg)),
        E::Math {fun:M::Abs,arg,..}=>absolute(get(arg)),
        E::Math {fun:M::Sqrt,arg,..}=>square_root(get(arg)),
        E::Math {fun:M::Exp,arg,..}=>exponential(get(arg)),
        E::Math {fun:M::Exp2,arg,..}=>exponential_base(get(arg),true),
        E::Math {fun:M::Min,arg,arg1:Some(b),..}=>extrema(get(arg),get(b),false),
        E::Math {fun:M::Max,arg,arg1:Some(b),..}=>extrema(get(arg),get(b),true),
        E::Math {fun:M::Clamp,arg,arg1:Some(lo),arg2:Some(hi),..}=>clamp(get(arg),get(lo),get(hi)),
        // No guessed error margin for transcendental or resource operations.
        E::Math {..}=>Value::Unknown,
        _=>return None,
    })
}

pub(super) fn failure(e:&E,v:&Value)->Option<String> {
    fn unknown(v:&Value)->bool {match v {Value::Unknown|Value::Position|Value::Time|Value::Independent=>true,Value::Parts(p)=>p.iter().any(unknown),_=>false}}
    if !unknown(v) {return None;}
    Some(match e {
        E::Math {fun,..}=>format!("math operation {fun:?} has no finite enclosure for these operands"),
        E::Binary {op:naga::BinaryOperator::Divide,..}=>"division denominator crosses zero or leaves the certified normal range, or its quotient overflows".into(),
        E::Binary {op,..}=>format!("arithmetic {op:?} has an unbounded/unsupported operand or overflows f32"),
        _=>"unsupported shader expression".into(),
    })
}

pub(super) fn majorant(module:&naga::Module,entry:&str,words:&[Option<u32>],position:[[f32;2];3],times:[f32;2])->Result<f32> {
    let (_,function)=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry)).context("medium coefficient bridge missing")?;
    ensure!(position.iter().chain(std::iter::once(&times)).all(|[a,b]|a.is_finite() && b.is_finite() && a<=b),"invalid certification domain");
    let p=Value::Parts(position.map(|[a,b]|Value::NumberRange(f64::from(a),f64::from(b))).to_vec());
    let time=Value::NumberRange(f64::from(times[0]),f64::from(times[1]));
    let words=words.iter().copied().collect();
    let (value,state)=evaluate(module,function,&[p,time,Value::Integer(0),Value::Position],0,false,Some(words),None,true);
    let mut maximum=0f32;
    for channel in 0..3 {
        let sigma=binary(naga::BinaryOperator::Add,value.part(0).part(channel),value.part(1).part(channel));
        let (_,upper)=bounds(&sigma).with_context(|| {
            let failures=state.proof_failures.lock().unwrap().iter().cloned().collect::<Vec<_>>();
            let reason=if failures.is_empty() {"unbounded dependency, resource read or unsupported shader expression".to_owned()}else{failures.join("; ")};
            format!("cannot prove finite extinction in channel {channel}: {reason}")
        })?;
        maximum=maximum.max(upper as f32);
    }
    ensure!(maximum.is_finite(),"extinction enclosure overflows f32");
    Ok(if maximum>0. {maximum.max(f32::MIN_POSITIVE)}else{maximum})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn discrete_bounds_and_float_branches() {
        use naga::BinaryOperator as B;
        assert_eq!(clamp(Value::Integer(7),Value::Integer(1),Value::Integer(4)),Value::Integer(4));
        assert_eq!(clamp(Value::IntegerRange(-2,3),Value::Integer(1),Value::Integer(4)),Value::IntegerRange(1,3));
        assert_eq!(absolute(Value::Integer(-3)),Value::Integer(3));
        assert_eq!(absolute(Value::Integer(i32::MIN as i64)),Value::Unknown);
        assert_eq!(floor(Value::Number(2.75)),Value::Number(2.));
        assert_eq!(floor(Value::NumberRange(-0.5,2.75)),Value::NumberRange(-1.,2.));
        for (op,a,b,expected) in [
            (B::Equal,Value::Number(0.),Value::Number(0.),Value::Bool(true)),
            (B::NotEqual,Value::Number(0.),Value::Number(0.),Value::Bool(false)),
            (B::NotEqual,Value::NumberRange(1.,2.),Value::Number(0.),Value::Bool(true)),
            (B::Equal,Value::NumberRange(1.,2.),Value::Number(0.),Value::Bool(false)),
            (B::Equal,Value::NumberRange(0.,2.),Value::Number(0.),Value::Unknown),
            (B::LessEqual,Value::NumberRange(0.,2.),Value::Number(2.),Value::Bool(true)),
            (B::GreaterEqual,Value::NumberRange(0.,2.),Value::Number(0.),Value::Bool(true)),
            (B::NotEqual,Value::Number(f64::NAN),Value::Number(0.),Value::Unknown),
        ] {assert_eq!(binary(op,a,b),expected);}
    }
    fn prove(body:&str)->Result<f32> {
        let source=format!("struct Coefficients {{absorption:vec3<f32>,scattering:vec3<f32>,emission:vec3<f32>,phase:f32}}\nfn point(p:vec3<f32>,time:f32,base:u32,world:vec3<f32>)->Coefficients {{var m:Coefficients;{body} return m;}}");
        let module=naga::front::wgsl::parse_str(&source)?;
        majorant(&module,"point",&[],[[-1.,1.];3],[0.,2.])
    }
    #[test]
    fn integer_conversion_encloses_backend_rounding() -> Result<()> {
        // Nearest rounds -16777219 down; toward zero rounds it up. The old
        // singleton conversion certified zero while a permitted result was two.
        let bound=prove("let n=select(-16777219i,-16777219i,p.x>0.);m.absorption=vec3(f32(n)+16777220.);")?;
        assert!(bound>=2. && bound<2.00001,"{bound}");
        assert_eq!(prove("m.absorption=vec3(f32(0i));")?,0.);
        Ok(())
    }
    #[test]
    fn positive_subnormal_bound_survives_gpu_flush() -> Result<()> {
        assert_eq!(prove("m.absorption=vec3(1e-40);")?,f32::MIN_POSITIVE);
        assert_eq!(prove("m.absorption=vec3(0.);")?,0.);
        Ok(())
    }
    #[test]
    fn complete_expression_and_narrow_peak_are_enclosed() -> Result<()> {
        let bound=prove("m.absorption=vec3(2.+p.x+time);m.scattering=vec3(3.*p.y*p.y);")?;
        assert!(bound>=8. && bound<8.00002,"{bound}");
        // All 9^3 spatial probes can miss this entire nonzero branch.
        let bound=prove("if p.x>0.01 && p.x<0.02 {m.scattering=vec3(1000.);}")?;
        assert!(bound>=1000. && bound<1000.01,"{bound}");
        Ok(())
    }
    #[test]
    fn nonlinear_extinction_is_not_a_substituted_input_maximum() -> Result<()> {
        let bound=prove("let density=clamp(p.x,0.,1.);m.absorption=vec3(density*(1.-density));")?;
        assert!(bound>=0.25 && bound<1.00001,"{bound}");
        Ok(())
    }
    #[test]
    fn zero_extinction_retains_uniform_emission_estimator() -> Result<()> {
        assert_eq!(prove("m.emission=vec3(1.+p.x);m.absorption=vec3(0.);m.scattering=vec3(0.);")?,0.);
        Ok(())
    }
    #[test]
    fn flushed_operands_and_division_accuracy_are_enclosed() {
        let division=binary(naga::BinaryOperator::Divide,Value::Number(1e-40),Value::Number(1e-30));
        let (lo,hi)=bounds(&division).unwrap();
        assert!(lo<=0. && hi>=1e-10);
        let root=square_root(Value::Number(1e-40));
        let (lo,hi)=bounds(&root).unwrap();
        assert!(lo==0. && hi>=1e-20);
    }
    #[test]
    fn exponential_extinction_encloses_the_complete_expression() -> Result<()> {
        let bound=prove("m.absorption=vec3(.75*exp(-5.*p.x));m.scattering=vec3(.25*exp(-5.*p.x));")?;
        let maximum=5f32.exp();
        assert!(bound>=maximum && bound<maximum*1.00001,"{bound} vs {maximum}");
        for x in [-80.,-5.,0.,5.,80.] {
            let Value::NumberRange(lo,hi)=exponential(Value::Number(x)) else {panic!("missing enclosure at {x}");};
            let exact=x.exp();
            assert!(lo<exact && hi>exact,"{lo} <= exp({x}) <= {hi}");
            let radius=(3.+2.*x.abs()).max(4.).ceil() as usize;
            let mut low=exact as f32;let mut high=low;
            for _ in 0..radius {low=low.next_down();high=high.next_up();}
            assert!(lo<=f64::from(low) && hi>=f64::from(high));
        }
        assert!(matches!(exponential(Value::Number(-100.)),Value::Unknown));
        assert!(matches!(exponential(Value::Number(100.)),Value::Unknown));
        Ok(())
    }
    #[test]
    fn exponential_extinction_encloses_complete_expression() -> Result<()> {
        for (expr,expected) in [("exp(-p.y)",1f64.exp()),("exp2(-p.y)",2.),
            ("exp(clamp(-1000.*p.y,-80.,80.))",80f64.exp())] {
            let bound=prove(&format!("m.absorption=vec3({expr});"))?;
            assert!(f64::from(bound)>=expected && f64::from(bound)<expected*1.0001,"{expr}: {bound}");
        }
        // exp(f32(ln(2))) lies just above 2. Its permitted result error
        // uses the larger ULP above 2, even when stepping down crosses 2.
        let x=f64::from(std::f32::consts::LN_2);
        let Value::NumberRange(lo,_)=exponential(Value::Number(x)) else {panic!("missing enclosure");};
        let ulp=f64::from(2f32.next_up()-2.);
        let permitted=x.exp()-(3.+2.*x.abs()).max(4.)*ulp;
        assert!(lo<=permitted,"{lo} excludes permitted result {permitted}");
        for x in [-127.,128.] {
            assert!(matches!(exponential_base(Value::Number(x),true),Value::Unknown));
        }
        Ok(())
    }
    #[test]
    fn singular_and_unsupported_operations_refuse() {
        for expression in ["1./p.x","exp(1000.*p.x)","exp2(1000.*p.x)","sin(p.x)","world.x","clamp(p.x,1.,0.)"] {
            let error=prove(&format!("m.absorption=vec3({expression});")).unwrap_err().to_string();
            assert!(error.contains("cannot prove finite extinction"),"{error}");
        }
    }
}

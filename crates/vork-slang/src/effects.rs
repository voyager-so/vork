//! Conservative constant evaluation of the compiler's closure-effect slice.
//! Unknown values retain a capability; only a proven zero removes one.
use super::*;
use naga::{Expression as E, Statement as S};
mod interval;
#[derive(Clone, Debug, PartialEq)]
enum Value {
    Unknown,
    Position,
    SpatialRange(f64, f64),
    Affine([f64;4]),
    Exponential(f64,[f64;3]),
    Independent,
    Time,
    Number(f64),
    Integer(i64),
    Bool(bool),
    IntegerRange(i64, i64),
    NumberRange(f64, f64),
    Parts(Vec<Value>),
}
impl Value {
    fn time(&self) -> bool {
        match self {
            Self::Time => true,
            Self::Parts(parts) => parts.iter().any(Self::time),
            _ => false,
        }
    }
    fn dependent(values: impl IntoIterator<Item = Self>) -> Self {
        if values.into_iter().any(|v| v.time()) {
            Self::Time
        } else {
            Self::Unknown
        }
    }

    fn part(&self, index: usize) -> Self {
        match self {
            Self::Parts(v) => v.get(index).cloned().unwrap_or(Self::Unknown),
            Self::Time => Self::Time,
            Self::Position => Self::Position,
            Self::Independent => Self::Independent,
            _ => Self::Unknown,
        }
    }
    fn widen(&self, other: &Self) -> Self {
        match (self,other) {
            (Self::Parts(a),Self::Parts(b)) if a.len()==b.len() => Self::Parts(a.iter().zip(b).map(|(a,b)|a.widen(b)).collect()),
            _ if self==other => self.clone(),
            _ if self.time() || other.time() => Self::Time,
            _ => Self::Unknown,
        }
    }
    fn is_zero(&self) -> bool {
        match self {
            Self::Number(n) => *n == 0.,
            Self::Parts(parts) => !parts.is_empty() && parts.iter().all(Self::is_zero),
            _ => false,
        }
    }
    /// Non-zero numbers (and ranges) become Unknown, componentwise.
    fn zero_or_unknown(self) -> Self {
        match self {
            Self::Parts(parts) => Self::Parts(parts.into_iter().map(Self::zero_or_unknown).collect()),
            Self::Number(n) if n == 0. => self,
            Self::Number(_) | Self::NumberRange(..) => Self::Unknown,
            v => v,
        }
    }
    /// Raw positions are finite scene coordinates; arithmetic results need a
    /// separate finite proof before zero multiplication may erase them.
    fn finite_spatial_operand(&self) -> bool {
        match self {
            Self::Position => true,
            Self::Number(n) => n.is_finite(),
            Self::NumberRange(lo, hi) => lo.is_finite() && hi.is_finite(),
            Self::Parts(parts) => parts.iter().all(Self::finite_spatial_operand),
            _ => false,
        }
    }
    fn independent(&self) -> bool {
        match self {Self::Unknown|Self::Position|Self::SpatialRange(..)|Self::Affine(..)|Self::Exponential(..)|Self::Time=>false,
            Self::Parts(parts)=>parts.iter().all(Self::independent),_=>true}
    }
    fn merge(&self, other: &Self) -> Self {
        if self==other {return self.clone();}
        match (self, other) {
            (Self::Integer(a),Self::Integer(b)) => Self::IntegerRange((*a).min(*b),(*a).max(*b)),
            (Self::IntegerRange(a,b),Self::Integer(c)) | (Self::Integer(c),Self::IntegerRange(a,b)) => Self::IntegerRange((*a).min(*c),(*b).max(*c)),
            (Self::IntegerRange(a,b),Self::IntegerRange(c,d)) => Self::IntegerRange((*a).min(*c),(*b).max(*d)),
            (Self::Number(a),Self::Number(b)) if a.is_finite() && b.is_finite() => Self::NumberRange(a.min(*b),a.max(*b)),
            (Self::NumberRange(a,b),Self::Number(c)) | (Self::Number(c),Self::NumberRange(a,b)) if c.is_finite() => Self::NumberRange(a.min(*c),b.max(*c)),
            (Self::NumberRange(a,b),Self::NumberRange(c,d)) => Self::NumberRange(a.min(*c),b.max(*d)),
            (Self::Parts(a), Self::Parts(b)) if a.len() == b.len() => {
                Self::Parts(a.iter().zip(b).map(|(a, b)| a.merge(b)).collect())
            }
            _ if self == other => self.clone(),
            _ if self.time() || other.time() => Self::Time,
            _ if self.independent() && other.independent() => Self::Independent,
            _ => Self::Unknown,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoopFlow { Break, Continue, Mixed }

/// The sources that can change between frames for an object that does not
/// move, or the wavelength alone for spectral output dependency proofs.
#[derive(Clone, Copy, Debug)]
struct Live { texels: bool, wavelength: bool, query_bounds: bool }
/// Bridge calls reading the camera.
const CAMERA_CALLS: [&str; 2] = ["vork_world_to_camera", "vork_world_to_window"];
/// Shading context fields that change between frames at a fixed point.
const LIVE_CONTEXT: [&str; 4] = ["time", "incoming", "incoming_dx", "incoming_dy"];
/// The shading context global of a live probe: its `LIVE_CONTEXT` fields are
/// Time. Spectral probes select only wavelength instead. An unrecognisable
/// context layout conservatively retains the dependency.
fn live_context(module: &naga::Module, ty: naga::Handle<naga::Type>, wavelength:bool) -> Value {
    // Slang suffixes WGSL identifiers with `_<n>`.
    let base = |name: &str| name.rsplit_once('_').filter(|(_, n)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())).map_or(name, |(base, _)| base).to_owned();
    match &module.types[ty].inner {
        naga::TypeInner::Struct { members, .. } if members.iter().any(|m| m.name.as_deref().is_some_and(|n| base(n) == if wavelength {"wavelength"} else {"time"})) => Value::Parts(
            members.iter().map(|m| if m.name.as_deref().is_some_and(|n| if wavelength {base(n)=="wavelength"} else {LIVE_CONTEXT.contains(&base(n).as_str())}) {Value::Time} else {Value::Unknown}).collect(),
        ),
        _ => Value::Time,
    }
}

#[derive(Default)]
struct QueryReads {filters: BTreeSet<QueryFilter>, incomplete: bool}
#[derive(Default)]
struct WorkProof { operations: u32, failure: Option<String> }
#[derive(Clone, Default)]
struct State {
    query_reads: Option<std::sync::Arc<std::sync::Mutex<QueryReads>>>,
    // Shared across explored branches and nested calls: summing both branches
    // is conservative, and a global analysis ceiling bounds proof construction.
    work: Option<std::sync::Arc<std::sync::Mutex<WorkProof>>>,
    image_reads: Option<std::sync::Arc<std::sync::Mutex<Option<BTreeSet<u32>>>>>,
    numeric: bool,
    certified: bool,
    proof_failures: std::sync::Arc<std::sync::Mutex<std::collections::BTreeSet<String>>>,
    dependency: bool,
    /// Follow output dependence through unrelated conditional computations.
    output_slice: bool,
    /// Material input words, `None` where unknown: set, word reads evaluate
    /// to the words (outside the dependency domain too) and bitcasts of known
    /// words keep their value.
    words: Option<std::sync::Arc<[Option<u32>]>>,
    simulate_loop: bool,
    loop_flow: Option<LoopFlow>,
    loop_breaks: Vec<std::sync::Arc<State>>,
    loop_continues: Vec<std::sync::Arc<State>>,
    /// Inside a switch case, `break` leaves the switch (`switch_exit`), not
    /// an enclosing loop.
    in_switch: bool,
    switch_exit: bool,
    observed: Option<(BTreeMap<usize, Value>, Vec<Value>)>,
    time_control: bool,
    /// Set: the analysis asks what varies while the object stays put, so the
    /// sources that do read as Time (`Live`).
    live: Option<Live>,
    arguments: Vec<Value>,
    locals: BTreeMap<usize, Value>,
    expressions: BTreeMap<usize, Value>,
    effects: BTreeMap<usize, Value>,
    exits: Option<(Value, Vec<Value>)>,
}
fn zero(module: &naga::Module, ty: naga::Handle<naga::Type>) -> Value {
    match &module.types[ty].inner {
        naga::TypeInner::Struct { members, .. } => {
            Value::Parts(members.iter().map(|m| zero(module, m.ty)).collect())
        }
        naga::TypeInner::Vector { size, .. } => {
            Value::Parts(vec![Value::Number(0.); *size as usize])
        }
        naga::TypeInner::Scalar(s) => match s.kind {
            naga::ScalarKind::Sint | naga::ScalarKind::Uint => Value::Integer(0),
            naga::ScalarKind::Bool => Value::Bool(false),
            _ => Value::Number(0.),
        },
        _ => Value::Unknown,
    }
}
// Integer expressions may be signed or unsigned. Prove only arithmetic whose
// result fits both representations when operands do not establish signedness.
fn integer_result(value: Option<i64>) -> Value {
    value.filter(|v| *v >= i32::MIN as i64 && *v <= i32::MAX as i64)
        .map(Value::Integer).unwrap_or(Value::Unknown)
}
/// Fold f32 arithmetic only when it is exact: the operands are f32 values, so
/// an exact sum or product of normal numbers is what every GPU computes,
/// for the declared operation. Certificates must preserve the expression graph
/// through native compilation; fast-math reassociation can invalidate exact steps.
/// Anything else stays unknown, so rounding never proves a zero.
fn exact_f32(op: naga::BinaryOperator, a: f64, b: f64) -> Value {
    let (r, exact) = match op {
        naga::BinaryOperator::Add => (a + b, (a + b) - a == b && (a + b) - b == a),
        naga::BinaryOperator::Subtract => return exact_f32(naga::BinaryOperator::Add, a, -b),
        // Products of two f32 values are exact in f64.
        naga::BinaryOperator::Multiply => (a * b, true),
        _ => return Value::Unknown,
    };
    // Subnormals are not exact either: a backend may flush them to zero.
    if exact && r.is_finite() && f64::from(r as f32) == r && [a, b, r].into_iter().all(normal_f32) { Value::Number(r) } else { Value::Unknown }
}
/// Zero or a normal f32 magnitude: no backend flushes it.
fn normal_f32(x: f64) -> bool { x == 0. || x.abs() >= f64::from(f32::MIN_POSITIVE) }
fn compare<T: PartialOrd>(op: naga::BinaryOperator, a: T, b: T) -> Option<Value> {
    use naga::BinaryOperator as B;
    Some(Value::Bool(match op {
        B::Equal => a == b, B::NotEqual => a != b, B::Less => a < b,
        B::LessEqual => a <= b, B::Greater => a > b, B::GreaterEqual => a >= b,
        _ => return None,
    }))
}
fn expression(
    module: &naga::Module,
    function: &naga::Function,
    args: &[Value],
    state: &State,
    handle: naga::Handle<E>,
) -> Value {
    if let Some(value) = state.expressions.get(&handle.index()) {
        return value.clone();
    }
    let get = |h| expression(module, function, args, state, h);
    if state.numeric {
        if let E::Binary {op,left,right}=&function.expressions[handle] {
            if let (Value::Number(a),Value::Number(b))=(get(*left),get(*right)) {
                let (a,b)=(a as f32,b as f32);
                let n=match op {
                    naga::BinaryOperator::Add=>Some(a+b), naga::BinaryOperator::Subtract=>Some(a-b),
                    naga::BinaryOperator::Multiply=>Some(a*b), naga::BinaryOperator::Divide=>Some(a/b), _=>None,
                };
                if let Some(n)=n {return if n.is_finite() {Value::Number(n.into())} else {Value::Unknown};}
            }
        }
    }
    if state.certified {
        if let Some(value)=interval::operator(&function.expressions[handle], &get) {
            if let Some(reason)=interval::failure(&function.expressions[handle],&value) {
                state.proof_failures.lock().unwrap().insert(reason);
            }
            return value;
        }
    }
    if state.dependency {
        if let Some(value)=spatial_operator(module,function,args,state,handle) {return value;}
    }
    match &function.expressions[handle] {
        E::Literal(naga::Literal::Bool(v)) => Value::Bool(*v),
        E::Literal(naga::Literal::F32(v)) => Value::Number(f64::from(*v)),
        E::Literal(naga::Literal::F64(_)) => Value::Unknown,
        E::Literal(naga::Literal::I32(v)) => Value::Integer(i64::from(*v)),
        E::Literal(naga::Literal::U32(v)) => Value::Integer(i64::from(*v)),
        E::ZeroValue(ty) => zero(module, *ty),
        E::GlobalVariable(g) if state.live.is_some() && module.global_variables[*g].name.as_deref().is_some_and(|n| n.starts_with("vork_context")) => {
            live_context(module, module.global_variables[*g].ty,state.live.unwrap().wavelength)
        }
        E::FunctionArgument(i) => state
            .arguments
            .get(*i as usize)
            .cloned()
            .unwrap_or(Value::Unknown),
        E::LocalVariable(h) => state
            .locals
            .get(&h.index())
            .cloned()
            .unwrap_or(Value::Unknown),
        E::Load { pointer } => {
            if let Some((root, path)) = place(function, *pointer) {
                path.iter().fold(
                    state.locals.get(&root).cloned().unwrap_or(Value::Unknown),
                    |v, i| v.part(*i),
                )
            } else if let Some((root, path)) = argument_place(function, *pointer) {
                path.iter().fold(
                    state.arguments.get(root).cloned().unwrap_or(Value::Unknown),
                    |v, i| v.part(*i),
                )
            } else {
                get(*pointer)
            }
        }
        E::As {
            expr,
            kind,
            convert: Some(width),
        } => {
            if matches!(kind,naga::ScalarKind::Float) && *width==4 {
                if let Value::IntegerRange(a,b)=get(*expr) {
                    return Value::NumberRange(f64::from(a as f32),f64::from(b as f32));
                }
            }
            if let Value::NumberRange(a,b)=get(*expr) {
                let limits=match kind {
                    naga::ScalarKind::Sint=>Some((i32::MIN as f64,i32::MAX as f64)),
                    naga::ScalarKind::Uint=>Some((0.,u32::MAX as f64)),
                    _=>None,
                };
                if *width==4 && limits.is_some_and(|(lo,hi)|a>=lo && b<=hi) {
                    let (a,b)=(a.trunc() as i64,b.trunc() as i64);
                    return if a==b {Value::Integer(a)}else{Value::IntegerRange(a,b)};
                }
            }
            let number = match get(*expr) {
                Value::Number(n) => Some(n),
                Value::Integer(n) => Some(n as f64),
                _ => None,
            };
            match (number, kind, *width) {
                (Some(n), naga::ScalarKind::Float, 4) => Value::Number(f64::from(n as f32)),
                (Some(n), naga::ScalarKind::Sint, 4)
                    if n >= i32::MIN as f64 && n <= i32::MAX as f64 =>
                {
                    Value::Integer(n as i32 as i64)
                }
                (Some(n), naga::ScalarKind::Uint, 4) if n >= 0. && n <= u32::MAX as f64 => {
                    Value::Integer(n as u32 as i64)
                }
                _ => Value::dependent([get(*expr)]),
            }
        }
        // Reinterpreting a known material word keeps its exact bits: a finite
        // f32 (the GPU reads the same word) or the 32-bit integer. Only with
        // real words: the other analyses stub word reads with their index.
        // Otherwise bitcasts preserve runtime dependencies like a conversion.
        E::As { expr, kind, convert: None } => match (get(*expr), kind) {
            (Value::Integer(n), kind) if state.words.is_some() => match kind {
                naga::ScalarKind::Float => Some(f32::from_bits(n as u32)).filter(|v| v.is_finite()).map_or(Value::Unknown, |v| Value::Number(f64::from(v))),
                naga::ScalarKind::Sint => Value::Integer(i64::from(n as u32 as i32)),
                naga::ScalarKind::Uint => Value::Integer(i64::from(n as u32)),
                _ => Value::Unknown,
            },
            (value, _) => Value::dependent([value]),
        },
        E::AccessIndex { base, index } => get(*base).part(*index as usize),
        E::Access { base, index } => match get(*index) {
            Value::Integer(i) if i >= 0 => get(*base).part(i as usize),
            _ => Value::dependent([get(*base),get(*index)]),
        },
        E::Unary { op: naga::UnaryOperator::LogicalNot, expr } => match get(*expr) {
            Value::Bool(v) => Value::Bool(!v),
            _ => Value::dependent([get(*expr)]),
        },
        E::Compose { components, .. } => Value::Parts(components.iter().map(|h| get(*h)).collect()),
        E::Splat { size, value } => Value::Parts(vec![get(*value); *size as usize]),
        E::Binary { op, left, right } => match (get(*left), get(*right)) {
            (Value::Bool(a),Value::Bool(b)) if state.numeric || state.certified => match op {
                naga::BinaryOperator::LogicalAnd => Value::Bool(a && b),
                naga::BinaryOperator::LogicalOr => Value::Bool(a || b),
                naga::BinaryOperator::Equal => Value::Bool(a == b),
                naga::BinaryOperator::NotEqual => Value::Bool(a != b),
                _ => Value::Unknown,
            },
            (a @ (Value::Integer(_)|Value::IntegerRange(..)),b @ (Value::Integer(_)|Value::IntegerRange(..)))
                if matches!(op,naga::BinaryOperator::Equal|naga::BinaryOperator::NotEqual|naga::BinaryOperator::Less|naga::BinaryOperator::LessEqual|naga::BinaryOperator::Greater|naga::BinaryOperator::GreaterEqual)=>interval::binary(*op,a,b),
            (Value::IntegerRange(a,b),Value::Integer(c)) | (Value::Integer(c),Value::IntegerRange(a,b)) if *op==naga::BinaryOperator::Add => {
                match (a.checked_add(c),b.checked_add(c)) {
                    (Some(a),Some(b)) if a>=i32::MIN as i64 && b<=i32::MAX as i64 => Value::IntegerRange(a,b),
                    _ => Value::Unknown,
                }
            },
            (Value::IntegerRange(a,b),Value::IntegerRange(c,d)) if *op==naga::BinaryOperator::Add => {
                match (a.checked_add(c),b.checked_add(d)) {
                    (Some(a),Some(b)) if a>=i32::MIN as i64 && b<=i32::MAX as i64 => Value::IntegerRange(a,b),
                    _ => Value::Unknown,
                }
            },
            (Value::Integer(a), Value::Integer(b)) => compare(*op, a, b).unwrap_or_else(|| match op {
                naga::BinaryOperator::Add => integer_result(a.checked_add(b)),
                naga::BinaryOperator::Subtract => if a >= 0 && b > a {Value::Unknown} else {integer_result(a.checked_sub(b))},
                naga::BinaryOperator::Multiply => integer_result(a.checked_mul(b)),
                naga::BinaryOperator::Divide=>integer_result(a.checked_div(b)),
                // A mask of non-negative (u32) words is never larger than either.
                naga::BinaryOperator::And if a >= 0 && b >= 0 => Value::Integer(a & b),
                naga::BinaryOperator::And=>integer_result(Some(a&b)),
                naga::BinaryOperator::ShiftLeft=>integer_result(u32::try_from(b).ok().filter(|b|*b<32).and_then(|b|a.checked_shl(b))),
                naga::BinaryOperator::ShiftRight=>integer_result(u32::try_from(b).ok().and_then(|b|a.checked_shr(b))),
                _ => Value::Unknown,
            }),
            // Numbers are exact f32 values, so comparing them is exact, except
            // that a backend may flush a subnormal operand to zero first.
            // Bounds must remain valid even for undefined floating inputs;
            // material-effect simplifications do not prove execution limits.
            (Value::Number(a), Value::Number(b)) if state.live.is_some_and(|l|l.query_bounds) =>
                [a,b].into_iter().all(normal_f32).then(||compare(*op,a,b)).flatten().unwrap_or(Value::Unknown),
            (Value::Number(a), Value::Number(b)) => [a, b].into_iter().all(normal_f32)
                .then(|| compare(*op, a, b)).flatten().unwrap_or_else(|| exact_f32(*op, a, b)),
            // A proven zero factor annihilates the other operand, including
            // unknown and time-varying ones, under any rounding or contraction,
            // wherever that operand is finite. It is not proven finite (1/x at
            // x = 0, an overflowing pow, an Inf texel), and there the GPU computes
            // NaN: a closure value with no defined shading, which a blended route
            // would propagate or clamp to either end depending on the backend.
            // Only such undefined pixels can differ; requiring bounded operands
            // instead would need every runtime value proven finite, which no
            // position-, time- or texel-driven factor can be. A scalar zero
            // scales a known vector to a zero vector; otherwise it stays the
            // scalar it was.
            (a, b) if !state.live.is_some_and(|l|l.query_bounds) && *op == naga::BinaryOperator::Multiply && (a.is_zero() || b.is_zero()) => {
                let (zero, other) = if a.is_zero() { (a, b) } else { (b, a) };
                match other {
                    Value::Parts(parts) if matches!(zero, Value::Number(_)) => Value::Parts(vec![Value::Number(0.); parts.len()]),
                    _ => zero,
                }
            }
            (a, b) => Value::dependent([a, b]),
        },
        E::Select { condition, accept, reject } => match get(*condition) {
            Value::Bool(true) => get(*accept),
            Value::Bool(false) => get(*reject),
            Value::Time => Value::Time,
            _ => get(*accept).merge(&get(*reject)),
        },
        E::Math { fun: naga::MathFunction::Min, arg, arg1: Some(other), .. } => {
            let bounds=|value:Value| match value {Value::Integer(n)=>Some((n,n)),Value::IntegerRange(a,b)=>Some((a.clone(),b.clone())),_=>None};
            match (bounds(get(*arg)),bounds(get(*other))) {
                (Some((a,b)),Some((c,d))) => {
                    let (a,b)=(a.min(c),b.min(d));
                    if a==b {Value::Integer(a)} else {Value::IntegerRange(a,b)}
                },
                _=>Value::dependent([get(*arg),get(*other)]),
            }
        },
        E::Unary { op: naga::UnaryOperator::Negate, expr } if state.numeric || state.certified => match get(*expr) {
            Value::Integer(n) => integer_result(n.checked_neg()),
            Value::IntegerRange(a,b) => match (b.checked_neg(),a.checked_neg()) {
                (Some(a),Some(b)) if a>=i32::MIN as i64 && b<=i32::MAX as i64=>Value::IntegerRange(a,b),
                _=>Value::Unknown,
            },
            Value::Number(n) => Value::Number(-n),
            v => Value::dependent([v]),
        },
        E::Swizzle { size, vector, pattern } if state.numeric || state.live.is_some_and(|l|l.query_bounds) => {
            let v=get(*vector);
            Value::Parts(pattern[..*size as usize].iter().map(|c|v.part(*c as usize)).collect())
        },
        E::Unary { expr, .. }
        | E::Derivative { expr, .. }
        | E::Relational { argument: expr, .. }
        | E::Swizzle { vector: expr, .. } => Value::dependent([get(*expr)]),
        // `lerp`/`mix` keeps a value both ends share only where it is exactly
        // zero: a*(1-t)+b*t of equal non-zero ends can round (or contract) to
        // a neighbour.
        E::Math {
            fun: naga::MathFunction::Mix,
            arg,
            arg1: Some(other),
            arg2: factor,
            ..
        } => {
            if state.live.is_some_and(|l|l.query_bounds) {return Value::Unknown;}
            let a = get(*arg);
            let b = get(*other);
            if factor.is_some_and(|h| get(h).time()) {
                Value::Time
            } else {
                a.merge(&b).zero_or_unknown()
            }
        }
        E::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => Value::dependent(
            std::iter::once(*arg)
                .chain(*arg1)
                .chain(*arg2)
                .chain(*arg3)
                .map(get),
        ),
        // A live probe cannot follow other expressions: they may carry a live
        // value. Globals and constants are not live (the context is above).
        E::GlobalVariable(_) | E::Constant(_) | E::Override(_) | E::Literal(_) => Value::Unknown,
        _ if state.live.is_some() => Value::Time,
        _ => Value::Unknown,
    }
}
// Pointer expressions are addresses, not snapshots of their pointee. Resolve
// loads against current locals, including a proven dynamic array index.
fn spatial_load(module:&naga::Module,function:&naga::Function,args:&[Value],state:&State,pointer:naga::Handle<E>)->Value {
    match &function.expressions[pointer] {
        E::LocalVariable(h)=>state.locals.get(&h.index()).cloned().unwrap_or(Value::Unknown),
        E::FunctionArgument(i)=>state.arguments.get(*i as usize).cloned().unwrap_or(Value::Unknown),
        E::AccessIndex {base,index}=>spatial_load(module,function,args,state,*base).part(*index as usize),
        E::Access {base,index}=> {
            let base=spatial_load(module,function,args,state,*base);
            match expression(module,function,args,state,*index) {
                Value::Integer(i) if i>=0=>base.part(i as usize),
                i if i.independent() && base.independent()=>Value::Independent,
                _=>Value::Unknown,
            }
        },
        _=>expression(module,function,args,state,pointer),
    }
}
// Spatial invariance is a separate abstract operator domain from the existing
// time/capacity analysis, sharing its calls, fields, stores and control flow.
fn spatial_operator(module:&naga::Module,function:&naga::Function,args:&[Value],state:&State,handle:naga::Handle<E>) -> Option<Value> {
    let get=|h|expression(module,function,args,state,h);
    if let Some(value)=profile_operator(&function.expressions[handle],&get) {return Some(value);}
    let value=match &function.expressions[handle] {
        E::Load {pointer}=>spatial_load(module,function,args,state,*pointer),
        E::Unary {op:naga::UnaryOperator::Negate,expr} if state.dependency => match get(*expr) {
            Value::SpatialRange(a,b)=>Value::SpatialRange(-b,-a),
            // Negation is an exact sign flip and preserves finite coordinates.
            Value::Position=>Value::Position,
            Value::Number(v)=>Value::Number(-v),Value::Integer(0)=>Value::Integer(0),
            v if v.independent()=>Value::Independent,_=>Value::Unknown,
        },
        E::As {expr,kind:naga::ScalarKind::Float,convert:None} if state.dependency => {
            match get(*expr) {Value::Integer(v)=>Value::Number(f64::from(f32::from_bits(v as u32))),
                v if v.independent()=>Value::Independent,_=>Value::Unknown}
        },
        E::As {
            expr,
            kind,
            convert: Some(width),
        } => {
            // Slang lowers default float vectors through integer-zero vectors.
            // Their conversion is exact, componentwise, including in structs.
            fn float_zero(value:Value)->Option<Value> {
                match value {
                    Value::Integer(0)|Value::Number(0.)=>Some(Value::Number(0.)),
                    Value::Parts(parts)=>parts.into_iter().map(float_zero).collect::<Option<Vec<_>>>().map(Value::Parts),
                    _=>None,
                }
            }
            if *kind==naga::ScalarKind::Float && *width==4 {
                if let Some(zero)=float_zero(get(*expr)) {return Some(zero);}
            }
            let number = match get(*expr) {
                Value::Number(n) => Some(n),
                Value::Integer(n) => Some(n as f64),
                _ => None,
            };
            match (number, kind, *width) {
                (Some(n), naga::ScalarKind::Float, 4) => Value::Number(f64::from(n as f32)),
                (Some(n), naga::ScalarKind::Sint, 4)
                    if n >= i32::MIN as f64 && n <= i32::MAX as f64 =>
                {
                    Value::Integer(n as i32 as i64)
                }
                (Some(n), naga::ScalarKind::Uint, 4) if n >= 0. && n <= u32::MAX as f64 => {
                    Value::Integer(n as u32 as i64)
                }
                _ if get(*expr).independent() => Value::Independent,
                _ => Value::Unknown,
            }
        }
        E::Binary { op, left, right } => match (get(*left), get(*right)) {
            (Value::Integer(a),Value::Integer(b)) if state.dependency => {
                use naga::BinaryOperator as B;
                match op {
                    B::Add | B::Subtract | B::Multiply => {
                        // A sign/width-erased value is exact only where signed
                        // and unsigned 32-bit arithmetic agree without overflow.
                        let n=match op {B::Add=>a.checked_add(b),B::Subtract=>a.checked_sub(b),_=>a.checked_mul(b)};
                        n.filter(|n|(0..=i64::from(i32::MAX)).contains(n)).map(Value::Integer).unwrap_or(Value::Independent)
                    },
                    B::And=>Value::Integer(a&b),B::InclusiveOr=>Value::Integer(a|b),
                    B::ExclusiveOr=>Value::Integer(a^b),B::Equal=>Value::Bool(a==b),B::NotEqual=>Value::Bool(a!=b),
                    B::Less=>Value::Bool(a<b),B::Greater=>Value::Bool(a>b),
                    _=>Value::Independent,
                }
            },
            // Only erase proven finite operands: Unknown may contain Inf/NaN
            // from position arithmetic, for which multiplication by zero is NaN.
            (a,b) if *op==naga::BinaryOperator::Multiply
                && ((a.is_zero() && b.finite_spatial_operand())
                    || (b.is_zero() && a.finite_spatial_operand())) => {
                let (zero,other)=if a.is_zero() {(a,b)} else {(b,a)};
                match other {
                    Value::Parts(parts) if matches!(zero,Value::Number(_))=>Value::Parts(vec![Value::Number(0.);parts.len()]),
                    _=>zero,
                }
            },
            (a,b) if *op==naga::BinaryOperator::Multiply && a.independent() && b.independent() => {
                // Retain only a finite bound, not an exact arithmetic value:
                // shader contraction can change the rounded result. A factor
                // of two leaves room for rounding and underflow; overflow and
                // unsupported operands remain unknown.
                fn magnitude(v:&Value)->Option<f64> {match v {
                    Value::Number(n)=>Some(n.abs()),
                    Value::NumberRange(lo,hi)=>Some(lo.abs().max(hi.abs())),
                    _=>None,
                }}
                match magnitude(&a).zip(magnitude(&b)) {
                    Some((a,b)) if a.is_finite() && b.is_finite() && a*b<=f64::from(f32::MAX)/2. => {
                        let bound=(2.*a*b).max(f64::from(f32::MIN_POSITIVE));
                        Value::NumberRange(-bound,bound)
                    },
                    _=>Value::Independent,
                }
            },
            (Value::SpatialRange(a,b),Value::Number(c)) | (Value::Number(c),Value::SpatialRange(a,b))
                if *op==naga::BinaryOperator::Multiply && normal_f32(c) && c!=0. => {
                let (lo,hi)=if c>0. {(a*c,b*c)} else {(b*c,a*c)};
                // Widen by a generous f32 rounding margin. These ranges are
                // spatial dependencies until a saturating operation erases them.
                let margin=lo.abs().min(hi.abs())*1.0e-6;
                Value::SpatialRange(lo-margin,hi+margin)
            },
            (Value::SpatialRange(a,b),Value::Number(c)) => {
                use naga::BinaryOperator as B;
                match op {
                    B::Less if b<c=>Value::Bool(true),B::Less if a>=c=>Value::Bool(false),
                    B::LessEqual if b<=c=>Value::Bool(true),B::LessEqual if a>c=>Value::Bool(false),
                    B::Greater if a>c=>Value::Bool(true),B::Greater if b<=c=>Value::Bool(false),
                    B::GreaterEqual if a>=c=>Value::Bool(true),B::GreaterEqual if b<c=>Value::Bool(false),
                    _=>Value::Unknown,
                }
            },
            // Comparisons of known normal f32 values (including zero) are
            // exact. Arithmetic still cannot fold across GPU contraction.
            (Value::Number(a),Value::Number(b)) => {
                if normal_f32(a) && normal_f32(b) {compare(*op,a,b).unwrap_or(Value::Independent)}
                else {Value::Independent}
            },
            (a,b) if a.independent() && b.independent() => Value::Independent,
            _ => Value::Unknown,
        },
        E::Constant(_)=>Value::Independent,
        E::Access {base,index}=>match get(*index) {
            Value::Integer(i) if i>=0=>get(*base).part(i as usize),
            i if i.independent() && get(*base).independent()=>Value::Independent,
            _=>Value::Unknown,
        },
        E::Select {condition,accept,reject}=> match get(*condition) {
            Value::Bool(true)=>get(*accept),Value::Bool(false)=>get(*reject),
            c if c.independent()=>get(*accept).merge(&get(*reject)),_=>Value::Unknown,
        },
        E::Math {fun,arg,arg1:Some(other),..} if matches!(fun,naga::MathFunction::Min|naga::MathFunction::Max)=> {
            fn extrema(a:Value,b:Value,maximum:bool)->Value {
                match (a,b) {
                    (Value::Number(a),Value::Number(b)) if normal_f32(a) && normal_f32(b)=>Value::Number(if maximum {a.max(b)} else {a.min(b)}),
                    (Value::Parts(a),Value::Parts(b)) if a.len()==b.len()=>Value::Parts(a.into_iter().zip(b).map(|(a,b)|extrema(a,b,maximum)).collect()),
                    (a,b) if a.independent() && b.independent()=>Value::Independent,
                    (a,b)=>Value::dependent([a,b]),
                }
            }
            extrema(get(*arg),get(*other),*fun==naga::MathFunction::Max)
        },
        E::Math {fun:naga::MathFunction::Clamp,arg,arg1:Some(lo),arg2:Some(hi),..} => {
            match (get(*arg),get(*lo),get(*hi)) {
                (Value::SpatialRange(a,b),Value::Number(lo),Value::Number(hi)) if lo<=hi => {
                    if b<=lo {Value::Number(lo)} else if a>=hi {Value::Number(hi)} else {Value::Unknown}
                },
                (a,b,c) if a.independent() && b.independent() && c.independent()=>Value::Independent,
                _=>Value::Unknown,
            }
        },
        // exp(0) is finite even when the backend uses an approximation.
        // Keep a loose bound instead of requiring an exact transcendental.
        E::Math {fun:naga::MathFunction::Exp,arg,..} if get(*arg).is_zero()=>Value::NumberRange(0.,2.),
        E::Math {arg,arg1,arg2,arg3,..}=>if std::iter::once(*arg).chain(*arg1).chain(*arg2).chain(*arg3).all(|h|get(h).independent()) {Value::Independent} else {Value::Unknown},
        E::Unary {expr,..}|E::Derivative {expr,..}|E::Swizzle {vector:expr,..}|E::Relational {argument:expr,..}=>if get(*expr).independent() {Value::Independent} else {Value::Unknown},
        _=>return None,
    };Some(value)
}

// A deliberately small algebraic proof: constant multiples of exp(affine
// position). Unsupported sums, clamps and control flow stay unknown. This is
// shape recognition, never a fit to sampled point values.
fn profile_operator(expression:&E,get:&impl Fn(naga::Handle<E>)->Value)->Option<Value> {
    fn profile(v:&Value)->bool {match v {Value::Affine(..)|Value::Exponential(..)=>true,Value::Parts(p)=>p.iter().any(profile),_=>false}}
    fn binary(op:naga::BinaryOperator,a:Value,b:Value)->Value {
        use naga::BinaryOperator as B;
        match (a,b) {
            (Value::Parts(a),Value::Parts(b)) if a.len()==b.len()=>Value::Parts(a.into_iter().zip(b).map(|(a,b)|binary(op,a,b)).collect()),
            (Value::Parts(a),b)=>Value::Parts(a.into_iter().map(|a|binary(op,a,b.clone())).collect()),
            (a,Value::Parts(b))=>Value::Parts(b.into_iter().map(|b|binary(op,a.clone(),b)).collect()),
            (Value::Number(a),Value::Number(b))=>match op {B::Multiply=>Value::Number(a*b),B::Add=>Value::Number(a+b),B::Subtract=>Value::Number(a-b),_=>Value::Unknown},
            (Value::Affine(a),Value::Number(b)) if op==B::Multiply=>Value::Affine(a.map(|a|a*b)),
            (Value::Number(a),Value::Affine(b)) if op==B::Multiply=>Value::Affine(b.map(|b|a*b)),
            (Value::Affine(mut a),Value::Number(b)) if op==B::Add || op==B::Subtract=>{a[3]+=if op==B::Add {b} else {-b};Value::Affine(a)},
            (Value::Affine(a),Value::Affine(b)) if op==B::Add || op==B::Subtract=>Value::Affine(std::array::from_fn(|i|if op==B::Add {a[i]+b[i]} else {a[i]-b[i]})),
            (Value::Exponential(a,g),Value::Number(b)) | (Value::Number(b),Value::Exponential(a,g)) if op==B::Multiply=>Value::Exponential(a*b,g),
            (Value::Exponential(a,g),Value::Number(0.)) | (Value::Number(0.),Value::Exponential(a,g)) if op==B::Add=>Value::Exponential(a,g),
            _=>Value::Unknown,
        }
    }
    match expression {
        E::Unary {op:naga::UnaryOperator::Negate,expr} if profile(&get(*expr))=>Some(binary(naga::BinaryOperator::Multiply,get(*expr),Value::Number(-1.))),
        E::Binary {op,left,right} if profile(&get(*left)) || profile(&get(*right))=>Some(binary(*op,get(*left),get(*right))),
        E::Math {fun:naga::MathFunction::Exp,arg,..}=>match get(*arg) {
            Value::Affine(a) if a.iter().all(|a|a.is_finite()) && a[3].exp().is_finite()=>Some(Value::Exponential(a[3].exp(),[a[0],a[1],a[2]])),
            _=>None,
        },
        E::Math {fun:naga::MathFunction::Max,arg,arg1:Some(other),..} if profile(&get(*arg)) || profile(&get(*other))=> {
            fn nonnegative(a:Value,b:Value)->Value {match (a,b) {
                (Value::Parts(a),Value::Parts(b)) if a.len()==b.len()=>Value::Parts(a.into_iter().zip(b).map(|(a,b)|nonnegative(a,b)).collect()),
                (Value::Exponential(a,g),Value::Number(0.)) | (Value::Number(0.),Value::Exponential(a,g)) if a>=0.=>Value::Exponential(a,g),
                (Value::Number(a),Value::Number(b))=>Value::Number(a.max(b)),
                _=>Value::Unknown,
            }}
            Some(nonnegative(get(*arg),get(*other)))
        },
        _=>None,
    }
}

fn place(function: &naga::Function, handle: naga::Handle<E>) -> Option<(usize, Vec<usize>)> {
    match &function.expressions[handle] {
        E::LocalVariable(h) => Some((h.index(), Vec::new())),
        E::Access { base, .. } => {
            let (root, mut path) = place(function, *base)?;
            path.push(usize::MAX);
            Some((root, path))
        }
        E::AccessIndex { base, index } => {
            let (root, mut path) = place(function, *base)?;
            path.push(*index as usize);
            Some((root, path))
        }
        _ => None,
    }
}
fn argument_place(function: &naga::Function, h: naga::Handle<E>) -> Option<(usize, Vec<usize>)> {
    match &function.expressions[h] {
        E::FunctionArgument(i) => Some((*i as usize, Vec::new())),
        E::AccessIndex { base, index } => {
            let (root, mut path) = argument_place(function, *base)?;
            path.push(*index as usize);
            Some((root, path))
        }
        E::Access { base, .. } => {
            let (root, mut path) = argument_place(function, *base)?;
            path.push(usize::MAX);
            Some((root, path))
        }
        _ => None,
    }
}
fn observe(state: &mut State) {
    if let Some((locals, arguments)) = &mut state.observed {
        for (key, value) in locals.iter_mut() {
            *value=value.widen(state.locals.get(key).unwrap_or(&Value::Unknown));
        }
        for (a,b) in arguments.iter_mut().zip(&state.arguments) { *a=a.widen(b); }
    }
}
fn merge_observed(state: &mut State, other: &State) {
    if let (Some((locals,args)), Some((others,other_args))) = (&mut state.observed,&other.observed) {
        for (key,value) in locals { *value=value.widen(others.get(key).unwrap_or(&Value::Unknown)); }
        for (a,b) in args.iter_mut().zip(other_args) {*a=a.widen(b);}
    }
}
fn unknown(state: &mut State) -> Option<Value> {
    if let Some(reads)=&state.query_reads {reads.lock().unwrap().incomplete=true;}
    if let Some(work)=&state.work {work.lock().unwrap().failure.get_or_insert_with(||"unsupported control flow in medium work proof".into());}
    if let Some(reads)=&state.image_reads {*reads.lock().unwrap()=None;}
    // A live probe abandons code that may read a live source.
    let value = if state.live.is_some() { Value::Time } else { Value::dependent(state.arguments.iter().chain(state.locals.values()).cloned()) };
    state.arguments.fill(value.clone());
    state.locals.values_mut().for_each(|v| *v = value.clone());
    Some(value)
}
fn exit(state: &mut State, value: Value, arguments: Vec<Value>) {
    state.exits = Some(if let Some((before, args)) = state.exits.take() {
        (
            before.merge(&value),
            args.iter()
                .zip(arguments)
                .map(|(a, b)| a.merge(&b))
                .collect(),
        )
    } else {
        (value, arguments)
    });
}
fn merge_exits(state: &mut State, other: &State) {
    merge_observed(state,other);
    if let Some((value, args)) = &other.exits {
        exit(state, value.clone(), args.clone());
    }
}
fn merge_state(state: &mut State, a: &State, b: &State) {
    if state.simulate_loop {
        state.loop_flow = if a.loop_flow == b.loop_flow {a.loop_flow} else {Some(LoopFlow::Mixed)};
    }
    let paths=|a:&[std::sync::Arc<State>],b:&[std::sync::Arc<State>]| {
        let mut result=a.to_vec();
        for path in b {if !result.iter().any(|p|std::sync::Arc::ptr_eq(p,path)) {result.push(path.clone());}}
        result
    };
    state.loop_breaks=paths(&a.loop_breaks,&b.loop_breaks);
    state.loop_continues=paths(&a.loop_continues,&b.loop_continues);
    state.time_control = a.time_control || b.time_control;
    state.effects.extend(a.effects.clone());
    state.effects.extend(b.effects.clone());
    for (key, value) in &mut state.locals {
        *value = a
            .locals
            .get(key)
            .unwrap_or(&Value::Unknown)
            .merge(b.locals.get(key).unwrap_or(&Value::Unknown));
    }
    for (i, value) in state.arguments.iter_mut().enumerate() {
        *value = a.arguments[i].merge(&b.arguments[i]);
    }
    merge_exits(state, a);
    merge_exits(state, b);
}
fn assign(value: &mut Value, path: &[usize], new: Value) {
    if path.is_empty() {
        *value = new;
        return;
    }
    if path[0] == usize::MAX {
        *value = Value::dependent([value.clone(), new]);
        return;
    }
    if let Value::Parts(parts) = value {
        if let Some(part) = parts.get_mut(path[0]) {
            assign(part, &path[1..], new);
        }
    }
}
fn block(
    module: &naga::Module,
    function: &naga::Function,
    args: &[Value],
    state: &mut State,
    body: &naga::Block,
    depth: usize,
) -> Option<Value> {
    let get = |state: &State, h| expression(module, function, args, state, h);
    for statement in body.iter() {
        if let Some(work)=&state.work {
            let mut work=work.lock().unwrap();
            let cost=1+if let S::Emit(range)=statement {range.clone().count() as u32} else {0};
            if cost>1_000_000 || work.operations>1_000_000-cost {
                work.failure=Some("medium work proof exceeds its analysis budget".into());
                return Some(Value::Unknown);
            }
            work.operations+=cost;
        }
        observe(state);
        if (state.simulate_loop && state.loop_flow.is_some()) || state.switch_exit { return None; }
        match statement {
            S::Break if state.in_switch => {state.switch_exit=true;return None;}
            S::Break if state.simulate_loop => {state.loop_flow=Some(LoopFlow::Break);return None;}
            S::Continue if state.simulate_loop => {state.loop_flow=Some(LoopFlow::Continue);return None;}
            S::Emit(range) => {
                for h in range.clone() {
                    state.expressions.remove(&h.index());
                    let value = get(state, h);
                    state.expressions.insert(h.index(), value);
                }
            }
            S::Store { pointer, value } => {
                let value = if state.time_control {
                    Value::Time
                } else {
                    get(state, *value)
                };
                if let Some((root, path)) = place(function, *pointer) {
                    assign(
                        state.locals.entry(root).or_insert(Value::Unknown),
                        &path,
                        value,
                    );
                } else if let Some((root, path)) = argument_place(function, *pointer) {
                    assign(&mut state.arguments[root], &path, value);
                } else if let E::AccessIndex { base, index } = &function.expressions[*pointer] {
                    if let E::GlobalVariable(global) = function.expressions[*base] {
                        if module.global_variables[global]
                            .name
                            .as_deref()
                            .is_some_and(|name| name.starts_with("vork_effects"))
                        {
                            state.effects.insert(*index as usize, value);
                        }
                    }
                } else if let E::Access { base, index } = &function.expressions[*pointer] {
                    if let E::GlobalVariable(global) = function.expressions[*base] {
                        if module.global_variables[global]
                            .name
                            .as_deref()
                            .is_some_and(|name| name.starts_with("vork_effects"))
                        {
                            if let Value::Integer(index) = get(state, *index) {
                                state.effects.insert(index as usize, value);
                            } else {
                                state.effects.insert(usize::MAX,Value::Unknown);
                            }
                        }
                    }
                }
            }
            S::Call {
                function: callee,
                arguments,
                result,
            } => {
                let values: Vec<_> = arguments
                    .iter()
                    .map(|h| {
                        if let Some((root, path)) = place(function, *h) {
                            path.iter().fold(
                                state.locals.get(&root).cloned().unwrap_or(Value::Unknown),
                                |v, i| v.part(*i),
                            )
                        } else if let Some((root, path)) = argument_place(function, *h) {
                            path.iter()
                                .fold(state.arguments.get(root).cloned().unwrap_or(Value::Unknown), |v, i| v.part(*i))
                        } else {
                            get(state, *h)
                        }
                    })
                    .collect();
                let (value, mut after) =
                    evaluate_mode(module, &module.functions[*callee], &values, depth + 1,state.dependency,state.words.clone(),state.live,state.certified,state.numeric,state.image_reads.clone(),state.query_reads.clone(),state.work.clone(),state.output_slice);
                if state.certified {
                    state.proof_failures.lock().unwrap().extend(after.proof_failures.lock().unwrap().iter().cloned());
                }
                // Writes of a call under time-dependent control are Time too.
                if state.time_control {
                    after.arguments.iter_mut().for_each(|a| *a = Value::Time);
                    after.effects.values_mut().for_each(|e| *e = Value::Time);
                }
                for (i, h) in arguments.iter().enumerate() {
                    if !matches!(
                        module.types[module.functions[*callee].arguments[i].ty].inner,
                        naga::TypeInner::Pointer { .. }
                    ) {
                        continue;
                    }
                    let written = after.arguments.get(i).cloned().unwrap_or(Value::Unknown);
                    if let Some((root, path)) = place(function, *h) {
                        assign(
                            state.locals.entry(root).or_insert(Value::Unknown),
                            &path,
                            written,
                        );
                    } else if let Some((root, path)) = argument_place(function, *h) {
                        assign(&mut state.arguments[root], &path, written);
                    }
                }
                state.effects.extend(after.effects);
                if let Some(result) = result {
                    state.expressions.insert(result.index(), value);
                }
            }
            S::Return { value } => {
                return Some(if state.time_control {
                    Value::Time
                } else {
                    value.map(|h| get(state, h)).unwrap_or(Value::Unknown)
                });
            }
            S::Block(body) => {
                if let Some(value) = block(module, function, args, state, body, depth) {
                    return Some(value);
                }
            }
            S::If { condition, accept, reject } => {
                if state.dependency && !state.output_slice && !get(state,*condition).independent() {return unknown(state);}
                if let Value::Bool(condition) = get(state, *condition) {
                    if let Some(value) = block(module,function,args,state,if condition {accept} else {reject},depth) {return Some(value);}
                    continue;
                }
                let dependent = if state.output_slice {!get(state,*condition).independent()} else {get(state, *condition).time()};
                let enclosing_control=state.time_control;
                let mut a = state.clone();
                let mut b = state.clone();
                a.time_control |= dependent;
                b.time_control |= dependent;
                let ra = block(module, function, args, &mut a, accept, depth);
                let rb = block(module, function, args, &mut b, reject, depth);
                // Keep a path leaving this iteration separate from the path
                // executing its remaining statements. Joining their counters
                // here would lose a finite bound (or execute skipped stores).
                if state.simulate_loop && state.certified && ra.is_none() && rb.is_none() && a.loop_flow != b.loop_flow {
                    let split=match (a.loop_flow,b.loop_flow) {
                        (Some(LoopFlow::Break|LoopFlow::Continue),None)=>Some((a.clone(),b.clone())),
                        (None,Some(LoopFlow::Break|LoopFlow::Continue))=>Some((b.clone(),a.clone())),
                        _=>None,
                    };
                    if let Some((mut stopped,mut running))=split {
                        let flow=stopped.loop_flow.take();
                        running.loop_breaks.append(&mut stopped.loop_breaks);
                        running.loop_continues.append(&mut stopped.loop_continues);
                        if flow==Some(LoopFlow::Break) {running.loop_breaks.push(std::sync::Arc::new(stopped));}
                        else {running.loop_continues.push(std::sync::Arc::new(stopped));}
                        *state=running;
                        continue;
                    } else {
                        // Both terminal paths skip the rest of this body.
                        // Preserve each destination until the iteration joins.
                        merge_state(state,&a,&b);
                        if matches!((a.loop_flow,b.loop_flow),(Some(LoopFlow::Break),Some(LoopFlow::Continue))|(Some(LoopFlow::Continue),Some(LoopFlow::Break))) {
                            for mut branch in [a,b] {
                                let flow=branch.loop_flow.take();
                                branch.loop_breaks.clear();branch.loop_continues.clear();
                                if flow==Some(LoopFlow::Break) {state.loop_breaks.push(std::sync::Arc::new(branch));}
                                else {state.loop_continues.push(std::sync::Arc::new(branch));}
                            }
                            state.loop_flow=Some(LoopFlow::Break);
                        }
                        return None;
                    }
                }
                match (ra, rb) {
                    (Some(x), Some(y)) => {
                        merge_state(state, &a, &b);
                        return Some(x.merge(&y));
                    }
                    (Some(value), None) => {
                        let args = a.arguments.clone();
                        exit(&mut a, value, args);
                        *state = b;
                        merge_exits(state, &a);
                    }
                    (None, Some(value)) => {
                        let args = b.arguments.clone();
                        exit(&mut b, value, args);
                        *state = a;
                        merge_exits(state, &b);
                    }
                    // One path leaving the switch while the other runs on
                    // cannot be joined statement by statement.
                    (None, None) if a.switch_exit != b.switch_exit => return unknown(state),
                    (None, None) => {
                        merge_state(state, &a, &b);
                        state.switch_exit = a.switch_exit;
                        // Both arms reconverge. Their writes retain dependence;
                        // subsequent unrelated computations do not inherit it.
                        if state.output_slice {state.time_control=enclosing_control;}
                    }
                }
            }
            S::Loop {
                body, continuing, break_if,
            } => {
                if let Some((after,returned)) = constant_loop(module,function,args,state,body,continuing,*break_if,depth) {
                    *state=after;
                    if returned.is_some() {return returned;}
                    continue;
                }
                if let Some(work)=&state.work {
                    work.lock().unwrap().failure=Some("medium loop count is not provably finite; use statically bounded loops".into());
                    return Some(Value::Unknown);
                }
                if state.dependency || state.certified {
                    if state.certified {state.proof_failures.lock().unwrap().insert(format!("{}: loop count is not provably finite; use a provably bounded loop",function.name.as_deref().unwrap_or("entry")));}
                    return unknown(state);
                }
                // Join every intermediate store, including paths before break,
                // continue and return. This preserves inductive struct fields
                // (e.g. zero Blend effects) without treating a later reset as proof.
                let had_observer=state.observed.is_some();
                let mut current = state.clone();
                current.in_switch = false;
                if current.observed.is_none() {
                    current.observed=Some((current.locals.clone(),current.arguments.clone()));
                }
                let mut stable = false;
                for _ in 0..64 {
                    let mut after = current.clone();
                    if let Some(value) = block(module, function, args, &mut after, body, depth) {
                        let args = after.arguments.clone();
                        exit(&mut after, value, args);
                    }
                    if let Some(value) =
                        block(module, function, args, &mut after, continuing, depth)
                    {
                        let args = after.arguments.clone();
                        exit(&mut after, value, args);
                    }
                    after.time_control |= break_if.is_some_and(|h| get(&after,h).time());
                    let mut merged = current.clone();
                    merge_state(&mut merged, &current, &after);
                    if let Some((locals,args))=&merged.observed {
                        merged.locals=locals.clone();
                        merged.arguments=args.clone();
                    }
                    stable = merged.locals == current.locals
                        && merged.arguments == current.arguments
                        && merged.exits == current.exits
                        && merged.time_control == current.time_control;
                    current = merged;
                    if stable {
                        break;
                    }
                }
                if !stable {
                    return unknown(state);
                }
                state.time_control = current.time_control;
                state.locals = current.locals;
                state.arguments = current.arguments;
                state.exits = current.exits;
                state.time_control |= current.time_control;
                state.observed=if had_observer {current.observed} else {None};
            }
            S::Switch { selector, cases } => {
                if state.dependency && !get(state,*selector).independent() {return unknown(state);}
                // A case body runs until its end or a `break` that leaves the switch.
                let case = |state: &mut State, body: &naga::Block| {
                    let outer = state.in_switch;
                    state.in_switch = true;
                    let value = block(module, function, args, state, body, depth);
                    (state.in_switch, state.switch_exit) = (outer, false);
                    value
                };
                // A known selector runs its own case (or the default) only.
                if let Value::Integer(n) = get(state, *selector) {
                    let taken = cases.iter().find(|case| match case.value {
                        naga::SwitchValue::I32(v) => v as u32 == n as u32,
                        naga::SwitchValue::U32(v) => v == n as u32,
                        naga::SwitchValue::Default => false,
                    }).or_else(|| cases.iter().find(|case| case.value == naga::SwitchValue::Default));
                    if let Some(taken) = taken.filter(|taken| !taken.fall_through) {
                        if let Some(value) = case(state, &taken.body) {return Some(value);}
                        continue;
                    }
                }
                state.time_control |= get(state, *selector).time();
                if cases.iter().any(|case| case.fall_through) {
                    return unknown(state);
                }
                let mut live: Option<State> = None;
                let mut terminal = state.clone();
                let mut returned: Option<Value> = None;
                for each in cases {
                    let mut branch = state.clone();
                    if let Some(value) = case(&mut branch, &each.body) {
                        let arguments = branch.arguments.clone();
                        exit(&mut branch, value.clone(), arguments);
                        returned = Some(returned.map_or(value.clone(), |v| v.merge(&value)));
                        merge_exits(&mut terminal, &branch);
                    } else if let Some(before) = live {
                        let mut merged = before.clone();
                        merge_state(&mut merged, &before, &branch);
                        live = Some(merged);
                    } else {
                        live = Some(branch);
                    }
                }
                if let Some(after) = live {
                    *state = after;
                    merge_exits(state, &terminal);
                } else {
                    *state = terminal;
                    return returned;
                }
            }
            S::Atomic {result,..} if state.query_reads.is_some() => {
                if let Some(result)=result {state.expressions.insert(result.index(),Value::Unknown);}
            }
            S::Atomic { .. } | S::ImageStore { .. } | S::Kill => return unknown(state),
            _ => {}
        }
    }
    observe(state);
    None
}
// Execute only loops whose control flow is proven at every iteration. Unknown
// control or the analysis budget fall back to conservative widening. Return
// paths are retained as exits while the remaining paths continue to termination.
fn constant_loop(
    module: &naga::Module, function: &naga::Function, args: &[Value], initial: &State,
    body: &naga::Block, continuing: &naga::Block, break_if: Option<naga::Handle<E>>, depth: usize,
) -> Option<(State,Option<Value>)> {
    let mut state=initial.clone();
    state.simulate_loop=true;
    state.loop_flow=None;
    state.loop_breaks.clear();
    state.loop_continues.clear();
    state.in_switch=false;
    let join=|states:Vec<std::sync::Arc<State>>|->Option<State> {
        let mut states=states.into_iter();
        let mut result=states.next()?.as_ref().clone();
        for other in states {let before=result.clone();merge_state(&mut result,&before,&other);}
        Some(result)
    };
    let leave=|mut state: State,returned:Option<Value>| {
        let mut breaks=std::mem::take(&mut state.loop_breaks);
        let returned=if breaks.is_empty() {returned} else {
            if let Some(value)=returned {let arguments=state.arguments.clone();exit(&mut state,value,arguments);}
            else {breaks.push(std::sync::Arc::new(state.clone()));}
            let mut after=join(breaks).unwrap();
            merge_exits(&mut after,&state);
            state=after;
            None
        };
        state.simulate_loop=initial.simulate_loop;
        state.in_switch=initial.in_switch;
        state.loop_flow=None;
        state.loop_breaks=initial.loop_breaks.clone();
        state.loop_continues=initial.loop_continues.clone();
        (state,returned)
    };
    for _ in 0..4096 {
        let before_locals=state.locals.clone();
        let before_arguments=state.arguments.clone();
        let returned=block(module,function,args,&mut state,body,depth);
        let mut continues=std::mem::take(&mut state.loop_continues);
        let running=if let Some(value)=returned {
            if continues.is_empty() {return Some(leave(state,Some(value)));}
            let arguments=state.arguments.clone();exit(&mut state,value,arguments);
            false
        }else {
            match state.loop_flow {
                Some(LoopFlow::Break) if continues.is_empty()=>return Some(leave(state,None)),
                Some(LoopFlow::Break)=> {
                    let mut stopped=state.clone();stopped.loop_breaks.clear();stopped.loop_continues.clear();
                    state.loop_breaks.push(std::sync::Arc::new(stopped));
                    false
                },
                Some(LoopFlow::Mixed)=>return None,
                _=>true,
            }
        };
        if !continues.is_empty() {
            let breaks=std::mem::take(&mut state.loop_breaks);
            let terminal=state.clone();
            if running {continues.push(std::sync::Arc::new(state));}
            state=join(continues)?;
            state.loop_breaks=breaks;
            merge_exits(&mut state,&terminal);
        }
        // Coalesce completed paths once per iteration, so a runtime early
        // exit does not retain a growing history of interpreter snapshots.
        if state.loop_breaks.len()>1 {
            let mut after=join(std::mem::take(&mut state.loop_breaks))?;
            after.loop_breaks.clear();after.loop_continues.clear();
            state.loop_breaks.push(std::sync::Arc::new(after));
        }
        state.loop_flow=None;
        if let Some(value)=block(module,function,args,&mut state,continuing,depth) {return Some(leave(state,Some(value)));}
        if let Some(condition)=break_if {
            match expression(module,function,args,&state,condition) {
                Value::Bool(true)=>return Some(leave(state,None)),
                Value::Bool(false)=>{},
                _=>return None,
            }
        }
        if state.loop_flow.is_some() {return None;}
        // An unchanged abstract store cannot establish a later exit. Avoid
        // spending the full iteration budget on a runtime/unbounded loop.
        if state.locals==before_locals && state.arguments==before_arguments {return None;}
    }
    None
}
fn evaluate(
    module: &naga::Module,
    function: &naga::Function,
    args: &[Value],
    depth: usize,
    dependency: bool,
    words: Option<std::sync::Arc<[Option<u32>]>>,
    live: Option<Live>,
    certified: bool,
) -> (Value, State) {
    evaluate_mode(module,function,args,depth,dependency,words,live,certified,false,None,None,None,false)
}
fn evaluate_mode(
    module: &naga::Module,
    function: &naga::Function,
    args: &[Value],
    depth: usize,
    dependency: bool,
    words: Option<std::sync::Arc<[Option<u32>]>>,
    live: Option<Live>,
    certified: bool,
    numeric: bool,
    image_reads: Option<std::sync::Arc<std::sync::Mutex<Option<BTreeSet<u32>>>>>,
    query_reads: Option<std::sync::Arc<std::sync::Mutex<QueryReads>>>,
    work: Option<std::sync::Arc<std::sync::Mutex<WorkProof>>>,
    output_slice: bool,
) -> (Value, State) {
    if function.name.as_deref() == Some("vork_trace_field") {
        if let Some(reads)=&query_reads {
            let mask=match args.get(4) {Some(Value::Integer(mask))=>(*mask as u32&255) as u8,_=>255};
            let object=match args.get(5) {Some(Value::Integer(object))=>u32::try_from(*object).ok().filter(|&object|object!=u32::MAX),_=>None};
            reads.lock().unwrap().filters.insert(QueryFilter {mask,object});
        }
        let value=if live.is_some_and(|live|!live.wavelength) {Value::Time} else {Value::dependent(args.iter().cloned())};
        return (value, State { arguments: args.to_vec(), ..Default::default() });
    }
    if let Some(reads)=&image_reads {
        let name=function.name.as_deref().unwrap_or("");
        if matches!(name,SAMPLE_IMAGE|"vork_compositor_load"|"vork_compositor_size"|"vork_compositor_context"|MATERIAL_WORD) {
            if matches!(name,SAMPLE_IMAGE|"vork_compositor_load"|"vork_compositor_size") {
                let mut reads=reads.lock().unwrap();
                match args.first() {
                    Some(Value::Integer(0)) if name=="vork_compositor_size"=>(),
                    Some(Value::Integer(slot)) if u32::try_from(*slot).is_ok()=>{if let Some(reads)=reads.as_mut() {reads.insert(*slot as u32);}},
                    _=>*reads=None,
                }
            }
            return (Value::Unknown,State {image_reads,arguments:args.to_vec(),..Default::default()});
        }
    }
    if let Some(live) = live {
        let name = function.name.as_deref().unwrap_or("");
        let value = match name {
            _ if CAMERA_CALLS.contains(&name) => Some(if live.wavelength {Value::dependent(args.iter().cloned())} else {Value::Time}),
            // A texel varies with live texels or with its coordinates, slot and gradients.
            SAMPLE_IMAGE if live.texels => Some(Value::Time),
            SAMPLE_IMAGE => Some(Value::dependent(args.iter().cloned())),
            _ => None,
        };
        if let Some(value) = value {
            return (value, State {arguments: args.to_vec(), live: Some(live), ..Default::default()});
        }
    }
    if (dependency || words.is_some()) && function.name.as_deref() == Some(MATERIAL_WORD) {
        let value=match (&words,args.first()) {
            (Some(words),Some(Value::Integer(index))) => match usize::try_from(*index).ok().and_then(|i| words.get(i)) {
                Some(Some(value)) => Value::Integer(i64::from(*value)),
                // An animated input has an unknown value, but its fixed word
                // address is independent of position during this invocation.
                Some(None) if dependency => Value::Independent,
                _ => Value::Unknown,
            },
            (None,_) if args.iter().all(Value::independent) => Value::Independent,
            _ => Value::Unknown,
        };
        return (value,State {dependency,words,live,arguments:args.to_vec(),..Default::default()});
    }
    // Dense sampling is a convex interpolant clamped to the reduced finite
    // range. These words are renderer-owned publication metadata, never an
    // authored maximum. Exterior samples are exactly zero.
    if function.name.as_deref().is_some_and(|n| n.starts_with("vork_dense_sample")) {
        let value = if certified {
            if let (Some(words),Some(Value::Integer(slot))) = (&words,args.first()) {
                let at=usize::try_from(*slot).unwrap_or(usize::MAX);
                let word=|i:usize| at.checked_add(i).and_then(|i|words.get(i)).copied().flatten();
                match (word(0),word(4),word(5)) {
                    (Some(u32::MAX),Some(lo),Some(hi)) => {
                        let (lo,hi)=(f32::from_bits(lo),f32::from_bits(hi));
                        if lo.is_finite() && hi.is_finite() && lo<=hi {
                            Value::Parts(vec![Value::NumberRange(f64::from(lo.min(0.)),f64::from(hi.max(0.)));3])
                        } else {Value::Unknown}
                    },
                    _=>Value::Unknown,
                }
            } else {Value::Unknown}
        } else {Value::dependent(args.iter().cloned())};
        return (value,State {dependency,words,live,arguments:args.to_vec(),..Default::default()});
    }
    // Runtime resources have unknown values, but retain time-dependent addresses
    // and coordinates. Their own revisions are tracked separately from sample time.
    // Validation bodies only check signatures and cannot prove resource values.
    // Image discovery executes its instrumented stub, so image calls stay below.
    if function.name.as_deref().is_some_and(|name| {
        matches!(
            name,
            "vork_attribute" | "vork_object_to_generated"
                | "vork_world_to_object"
                | "vork_world_normal_to_object"
                | "vork_object_direction_to_world"
                | "vork_object_normal_to_world"
                | "vork_world_to_camera"
                | "vork_world_to_window"
        )
    }) {
        return (
            Value::dependent(args.iter().cloned()),
            State {
                arguments: args.to_vec(),
                ..Default::default()
            },
        );
    }
    let mut state = State {
        work,
        image_reads,
        query_reads,
        numeric,
        certified,
        dependency,
        output_slice,
        words,
        live,
        arguments: args.to_vec(),
        ..Default::default()
    };
    // Validated Naga functions form an acyclic call graph. Its size bounds
    // every finite call chain; the smaller budget belongs to other analyses.
    if depth > if live.is_some_and(|l|l.query_bounds) {module.functions.len()}else{32} {
        if certified {state.proof_failures.lock().unwrap().insert(format!("{}: call depth is not provably finite; remove recursive calls",function.name.as_deref().unwrap_or("entry")));}
        unknown(&mut state);
        return (
            state.arguments.first().cloned().unwrap_or(Value::Unknown),
            state,
        );
    }
    for (h, local) in function.local_variables.iter() {
        let value = local
            .init
            .map(|h| expression(module, function, args, &state, h))
            .unwrap_or_else(|| zero(module, local.ty));
        state.locals.insert(h.index(), value);
    }
    let value =
        block(module, function, args, &mut state, &function.body, depth).unwrap_or(Value::Unknown);
    let arguments = state.arguments.clone();
    exit(&mut state, value, arguments);
    let (value, arguments) = state.exits.as_ref().unwrap().clone();
    state.arguments = arguments;
    (value, state)
}
/// Use the same interprocedural output slice as closure effects. A context may
/// carry time through helpers without ever reading it; signature/field presence
/// alone cannot decide whether an environment needs per-sample evaluation.
pub(super) fn environment_uses_time(wgsl: &str, entry: &str) -> Result<bool> {
    environment_depends_on(wgsl, entry, false)
}
pub(super) fn environment_uses_origin(wgsl: &str, entry: &str) -> Result<bool> {
    environment_depends_on(wgsl, entry, true)
}
pub(super) fn environment_constant_camera(wgsl: &str, entry: &str, uniform_images: bool) -> Result<bool> {
    // Uniform image sampling returns the same bits for every coordinate. Keep
    // slot dependence: a direction-dependent choice between constants varies.
    // Use an unknown word, never a made-up numeric texel: zero could otherwise
    // incorrectly erase a procedural contribution multiplied by the image.
    let texel=format!("bitcast<f32>({MATERIAL_WORD}(slot))");
    let sample = if uniform_images { texel } else { format!("{texel}+dot(uv+dx+dy,vec2<f32>(1.))") };
    let source = format!("fn {MATERIAL_WORD}(i:u32)->u32{{return i;}}\nfn {SAMPLE_IMAGE}(slot:u32,uv:vec2<f32>,dx:vec2<f32>,dy:vec2<f32>)->vec4<f32>{{return vec4<f32>({sample});}}\n{wgsl}");
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|e| anyhow::anyhow!(e.emit_to_string(&source)))?;
    let function = module.functions.iter().find(|(_,f)| f.name.as_deref()==Some(entry))
        .context("environment entry missing")?.1;
    let spatial=Value::Parts(vec![Value::Position;3]);
    let (_,state)=evaluate_mode(&module,function,&[spatial.clone(),spatial,Value::Position,Value::Independent,
        Value::Parts(vec![Value::Parts(vec![Value::Number(0.);4]);2])],0,true,None,None,false,false,None,None,None,true);
    Ok(state.arguments[4].part(1).independent())
}
fn environment_depends_on(wgsl: &str, entry: &str, origin: bool) -> Result<bool> {
    let unknown=Value::Parts(vec![Value::Unknown;3]);
    output_uses_time(wgsl,entry,&[
        if origin {Value::Parts(vec![Value::Time;3])} else {unknown.clone()},
        unknown, if origin {Value::Unknown} else {Value::Time}, Value::Unknown,
        Value::Parts(vec![Value::Parts(vec![Value::Number(0.);4]);2]),
    ],4)
}
fn output_uses_time(wgsl: &str, entry: &str, args:&[Value], output:usize) -> Result<bool> {
    let source = format!("fn {MATERIAL_WORD}(i:u32)->u32{{return i;}}\nfn {SAMPLE_IMAGE}(slot:u32,uv:vec2<f32>,dx:vec2<f32>,dy:vec2<f32>)->vec4<f32>{{return vec4<f32>(f32(slot)+dot(uv+dx+dy,vec2<f32>(1.)));}}\n{wgsl}");
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|e| anyhow::anyhow!(e.emit_to_string(&source)))?;
    let function = module.functions.iter()
        .find(|(_, f)| f.name.as_deref() == Some(entry))
        .context("environment entry missing")?.1;
    let (_, state) = evaluate(&module, function, args, 0,false,None,None,false);
    Ok(state.arguments[output].time())
}
/// Follow compiler-resolved calls, aggregate fields and local stores. Unknown
/// operations and spatial control flow conservatively retain the general path.
pub(super) fn medium_homogeneous(module: &naga::Module, entry: &str, words:Option<&[Option<u32>]>) -> Result<bool> {
    let (_, function)=module.functions.iter().find(|(_, f)|f.name.as_deref()==Some(entry))
        .context("medium coefficient bridge missing")?;
    let base=if words.is_some() {Value::Integer(0)} else {Value::Independent};
    let words=words.map(|words| words.iter().copied().collect());
    let (value, _)=evaluate(module,function,&[Value::Position,Value::Independent,base,Value::Position,Value::Independent],0,true,words,None,false);
    Ok(value.independent())
}
/// The same complete-expression proof with time as the varying coordinate.
pub(super) fn medium_time_dependent(module:&naga::Module,entry:&str,words:&[Option<u32>])->Result<bool> {
    let (_,function)=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry))
        .context("medium coefficient bridge missing")?;
    let words=words.iter().copied().collect();
    let (value,_)=evaluate(module,function,&[Value::Independent,Value::Position,Value::Integer(0),Value::Independent,Value::Independent],0,true,Some(words),None,false);
    Ok(!value.independent())
}
pub(super) fn medium_emission_free(module:&naga::Module,entry:&str,words:&[Option<u32>])->Result<bool> {
    let (_,function)=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry))
        .context("medium coefficient bridge missing")?;
    let words=words.iter().copied().collect();
    let (value,_)=evaluate(module,function,&[Value::Position,Value::Independent,Value::Integer(0),Value::Position,Value::Independent],0,true,Some(words),None,false);
    Ok(value.part(2).is_zero())
}
/// Prove scattering is identically zero, including position, time and wavelength.
/// Animated input words remain unknown; no finite point probes establish this.
pub(super) fn medium_scattering_free(module:&naga::Module,entry:&str,words:&[Option<u32>])->Result<bool> {
    let (_,function)=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry))
        .context("medium coefficient bridge missing")?;
    let words=words.iter().copied().collect();
    let (value,_)=evaluate(module,function,&[Value::Position,Value::Position,Value::Integer(0),Value::Position,Value::Position],0,true,Some(words),None,false);
    Ok(value.part(1).is_zero())
}
/// Prove a one-axis profile constant on both exterior half-spaces. This is
/// interval dependency analysis, never equality of finite point probes.
pub(super) fn medium_tail_slab(module:&naga::Module,entry:&str,words:&[Option<u32>])->Result<Option<(usize,f32)>> {
    let (_,function)=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry))
        .context("medium coefficient bridge missing")?;
    let words:std::sync::Arc<[Option<u32>]>=words.iter().copied().collect();
    let independent=|position| evaluate(module,function,&[position,Value::Independent,Value::Integer(0),Value::Position,Value::Independent],0,true,Some(words.clone()),None,false).0.independent();
    for axis in 0..3 {
        let mut position=vec![Value::Position;3];position[axis]=Value::Independent;
        if !independent(Value::Parts(position)) {continue;}
        for exponent in -126..=126 {
            let bound=2f32.powi(exponent);
            let exterior=|positive| {
                let mut p=vec![Value::Position;3];
                p[axis]=if positive {Value::SpatialRange(f64::from(bound),f64::INFINITY)}
                    else {Value::SpatialRange(f64::NEG_INFINITY,-f64::from(bound))};
                independent(Value::Parts(p))
            };
            if exterior(true) && exterior(false) {return Ok(Some((axis,bound)));}
        }
    }
    Ok(None)
}
/// Prove the complete absorption/scattering fields share one exponential
/// profile, with spatially constant emission and phase. Numeric amplitudes are
/// immutable-input specializations; no library path or input name is inspected.
pub(super) fn medium_exponential(module:&naga::Module,entry:&str,words:&[Option<u32>])->Result<Option<([f32;3],[f32;3],[f32;3])>> {
    medium_connection_profile(module,entry,words,false)
}
pub(super) fn medium_connection_profile(module:&naga::Module,entry:&str,words:&[Option<u32>],constants:bool)->Result<Option<([f32;3],[f32;3],[f32;3])>> {
    let (_,function)=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry)).context("medium coefficient bridge missing")?;
    let position=Value::Parts((0..3).map(|axis|Value::Affine(std::array::from_fn(|i|if i==axis {1.} else {0.}))).collect());
    let words=words.iter().copied().collect();
    // The world consumer supplies identity-space local coordinates, so both
    // context positions describe this affine world point. Finite-domain
    // extinction bounds are a separate proof and do not use this gradient.
    let (value,_)=evaluate(module,function,&[position.clone(),Value::Independent,Value::Integer(0),position,Value::Independent],0,true,Some(words),None,false);
    if !value.part(2).independent() || !value.part(3).independent() {return Ok(None);}
    let mut gradient=None;let mut coefficients=[[0f32;3];2];
    for (field,channels) in coefficients.iter_mut().enumerate() {
        for (channel,result) in channels.iter_mut().enumerate() {
            match value.part(field).part(channel) {
                // A vanished positive amplitude is not vacuum: its field can
                // become representable away from the origin. Refuse this
                // certificate instead of erasing an exponential world tail.
                Value::Exponential(a,g) if a>0. && (a as f32)>0. && a.is_finite() && (a as f32).is_finite() && (!constants || normal_f32(a))=> {
                    *result=a as f32;
                    if a>0. {if gradient.is_some_and(|previous|previous!=g) {return Ok(None);}gradient=Some(g);}
                },
                Value::Number(0.)=>{},
                Value::Number(a) if constants && a>0. && normal_f32(a)=> {
                    if gradient.is_some_and(|previous|previous!=[0.;3]) {return Ok(None);}
                    gradient=Some([0.;3]);*result=a as f32;
                },
                _=>return Ok(None),
            }
        }
    }
    Ok(gradient.or_else(||constants.then_some([0.;3])).map(|g|(coefficients[0],coefficients[1],g.map(|x|x as f32))))
}

fn calls(block: &naga::Block, out: &mut Vec<naga::Handle<naga::Function>>) {
    for statement in block.iter() {
        match statement {
            S::Call { function, .. } => out.push(*function),
            S::Block(body) => calls(body, out),
            S::If { accept, reject, .. } => {
                calls(accept, out);
                calls(reject, out);
            }
            S::Switch { cases, .. } => {
                for case in cases {
                    calls(&case.body, out);
                }
            }
            S::Loop {
                body, continuing, ..
            } => {
                calls(body, out);
                calls(continuing, out);
            }
            _ => (),
        }
    }
}

pub(super) fn displacement_depends_on(wgsl: &str, entry: &str, parameter: usize) -> Result<bool> {
    let source = super::coordinates::validation_stubs().to_owned()
        + &format!(
            "fn vork_blend_overflow(count:u32,capacity:u32) {{}} fn {WAVELENGTH_EFFECT}(v:f32)->f32{{return v;}} fn {MATERIAL_WORD}(i:u32)->u32{{return i;}} fn {SAMPLE_IMAGE}(s:u32,u:vec2<f32>,x:vec2<f32>,y:vec2<f32>)->vec4<f32>{{return vec4(f32(s)+dot(u+x+y,vec2<f32>(1.)));}} fn vork_attribute(s:u32,d:u32)->vec4<f32>{{return vec4(0.);}}\n{wgsl}"
        );
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|e| anyhow::anyhow!(e.emit_to_string(&source)))?;
    let function = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some(entry))
        .context("displacement wrapper missing")?
        .1;
    if parameter == 1 {
        let tag = entry
            .strip_prefix("vork_displace_")
            .and_then(|s| s.split_once("_0_"))
            .context("displacement wrapper tag missing")?
            .0;
        let surface = format!("surface_0_{tag}");
        let mut pending = Vec::new();
        calls(&function.body, &mut pending);
        let mut visited = BTreeSet::new();
        while let Some(callee) = pending.pop() {
            if !visited.insert(callee.index()) {
                continue;
            }
            let callee = &module.functions[callee];
            let name = callee.name.as_deref().unwrap_or("");
            if name == format!("displace_0_{tag}") {
                let ty = callee
                    .result
                    .as_ref()
                    .context("displace must return float3")?
                    .ty;
                ensure!(
                    matches!(
                        module.types[ty].inner,
                        naga::TypeInner::Vector {
                            size: naga::VectorSize::Tri,
                            scalar: naga::Scalar {
                                kind: naga::ScalarKind::Float,
                                width: 4
                            }
                        }
                    ),
                    "displace must return float3"
                );
            }

            ensure!(
                name != surface
                    && !matches!(
                        name,
                        "vork_world_to_camera"
                            | "vork_world_to_window"
                            | "vork_object_to_generated"
                            | "vork_world_to_object"
                            | "vork_world_normal_to_object"
                            | "vork_object_direction_to_world"
                            | "vork_object_normal_to_world"
                    ),
                "displace cannot call surface or read scene transforms, cameras or rendered images ({name})"
            );
            calls(&callee.body, &mut pending);
        }
    }
    let mut args = vec![
        Value::Parts(vec![Value::Unknown; 3]),
        Value::Parts(vec![Value::Unknown; 3]),
        Value::Unknown,
        Value::Unknown,
    ];
    args.resize(function.arguments.len(),Value::Unknown);
    args[parameter] = if parameter < 2 {
        Value::Parts(vec![Value::Time; 3])
    } else {
        Value::Time
    };
    Ok(evaluate(&module, function, &args, 0, false, None, None, false).0.time())
}
/// Trace lowered image handles through structs, helpers and control flow with
/// the same interprocedural evaluator as material proofs. The shared observation
/// survives branch returns and loop widening. Unknown flow/addresses fail closed.
pub(super) fn composite_reads(wgsl: &str, entry: &str, abi: &Abi) -> Result<Option<Vec<u32>>> {
    let module=super::validated_module(wgsl,abi)?;
    let function=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry)).context("compositor wrapper missing")?.1;
    let mut pending=Vec::new();calls(&function.body,&mut pending);
    let mut visited=BTreeSet::new();let mut image_call=false;
    while let Some(callee)=pending.pop() {
        if !visited.insert(callee.index()) {continue;}
        let callee=&module.functions[callee];
        image_call |= matches!(callee.name.as_deref(),Some(SAMPLE_IMAGE|"vork_compositor_load"|"vork_compositor_size"));
        calls(&callee.body,&mut pending);
    }
    if !image_call {return Ok(Some(Vec::new()));}
    let reads=std::sync::Arc::new(std::sync::Mutex::new(Some(BTreeSet::new())));
    let args=vec![Value::Unknown;function.arguments.len()];
    evaluate_mode(&module,function,&args,0,false,None,None,false,false,Some(reads.clone()),None,None,false);
    let result=reads.lock().unwrap().clone().map(|reads|reads.into_iter().collect());
    Ok(result)
}

pub(super) fn displacement_reads(wgsl: &str, count: usize) -> Result<Vec<bool>> {
    let wgsl = wgsl.replace("vork_attribute", "vork_displacement_attribute_probe");
    let source = super::coordinates::validation_stubs().to_owned()
        + &format!(
            "@group(9) @binding(0) var<storage,read_write> vork_effects_displacement:array<u32>; fn vork_displacement_attribute_probe(slot:u32,differential:u32)->vec4<f32> {{vork_effects_displacement[slot]=1u;return vec4(f32(slot));}} fn vork_blend_overflow(count:u32,capacity:u32) {{}} fn {WAVELENGTH_EFFECT}(v:f32)->f32{{return v;}} fn {MATERIAL_WORD}(i:u32)->u32 {{return i/5u;}} fn {SAMPLE_IMAGE}(s:u32,u:vec2<f32>,x:vec2<f32>,y:vec2<f32>)->vec4<f32>{{vork_effects_displacement[s]=1u;return vec4(u+x+y,f32(s),1.);}}\n{wgsl}"
        );
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|e| anyhow::anyhow!(e.emit_to_string(&source)))?;
    let entry = module
        .entry_points
        .iter()
        .find(|e| e.name == "vork_entry")
        .context("displacement probe entry missing")?;
    let (_, state) = evaluate(&module, &entry.function, &[], 0, false, None, None, false);
    Ok((0..count)
        .map(|i| state.effects.contains_key(&i) || state.effects.contains_key(&usize::MAX))
        .collect())
}

pub(super) fn calls_trace(wgsl: &str, entry: &str, abi: &Abi) -> Result<bool> {
    let module = super::validated_module(wgsl, abi)?;
    let function = module.functions.iter().find(|(_, f)| f.name.as_deref() == Some(entry)).map(|(_, f)|f)
        .or_else(||module.entry_points.iter().find(|e|e.name==entry).map(|e|&e.function)).context("query analysis entry missing")?;
    let mut pending=Vec::new(); calls(&function.body,&mut pending);
    let mut visited=BTreeSet::new();
    while let Some(handle)=pending.pop() {
        if !visited.insert(handle.index()) {continue;}
        let callee=&module.functions[handle];
        if callee.name.as_deref()==Some("vork_trace_field") {return Ok(true);}
        calls(&callee.body,&mut pending);
    }
    Ok(false)
}

fn query_bound_failure(failures:&BTreeSet<String>)->Option<&str> {
    failures.iter().find(|reason| reason.contains(": loop count is not provably finite") || reason.contains(": call depth is not provably finite")).map(String::as_str)
}

pub(super) fn compute_query_filters(wgsl:&str,abi:&Abi)->Result<Vec<QueryFilter>> {
    let module=super::validated_module(wgsl,abi)?;
    let function=&module.entry_points.iter().find(|entry|entry.name=="vork_entry").context("compute query entry missing")?.function;
    let reads=std::sync::Arc::new(std::sync::Mutex::new(QueryReads::default()));
    let args=vec![Value::Unknown;function.arguments.len()];
    let (_,state)=evaluate_mode(&module,function,&args,0,false,Some(Vec::<Option<u32>>::new().into()),Some(Live {texels:false,wavelength:true,query_bounds:true}),true,false,None,Some(reads.clone()),None,false);
    let failures=state.proof_failures.lock().unwrap();
    if let Some(reason)=query_bound_failure(&failures) {anyhow::bail!("trace: {reason}");}
    ensure!(!reads.lock().unwrap().incomplete,"trace: compute entry query filters are not provably finite; use fixed query filters");
    let filters=reads.lock().unwrap().filters.iter().copied().collect();
    Ok(filters)
}

/// Admission requires finite caller loops, independently of the runtime ray budget.
pub(super) fn require_bounded_query_caller(wgsl: &str, entry: &str, abi: &Abi) -> Result<()> {
    let module = super::validated_module(wgsl, abi)?;
    let function = module.functions.iter().find(|(_, f)| f.name.as_deref() == Some(entry)).context("query entry missing")?.1;
    let args = vec![Value::Unknown; function.arguments.len()];
    // Signature stubs are not runtime values: inputs and image samples must
    // remain unknown. The non-time live domain keeps fixed loop counters exact.
    let (_, state) = evaluate(&module, function, &args, 0, false, Some(Vec::<Option<u32>>::new().into()), Some(Live {texels:false,wavelength:true,query_bounds:true}), true);
    let failures = state.proof_failures.lock().unwrap();
    if let Some(reason)=query_bound_failure(&failures) {anyhow::bail!("trace: {reason}");}
    Ok(())
}

/// Follow compiler-resolved calls, including imported helpers and nested control
/// flow. Retained calls count even when their result is unused: conservative.
pub(super) fn program_dependencies(wgsl: &str, entry: &str, abi: &Abi) -> Result<ProgramDependencies> {
    let source = super::coordinates::validation_stubs().to_owned() + &format!(
        "fn vork_blend_overflow(count:u32,capacity:u32) {{}} fn {WAVELENGTH_EFFECT}(v:f32)->f32{{return v;}} fn {MATERIAL_WORD}(i:u32)->u32{{return i;}} fn {SAMPLE_IMAGE}(s:u32,u:vec2<f32>,x:vec2<f32>,y:vec2<f32>)->vec4<f32>{{return vec4(f32(s)+dot(u+x+y,vec2<f32>(1.)));}} fn vork_attribute(s:u32,d:u32)->vec4<f32>{{return vec4(0.);}}\n{wgsl}");
    let module = naga::front::wgsl::parse_str(&source).map_err(|e| anyhow::anyhow!(e.emit_to_string(&source)))?;
    let function = module.functions.iter().find(|(_, f)| f.name.as_deref() == Some(entry)).context("surface entry missing")?.1;
    let mut dependencies = ProgramDependencies::default();
    let mut pending = Vec::new();
    calls(&function.body, &mut pending);
    let mut visited = BTreeSet::new();
    while let Some(handle) = pending.pop() {
        if !visited.insert(handle.index()) { continue; }
        let callee = &module.functions[handle];
        match callee.name.as_deref().unwrap_or("") {
            name if CAMERA_CALLS.contains(&name) => dependencies.camera = true,
            "vork_object_to_generated" | "vork_world_to_object" | "vork_world_normal_to_object"
                | "vork_object_direction_to_world"
                | "vork_object_normal_to_world" => dependencies.object_transforms = true,
            "vork_attribute" => dependencies.attributes = true,
            "vork_trace_field" => dependencies.trace = true,
            name if name == SAMPLE_IMAGE => dependencies.images = true,
            _ => (),
        }
        calls(&callee.body, &mut pending);
    }
    let mut context = vec![Value::Parts(vec![Value::Unknown; 4]); abi.context_vec4s()];
    for (i, field, _) in slots(&abi.context) {
        if field.name == "time" {
            if let Value::Parts(parts) = &mut context[i / 4] { parts[i % 4] = Value::Time; }
        }
    }
    fn initial(module: &naga::Module, ty: naga::Handle<naga::Type>) -> Value {
        match module.types[ty].inner {
            naga::TypeInner::Pointer { base, .. } => initial(module, base),
            naga::TypeInner::Array { base, size: naga::ArraySize::Constant(n), .. } => Value::Parts(vec![initial(module, base); n.get() as usize]),
            _ => zero(module, ty),
        }
    }
    let mut args: Vec<_> = function.arguments.iter().map(|a| {
        if matches!(module.types[a.ty].inner, naga::TypeInner::Pointer { .. }) {
            initial(&module, a.ty)
        } else { Value::Unknown }
    }).collect();
    args[0] = Value::Parts(context);
    args[1] = Value::Unknown;
    let (_, state) = evaluate(&module, function, &args, 0, false, None, None, false);
    // Resource reads do not imply clock reads. Their changing coordinates and
    // addresses propagate Time through opaque calls; consumers track camera,
    // object/geometry and image changes through the separate dependency facts.
    dependencies.time = state.arguments.iter().skip(2).any(Value::time);
    Ok(dependencies)
}

/// A program's closure-effect probe (`Program::effects`), parsed once. Its
/// inputs are decoded from material words exactly as in the renderer's
/// wrappers, so evaluating it with an instance's words needs no Slang run.
pub struct EffectProbe {
    module: naga::Module,
    entry: usize,
    lit: bool,
}
impl EffectProbe {
    /// `lit`: the program declares a `lit` stage.
    pub(super) fn parse(wgsl: &str, lit: bool) -> Result<Self> {
        // Texels are unknown: the stub reads a buffer nothing writes.
        let source = super::coordinates::validation_stubs().to_owned()
            + "fn vork_attribute(slot:u32,differential:u32)->vec4<f32>{return vec4(f32(slot+differential));}\n"
            + &format!("@group(9) @binding(0) var<storage,read> vork_texels:array<vec4<f32>>; fn vork_blend_overflow(count:u32,capacity:u32) {{}} fn {WAVELENGTH_EFFECT}(v:f32)->f32{{return v;}} fn {MATERIAL_WORD}(i:u32)->u32 {{return i;}} fn {SAMPLE_IMAGE}(s:u32,u:vec2<f32>,x:vec2<f32>,y:vec2<f32>)->vec4<f32>{{return vork_texels[s];}}\n{wgsl}");
        let module = naga::front::wgsl::parse_str(&source)
            .map_err(|e| anyhow::anyhow!(e.emit_to_string(&source)))?;
        let entry = module.entry_points.iter().position(|entry| entry.name == "vork_entry").context("closure effect probe entry missing")?;
        Ok(Self { module, entry, lit })
    }
    /// The closure effects with material words `words` (`InputLayout::
    /// effect_words`; None, or past the end: unknown). The words are the
    /// exact bits the GPU reads, so f32 arithmetic on them folds only where
    /// exact (`exact_f32`). Unknown words, resources and the shading context
    /// stay unknown, so only a proof that holds for every point, time and
    /// texel removes an effect.
    pub fn capabilities(&self, words: &[Option<u32>]) -> ProgramCapabilities {
        let lit = self.lit;
        let (_, state) = evaluate(&self.module, &self.module.entry_points[self.entry].function, &[], 0, false, Some(words.into()), None, false);
        let active = |index| state.effects.get(&index) != Some(&Value::Number(0.));
        ProgramCapabilities {
            dependencies: ProgramDependencies::default(),
            hair: active(5),
            physical_hair: active(17),
            subsurface: active(0),
            spatial_subsurface: active(18),
            opacity: active(1),
            transmission: active(2),
            anisotropy: active(7) || active(9),
            coat_anisotropy: active(registry::EFFECT_SURFACE_COAT_ANISOTROPY as usize) || active(registry::EFFECT_LIT_COAT_ANISOTROPY as usize),
            film: active(8) || active(10),
            coat: active(11) || active(14),
            fuzz: active(12) || active(15),
            tint: active(13) || active(16),
            weighted_ior: active(registry::EFFECT_SURFACE_WEIGHTED_IOR as usize) || active(registry::EFFECT_LIT_WEIGHTED_IOR as usize),
            lobes: active(3),
            blend_lobes: match state.effects.get(&6) {
                Some(Value::NumberRange(a,b)) if *a>=1. && b.is_finite() && *b<=u32::MAX as f64 && b.fract()==0. => Some(*b as u32),
                Some(Value::Number(n)) if n.is_finite() && *n >= 1. && *n <= u32::MAX as f64 && n.fract() == 0. => Some(*n as u32),
                _ => None,
            },
            lit,
            emission: active(registry::EFFECT_SURFACE_EMISSION as usize),
            displacement: false,
            lit_reflects: lit && active(4),
        }
    }
    /// Preserve output dependencies without interpreting selected-lobe arrays.
    /// Every closure contributes dispersion and wavelength summaries; weights
    /// contribute wavelength dependencies too.
    pub(super) fn needs_spectral(&self, words:&[Option<u32>]) -> bool {
        let live=Some(Live {texels:false,wavelength:true,query_bounds:false});
        let (_,state)=evaluate(&self.module,&self.module.entry_points[self.entry].function,&[],0,false,Some(words.into()),live,false);
        [registry::EFFECT_SURFACE_DISPERSION,registry::EFFECT_LIT_DISPERSION].into_iter()
            .any(|index|state.effects.get(&(index as usize))!=Some(&Value::Number(0.)))
            || state.effects.contains_key(&usize::MAX)
            || [registry::EFFECT_SURFACE_WAVELENGTH,registry::EFFECT_LIT_WAVELENGTH].into_iter()
                .any(|index|state.effects.get(&(index as usize)).is_none_or(Value::time))
    }
    /// Whether coverage (geometry opacity) can change between frames on an
    /// object that does not move, with material words `words` as in
    /// `capabilities`: it reads the clock, the camera or,
    /// when `live_texels` (an image that can change), a texel. Unproven is true.
    pub fn coverage_varies(&self, words: &[Option<u32>], live_texels: bool) -> bool {
        let live = Some(Live { texels: live_texels, wavelength:false, query_bounds:false });
        let (_, state) = evaluate(&self.module, &self.module.entry_points[self.entry].function, &[], 0, false, Some(words.into()), live, false);
        state.effects.contains_key(&usize::MAX) || state.effects.get(&1).is_none_or(Value::time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// One instance's capabilities: `values` packed into material words.
    fn instance_capabilities(program: &Program, values: serde_json::Value, dynamic: &[&str]) -> Result<ProgramCapabilities> {
        let values: BTreeMap<String, serde_json::Value> = serde_json::from_value(values)?;
        let dynamic: BTreeSet<String> = dynamic.iter().map(|name| name.to_string()).collect();
        Ok(program.effect_probe()?.capabilities(&program.inputs.effect_words(&values, &dynamic)?))
    }
    #[test]
    #[ignore = "requires pinned Slang"]
    fn spatial_subsurface_proof_preserves_thin_lighting_and_unknown_interiors() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        for (body, expected) in [
            ("c.geometry_thin_walled=true; return c;", false),
            ("c.geometry_thin_walled=false; return c;", true),
            ("c.geometry_thin_walled=ctx.POSITION.x>0.; return c;", true),
            ("c.geometry_thin_walled=true; return add(c,c);", false),
            ("c.geometry_thin_walled=true; Closure b=c;b.geometry_thin_walled=false;return add(c,b);", true),
        ] {
            let result = if body.contains("add(") {"Blend"} else {"Closure"};
            let source = format!("struct Inputs {{}}; {result} surface(ShadingContext ctx,Inputs i) {{Closure c;c.subsurface_weight=1.;{body}}}");
            let program = Compiler::discover().compile_surface(&abi,&source,Path::new("thin.slang"),"thin")?;
            assert!(program.capabilities.subsurface, "thin sheets still scatter: {body}");
            assert_eq!(program.capabilities.spatial_subsurface,expected,"{body}");
        }
        let source="struct Inputs {bool thin=true;}; Closure surface(ShadingContext ctx, Inputs i) {Closure c;c.subsurface_weight=1.;c.geometry_thin_walled=i.thin;return c;}";
        let program=Compiler::discover().compile_surface(&abi,source,Path::new("thin.slang"),"thin")?;
        for (value,dynamic,expected) in [(true,false,false),(false,false,true),(true,true,true)] {
            let caps=instance_capabilities(&program,serde_json::json!({"thin":value}),if dynamic {&["thin"]} else {&[]})?;
            assert!(caps.subsurface);
            assert_eq!(caps.spatial_subsurface,expected);
        }
        Ok(())
    }

    #[test]
    fn animated_uniform_words_keep_position_independence() -> Result<()> {
        let source=format!("struct M {{a:vec3<f32>,s:vec3<f32>,e:vec3<f32>,g:f32}}; fn {MATERIAL_WORD}(i:u32)->u32 {{return i;}} fn point(p:vec3<f32>,time:f32,base:u32)->M {{let density=bitcast<f32>({MATERIAL_WORD}(base));let height=bitcast<f32>({MATERIAL_WORD}(base+1u));return M(vec3(density*exp(-p.y*height)),vec3(0.),vec3(0.),0.);}}");
        let module=naga::front::wgsl::parse_str(&source).unwrap();
        assert!(medium_homogeneous(&module,"point",Some(&[None,Some(0f32.to_bits())]))?);
        assert!(!medium_homogeneous(&module,"point",Some(&[Some(1f32.to_bits()),None]))?);
        Ok(())
    }
    #[test]
    fn exponential_proofs_keep_unused_animated_words_unknown() -> Result<()> {
        let source=format!("struct M {{a:vec3<f32>,s:vec3<f32>,e:vec3<f32>,g:f32}}; fn {MATERIAL_WORD}(i:u32)->u32 {{return i;}} fn point(p:vec3<f32>,time:f32,base:u32)->M {{let density=bitcast<f32>({MATERIAL_WORD}(base));return M(vec3(density*exp(-0.5*p.y)),vec3(0.),vec3(0.),0.);}}");
        let module=naga::front::wgsl::parse_str(&source)?;
        assert_eq!(medium_exponential(&module,"point",&[Some(2f32.to_bits()),None])?,
            Some(([2.;3],[0.;3],[0.,-0.5,0.])));
        assert!(medium_exponential(&module,"point",&[None,Some(2f32.to_bits())])?.is_none());
        Ok(())
    }
    #[test]
    #[cfg(vork_private_tests)] // Uses the private compute-query fixture.
    #[ignore = "requires pinned Slang"]
    fn query_dependent_physical_hair_is_retained() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        let fixture:serde_json::Value=serde_json::from_str(include_str!("../../vork-render/tests/fixtures/compute-trace.json"))?;
        let source=fixture["programs"]["bake"]["source"].as_str().unwrap();
        let body=source.split_once("bool good=true;").unwrap().1.split_once("#endif").unwrap().0;
        let helper=format!("bool query_contract(){{bool good=true;{body}\n#endif\nreturn good;}}\n");
        let hair=module_source("std/hair").unwrap().replace("Hair surface",&format!("{helper}Hair surface")).replace("h.scattering = true;","h.scattering = query_contract();");
        let program=Compiler::discover().compile_surface(&abi,&hair,Path::new("query_hair.slang"),"queryhair")?;
        let capabilities=instance_capabilities(&program,serde_json::json!({"color":[0.5,0.3,0.1],"offset":0.}),&[])?;
        assert!(capabilities.hair && capabilities.physical_hair,"{capabilities:?}");
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang"]
    fn physical_hair_follows_material_words() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        let compiler = Compiler::discover();
        // (source, whether it declares the `model` enum, expected per case)
        for (source, model, expected) in [
            (module_source("std/hair").unwrap().to_owned(), false, [true;4]),
            // factor stays 0: only Chiang or an animated model can scatter.
            ("enum Model { Diffuse, Chiang }; struct Inputs { Model model=Model::Diffuse; float factor=0.; }; Hair surface(ShadingContext ctx,Inputs i) {Hair h; h.scattering=i.model==Model::Chiang || i.factor>0.;return h;}".into(), true, [false,false,true,true]),
            ("struct Inputs {}; Hair surface(ShadingContext ctx,Inputs i) {Hair h;h.scattering=false;return h;}".into(), false, [false;4]),
        ] {
            let origin=Path::new("physical.slang");
            let program=compiler.compile_surface(&abi,&source,origin,"physical")?;
            for (index, (token, dynamic)) in [(None,false),(Some("diffuse"),false),(Some("chiang"),false),(Some("diffuse"),true)].into_iter().enumerate() {
                let values=token.filter(|_| model).map(|token| ("model",token)).into_iter().collect::<BTreeMap<_,_>>();
                let dynamic: &[&str]=if dynamic {&["model"]} else {&[]};
                assert_eq!(instance_capabilities(&program,serde_json::json!(values),dynamic)?.physical_hair,expected[index],"{source}: {index}");
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang"]
    fn compiled_programs_report_camera_dependencies() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        for (value, camera, time) in [
            ("float3(.5)", false, false),
            ("world_to_camera(ctx.POSITION)", true, false),
            ("world_to_window(ctx.POSITION)", true, false),
            ("float3(float(asuint(ctx.time)&255u))", false, true),
        ] {
            let source = format!("struct Inputs {{}}; Closure surface(ShadingContext ctx,Inputs inputs) {{ Closure c={{}};c.emission_luminance=1.;c.emission_color={value};return c;}}");
            let program = Compiler::discover().compile_surface(&abi, &source, Path::new("dependencies.slang"), "dependencies")?;
            assert_eq!((program.capabilities.dependencies.camera,program.capabilities.dependencies.time), (camera,time), "{value}");
        }
        Ok(())
    }

    #[test]
    fn source_time_survives_image_channel_selection() -> Result<()> {
        let wgsl=format!("fn emission(p:vec3<f32>,n:vec3<f32>,time:f32,base:u32)->vec3<f32>{{return vec3<f32>({SAMPLE_IMAGE}(0u,vec2<f32>(time),vec2<f32>(0.),vec2<f32>(0.)).w);}}");
        assert!(displacement_depends_on(&wgsl,"emission",2)?);
        Ok(())
    }

    #[test]
    fn program_time_in_any_shader_lobe_remains_live() -> Result<()> {
        let abi = Abi { aov_count: 0, context: vec![Field::new("time", Type::Float, None)], closure: vec![] };
        let wgsl = "fn surface(a:array<vec4<f32>,1>,base:u32,lobe:u32,u:f32,r:ptr<function,array<vec4<f32>,1>>) { if lobe==0u { (*r)[0]=vec4<f32>(1.); } else { (*r)[0]=vec4<f32>(a[0].x); } }";
        assert!(program_dependencies(wgsl, "surface", &abi)?.time);
        Ok(())
    }
    #[test]
    fn helper_writes_under_a_clock_branch_depend_on_time() -> Result<()> {
        let abi = Abi { aov_count: 0, context: vec![Field::new("time", Type::Float, None)], closure: vec![] };
        let wgsl = "fn fade(r:ptr<function,array<vec4<f32>,1>>) { (*r)[0]=vec4<f32>(0.); } \
            fn surface(a:array<vec4<f32>,1>,base:u32,r:ptr<function,array<vec4<f32>,1>>) { (*r)[0]=vec4<f32>(1.); if a[0].x>1. { fade(r); } }";
        assert!(program_dependencies(wgsl, "surface", &abi)?.time);
        Ok(())
    }

    #[test]
    fn program_runtime_dependencies_follow_compiled_helpers_and_bitcasts() -> Result<()> {
        let abi = Abi { aov_count: 0, context: vec![Field::new("time", Type::Float, None)], closure: vec![] };
        for (value, time, camera, transforms, attributes, images) in [
            ("vec4<f32>(1.)", false, false, false, false, false),
            ("vec4<f32>(f32(bitcast<u32>(a[0].x) & 255u))", true, false, false, false, false),
            ("vec4<f32>(vork_world_to_camera(vec3<f32>(0.)),1.)", false, true, false, false, false),
            ("vec4<f32>(vork_world_to_window(vec3<f32>(0.)),1.)", false, true, false, false, false),
            ("vec4<f32>(vork_world_to_object(0u,vec3<f32>(0.)),1.)", false, false, true, false, false),
            ("vec4<f32>(vork_object_direction_to_world(0u,vec3<f32>(1.)),1.)", false, false, true, false, false),
            ("vec4<f32>(vork_object_normal_to_world(0u,vec3<f32>(1.)),1.)", false, false, true, false, false),
            ("vork_attribute(0u,0u)", false, false, false, true, false),
            ("vork_attribute(u32(a[0].x),0u)", true, false, false, true, false),
            ("vec4<f32>(vork_world_to_camera(vec3<f32>(a[0].x)),1.)", true, true, false, false, false),
            ("vork_sample_image(0u,vec2<f32>(0.),vec2<f32>(0.),vec2<f32>(0.))", false, false, false, false, true),
            ("vork_sample_image(0u,vec2<f32>(a[0].x),vec2<f32>(0.),vec2<f32>(0.))", true, false, false, false, true),
            ("vec4<f32>(vork_sample_image(0u,vec2<f32>(a[0].x),vec2<f32>(0.),vec2<f32>(0.)).z)", true, false, false, false, true),
        ] {
            let value = value.replace("vork_sample_image", SAMPLE_IMAGE);
            let wgsl = format!("fn helper(a:array<vec4<f32>,1>)->vec4<f32>{{return {value};}} fn surface(a:array<vec4<f32>,1>,base:u32,r:ptr<function,array<vec4<f32>,1>>){{(*r)[0]=helper(a);}}");
            let d = program_dependencies(&wgsl, "surface", &abi)?;
            assert_eq!((d.time,d.camera,d.object_transforms,d.attributes,d.images), (time,camera,transforms,attributes,images), "{value}");
        }
        Ok(())
    }

    #[test]
    fn environment_camera_constants_require_an_output_proof() -> Result<()> {
        for (value,uniform_images,expected) in [
            ("vec4<f32>(2.)",false,true),
            ("vec4<f32>(direction,1.)",true,false),
            ("vec4<f32>(origin,1.)",true,false),
            ("vec4<f32>(time)",true,false),
            ("vork_sample_image(0u,direction.xy,vec2(0.),vec2(0.))",false,false),
            ("vork_sample_image(0u,direction.xy,vec2(0.),vec2(0.))",true,true),
            ("vork_sample_image(0u,vec2(0.),vec2(0.),vec2(0.))*vec4(direction,1.)",true,false),
            ("vork_sample_image(u32(direction.x),vec2(0.),vec2(0.),vec2(0.))",true,false),
            ("select(vec4(1.),vec4(2.),direction.x>0.)",true,false),
        ] {
            let value=value.replace("vork_sample_image",SAMPLE_IMAGE);
            let source=format!("fn environment(origin:vec3<f32>,direction:vec3<f32>,time:f32,base:u32,result:ptr<function,array<vec4<f32>,2>>) {{ (*result)[0]=vec4(direction,1.); (*result)[1]={value}; }}");
            assert_eq!(environment_constant_camera(&source,"environment",uniform_images)?,expected,"{value}");
        }
        for (body,expected) in [
            ("var uv=0.; if direction.x>0. {uv=direction.y;} (*result)[1]=vork_sample_image(0u,vec2(uv),vec2(0.),vec2(0.));",true),
            ("var v=0.; if direction.x>0. {v=1.;} (*result)[1]=vec4(v);",false),
            ("if direction.x>0. {(*result)[1]=vec4(1.); return;} (*result)[1]=vec4(2.);",false),
            ("var slot=0u; if direction.x>0. {slot=1u;} (*result)[1]=vork_sample_image(slot,vec2(0.),vec2(0.),vec2(0.));",false),
            ("var v=0.; loop {if v>direction.x {break;} v+=.1;} (*result)[1]=vec4(v);",false),
        ] {
            let body=body.replace("vork_sample_image",SAMPLE_IMAGE);
            let source=format!("fn environment(origin:vec3<f32>,direction:vec3<f32>,time:f32,base:u32,result:ptr<function,array<vec4<f32>,2>>) {{ {body} }}");
            assert_eq!(environment_constant_camera(&source,"environment",true)?,expected,"{body}");
        }
        Ok(())
    }
    #[test]
    fn environment_time_survives_bitcasts() -> Result<()> {
        for (value, expected) in [
            ("f32(bitcast<u32>(time) & 255u)", true),
            ("bitcast<f32>(bitcast<u32>(time))", true),
            ("f32(bitcast<u32>(direction.x) & 255u)", false),
        ] {
            let source = format!(
                "fn environment(origin:vec3<f32>,direction:vec3<f32>,time:f32,base:u32,result:ptr<function,array<vec4<f32>,1>>) {{ (*result)[0]=vec4<f32>({value}); }}"
            );
            assert_eq!(environment_uses_time(&source, "environment")?, expected, "{value}");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang"]
    fn lit_stages_report_whether_their_closure_reflects() -> Result<()> {
        let abi = Abi { aov_count: 0,
            context: Abi::spec_context(),
            closure: openpbr_fields(),
        };
        for (lit, expected) in [
            ("", (false, false)),
            (
                "Closure lit(ShadingContext ctx,Inputs inputs,Lighting light) {return unlit(light.color);}",
                (true, false),
            ),
            (
                "Closure lit(ShadingContext ctx,Inputs inputs,Lighting light) {return unlit(step(inputs.cut,light.direct.g)*float3(1.),light.alpha);}",
                (true, false),
            ),
            (
                "Closure lit(ShadingContext ctx,Inputs inputs,Lighting light) {Closure c={};c.base_color=light.color;return c;}",
                (true, true),
            ),
            (
                "Closure lit(ShadingContext ctx,Inputs inputs,Lighting light) {Closure c=unlit(light.color);c.specular_weight=inputs.cut;return c;}",
                (true, true),
            ),
        ] {
            let source = format!(
                "struct Inputs {{float cut=.5;}}; Closure surface(ShadingContext ctx,Inputs inputs) {{Closure c={{}};return c;}} {lit}"
            );
            let program = Compiler::discover().compile_surface(
                &abi,
                &source,
                Path::new("lit.slang"),
                "lit",
            )?;
            let c = program.capabilities;
            assert_eq!((c.lit, c.lit_reflects), expected, "{lit}");
            assert_eq!(
                program.slang.contains("void vork_lit_lit("),
                c.lit,
                "{lit}"
            );
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang"]
    fn shader_programs_report_lobes_and_any_lobe_effect() -> Result<()> {
        let abi = Abi { aov_count: 0,
            context: Abi::spec_context(),
            closure: openpbr_fields(),
        };
        for (body, expected) in [
            ("return c;", (false, false, false, false)),
            (
                "return blend(c,m,inputs.factor);",
                (false, false, false, true),
            ),
            (
                "return add(c,blend(m,c,inputs.factor));",
                (false, false, false, true),
            ),
            (
                "Closure t=c;t.geometry_opacity=0.;return blend(t,m,inputs.factor);",
                (false, true, false, true),
            ),
            (
                "m.transmission_weight=1.;return blend(c,m,.5);",
                (false, false, true, true),
            ),
            (
                "return add(add(c,c),add(c,blend(c,m,inputs.factor)));",
                (false, false, false, true),
            ),
        ] {
            let ty = if body == "return c;" {
                "Closure"
            } else {
                "Blend"
            };
            let source = format!(
                "struct Inputs {{float factor=.5;}}; {ty} surface(ShadingContext ctx,Inputs inputs) {{Closure c={{}};Closure m={{}};m.base_metalness=1.;m.specular_roughness=.1;{body}}}"
            );
            let program = Compiler::discover().compile_surface(
                &abi,
                &source,
                Path::new("lobes.slang"),
                "lobes",
            )?;
            let c = program.capabilities;
            assert_eq!(
                (c.subsurface, c.opacity, c.transmission, c.lobes),
                expected,
                "{body}"
            );
            assert_eq!(
                program.slang.contains("uint lobe, float u"),
                c.lobes,
                "{body}"
            );
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang"]
    fn closure_effects_follow_outputs_instead_of_input_names() -> Result<()> {
        let abi = Abi { aov_count: 0,
            context: Abi::spec_context(),
            closure: openpbr_fields(),
        };
        for (body, expected) in [
            ("c.base_color=inputs.color;", (false, false, false)),
            ("c.geometry_opacity=0.5;", (false, true, false)),
            ("c.transmission_weight=1.;", (false, false, true)),
            ("adjust(c,inputs.factor);", (false, true, false)),
            (
                "if(ctx.POSITION.x>0.) return unlit(inputs.color);c.base_color=inputs.color;",
                (false, false, false),
            ),
            (
                "float a=1.;for(int i=0;i<4;i++){c.geometry_opacity=a;a=inputs.factor;}",
                (false, true, false),
            ),
            (
                "switch(int(ctx.POSITION.x)){case 0:c.base_color=inputs.color;break;default:c.base_color=float3(.5);break;}",
                (false, false, false),
            ),
            (
                "for(int i=0;i<4;i++) c.base_color*=inputs.color;",
                (false, false, false),
            ),
            ("c.geometry_opacity=asfloat(1u);", (false, true, false)),
            (
                "Closure m=c;m.base_color=inputs.color;c=mix_closure(c,m,inputs.factor);",
                (false, false, false),
            ),
            (
                "Closure m=c;m.geometry_opacity=0.;c=mix_closure(c,m,inputs.factor);",
                (false, true, false),
            ),
            (
                "for(int i=0;i<4;i++){c.geometry_opacity=.5;if(ctx.POSITION.x>0.)break;c.geometry_opacity=1.;}",
                (false, true, false),
            ),
            (
                "float3 v=float3(0);v[int(ctx.POSITION.x)&1]=1.;c.subsurface_weight=v.x;",
                (true, false, false),
            ),
            (
                "if(ctx.POSITION.x>0.) c.subsurface_weight=inputs.factor;",
                (true, false, false),
            ),
        ] {
            let source = format!(
                "struct Inputs {{float3 color=float3(1);float factor=1.;}}; void adjust(inout Closure c,float a) {{if(a>0.){{c.geometry_opacity=.25;return;}} c.geometry_opacity=.75;}} Closure surface(ShadingContext ctx,Inputs inputs) {{Closure c={{}};{body}return c;}}"
            );
            let program = Compiler::discover().compile_surface(
                &abi,
                &source,
                Path::new("effects.slang"),
                "effects",
            )?;
            let c = program.capabilities;
            assert_eq!(
                (c.subsurface, c.opacity, c.transmission),
                expected,
                "{body}"
            );
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang"]
    fn stable_inputs_prove_constant_coverage_absent() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        let compiler = Compiler::discover();
        let origin = Path::new("bound.slang");
        let check = |source: &str, cases: &[(serde_json::Value, &[&str], (bool, bool))]| -> Result<()> {
            let program = compiler.compile_surface(&abi, source, origin, "bound")?;
            assert!(program.capabilities.transmission && program.capabilities.opacity, "unbound inputs keep both effects: {source}");
            for (values, dynamic, expected) in cases {
                let c = instance_capabilities(&program, values.clone(), dynamic)?;
                assert_eq!((c.transmission, c.opacity), *expected, "{source}: {values:?} {dynamic:?}");
            }
            Ok(())
        };
        // The sizzle ocean's shape: transmission scaled by a per-point factor.
        check("struct Inputs {float transmission=0.;float opacity=1.;float foam=1.;float3 tint=float3(1);Image map;}; \
            Closure surface(ShadingContext ctx,Inputs i) {Closure c={};float foam=i.foam*smoothstep(.55,.15,sin(ctx.POSITION.x*ctx.time)); \
            c.transmission_weight=i.transmission*(1.-foam);c.geometry_opacity=i.opacity;c.transmission_color=i.tint;return c;}", &[
            (serde_json::json!({}), &[], (false, false)),
            (serde_json::json!({"transmission":0,"opacity":1,"foam":0.5}), &[], (false, false)),
            (serde_json::json!({"transmission":0.25}), &[], (true, false)),
            (serde_json::json!({"opacity":0.5}), &[], (false, true)),
            (serde_json::json!({"transmission":0}), &["transmission"], (true, false)),
            (serde_json::json!({"opacity":1}), &["opacity"], (false, true)),
        ])?;
        // Texture-driven coverage stays whatever the material words are.
        check("struct Inputs {float transmission=0.;Image map;}; Closure surface(ShadingContext ctx,Inputs i) {Closure c={}; \
            c.transmission_weight=i.transmission+sample(i.map,ctx.POSITION.xy,ctx).r;c.geometry_opacity=1.-sample(i.map,ctx.POSITION.xy,ctx).g;return c;}", &[
            (serde_json::json!({}), &[], (true, true)),
        ])?;
        // A zero only by rounding is no proof: a*b+c is 0 with separately rounded
        // f32 steps but 2^-46 under FMA contraction. Exact cancellation is.
        check("struct Inputs {float a=0.;float b=0.;float c=0.;}; Closure surface(ShadingContext ctx,Inputs i) {Closure c={}; \
            c.transmission_weight=(i.a*i.b+i.c)*35184372088832.;c.geometry_opacity=1.-(i.a*i.b+i.c);return c;}", &[
            (serde_json::json!({"a":1.00000011920928955078125,"b":1.00000011920928955078125,"c":-1.0000002384185791015625}), &[], (true, true)),
            (serde_json::json!({"a":0.5,"b":4,"c":-2}), &[], (false, false)),
            (serde_json::json!({"a":0.5,"b":2,"c":0}), &[], (true, true)),
        ])?;
        // An exact zero through a subnormal is no proof either: flushing
        // 2^-127 to zero leaves 0.5 transmission.
        check("struct Inputs {float a=0.;float b=0.;float s=0.;}; Closure surface(ShadingContext ctx,Inputs i) {Closure c={}; \
            c.transmission_weight=.5-(i.a*i.b)*i.s;c.geometry_opacity=1.-(i.a*i.b)*i.s;return c;}", &[
            (serde_json::json!({"a":1.1754943508222875e-38,"b":0.5,"s":8.507059173023462e37}), &[], (true, true)),
            (serde_json::json!({"a":0.25,"b":0.5,"s":4}), &[], (false, true)),
        ])?;
        // Texel alpha is unknown, whatever the coordinates.
        check("struct Inputs {Image map;}; Closure surface(ShadingContext ctx,Inputs i) {Closure c={}; \
            c.transmission_weight=sample(i.map,float2(.5),ctx).w;c.geometry_opacity=sample(i.map,ctx.POSITION.xy,ctx).a;return c;}", &[
            (serde_json::json!({}), &[], (true, true)),
        ])?;
        Ok(())
    }
    /// Inputs reach the probe only as runtime material words, so Slang folds
    /// none of them, and the evaluator folds only exact f32 arithmetic: a zero
    /// that exists only by rounding or contraction is never a proof.
    #[test]
    #[ignore = "requires pinned Slang"]
    fn bound_probe_counterexamples() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        let compiler = Compiler::discover();
        let wide: &str = &(0..40).map(|n| format!("float x{n}=1.;")).collect::<String>();
        // (inputs, transmission_weight, geometry_opacity, values, expected (transmission, opacity))
        let cases: &[(&str, &str, &str, serde_json::Value, (bool, bool))] = &[
            // 0.3-3*0.1 is 0 rounded per step and 7.45e-9 under FMA; ints, bools
            // and enums used to be bound as literals Slang could fold.
            ("int n=0;", "(0.3-float(i.n)*0.1)*1e9", "1.", serde_json::json!({"n":3}), (true, false)),
            ("int n=3;", "(0.3-float(i.n)*0.1)*1e9", "1.", serde_json::json!({}), (true, false)),
            ("bool b=false;", "(0.3-(i.b?3.:1.)*0.1)*1e9", "1.", serde_json::json!({"b":true}), (true, false)),
            ("M m=M::A;", "(0.3-(i.m==M::B?3.:1.)*0.1)*1e9", "1.", serde_json::json!({"m":"b"}), (true, false)),
            ("M m=M::A;", "i.m==M::B?0.:1.", "1.", serde_json::json!({"m":"b"}), (false, false)),
            ("M m=M::A;", "i.m==M::B?0.:1.", "1.", serde_json::json!({}), (true, false)),
            ("float a=0.;float b=0.;", "(i.a-i.b)*-1e8", "1.", serde_json::json!({"a":0.3,"b":0.30000004}), (true, false)),
            ("float a=0.;float b=0.;", "(i.a-i.b)*-1e8", "1.", serde_json::json!({"a":1.2345678e-5,"b":1.2345679e-5}), (true, false)),
            ("float t=0.;", "i.t", "1.", serde_json::json!({"t":1e-45}), (true, false)),
            ("float t=0.;", "i.t-3.4028235e38", "1.", serde_json::json!({"t":3.4028235e38}), (false, false)),
            ("float t=0.;", "i.t*1e7-1.", "1.", serde_json::json!({"t":1e-7}), (true, false)),
            ("float t=0.;", "max(i.t,0.)", "1.", serde_json::json!({}), (true, false)),
            ("float t=0.;", "ctx.POSITION.x>0.?i.t:0.", "1.", serde_json::json!({}), (false, false)),
            // 0 times a runtime value that may be infinite is NaN there: no
            // defined shading, not coverage (effects.rs, Binary Multiply).
            ("float t=0.;", "i.t*(1./ctx.POSITION.x)", "1.", serde_json::json!({}), (false, false)),
            ("float a=1.;", "0.", "i.a*(1.-0.*ctx.POSITION.x)", serde_json::json!({}), (false, false)),
            // lerp of equal non-zero ends can round to 1 - 1 ulp; of zeros it is exact.
            ("float o=1.;", "0.", "lerp(i.o,i.o,ctx.POSITION.x)", serde_json::json!({}), (false, true)),
            ("float t=0.;", "lerp(i.t,i.t,ctx.POSITION.x)", "1.", serde_json::json!({}), (false, false)),
            ("float a=0.;float b=0.;float t=0.;", "0.", "lerp(i.a,i.b,i.t)", serde_json::json!({"a":1,"b":1,"t":67108864}), (false, true)),
            // Comparisons fold exactly, but a backend may flush a subnormal
            // operand before comparing it.
            ("float t=0.;", "0.", "i.t>0.?1.:.5", serde_json::json!({"t":0.25}), (false, false)),
            ("float t=0.;", "0.", "i.t>0.?1.:.5", serde_json::json!({"t":1e-40}), (false, true)),
            ("float t=0.;", "0.", "i.t==0.?.5:1.", serde_json::json!({"t":1e-40}), (false, true)),
            // Inputs past the first supplied-mask word.
            (wide, "1.-i.x39", "i.x31", serde_json::json!({"x39":1,"x31":1}), (false, false)),
            (wide, "1.-i.x39", "i.x31", serde_json::json!({"x39":0.5,"x31":0.5}), (true, true)),
        ];
        for (inputs, transmission, opacity, values, expected) in cases {
            let source = format!("enum M {{ A, B }}; struct Inputs {{{inputs}}}; Closure surface(ShadingContext ctx,Inputs i) {{Closure c={{}}; c.transmission_weight={transmission}; c.geometry_opacity={opacity}; return c;}}");
            let program = compiler.compile_surface(&abi, &source, Path::new("review.slang"), "review")?;
            let c = instance_capabilities(&program, values.clone(), &[])?;
            assert_eq!((c.transmission, c.opacity), *expected, "{source} {values}");
        }
        // A `break` in a switch case leaves the switch, not the enclosing loop,
        // and skips the rest of its case.
        for (body, values, transmission) in [
            ("[loop] for(int k=0;k<2;++k){switch(i.mode){case 0: if(i.stop) break; c.base_color=float3(.2); break; default: c.base_color=float3(.8); break;} c.transmission_weight=.5;}",
                serde_json::json!({"stop":true}), true),
            ("switch(i.mode){case 0: c.transmission_weight=.5; if(i.stop) break; c.transmission_weight=0.; break; default: break;}",
                serde_json::json!({"stop":true}), true),
            ("switch(i.mode){case 0: c.transmission_weight=.5; if(i.stop) break; c.transmission_weight=0.; break; default: break;}",
                serde_json::json!({"stop":false}), false),
            ("switch(i.mode){case 0: c.transmission_weight=.5; if(ctx.POSITION.x>0.) break; c.transmission_weight=0.; break; default: break;}",
                serde_json::json!({}), true),
        ] {
            let source = format!("struct Inputs {{int mode=0;bool stop=false;}}; Closure surface(ShadingContext ctx,Inputs i) {{Closure c={{}}; c.transmission_weight=0.; {body} return c;}}");
            let program = compiler.compile_surface(&abi, &source, Path::new("review.slang"), "review")?;
            assert_eq!(instance_capabilities(&program, values.clone(), &[])?.transmission, transmission, "{source} {values}");
        }
        Ok(())
    }
    /// Coverage varies between frames only through the clock, the camera,
    /// an image that can change; unproven is varying.
    #[test]
    #[ignore = "requires pinned Slang"]
    fn coverage_varies_only_with_live_sources() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        let compiler = Compiler::discover();
        let helper = "void fade(inout Closure c, float o) { c.geometry_opacity = o; }";
        // (opacity statement, texels live, expected)
        for (body, live_texels, expected) in [
            ("c.geometry_opacity=i.o;", false, false),
            ("c.geometry_opacity=i.o*ctx.POSITION.x;", false, false),
            ("c.geometry_opacity=sample(i.map,ctx.TEXCOORD_0,ctx).a;", false, false),
            ("c.geometry_opacity=sample(i.map,ctx.TEXCOORD_0,ctx).a;", true, true),
            ("c.geometry_opacity=sample(i.map,ctx.TEXCOORD_0+float2(ctx.time,0.),ctx).a;", false, true),
            ("c.geometry_opacity=i.o;c.base_color=sample(i.map,ctx.TEXCOORD_0,ctx).rgb;", true, false),
            ("c.geometry_opacity=i.o*(.5+.5*sin(ctx.time));", false, true),
            ("c.geometry_opacity=ctx.time>1.?i.o:1.;", false, true),
            ("c.geometry_opacity=i.o; if (ctx.time>1.) {fade(c,0.);}", false, true),
            ("c.geometry_opacity=abs(dot(ctx.NORMAL,ctx.incoming));", false, true),
            ("c.geometry_opacity=i.o;c.base_color=float3(sin(ctx.time));", false, false),
        ] {
            let source = format!("struct Inputs {{float o=.5;Image map;}}; {helper} Closure surface(ShadingContext ctx,Inputs i) {{Closure c={{}}; {body} return c;}}");
            let program = compiler.compile_surface(&abi, &source, Path::new("varies.slang"), "varies")?;
            let words = program.inputs.effect_words(&BTreeMap::new(), &BTreeSet::new())?;
            assert_eq!(program.effect_probe()?.coverage_varies(&words, live_texels), expected, "{body} live texels {live_texels}");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang"]
    fn coat_anisotropy_and_weighted_ior_proofs_are_independent() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        for coat in [false, true] {
            for weighted in [false, true] {
                for stage in ["surface", "blend", "lit"] {
                    let body = format!("Closure c={{}};c.coat_weight=1.;c.coat_roughness_anisotropy={};c.specular_weight={};",
                        if coat {".7"} else {"0."}, if weighted {".25"} else {"1."});
                    let source = match stage {
                        "blend" => format!("struct Inputs {{}}; Blend surface(ShadingContext ctx,Inputs i) {{{body}return add(c,c);}}"),
                        "lit" => format!("struct Inputs {{}}; Closure surface(ShadingContext ctx,Inputs i) {{Closure c={{}};return c;}} Closure lit(ShadingContext ctx,Inputs i,Lighting light) {{{body}return c;}}"),
                        _ => format!("struct Inputs {{}}; Closure surface(ShadingContext ctx,Inputs i) {{{body}return c;}}"),
                    };
                    let c = Compiler::discover().compile_surface(&abi,&source,Path::new("independent.slang"),"independent")?.capabilities;
                    assert_eq!((c.coat_anisotropy,c.weighted_ior),(coat,weighted),"{stage}: {source}");
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang"]
    fn unit_weight_effects_cover_surface_lobes_and_lit_returns() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        for (result, body, lit, expected) in [
            ("Closure", "return c;", "", false),
            ("Closure", "c.specular_weight=.25;return c;", "", true),
            ("Closure", "c.specular_weight=inputs.factor;return c;", "", true),
            ("Blend", "return blend(c,c,inputs.factor);", "", false),
            ("Blend", "Closure d=c;d.specular_weight=0.;return add(c,d);", "", true),
            ("Closure", "return c;", "Closure lit(ShadingContext ctx,Inputs inputs,Lighting light) {Closure c={};c.specular_weight=inputs.factor;return c;}", true),
        ] {
            let source=format!("struct Inputs {{float factor=1.;}}; {result} surface(ShadingContext ctx,Inputs inputs) {{Closure c={{}};{body}}} {lit}");
            let program=Compiler::discover().compile_surface(&abi,&source,Path::new("weight.slang"),"weight")?;
            assert_eq!(serde_json::to_value(program.capabilities)?["weighted_ior"],serde_json::json!(expected),"{source}");
        }
        let source="struct Inputs {float weight=1.;}; Closure surface(ShadingContext ctx,Inputs i) {Closure c={};c.specular_weight=i.weight;return c;}";
        let program=Compiler::discover().compile_surface(&abi,source,Path::new("bound_weight.slang"),"boundweight")?;
        for (values,animated,expected) in [
            (serde_json::json!({}),false,false),
            (serde_json::json!({"weight":1}),false,false),
            (serde_json::json!({"weight":0}),false,true),
            (serde_json::json!({"weight":0.25}),false,true),
            (serde_json::json!({"weight":1}),true,true),
        ] {
            assert_eq!(instance_capabilities(&program,values,if animated {&["weight"]} else {&[]})?.weighted_ior,expected);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang"]
    fn layer_effects_cover_surface_lobes_and_lit_returns() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        for (result, body, lit, expected) in [
            ("Closure", "return c;", "", (false,false,false)),
            ("Closure", "c.coat_weight=inputs.factor;return c;", "", (true,false,false)),
            ("Closure", "c.fuzz_weight=inputs.factor;return c;", "", (false,true,false)),
            ("Closure", "c.specular_color=float3(inputs.factor);return c;", "", (false,false,true)),
            ("Blend", "return blend(c,c,inputs.factor);", "", (false,false,false)),
            ("Blend", "Closure d=c;d.coat_weight=inputs.factor;return blend(c,d,inputs.factor);", "", (true,false,false)),
            ("Blend", "Closure d=c;d.fuzz_weight=inputs.factor;return add(c,d);", "", (false,true,false)),
            ("Blend", "Closure d=c;d.specular_color=float3(inputs.factor);return add(c,d);", "", (false,false,true)),
            ("Closure", "return c;", "Closure lit(ShadingContext ctx,Inputs inputs,Lighting light) {Closure c={};c.coat_weight=inputs.factor;return c;}", (true,false,false)),
            ("Closure", "return c;", "Closure lit(ShadingContext ctx,Inputs inputs,Lighting light) {return unlit(light.color);}", (false,false,false)),
        ] {
            let source=format!("struct Inputs {{float factor=1.;}}; {result} surface(ShadingContext ctx,Inputs inputs) {{Closure c={{}};{body}}} {lit}");
            let program=Compiler::discover().compile_surface(&abi,&source,Path::new("layers.slang"),"layers")?;
            let c=program.capabilities;
            assert_eq!((c.coat,c.fuzz,c.tint),expected,"{source}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang"]
    fn response_effects_cover_closures_shader_lobes_and_lit() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        for (result, body, lit, expected) in [
            ("Closure", "return c;", "", (false,false,false)),
            ("Closure", "c.coat_roughness_anisotropy=inputs.factor;return c;", "", (false,false,false)),
            ("Closure", "c.coat_weight=1.;c.coat_roughness_anisotropy=inputs.factor;return c;", "", (true,false,true)),
            ("Closure", "c.coat_weight=1.;c.specular_roughness_anisotropy=inputs.factor;return c;", "", (true,false,false)),
            ("Blend", "Closure d=c;d.coat_weight=1.;d.coat_roughness_anisotropy=inputs.factor;return add(c,d);", "", (true,false,true)),
            ("Closure", "return c;", "Closure lit(ShadingContext ctx,Inputs inputs,Lighting light) {Closure c={};c.coat_weight=1.;c.coat_roughness_anisotropy=inputs.factor;return c;}", (true,false,true)),
            ("Closure", "c.specular_roughness_anisotropy=inputs.factor;return c;", "", (true,false,false)),
            ("Closure", "c.thin_film_weight=inputs.factor;return c;", "", (false,true,false)),
            ("Blend", "return blend(c,c,inputs.factor);", "", (false,false,false)),
            ("Blend", "Closure d=c;d.thin_film_weight=inputs.factor;return blend(c,d,inputs.factor);", "", (false,true,false)),
            ("Blend", "Closure d=c;d.specular_roughness_anisotropy=inputs.factor;return add(c,d);", "", (true,false,false)),
            ("Closure", "return c;", "Closure lit(ShadingContext ctx,Inputs inputs,Lighting light) {Closure c={};c.thin_film_weight=inputs.factor;return c;}", (false,true,false)),
            ("Closure", "return c;", "Closure lit(ShadingContext ctx,Inputs inputs,Lighting light) {return unlit(light.color);}", (false,false,false)),
        ] {
            let source=format!("struct Inputs {{float factor=1.;}}; {result} surface(ShadingContext ctx,Inputs inputs) {{Closure c={{}};{body}}} {lit}");
            let program=Compiler::discover().compile_surface(&abi,&source,Path::new("response.slang"),"response")?;
            assert_eq!((program.capabilities.anisotropy,program.capabilities.film,program.capabilities.coat_anisotropy),expected,"{source}");
        }
        Ok(())
    }

}

#[cfg(test)]
mod tail_interval_tests {
    use super::*;
    #[test]
    fn zero_does_not_erase_nonfinite_spatial_arithmetic() -> Result<()> {
        for (expression, expected) in [
            ("p.y * 0.0", true),
            ("0.0 * p.y", true),
            ("(p.y / 0.0) * 0.0", false),
            ("0.0 * (p.y / 0.0)", false),
            ("(p.y * 1e38) * 0.0", false),
        ] {
            let source = format!("fn point(p:vec3<f32>,time:f32,base:u32)->vec3<f32> {{return vec3<f32>({expression});}}");
            let module = naga::front::wgsl::parse_str(&source)?;
            assert_eq!(medium_homogeneous(&module, "point", Some(&[]))?, expected, "{expression}");
        }
        Ok(())
    }
    #[test]
    fn immutable_numeric_guards_prune_spatial_branches() -> Result<()> {
        let source=format!("struct M {{a:vec3<f32>,s:vec3<f32>,e:vec3<f32>,g:f32}}; fn {MATERIAL_WORD}(i:u32)->u32 {{return i;}} fn point(p:vec3<f32>,time:f32,base:u32)->M {{var m=M(vec3<f32>(0.2),vec3<f32>(0.0),vec3<f32>(0.0),0.0);if(bitcast<f32>({MATERIAL_WORD}(base))>0.0) {{m.a=p;}} return m;}}");
        let module=naga::front::wgsl::parse_str(&source)?;
        for (guard,expected) in [(0f32,true),(-1.,true),(1.,false)] {
            assert_eq!(medium_homogeneous(&module,"point",Some(&[Some(guard.to_bits())]))?,expected);
        }
        Ok(())
    }
    #[test]
    fn fixed_loops_retain_spatial_dependencies() -> Result<()> {
        for (term,table,expected) in [
            ("vec3<f32>(0.1)","0.1,0.2,0.3",true),("p","0.1,0.2,0.3",false),
            ("vec3<f32>(table[j])","0.1,0.2,0.3",true),
            ("vec3<f32>(table[j])","p.x,0.2,0.3",false),
            ("vec3<f32>(table[u32(p.y)])","0.1,0.2,0.3",false),
        ] {
            let source=format!("struct M {{a:vec3<f32>,s:vec3<f32>,e:vec3<f32>,g:f32}};fn point(p:vec3<f32>,time:f32,base:u32)->M {{var m=M(vec3<f32>(0.2),vec3<f32>(0.0),vec3<f32>(0.0),0.0);var table=array<f32,3>({table});for(var j=0u;j<3u;j+=1u) {{m.a=m.a+{term};}}return m;}}");
            let module=naga::front::wgsl::parse_str(&source)?;
            assert_eq!(medium_homogeneous(&module,"point",Some(&[]))?,expected);
        }
        Ok(())
    }
    #[test]
    fn medium_exponential_is_algebraic_not_a_point_fit() -> Result<()> {
        for (expression,expected) in [
            ("2.0*exp(-0.5*p.y)",true),
            ("exp(-0.5*p.y+0.25*p.x)",true),
            ("exp(clamp(-0.5*p.y,-80.0,80.0))",false),
            ("exp(-0.5*p.y)+sin(p.x)",false),
            ("exp(-0.5*p.y)+0.01",false),
            ("exp(-0.5*p.y*p.y)",false),
        ] {
            let source=format!("struct M {{a:vec3<f32>,s:vec3<f32>,e:vec3<f32>,g:f32}}; fn point(p:vec3<f32>,time:f32,base:u32)->M {{return M(max(vec3<f32>({expression}),vec3<f32>(0.0)),vec3<f32>(0.0),vec3<f32>(0.01),0.0);}}");
            let module=naga::front::wgsl::parse_str(&source)?;
            let result=medium_exponential(&module,"point",&[])?;
            assert_eq!(result.is_some(),expected,"{expression}: {result:?}");
            if expression.starts_with("2.0") {assert_eq!(result,Some(([2.;3],[0.;3],[0.,-0.5,0.])));}
        }
        Ok(())
    }

    #[test]
    fn constant_exteriors_are_proved_over_half_spaces() -> Result<()> {
        for (expression,axis) in [
            ("exp(clamp(-0.04*p.y,-80.0,80.0))",Some(1)),
            ("exp(clamp(-1e-20*p.z,-80.0,80.0))",Some(2)),
            ("select(0.0,0.2,p.z < -10.0)",Some(2)),
            ("exp(-0.04*p.y)",None),
            ("exp(clamp(-0.04*p.y,-80.0,80.0))+sin(p.x)",None),
            ("sin(p.y)",None),
        ] {
            let source=format!("struct M {{a:vec3<f32>,s:vec3<f32>,e:vec3<f32>,g:f32}}; fn point(p:vec3<f32>,time:f32,base:u32)->M {{return M(vec3<f32>({expression}),vec3<f32>(0.0),vec3<f32>(0.01),0.0);}}");
            let module=naga::front::wgsl::parse_str(&source)?;
            assert_eq!(medium_tail_slab(&module,"point",&[])?.map(|(a,_)|a),axis,"{expression}");
            assert!(!medium_emission_free(&module,"point",&[])?);
        }
        let module=naga::front::wgsl::parse_str("struct M {a:vec3<f32>,s:vec3<f32>,e:vec3<f32>,g:f32}; fn point(p:vec3<f32>,time:f32,base:u32)->M {return M(vec3<f32>(sin(p.y)),vec3<f32>(0.0),vec3<f32>(0.0),0.0);}")?;
        assert!(medium_emission_free(&module,"point",&[])?);
        Ok(())
    }
}

pub(super) fn medium_majorant(module:&naga::Module,entry:&str,words:&[Option<u32>],position:[[f32;2];3],times:[f32;2])->Result<f32> {
    interval::majorant(module,entry,words,position,times)
}

/// Trace the clock through all camera outputs, including calibration and sensor.
pub(super) fn camera_uses_time(module: &naga::Module, camera: Option<&str>, sensor: Option<&str>, calibration: Option<&str>) -> Result<bool> {
    for (entry, fields, clock, count, output) in [(camera,9,3,3,2),(sensor,4,3,1,2),(calibration,0,2,4,4)] {
        let Some(entry)=entry else {continue};
        let (_,function)=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry)).context("camera stage bridge missing")?;
        let mut args=vec![Value::Unknown;function.arguments.len()];
        if fields==0 {args[clock]=Value::Time;} else {
            let mut context=vec![Value::Unknown;fields];context[clock]=Value::Time;args[0]=Value::Parts(context);
        }
        args[output]=Value::Parts(vec![Value::Parts(vec![Value::Number(0.);4]);count]);
        let (_,state)=evaluate(module,function,&args,0,false,None,None,false);
        if state.arguments[output].time() {return Ok(true);}
    }
    Ok(false)
}

pub(super) fn camera_needs_spectral(module: &naga::Module, camera: Option<&str>, sensor: Option<&str>, words: &[Option<u32>]) -> Result<bool> {
    let words: std::sync::Arc<[Option<u32>]> = words.iter().copied().collect();
    let output = |entry: &str, context: Value, count: usize| -> Result<Value> {
        let (_, function) = module.functions.iter().find(|(_, f)| f.name.as_deref()==Some(entry))
            .context("camera stage bridge missing")?;
        let (_, state) = evaluate(module,function,&[context,Value::Integer(0),
            Value::Parts(vec![Value::Parts(vec![Value::Number(0.);4]);count])],0,false,Some(words.clone()),None,false);
        Ok(state.arguments[2].clone())
    };
    if let Some(entry)=camera {
        let context=Value::Unknown;
        let result=output(entry,context,3)?;
        if result.part(0).part(3)!=Value::Number(1.) {return Ok(true);}
    }
    if let Some(entry)=sensor {
        let context=Value::Parts(vec![Value::Unknown,Value::Time,Value::Unknown,Value::Unknown]);
        if output(entry,context,1)?.time() {return Ok(true);}
    }
    Ok(false)
}


pub(super) fn scalar_argument_spectral(module: &naga::Module, entry: &str, wavelength: usize, base: usize, words: &[Option<u32>]) -> Result<bool> {
    let (_,function)=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry)).context("spectral stage bridge missing")?;
    let mut args=vec![Value::Unknown;function.arguments.len()];args[wavelength]=Value::Time;args[base]=Value::Integer(0);
    let (value,_)=evaluate(module,function,&args,0,false,Some(words.into()),None,false);
    Ok(value.time())
}

/// Numeric initializer accessors share camera calibration's exact CPU evaluator.
pub(super) fn input_value(module: &naga::Module, entry: &str, words: &[u32]) -> Option<serde_json::Value> {
    fn json(value: Value) -> Option<serde_json::Value> {
        match value {
            Value::Number(v) => serde_json::Number::from_f64(v).map(serde_json::Value::Number),
            Value::Integer(v) => Some(v.into()),
            Value::Bool(v) => Some(v.into()),
            Value::Parts(v) => v.into_iter().map(json).collect::<Option<Vec<_>>>().map(serde_json::Value::Array),
            _ => None,
        }
    }
    let function=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry))?.1;
    let words=words.iter().copied().map(Some).collect::<Vec<_>>();
    let (value,_)=evaluate_mode(module,function,&[Value::Integer(0)],0,false,Some(words.into()),None,false,true,None,None,None,false);
    json(value)
}

/// Evaluate an acceleration certificate from packed, sampled material inputs.
/// Unsupported or non-finite expressions fail admission instead of borrowing optics.
pub(super) fn camera_calibration(module: &naga::Module, entry: &str, words: &[u32], aspect: f32, focus: f32, time: f32) -> Result<[[f32;4];4]> {
    let function=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry))
        .context("camera calibration entry missing")?.1;
    let args=[Value::Number(aspect.into()),Value::Number(focus.into()),Value::Number(time.into()),Value::Integer(0),
        Value::Parts(vec![Value::Parts(vec![Value::Number(0.);4]);4])];
    let words=words.iter().copied().map(Some).collect::<Vec<_>>();
    let (_,state)=evaluate_mode(module,function,&args,0,false,Some(words.into()),None,false,true,None,None,None,false);
    let mut result=[[0.;4];4];
    for row in 0..4 {for col in 0..4 {
        let Value::Number(v)=state.arguments[4].part(row).part(col) else {
            anyhow::bail!("camera calibration must evaluate to finite numeric projective facts: row {row}, column {col}: {:?}",state.arguments[4]);
        };
        ensure!(v.is_finite(),"non-finite camera calibration");
        result[row][col]=v as f32;
    }}
    Ok(result)
}

/// Prove aperture absence over the camera clock and varying input domain.
/// Unknown outputs retain refinement; a sampled zero is not a domain proof.
pub(super) fn camera_aperture_possible(module: &naga::Module, entry: &str, words: &[Option<u32>], aspect: f32, focus: Option<f32>) -> Result<bool> {
    let function=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry))
        .context("camera calibration entry missing")?.1;
    let args=[Value::Number(aspect.into()),focus.map_or(Value::Unknown,|f|Value::Number(f.into())),Value::Unknown,Value::Integer(0),
        Value::Parts(vec![Value::Parts(vec![Value::Number(0.);4]);4])];
    let (_,state)=evaluate(module,function,&args,0,false,Some(words.into()),None,false);
    Ok(state.arguments[4].part(2).part(0)!=Value::Number(0.))
}

pub(super) fn camera_unit_weight(module: &naga::Module, entry: &str, words: &[Option<u32>]) -> Result<bool> {
    let function=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry)).context("camera stage bridge missing")?.1;
    let (_,state)=evaluate(module,function,&[Value::Unknown,Value::Integer(0),Value::Parts(vec![Value::Parts(vec![Value::Number(0.);4]);3])],0,false,Some(words.into()),None,false);
    Ok((0..3).all(|c|state.arguments[2].part(2).part(c)==Value::Number(1.)))
}

#[cfg(test)]
mod query_loop_tests {
    use super::*;

    #[test]
    fn finite_call_graph_has_no_fixed_depth_ceiling() -> Result<()> {
        let mut source=String::from("fn chain0(x:f32)->f32{return x;}");
        for i in 1..48 {source+=&format!("fn chain{i}(x:f32)->f32{{return chain{}(x);}}",i-1);}
        let module=naga::front::wgsl::parse_str(&source)?;
        let function=module.functions.iter().last().unwrap().1;
        let (_,state)=evaluate(&module,function,&[Value::Unknown],0,false,None,Some(Live {texels:false,wavelength:true,query_bounds:true}),true);
        assert!(query_bound_failure(&state.proof_failures.lock().unwrap()).is_none());
        // Recursive call graphs cannot enter the validated Naga interpreter.
        let error=naga::front::wgsl::parse_str("fn recursiveHelper(x:f32)->f32{return recursiveHelper(x);}").unwrap_err();
        assert!(error.emit_to_string("fn recursiveHelper(x:f32)->f32{return recursiveHelper(x);}").contains("recursiveHelper"));
        Ok(())
    }

    #[test]
    fn finite_paths_and_unbounded_survivors() -> Result<()> {
        for (body,finite) in [
            ("for(var i=0;i<4;i++){if(i==3){return 0.5;}}return 0.;",true),
            ("for(var i=0;i<4;i++){if(x>0.){continue;}if(x<0.){break;}}return 0.;",true),
            ("for(var i=0;i<4;i++){if(x>0.){break;}else{continue;}}return 0.;",true),
            ("for(var i=0;i<4;i++){if(x>0.){return 0.5;}}return 0.;",true),
            ("for(var i=0;i<4;i++){if(x>0.){continue;}i-=1;}return 0.;",false),
            ("loop {if(x>0.){break;}}return 0.;",false),
            ("loop {if(x>0.){return 0.5;}}return 0.;",false),
        ] {
            let source=format!("fn helper(x:f32)->f32 {{{body}}}");
            let module=naga::front::wgsl::parse_str(&source)?;
            let function=&module.functions.iter().next().unwrap().1;
            let (_,state)=evaluate(&module,function,&[Value::Unknown],0,false,None,Some(Live {texels:false,wavelength:true,query_bounds:true}),true);
            let failures=state.proof_failures.lock().unwrap();
            assert_eq!(query_bound_failure(&failures).is_none(),finite,"{body}: {failures:?}");
            if !finite {assert!(query_bound_failure(&failures).unwrap().starts_with("helper: loop"));}
        }
        Ok(())
    }
}

#[cfg(test)]
mod volume_tests {
    use super::*;
    #[test]
    fn production_volume_helper_reaches_stage_discovery() -> Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let source="import blender.volume; struct Inputs {}; MediumCoefficients medium(MediumContext c,Inputs i){MediumPoint p={};return principled_volume(p);}";
        let (stages,_)=Compiler::discover().stages(&abi,source,Path::new("production-volume.slang"))?;
        assert!(stages.contains("medium"));
        Ok(())
    }
}

/// Bound a point callback independently of its output ranges. No failed loop
/// proof can be hidden by multiplying or clamping its result to a constant.
pub(super) fn medium_work(module:&naga::Module,entry:&str,words:&[Option<u32>])->Result<u32> {
    let (_,function)=module.functions.iter().find(|(_,f)|f.name.as_deref()==Some(entry)).context("medium coefficient bridge missing")?;
    let proof=std::sync::Arc::new(std::sync::Mutex::new(WorkProof::default()));
    evaluate_mode(module,function,&[Value::Position,Value::Time,Value::Integer(0),Value::Position,Value::Independent],0,false,Some(words.iter().copied().collect()),None,false,false,None,None,Some(proof.clone()),false);
    let proof=proof.lock().unwrap();
    ensure!(proof.failure.is_none(),"{}",proof.failure.as_deref().unwrap_or("medium work proof failed"));
    Ok(proof.operations.max(1))
}

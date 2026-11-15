//! Constant loop bounds, checked on naga's IR.
//!
//! A shader's loops must end in at most [`MAX_LOOP_ITERATIONS`] iterations, provably from the source: a GPU has no
//! way to stop a runaway shader short of a device reset. The accepted form is the counted `for` loop,
//!
//! ```wgsl
//! for (var i = 0; i < 16; i++) { if (i >= count()) { break; } … }
//! ```
//!
//! which naga lowers to `loop { if (cond) {} else { break; } body; continuing { i = i ± step; } }`. The rules: the
//! loop's first statement is that test, comparing the counter (a local variable) with a constant; the counter
//! starts at a constant; `continuing` changes it by a constant step and nothing else writes it; and the count from
//! those constants is at most the limit. Anything else (`while`, a bare `loop`, a bound read from a parameter) is
//! rejected: a parameter-driven count is a constant bound plus an early `break`.

use crate::MAX_LOOP_ITERATIONS;
use naga::{BinaryOperator, Block, Expression, Function, Handle, Literal, LocalVariable, Module, Statement};

pub(crate) struct Unbounded {
    pub span: naga::Span,
    pub reason: String,
}

/// Every loop in `f` that doesn't have a provable constant bound.
pub(crate) fn unbounded(m: &Module, f: &Function) -> Vec<Unbounded> {
    let mut out = Vec::new();
    walk(m, f, &f.body, &mut out);
    out
}

fn walk(m: &Module, f: &Function, block: &Block, out: &mut Vec<Unbounded>) {
    let stmts: Vec<(&Statement, &naga::Span)> = block.span_iter().collect();
    for (i, (s, span)) in stmts.iter().enumerate() {
        match s {
            Statement::Loop { body, continuing, break_if } => {
                if let Err(reason) = bounded(m, f, &stmts[..i], body, continuing, break_if.as_ref()) {
                    out.push(Unbounded { span: **span, reason });
                }
                walk(m, f, body, out);
                walk(m, f, continuing, out);
            }
            Statement::Block(b) => walk(m, f, b, out),
            Statement::If { accept, reject, .. } => {
                walk(m, f, accept, out);
                walk(m, f, reject, out);
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    walk(m, f, &c.body, out);
                }
            }
            _ => {}
        }
    }
}

/// A constant's value (literals, module constants, conversions, negation and arithmetic of those).
fn constant(m: &Module, arena: &naga::Arena<Expression>, e: Handle<Expression>) -> Option<f64> {
    match &arena[e] {
        Expression::Literal(l) => literal(l),
        Expression::Constant(c) => constant(m, &m.global_expressions, m.constants[*c].init),
        Expression::ZeroValue(_) => Some(0.0),
        // Conversions need their own typed proof; never erase them.
        Expression::As { .. } => None,
        Expression::Unary { op: naga::UnaryOperator::Negate, expr } => constant(m, arena, *expr).map(|v| -v),
        Expression::Binary { op, left, right } => {
            let (a, b) = (constant(m, arena, *left)?, constant(m, arena, *right)?);
            match op {
                BinaryOperator::Add => Some(a + b),
                BinaryOperator::Subtract => Some(a - b),
                BinaryOperator::Multiply => Some(a * b),
                BinaryOperator::Divide if b != 0.0 => Some(a / b),
                _ => None,
            }
        }
        _ => None,
    }
}

fn literal(l: &Literal) -> Option<f64> {
    Some(match *l {
        Literal::F32(v) => v as f64,
        Literal::F16(v) => f32::from(v) as f64,
        Literal::F64(v) => v,
        Literal::U32(v) => v as f64,
        Literal::I32(v) => v as f64,
        Literal::U64(v) => v as f64,
        Literal::I64(v) => v as f64,
        Literal::AbstractInt(v) => v as f64,
        Literal::AbstractFloat(v) => v,
        Literal::Bool(_) => return None,
    })
}

/// The local variable `e` loads, if it's `Load(LocalVariable(v))`.
fn loaded_local(f: &Function, e: Handle<Expression>) -> Option<Handle<LocalVariable>> {
    match f.expressions[e] {
        Expression::Load { pointer } => match f.expressions[pointer] {
            Expression::LocalVariable(v) => Some(v),
            _ => None,
        },
        _ => None,
    }
}

fn points_at(f: &Function, e: Handle<Expression>, v: Handle<LocalVariable>) -> bool {
    match f.expressions[e] {
        Expression::LocalVariable(w) => w == v,
        Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => points_at(f, base, v),
        _ => false,
    }
}

/// Statements in `block` (and nested) that write `v`: stores, and calls handed a pointer to it.
fn writes(f: &Function, block: &Block, v: Handle<LocalVariable>) -> usize {
    let mut n = 0;
    for s in block.iter() {
        n += match s {
            Statement::Store { pointer, .. } => points_at(f, *pointer, v) as usize,
            Statement::Call { arguments, .. } => arguments.iter().filter(|a| points_at(f, **a, v)).count(),
            Statement::Block(b) => writes(f, b, v),
            Statement::If { accept, reject, .. } => writes(f, accept, v) + writes(f, reject, v),
            Statement::Switch { cases, .. } => cases.iter().map(|c| writes(f, &c.body, v)).sum(),
            Statement::Loop { body, continuing, .. } => writes(f, body, v) + writes(f, continuing, v),
            _ => 0,
        };
    }
    n
}

fn is_break_only(b: &Block) -> bool {
    matches!(b.iter().collect::<Vec<_>>().as_slice(), [Statement::Break])
}

const HOW: &str = "a loop needs a counter compared with a constant: `for (var i = 0; i < N; i++)`";

fn bounded(
    m: &Module,
    f: &Function,
    before: &[(&Statement, &naga::Span)],
    body: &Block,
    continuing: &Block,
    break_if: Option<&Handle<Expression>>,
) -> Result<(), String> {
    if break_if.is_some() {
        return Err(format!("`break if` isn't a checked bound; {HOW}"));
    }
    // The loop test: the first statement that isn't an Emit.
    let test = body.iter().find(|s| !matches!(s, Statement::Emit(_) | Statement::Call { .. }));
    let Some(Statement::If { condition, accept, reject }) = test else {
        return Err(format!("a `loop` has no bound; {HOW}"));
    };
    if !(accept.is_empty() && is_break_only(reject)) {
        return Err(format!("a `loop` has no bound; {HOW}"));
    }
    let Expression::Binary { op, left, right } = f.expressions[*condition] else {
        return Err(format!("the loop condition isn't a comparison of a counter with a constant; {HOW}"));
    };
    // counter OP bound, or bound OP counter (flipped).
    let (v, bound, op) = match (loaded_local(f, left), loaded_local(f, right)) {
        (Some(v), _) if constant(m, &f.expressions, right).is_some() => (v, constant(m, &f.expressions, right).unwrap(), op),
        (_, Some(v)) if constant(m, &f.expressions, left).is_some() => {
            let flipped = match op {
                BinaryOperator::Less => BinaryOperator::Greater,
                BinaryOperator::LessEqual => BinaryOperator::GreaterEqual,
                BinaryOperator::Greater => BinaryOperator::Less,
                BinaryOperator::GreaterEqual => BinaryOperator::LessEqual,
                other => other,
            };
            (v, constant(m, &f.expressions, left).unwrap(), flipped)
        }
        _ => {
            return Err(format!(
                "the loop's bound isn't a constant (a parameter or a computed value can't bound a loop); {HOW}, then `if (i >= n) {{ break; }}`"
            ))
        }
    };
    let name = f.local_variables[v].name.clone().unwrap_or_else(|| "counter".into());
    // The counter's start: its constant initialiser, or the last constant store before the loop.
    let mut start = f.local_variables[v].init.and_then(|e| constant(m, &f.expressions, e));
    for (s, _) in before {
        if let Statement::Store { pointer, value } = s {
            if points_at(f, *pointer, v) {
                start = constant(m, &f.expressions, *value);
            }
        }
    }
    let Some(start) = start else {
        return Err(format!("`{name}` doesn't start at a constant; {HOW}"));
    };
    // The step: `continuing { v = v ± c; }`, the loop's only write to v.
    let mut step = None;
    for s in continuing.iter() {
        if let Statement::Store { pointer, value } = s {
            if points_at(f, *pointer, v) {
                if let Expression::Binary { op, left, right } = f.expressions[*value] {
                    if loaded_local(f, left) == Some(v) {
                        let c = constant(m, &f.expressions, right);
                        step = match (op, c) {
                            (BinaryOperator::Add, Some(c)) => Some(c),
                            (BinaryOperator::Subtract, Some(c)) => Some(-c),
                            _ => None,
                        };
                    }
                }
            }
        }
    }
    let Some(step) = step else {
        return Err(format!(
            "`{name}` isn't stepped by a constant in the loop's header (a while loop's bound isn't checked); {HOW}"
        ));
    };
    if writes(f, body, v) + writes(f, continuing, v) != 1 {
        return Err(format!("`{name}` is changed inside the loop; only the header's step may change the counter"));
    }
    let unsigned = matches!(
        m.types[f.local_variables[v].ty].inner,
        naga::TypeInner::Scalar(naga::Scalar { kind: naga::ScalarKind::Uint, .. })
    );
    let away = || format!("`{name}` moves away from its bound, so the loop never ends");
    let count = match op {
        BinaryOperator::Less | BinaryOperator::LessEqual => {
            if start > bound || (op == BinaryOperator::Less && start >= bound) {
                0.0
            } else if step <= 0.0 {
                return Err(away());
            } else if op == BinaryOperator::Less {
                ((bound - start) / step).ceil()
            } else {
                ((bound - start) / step).floor() + 1.0
            }
        }
        BinaryOperator::Greater | BinaryOperator::GreaterEqual => {
            if start < bound || (op == BinaryOperator::Greater && start <= bound) {
                0.0
            } else if step >= 0.0 {
                return Err(away());
            } else {
                let n = if op == BinaryOperator::Greater {
                    ((start - bound) / -step).ceil()
                } else {
                    ((start - bound) / -step).floor() + 1.0
                };
                // An unsigned counter must not wrap below zero before the test fails.
                if unsigned && start + n * step < 0.0 {
                    return Err(format!("`{name}` is unsigned and wraps below 0 before the test fails, so the loop never ends"));
                }
                n
            }
        }
        _ => return Err(format!("the loop compares `{name}` with `==` or `!=`; use <, <=, > or >=; {HOW}")),
    };
    if !count.is_finite() || count > MAX_LOOP_ITERATIONS as f64 {
        return Err(format!(
            "the loop over `{name}` runs {} times; a loop runs at most {MAX_LOOP_ITERATIONS}",
            if count.is_finite() { format!("{count}") } else { "unboundedly many".into() }
        ));
    }
    // A real-number count alone does not prove termination: the header update
    // must also stay representable and make progress in the counter's type.
    let naga::TypeInner::Scalar(scalar) = m.types[f.local_variables[v].ty].inner else {
        return Err(format!("`{name}` needs a scalar counter"));
    };
    let representable = |x: f64| match (scalar.kind, scalar.width) {
        (naga::ScalarKind::Uint, 4) => x >= 0.0 && x <= u32::MAX as f64 && x.fract() == 0.0,
        (naga::ScalarKind::Sint, 4) => x >= i32::MIN as f64 && x <= i32::MAX as f64 && x.fract() == 0.0,
        // Shader backends may flush subnormals to zero. Such a counter
        // or step cannot supply a portable progress proof.
        (naga::ScalarKind::Float, 4) => x.is_finite() && (x as f32) as f64 == x
            && (x == 0.0 || x.abs() >= f32::MIN_POSITIVE as f64),
        _ => false,
    };
    if !representable(start) || !representable(bound) || !step.is_finite()
        || (scalar.kind == naga::ScalarKind::Float && !representable(step)) {

        return Err(format!("`{name}` has constants whose machine arithmetic cannot be proved"));
    }
    let test = |x: f64| match op {
        BinaryOperator::Less => x < bound,
        BinaryOperator::LessEqual => x <= bound,
        BinaryOperator::Greater => x > bound,
        BinaryOperator::GreaterEqual => x >= bound,
        _ => false,
    };
    let mut value = start;
    for _ in 0..MAX_LOOP_ITERATIONS {
        if !test(value) { return Ok(()); }
        let next = if scalar.kind == naga::ScalarKind::Float {
            ((value as f32) + (step as f32)) as f64
        } else {
            value + step
        };
        if !representable(next) || next == value {
            return Err(format!("`{name}` overflows or stops progressing before the loop test fails"));
        }
        value = next;
    }
    if test(value) {
        return Err(format!("the loop over `{name}` exceeds {MAX_LOOP_ITERATIONS} iterations in machine arithmetic"));
    }
    Ok(())
}

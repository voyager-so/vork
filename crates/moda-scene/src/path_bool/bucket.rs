//! Spatial bucketing and the exact lane's budget: what lets a request carry hundreds of operands.
//!
//! Operands whose control-hull boxes do not overlap (directly or through a chain of others) cannot interact
//! in any op, so a union, an exclude or a Pathfinder op solves each connected group on its own and
//! concatenates the answers; subtract, crop and minus back drop the operands that miss the one they cut or
//! clip. Each group then runs the exact lane (flo_curves, checked against the polygon lane) while the request's
//! [`ExactBudget`] lasts, and the polygon lane alone after that. The budget is what the solver used to cap the
//! whole request at (64 operands, 8,000 cubics), so the exact lane's cost, and its cross-check's, stay where
//! they were, whatever the operand count. The polygon lane (`i_overlay`, integer, O(n log n), folded as a
//! balanced tree) answers everything above it.

use super::solve::{cubic_count, Region};

/// Most operands one group may send down the exact lane: flo_curves folds left to right, so a group's cost
/// grows with the square of its size (a 200-operand grid's union took 1 s, a 500-operand one 12.6 s).
pub(crate) const EXACT_MAX_OPERANDS: usize = 64;
/// Most cubics the exact lane may see across the whole request (all its groups together).
pub(crate) const EXACT_MAX_CUBICS: usize = 8_000;

/// Axis-aligned box of a region's control points (a superset of the curve's own bounds).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Hull {
    pub(crate) min_x: f64,
    pub(crate) min_y: f64,
    pub(crate) max_x: f64,
    pub(crate) max_y: f64,
}

impl Hull {
    /// Overlap, touching included: two regions whose boxes only touch can still share an edge.
    pub(crate) fn overlaps(&self, other: &Hull) -> bool {
        self.min_x <= other.max_x
            && other.min_x <= self.max_x
            && self.min_y <= other.max_y
            && other.min_y <= self.max_y
    }
}

/// `None` for an empty region.
pub(crate) fn hull(region: &Region) -> Option<Hull> {
    let mut h = Hull {
        min_x: f64::INFINITY,
        min_y: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        max_y: f64::NEG_INFINITY,
    };
    for (start, segs) in region {
        for p in std::iter::once(start).chain(segs.iter().flat_map(|(a, b, c)| [a, b, c])) {
            h.min_x = h.min_x.min(p.0);
            h.min_y = h.min_y.min(p.1);
            h.max_x = h.max_x.max(p.0);
            h.max_y = h.max_y.max(p.1);
        }
    }
    (h.min_x <= h.max_x).then_some(h)
}

/// Connected groups of the non-empty regions under box overlap, each ascending (stacking order kept), the
/// groups ordered by their first member. A sweep over the boxes sorted by `min_x`: O(n log n) plus the
/// overlapping pairs.
pub(crate) fn components(hulls: &[Option<Hull>]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..hulls.len()).filter(|&i| hulls[i].is_some()).collect();
    let h = |i: usize| hulls[i].expect("filtered to Some");
    order.sort_by(|&a, &b| h(a).min_x.total_cmp(&h(b).min_x));

    let mut parent: Vec<usize> = (0..hulls.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    // Boxes whose x range is still open at the sweep position.
    let mut active: Vec<usize> = Vec::new();
    for &i in &order {
        let hi = h(i);
        active.retain(|&j| h(j).max_x >= hi.min_x);
        for &j in &active {
            if h(j).overlaps(&hi) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
        active.push(i);
    }

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut slot: Vec<Option<usize>> = vec![None; hulls.len()];
    for i in (0..hulls.len()).filter(|&i| hulls[i].is_some()) {
        let root = find(&mut parent, i);
        match slot[root] {
            Some(g) => groups[g].push(i),
            None => {
                slot[root] = Some(groups.len());
                groups.push(vec![i]);
            }
        }
    }
    groups
}

/// `target` and, in order, every other non-empty region in `among` whose box overlaps its box.
pub(crate) fn touching(hulls: &[Option<Hull>], target: usize, among: &[usize]) -> Vec<usize> {
    let Some(t) = hulls[target] else {
        return Vec::new();
    };
    let mut out: Vec<usize> = among
        .iter()
        .copied()
        .filter(|&i| i == target || hulls[i].is_some_and(|h| h.overlaps(&t)))
        .collect();
    if !out.contains(&target) {
        out.push(target);
        out.sort_unstable();
    }
    out
}

/// What is left of the exact lane's allowance for one request.
pub(crate) struct ExactBudget {
    cubics_left: usize,
}

impl ExactBudget {
    pub(crate) fn new() -> ExactBudget {
        ExactBudget {
            cubics_left: EXACT_MAX_CUBICS,
        }
    }

    /// Whether the group of `regions` may take the exact lane; if so, its cubics are spent.
    pub(crate) fn admit<'a>(&mut self, regions: impl IntoIterator<Item = &'a Region>) -> bool {
        let (mut n, mut cubics) = (0usize, 0usize);
        for r in regions {
            n += 1;
            cubics += cubic_count(r);
        }
        if n > EXACT_MAX_OPERANDS || cubics > self.cubics_left {
            return false;
        }
        self.cubics_left -= cubics;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hb(x0: f64, y0: f64, x1: f64, y1: f64) -> Option<Hull> {
        Some(Hull {
            min_x: x0,
            min_y: y0,
            max_x: x1,
            max_y: y1,
        })
    }

    #[test]
    fn components_join_chains_and_keep_order() {
        let hulls = vec![
            hb(0.0, 0.0, 10.0, 10.0),
            hb(100.0, 0.0, 110.0, 10.0),
            None,
            hb(10.0, 10.0, 20.0, 20.0),  // touches 0 at a corner
            hb(19.0, 0.0, 30.0, 5.0), // overlaps 3 in x only at [19, 20], and in y [0, 5] vs [10, 20]: no
            hb(25.0, 4.0, 40.0, 12.0), // overlaps 4
            hb(105.0, 5.0, 120.0, 30.0), // overlaps 1
        ];
        assert_eq!(components(&hulls), vec![vec![0, 3], vec![1, 6], vec![4, 5]]);
    }

    #[test]
    fn touching_keeps_the_target_and_its_overlaps() {
        let hulls = vec![
            hb(0.0, 0.0, 10.0, 10.0),
            hb(50.0, 50.0, 60.0, 60.0),
            hb(5.0, 5.0, 15.0, 15.0),
            None,
        ];
        assert_eq!(touching(&hulls, 0, &[1, 2, 3]), vec![0, 2]);
        assert_eq!(touching(&hulls, 3, &[0, 1, 2]), Vec::<usize>::new());
    }
}

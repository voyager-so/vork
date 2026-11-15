//! The scene's one record of what changed: a stamp per canonical node (its row, the node's index in
//! `doc.nodes`) and column group, and dirty lists for the consumers that follow changes.
//!
//! The rule: a write bumps the clock, stamps the row's group with it and adds the row to that group's dirty
//! lists. Anything derived from a row remembers the clock it was built at, and is valid if and only if the
//! stamps of the groups it reads are not newer ([`Stamps::current`]). Nothing derived keeps its own invalidation
//! rule.
//!
//! Stamps record writes; they are not a cache and hold nothing derived. Writers that go through [`crate::Scene`]'s
//! methods mark as they write. Code that writes the scene's public fields directly calls [`Stamps::mark_all`].

/// What a write changes, from the cheapest to follow to the dearest.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Group {
    /// The node's own transform, or the opacity of a body it draws itself: prepared geometry stands.
    Placement = 0,
    /// A paint's values in the form it already has (a colour, a gradient's stops): geometry stands.
    Paint = 1,
    /// Anything else of the node: it is prepared again.
    Content = 2,
    /// Membership, hierarchy, type or order.
    Structure = 3,
}

const GROUPS: usize = 4;

/// One consumer's dirty rows of one group: each row once, until the consumer clears it.
#[derive(Clone, Debug, Default)]
struct DirtyList {
    rows: Vec<u32>,
    listed: Vec<bool>,
}

/// A consumer's dirty list ([`Stamps::subscribe`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Subscription {
    group: Group,
    list: usize,
}

#[derive(Clone, Debug)]
pub struct Stamps {
    clock: u64,
    /// Per row, the clock of the last write to each group.
    rows: Vec<[u64; GROUPS]>,
    /// The clock of the last write to each group, in any row.
    latest: [u64; GROUPS],
    /// The clock of the last change not attributed to rows (a reload, a structural edit, a direct write): every
    /// row is as new as this.
    all: u64,
    lists: [Vec<DirtyList>; GROUPS],
}

impl Default for Stamps {
    fn default() -> Self {
        Stamps::new(0, 0)
    }
}

impl Stamps {
    /// Stamps for `rows` nodes that all changed at `clock`.
    pub fn new(rows: usize, clock: u64) -> Stamps {
        Stamps { clock, rows: vec![[0; GROUPS]; rows], latest: [0; GROUPS], all: clock, lists: Default::default() }
    }

    /// The clock: it advances on every write, so an unchanged clock proves an unchanged scene.
    pub fn clock(&self) -> u64 {
        self.clock
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Record a write to `group` of `row`. Returns the new clock.
    pub fn mark(&mut self, row: u32, group: Group) -> u64 {
        self.clock += 1;
        let g = group as usize;
        self.latest[g] = self.clock;
        if let Some(stamps) = self.rows.get_mut(row as usize) {
            stamps[g] = self.clock;
            for list in &mut self.lists[g] {
                if let Some(listed) = list.listed.get_mut(row as usize) {
                    if !std::mem::replace(listed, true) {
                        list.rows.push(row);
                    }
                }
            }
        } else {
            // Not a row: nothing can be attributed, so everything is dirty.
            self.all = self.clock;
        }
        self.clock
    }

    /// A write has begun whose rows are stamped when it ends (a batch edit, and a row write that re-derives,
    /// know what they changed only then). The clock advances now, so that nothing valid only while the clock
    /// stands (a derived column refreshed at the old clock) is read as current while the scene is written and
    /// not yet stamped. The stamps that follow say what changed. Returns the new clock.
    pub fn begin(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Record a write to `row` that may have changed any of its own fields (an unrestricted write: a mutable
    /// borrow of the node, a whole-node edit): its placement, paint and content are all stamped, so whatever was
    /// derived from any of them is stale. Returns the new clock.
    pub fn mark_row(&mut self, row: u32) -> u64 {
        for group in [Group::Placement, Group::Paint, Group::Content] {
            self.mark(row, group);
        }
        self.clock
    }

    /// Record a change that is not a write to known rows: every row, in every group, is dirty. Dirty lists are
    /// emptied (their rows may no longer exist); consumers see [`Stamps::all`] move.
    pub fn mark_all(&mut self) -> u64 {
        self.clock += 1;
        self.all = self.clock;
        self.latest = [self.clock; GROUPS];
        // Only the listed rows are unlisted: the cost follows what was dirty, not the row count.
        for list in self.lists.iter_mut().flatten() {
            for row in list.rows.drain(..) {
                if let Some(listed) = list.listed.get_mut(row as usize) {
                    *listed = false;
                }
            }
        }
        self.clock
    }

    /// The row count changed (a structural edit): call with the new count, then [`Stamps::mark_all`].
    pub fn resize(&mut self, rows: usize) {
        self.rows.resize(rows, [0; GROUPS]);
        for list in self.lists.iter_mut().flatten() {
            list.listed.resize(rows, false);
        }
    }

    /// Start again for a replaced scene of `rows` nodes, later than everything stamped before: derived data built
    /// at any earlier clock is stale. Subscriptions are kept, with empty lists.
    pub fn restart(&mut self, rows: usize) {
        self.rows.clear();
        self.resize(rows);
        self.mark_all();
    }

    /// Carry the clock on from a scene this one replaces, so clocks never repeat across the replacement.
    pub fn continue_from(&mut self, earlier: &Stamps) {
        self.clock = self.clock.max(earlier.clock) + 1;
        self.all = self.clock;
        self.latest = [self.clock; GROUPS];
    }

    /// The clock of the last write to `group` of `row` (a change to everything counts).
    pub fn stamp(&self, row: u32, group: Group) -> u64 {
        self.rows.get(row as usize).map_or(self.all, |stamps| stamps[group as usize].max(self.all))
    }

    /// Whether something built from `group` of `row` at clock `built_at` is still valid.
    pub fn current(&self, row: u32, group: Group, built_at: u64) -> bool {
        self.stamp(row, group) <= built_at
    }

    /// The clock of the last write to `group` in any row (a change to everything counts).
    pub fn latest(&self, group: Group) -> u64 {
        self.latest[group as usize].max(self.all)
    }

    /// The clock of the last change to everything.
    pub fn all(&self) -> u64 {
        self.all
    }

    /// A dirty list of `group` for one consumer: from now on it holds every row written in that group, once,
    /// until the consumer clears it.
    pub fn subscribe(&mut self, group: Group) -> Subscription {
        let lists = &mut self.lists[group as usize];
        lists.push(DirtyList { rows: Vec::new(), listed: vec![false; self.rows.len()] });
        Subscription { group, list: lists.len() - 1 }
    }

    /// The consumer's dirty rows, in the order they were first written.
    pub fn dirty(&self, subscription: Subscription) -> &[u32] {
        self.lists[subscription.group as usize].get(subscription.list).map_or(&[], |list| &list.rows)
    }

    /// The consumer has followed its dirty rows: empty its list.
    pub fn clear(&mut self, subscription: Subscription) {
        if let Some(list) = self.lists[subscription.group as usize].get_mut(subscription.list) {
            for row in list.rows.drain(..) {
                if let Some(listed) = list.listed.get_mut(row as usize) {
                    *listed = false;
                }
            }
        }
    }

    /// Take the subscriptions of a scene this one replaces (their lists start empty here).
    pub fn adopt_subscriptions(&mut self, earlier: &Stamps) {
        for (mine, theirs) in self.lists.iter_mut().zip(&earlier.lists) {
            mine.clear();
            mine.extend(theirs.iter().map(|_| DirtyList { rows: Vec::new(), listed: vec![false; self.rows.len()] }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_write_invalidates_exactly_what_was_built_before_it() {
        let mut stamps = Stamps::new(3, 0);
        let built = stamps.clock();
        assert!((0..3).all(|row| stamps.current(row, Group::Placement, built)));
        stamps.mark(1, Group::Placement);
        assert!(stamps.current(0, Group::Placement, built) && stamps.current(2, Group::Placement, built));
        assert!(!stamps.current(1, Group::Placement, built));
        assert!(stamps.current(1, Group::Content, built), "another group of the row stands");
        let rebuilt = stamps.clock();
        assert!(stamps.current(1, Group::Placement, rebuilt));
        stamps.mark_all();
        assert!((0..3).all(|row| !stamps.current(row, Group::Content, rebuilt)));
        assert!(stamps.latest(Group::Paint) > rebuilt);
    }

    #[test]
    fn an_unrestricted_write_invalidates_every_group_of_the_row() {
        let mut stamps = Stamps::new(2, 0);
        let (moved, painted) = (stamps.subscribe(Group::Placement), stamps.subscribe(Group::Paint));
        let built = stamps.clock();
        stamps.mark_row(1);
        for group in [Group::Placement, Group::Paint, Group::Content] {
            assert!(!stamps.current(1, group, built), "{group:?}");
            assert!(stamps.current(0, group, built), "{group:?}");
        }
        assert!(stamps.current(1, Group::Structure, built));
        assert_eq!((stamps.dirty(moved), stamps.dirty(painted)), (&[1u32][..], &[1u32][..]));
    }

    #[test]
    fn each_consumer_sees_each_dirty_row_once_until_it_clears() {
        let mut stamps = Stamps::new(4, 0);
        let (a, b) = (stamps.subscribe(Group::Placement), stamps.subscribe(Group::Placement));
        let content = stamps.subscribe(Group::Content);
        for row in [2, 0, 2, 2, 3] {
            stamps.mark(row, Group::Placement);
        }
        assert_eq!(stamps.dirty(a), [2, 0, 3]);
        assert_eq!(stamps.dirty(b), [2, 0, 3]);
        assert!(stamps.dirty(content).is_empty());
        stamps.clear(a);
        stamps.mark(2, Group::Placement);
        assert_eq!(stamps.dirty(a), [2], "a cleared consumer sees the row again");
        assert_eq!(stamps.dirty(b), [2, 0, 3], "a consumer that has not cleared keeps one entry");
        stamps.resize(2);
        stamps.mark_all();
        assert!(stamps.dirty(a).is_empty() && stamps.dirty(b).is_empty());
        // A change to everything unlisted every row: each is listed again by its next write.
        stamps.mark(1, Group::Placement);
        stamps.mark(0, Group::Placement);
        assert_eq!(stamps.dirty(a), [1, 0]);
        assert_eq!(stamps.dirty(b), [1, 0]);
        stamps.clear(b);
        stamps.mark(1, Group::Placement);
        assert_eq!(stamps.dirty(b), [1]);
        // A row that does not exist dirties everything.
        let before = stamps.all();
        stamps.mark(9, Group::Placement);
        assert!(stamps.all() > before);
    }

    #[test]
    fn a_replacement_scene_never_repeats_a_clock() {
        let mut old = Stamps::new(2, 0);
        let moved = old.subscribe(Group::Placement);
        old.mark(0, Group::Placement);
        old.mark(1, Group::Content);
        let built = old.clock();
        let mut new = Stamps::new(5, 0);
        new.adopt_subscriptions(&old);
        new.continue_from(&old);
        assert!(new.clock() > built);
        assert!((0..5).all(|row| !new.current(row, Group::Placement, built)));
        new.mark(4, Group::Placement);
        assert_eq!(new.dirty(moved), [4]);
    }
}

// Coverage draws from the invocation's transport allowance on demand. Initial
// containment, contact and shadow reservations each cover at most two surface
// evaluations; camera candidates cover one. Counting each reservation as one
// step therefore bounds coverage by twice the work limit, without a second
// fixed quota that can run out during finite containment or repeated queries.
// The invocation counter prevents auxiliary ledgers from renewing the reserve.
static uint invocation_medium_steps=0u;
struct PathMediumWork { uint remaining; bool failed; };
// Watch actual work, not hypothetical instructions. Roulette and the profile's
// bounce budgets own normal termination; the hard ceiling catches runaway work.
// Preserve the existing transport accounting, including one step per coverage
// reservation, so previously completing paths encounter the same hard ceiling.
bool medium_work_step(inout PathMediumWork ledger) {
    let cost=1u;
    if(ledger.failed) {return false;}
    bool exhausted=cost>ledger.remaining;
#ifdef VORK_MEDIUM_WORK_LIMIT
    exhausted=exhausted || cost>VORK_MEDIUM_WORK_LIMIT-invocation_medium_steps;
#endif
    if(exhausted) {
        ledger.failed=true;
        return false;
    }
    invocation_medium_steps+=cost;ledger.remaining-=cost;return true;
}

# GDN delta broadcast lifetime

The tiled recurrence first reduces the retained matrix projection and computes
one delta per value column. All four waves of its 128-thread workgroup consume
that delta before constructing the second reduction.

The previous kernel stored the broadcast in the first row of the reduction
array. Lane zero could replace that row with its second-projection terms before
another wave had read the broadcast. A barrier after the update loop did not
order these conflicting accesses.

The kernel now publishes the deltas in a separate immutable shared array. The
existing barrier makes the broadcast visible; no thread overwrites it while
the reduction array is reused. This adds 128 bytes of shared memory per block
without changing the recurrence equations, FP32 accumulation order, persistent
state ownership or graph status checks. The launcher and compiler identities
advance so older numerical plans cannot silently reuse the implementation.

The regression fixture uses the canonical 16 key heads and 48 value heads with
128-wide heads, asymmetric nonzero input and state, and updates large enough to
test persistent state directly. It checks independent scalar recurrence
equations, repeated identical initial conditions and captured/eager execution.
State tolerance is smaller than the broadcast error; BF16 output equality
alone is insufficient to detect it. The strengthened fixture fails with the
previous kernel and passes after the correction on gfx1151.

The complete 174-test suite passes on the first target. Full-checkpoint paired
logit/state determinism across modes is a separate qualification gate; results
remain private until that gate has completed. No workload or distributed
qualification is implied by the operator tests.

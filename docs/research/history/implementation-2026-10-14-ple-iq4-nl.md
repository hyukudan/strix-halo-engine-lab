# File-mapped IQ4_NL PLE lookup

The file-mapped PLE backend accepts BF16 and IQ4_NL tables. The request and
destination retain the BF16 activation contract; compressed storage is internal
to the backend. No whole-table expansion or checkpoint conversion is required.

An IQ4_NL row consists of complete 32-element blocks, each containing a binary16
scale and 16 packed bytes. Low nibbles represent the first 16 elements, high
nibbles the next 16. The nonlinear codebook is multiplied by the scale and each
result is rounded to BF16 with round-to-nearest, ties-to-even. Nonfinite scales
are rejected. A 160-element row occupies 90 stored bytes and produces 320 output
bytes. Requested row order and duplicates are preserved.

The backend checks two-dimensional nonzero geometry, complete blocks, both
manifest and mapping bounds, host-address overflow, output size and row IDs.
Its identity changed to distinguish the new interpretation. It remains C1:
one pending lookup owns the handle until completion or cancellation. Failed
completion retains that ownership; the prefetch wrapper cancels and clears its
state before propagating the failure. Partially written output from a failed
lookup must never be consumed.

`strix-ple-iq4-nl-tests` checks every finite binary16 scale against every codebook
entry in both nibble planes, using independent mathematical binary16 decoding
and BF16 rounding. Additional checks cover row ordering, duplicates, five-block
rows, unaligned destinations, guard bytes, malformed geometry and retained
ownership after failure. Existing BF16 lookup and workspace tests remain valid.
These three tests and the complete 166-entry CTest suite passed on the target
Linux host with gfx1151. Both changed C++ translation units also compiled with
the project's strict warning set and `-Werror`. The integration build retained
its existing warnings-as-errors override for an unrelated compiler diagnostic.

This qualifies the lookup conversion in isolation. Full-model logits, state,
generation and distributed model continuity still require independent native
execution evidence. The planned-I/O backend and its separate storage contract
are unchanged.

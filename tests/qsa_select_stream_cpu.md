# CPU selection proof fixture

Build and run from the repository root (C++20; no HIP headers or GPU):

```sh
g++ -O2 -std=c++20 tests/qsa_select_stream_cpu.cpp -o /tmp/x && /tmp/x
```

The reference ports `src/hip_indexed_attention_launcher.hip:1693-1793`
(`select_key`, the three 21/10/0 radix passes with 2048/2048/1024 digits,
prefix/mask/need and early exits, ordered compaction, fill and score writes).
It also models `src/hip_qsa_prefill_flash.inc:247-351`: dense rows,
group-maxima filtering, the 6144-candidate capacity and both fallback reasons.
Every row additionally compares the group shortcut against the full radix path.
Scalar exact integer sums replace device ballots/atomics/packed scans; ascending
iteration gives the same exclusive greater/equal ranks without changing order.
Dynamic graph arguments, statistics atomics and score computation are out of scope.

Group maxima use selection-equivalent numeric maxima for the producer's four-block
tree at `src/hip_qsa_prefill_flash.inc:224-229`. They ignore NaNs, fold zero signs
through `select_key`, and may represent an all-ineligible group by `-inf`.
Their raw bits are not outputs. Host `std::fmax` is deliberately avoided: its
signaling-NaN behavior initially caused a host-port mismatch on random bit patterns.
The full radix reference consumes original score bits directly; this additional
cross-check prevents the group optimization from silently changing selection.

Both candidates rank only `(select_key descending, block ascending)` integer
records. The streaming implementation traverses ascending slabs (default 256),
splitting traversal at absolute 32-block units. Each merge sorts the new slab by
integer rank and retains the first 512 records of its merge with the prior
winners. Discarded records cannot re-enter a later top 512. The two-launch model
first stores every partition's top 512, then merges those lists. Final emission
sorts winners by ascending block id. These are CPU algorithm models, not device
kernels or CPU performance benchmarks.

With fewer than 512 eligible scores, missing slots precede winners and each
contains tokens `0,1,2,3`; block 0 can consequently appear repeatedly and also
as an eligible winner. Selected scores and block 0 on fill become `-inf`, even
if block 0 was NaN. Partial tail tokens follow slot 511. For at most 512 complete
blocks, all visible tokens are emitted densely and **no scores change**, regardless
of eligibility. Eligibility is exactly `score > -inf`: positive infinity qualifies,
NaNs and negative infinity do not, and both zero signs have the same key.

Coverage uses seed `0x11c0512`, 616 directed rows (14 complete-block counts from
0 through 65536, every tail 0..3, eleven score classes), and 2400 randomized rows.
Each directed row tests all 24 slab/partition combinations: slabs
`1,7,32,64,255,256,257,1024`, partitions `512,1024,4096`. Random rows choose these
parameters independently. Classes cover all equal, boundary ties, signed zero,
infinities/mixed NaNs, all negative infinity, all NaNs (quiet/signaling, distinct
payloads/signs), 173 eligible, exactly 512 eligible (clipped for smaller rows),
denormals, random finite normals and arbitrary score bits. Every comparison checks
the entire selected capacity, including `0xa5a5a5a5` canaries, and all post-selection
score bytes. Every two-launch case also checks its actual packed-record byte count.
Exit status is zero only when all comparisons and reference branch coverage pass.

The printed byte table uses fully eligible rows at 8192, 32768 and 65536 blocks.
It reports 8 bytes per partial record, separate write/read totals, their roundtrip,
the full score tensor's 4 bytes/block, and partial-write minus full-score-write.
This is payload accounting: count/offset metadata and other buffers are excluded.
Rows with fewer eligible values store fewer records; dense rows store none.
Partition sizes 512/1024/4096 give partial writes of 2x/1x/0.25x the full score
tensor, respectively; reading the partials doubles those byte totals. These
figures establish no launch deletion, compiler resources, native numerical
fidelity, throughput, or complete-workload qualification. F11 remains a draft.

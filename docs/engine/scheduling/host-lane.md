# Host lane and byte budget

The host lane is the one CPU pool shared by every engine and by the serving layer. Engines do not own
threads for request preprocessing or background work; they submit jobs to the lane.

Sources: `src/core/host_lane.hpp`, `src/core/byte_budget.hpp`. Test: `strix-host-lane-tests`.

## Two priorities

| Priority | Jobs | Threads |
| --- | --- | --- |
| `critical` | short, latency-bound: tokenizer, row folds, constraint masks | at most two; each keeps polling for a short window (200 µs by default) after a job, so back-to-back jobs do not pay a wake-up |
| `background` | byte-heavy: snapshot packing, checkpoint CRC, cache hashing | sleeping threads (two by default); every job draws the bytes it reads from the byte budget |

`submit(priority, job)` returns a `std::future` of the job's result; an exception thrown by the job
travels through that future. Jobs still queued when the lane is destroyed are completed first.

When cores are declared, each critical thread is pinned to one of the first cores and the background
threads to the remaining ones. Without a declaration nothing is pinned.

## Byte budget

All compute units of the machine share one memory bus, and the decode weight stream uses nearly all of
it. A background job that reads memory at full speed therefore slows decode by the share of the bus it
takes. `ByteBudget` is a token bucket in bytes per second:

- no session decoding: unlimited;
- at least one session decoding (`ByteBudget::DecodeScope`): `byte_cap_gbytes_ps` (10 GB/s by default).

Decode is marked per call, so the cap stays in force for a short linger (250 ms) after the last decode
call ends: the gaps between the steps of one generation neither lift the cap nor refill the bucket.

`shared_byte_budget()` is the process-wide budget. Decode loops mark it, shared background paths draw
from it, and the process-wide lane uses it as its budget.

A reservation is always granted; the returned duration is the wait owed before the bytes are read, so a
request larger than the burst is legal. `acquire()` is the blocking form and returns early when the last
decode ends. `governed_slices()` walks a range in slices so a long pass reacts to a decode that starts
in the middle of it.

## Governed background work

- The session runtime marks every decode (and mixed) round on the process-wide budget.
- Checkpoint publication and reads (`strix::session::checkpoint_io`, the path used by session park, prefix
  checkpoints and their restores in every engine) draw each chunk they move from the budget.
  `atomic_write_background()` runs a publication as a background lane job.
- Prefix-cache key hashing draws its bytes from the budget.

Outside decode none of this waits. Test: `strix-governed-checkpoint-io-tests`.

`strix-host-lane-tests --load SECONDS THREADS CAP_GBPS [BUFFER_MIB]` is a load generator: background lane
jobs that read memory under the budget with decode marked active (`CAP_GBPS` 0: uncapped). It is used to
measure decode speed next to capped and uncapped background work.

## Configuration

The process-wide lane (`shared_host_lane()`) is on by default and reads:

| Variable | Meaning | Default |
| --- | --- | --- |
| `STRIX_HOST_LANE=0` | run every job inline in the caller (previous behaviour) | on |
| `STRIX_HOST_LANE_CORES=2-15` | cores the lane may pin | no pinning |
| `STRIX_HOST_LANE_BACKGROUND=N` | background threads | 2 |
| `STRIX_HOST_BYTE_CAP_GBPS=10` | background ceiling while a session decodes; `0` removes the cap | 10 |

## Receipt fields

`HostLane::receipt_json()` reports `host_lane_busy_ms`, `host_lane_jobs`, `host_lane_bytes_granted`,
`host_lane_cap_hits` and `host_lane_throttled_ms`.

# Native MTP fixture runner

With HIP and `BUILD_TESTING`, `strix-hip-mtp-model-qualification` uses the same
target, PLE, local sidecar and complete independent forward-replay setup as the
decoder. It does not duplicate a model loader or bypass qualification. Pass the
ordinary decoder arguments, including `--mtp`, prompt tokens and all asset,
conversion, rounding, replay and memory pins, plus:

| Option | Meaning |
| --- | --- |
| `--qualification-reference FILE` | Independently supplied fixture payload |
| `--qualification-sha256 HEX` | Expected complete file SHA256 |
| `--qualification-file-bytes N` | File bound, 160 bytes through 1 GiB |
| `--qualification-host-bytes N` | Fixture host budget, at least three times the file bound plus 256 KiB |
| `--qualification-coverage-mask N` | Cycle coverage bits: accepted window 1, restored rejection 2, EOS 4, budget 8, continuation 16; default 31 |

Both `--option value` and `--option=value` work. Duplicate, unknown, missing and
out-of-bound fixture options reject before model setup. Reference bytes are
read and completely SHA-checked before parsing or target execution. Matching a
digest binds bytes; it does not establish independent authorship of an oracle.

After actual complete forward replay and activation, the runner dispatches the
selected fixture and exits without ordinary generation. Its JSON records a
completed fixture, not general hardware, model fidelity or performance approval.
There is no automatic model CTest without externally supplied assets/references.

The selected kind is stored in the reference:

| Kind | Fixture |
| --- | --- |
| 1 | Consecutive greedy cycles, including requested coverage |
| 2 | History-capacity exhaustion with an otherwise eligible actual window and serial-only trace |
| 3 | All HC roles and routed experts with complete component outputs/workspaces |
| 4 | Shared expert with complete output/workspace |
| 5 | Real closed-window capture, publication and physical-pool reuse |
| 6 | Actual selected LM-head recipe and every vocabulary logit |
| 7 | Bounded target rollback after a real window, checking all device/host state bytes, D1 and exact session metadata |

Kinds 1/2/5/7 require the ordinary actual window options when testing a window.
For kind 2, choose a history bound already exhausted by the real prompt history
and a trace requiring no window attempts. Its cycle coverage mask normally
excludes accepted/rejected-window bits. No synthetic history is substituted.
Kind 7 uses supplied window inputs to mutate the real target; its byte-for-byte
restoration oracle is the actual pre-mutation state, not a generated model golden.

## Reference format

`tests/hip_mtp_qualification_reference.hpp` exposes a bounded encoder for external
reference producers. It serializes values only and never computes golden outputs.
`MTPQREF2` version 2 is little-endian, with no native struct serialization. The
160-byte header contains magic (8), version (uint32), kind (uint32), full file
length (uint64), three 32-byte content digests (independent source, target,
sidecar), then five uint64 identities (reference, proposal, structural plan,
semantic contract, actual native numerical profile). Only the selected payload
follows. Unknown kinds, excess counts, truncated arrays and trailing bytes reject.

Strings and vectors carry uint64 counts, followed by exact bytes or fixed-width
elements; strings are capped at 4096 bytes and records at 256 (window rows at 8).
Booleans occupy one byte and accept only 0/1. Floats retain their binary32 bits;
BF16 values retain their uint16 bits. Component serializers follow the field
order in `selected` and `cycle_row`; the encoder is the executable format contract.
Every fixture additionally checks its actual geometry and complete required
payload before launching that fixture.

Cycle references use complete final target wide values and the predictor's
occupied KV/raw-index prefixes. Reconciliation does not publish hidden/logit
outputs. Future cache rows and temporary wide/logit buffers are unspecified and
are excluded from that oracle; subsequent cycles test continuation. Independent
full-forward references remain necessary for the actual head's arithmetic profile.

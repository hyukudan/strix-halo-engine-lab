# Prefix state cache

Agent requests resend long, mostly identical prompts: a system prompt, tool schemas and the
conversation so far. `--prefix-cache-host-bytes N` keeps the complete model state at block
boundaries of the tiled prefill, and a later request whose prompt starts with a cached prefix
restores that state and prefills only the rest. The output is bit-identical to prefilling the whole
prompt with the same configuration.

It runs on the tiled known-prefix prefill
(`--known-prefix-window --known-prefix-full-rows --known-prefix-prefill-rows T`), with serial
greedy decode or with MTP decode (see [MTP](#mtp)). Every other route rejects the options.

## Options

| Option | Meaning |
|---|---|
| `--prefix-cache-host-bytes N` | Enables the cache; LRU byte budget of the host tier (pinned host memory). |
| `--prefix-cache-dir DIR` | Optional disk tier: checkpoints are written behind the request and found by later processes. |
| `--prefix-cache-disk-bytes N` | Optional disk budget (oldest checkpoints are deleted first; 0 = unbounded). |
| `--prefix-cache-every N` | Capture interval in tokens, a multiple of T (default: 4096 rounded up to a multiple of T). The last block boundary of every prompt is always captured. |
| `--prefix-cache-then-prompt FILE` | Repeatable. After the main request, run these prompts (token IDs separated by commas or whitespace) in the same process, each from a fresh session, through the same cache. Serial-decode route only. |
| `--known-prefix-tail-rows T2[,T3...]\|none` | Independent of the cache: after the last complete T-row block, run the remainder as complete blocks of the largest tail size that fits (descending sizes, 16..1024, multiples of 8, less than T), and only the rest as serial rows. Default (D-056): the sizes of 1024, 256, 64, 16 below T, whenever the tiled prefill has full rows and an output head; `none` keeps the remainder serial. |

Diagnostics: `STRIX_PREFIX_CACHE_LOOKUP=0` captures without ever restoring (every request is cold);
`STRIX_PREFIX_CACHE_DIGEST=1` adds a SHA-256 of the whole prompt state to the MTP receipt.

The receipt gains `prefix_state_cache` (serial route: identity, geometry, main request, follow-ups,
store counters) or `mtp_tiled_prompt.prefix_cache` (MTP route).

## What a checkpoint holds

A checkpoint at block boundary P = k·T is the session state after the first P prompt tokens:

- per-position rows of every attention layer: KV rows and raw indexer keys, positions [0, P);
- the state overwritten in place: GDN recurrent matrices, GDN and PLE convolution histories, the
  n-gram history and the position history;
- the session counters, state generations and the arithmetic lineage (every block interval with its
  producer identities);
- on the MTP route, the predictor's cache rows of the pairs (h_i, t_{i+1}) for i < P-1 and the target's
  wide hidden row h_{P-1}.

Per-position rows never change once written, so they are stored once per block (a segment) and the
checkpoints of one prompt share their common segments. The rest (about 111 MiB for this model) is
stored per checkpoint. Lookup returns the longest cached boundary P < prompt length (at least one
prompt token always runs, to produce the first output's logits); restore copies the rows and the
fixed state into the fresh session and republishes the recorded counters and lineage.

## Keys and identity

`identity = SHA-256` of: model content identity, executable digest (disk tier only), state layout
identity (includes the context), runtime compatibility identity, decode execution and kernel-plan
identities, prefill (sibling) execution and kernel-plan identities, tiled prefill plan identities,
prefill and decode numerical profile identities and their pair identity, T, the tail sizes and the context size.
On the MTP route it also covers the sidecar content, the MTP plan and native profile identities and
the draft-kernel switches (the predictor's cache rows are built with the draft kernels unless
`--mtp-draft-exact`).

`digest_k = SHA-256(digest_{k-1} || tokens of block k)`, `digest_0 = identity`. A digest names one
identity and one exact token prefix; the stored tokens are compared as well. Why each field matters:

- the tiled prefill reproduces serial rows bit for bit within one numerical profile (D-034), but with
  a split pair (D-033) complete blocks run the prefill profile and the unpadded remainder the decode
  profile, so which rows run which arithmetic depends on T (and the tail sizes). Checkpoints are only taken at
  block boundaries and the suffix of a restored request keeps the same block alignment as the full
  prompt, so the row-to-profile assignment is the same as without the cache;
- the decode identities do not change the state at a block boundary, but are part of the key so that
  no two configurations whose outputs could differ ever share an entry;
- context size and layout fix the state geometry; the executable digest keeps a disk tier from
  serving a state that another build produced.

## Storage

Host tier: pinned host memory, LRU over checkpoints within the byte budget; a segment is freed when
the last checkpoint that references it is evicted. A disk hit is loaded into the host tier.

Disk tier: `seg-<digest>.tmps` (one block's rows) and `ckpt-<digest>.tmpc` (fixed state, metadata
and the list of segment digests), written to a temporary name and renamed. Headers carry identity,
digests and a 64-bit payload checksum; segment tokens are re-hashed on load and must chain to the
requested digest. A file that fails any check is skipped (counted) and the next shorter prefix is
tried. Writes run on a background thread; the process waits for them before exiting.

## MTP

With `--mtp` and the tiled prefill flags the prompt itself runs through the tiled prefill:

1. Calibration: the first R prompt pairs (`--mtp-calibration-rows R`, default 8) run serially through
   the MTP target producer and are replayed against a golden of exactly those pairs (dumped with
   `--mtp-golden-dump` under the same flags). This qualifies and activates the predictor as before.
2. The target state and the predictor are reset; the longest cached prefix is restored.
3. Complete T-row blocks (then complete tail blocks) run tiled. A step observer copies every
   row's wide hidden state (the MTP target tap) out of the block arena, and the predictor's cache
   rows are built from those copies. Block boundaries are captured.
4. A block (T rows or a tail size) that ends exactly at the last prompt row also computes that
   row's head (terminal-row output policy). The first draft cycle starts from that row: its wide
   hidden state and FP32 logits mint the pending target ticket (`HipMtpTargetHidden::adopt_prefill_row`,
   same first-max selection as a serial row), so no prompt row runs serially.
5. Otherwise the remaining rows, at least the last prompt row, run serially through the MTP target
   producer, whose last ticket starts the first draft cycle. Decode is unchanged (adaptive windows).

`STRIX_MTP_TILED_START=0` restores the previous handoff (the last prompt row always serial; a prompt
of exactly k·T tokens then runs its last T rows through the tail blocks and serial rows). The receipt's
`mtp_tiled_prompt.tiled_start` says which handoff ran. A split numerical-profile pair is rejected on
this route.

## Measured

See docs/decisions/decision-register.md (D-035).

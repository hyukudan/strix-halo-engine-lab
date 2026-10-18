# Prefill numerical profile

`--prefill-numerical-profile NAME` selects the arithmetic of the prompt prefill separately from
`--numerical-profile`, which keeps selecting the arithmetic of decode: serial target rows, MTP target
windows and their checkpoints. Without the flag the prefill runs `--numerical-profile`, and every route,
token and existing receipt field is unchanged. An explicit value equal to `--numerical-profile` is the
same single-profile run.

## Admitted pairs and routes

A split pair is admitted only where it runs with existing kernels:

| Decode (`--numerical-profile`) | Prefill (`--prefill-numerical-profile`) |
| --- | --- |
| `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1` | `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1` |
| `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1` | `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1` |
| `hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1` | `hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1` |
| `hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1` | `hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1` |

The two profiles of a pair differ only in the dense Q8_0 projections (GDN, QSA, shared expert): FP32
activations or int8 activations per group of 32. Their HC, MoE and LM-head kernels, arena layout and
state layout are the same, so a state produced by one is an input state of the other. The decode
profile must be explicit. Exactly one prefill route is required:

- the tiled known-prefix prefill: `--known-prefix-window --known-prefix-full-rows
  --known-prefix-prefill-rows T`. Every complete T-row block runs the prefill profile. The unpadded
  suffix of fewer than T rows and generation run the decode profile, as serial rows;
- the MTP prompt (`--mtp`): every prompt row runs the prefill profile. MTP replay, draft cycles,
  target windows, checkpoints and generated rows run the decode profile.

Model copy tiers, final-state export, GDN islands, prepared or overlapped serial execution and the MTP
metadata diagnostic are rejected with a split pair. Every other admission rule of `--numerical-profile`
applies to both profiles. Other pairs fail at option parsing.

## Execution

The engine compiles a second serial execution for the prefill profile from the same binding and buffer
requests with only the dense A8 flag changed. Its dense projection kernel IDs carry (or lack) the
`.dense-a8-v1` marker, so the tiled prefill plan derived from it selects the matching known-window
verify IDs, and its launchers are configured from the prefill profile. The two executions are checked
to be siblings (same steps, dependencies, state access, buffer IO, arena layout and geometry; different
execution and kernel-plan identities).

A session is owned by one executor at a time. The prefill execution runs on the serial scratch arena of
the decode execution under its own execution identity (the layouts are identical and the two never run
concurrently), so the split needs no additional device memory; the extra cost is host-side: a second
compiled plan, and on the MTP route a second launcher catalog and launch plan. Before a prefill block (or
the MTP prompt) the session is handed to the prefill execution and afterwards back to decode
(`hand_off_serial_session`): only the executor identities change. The arithmetic lineage keeps the
producer identity of every row, so a session prefilled under a split pair can never be taken for a
decode-only serial history (prefix caches require one).

On the MTP route the target producer runs the prompt rows on the prefill execution. After the replay the
pending prompt row is re-minted with the handed-off session; the MTP chain check admits exactly that one
executor change and still proves the exact lineage continuation. MTP exactness holds from the handoff
state: replay against a golden dumped with the same pair has error 0, and every generated token is
verified by the decode target from the state the prefill produced. It does not reproduce the tokens of
a single-profile run from the start.

## Identity

- Receipt: `prefill_numerical_profile`, `prefill_numerical_profile_source` (`explicit` or
  `numerical-profile`) and `numerical_profile_pair_identity` in every run; with a split pair also
  `prefill_dense_a8_configured`, `prefill_serial_execution_identity` and
  `prefill_serial_kernel_plan_identity`, and `optimization_profile_selected` gains `+prefill:NAME`.
  MTP receipts add `target_prefill_numerical_profile`, `target_decode_numerical_profile`,
  `target_prefill_execution_identity` and `mtp_golden_target_profile_pair_recorded`.
- `numerical_profile_pair_identity` of a single-profile pair is that profile's identity; a split pair
  has a distinct identity that also differs from the reversed pair.
- MTP goldens (version 2) record the prefill and decode profile identities; see
  [the golden format](mtp-forward-golden-file.md). A golden dumped under one pair is rejected under
  another before any target work; a version-1 golden is admitted only for a single-profile run.

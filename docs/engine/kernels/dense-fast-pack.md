# Dense FAST pack (`--dense-fast-pack`)

Status: **experimental, opt-in, not qualified**. The pack changes the values of the dense weights
that decode reads, so a run with it is a numerical identity of its own. No pack measured so far
passes the FAST quality gate (numbers below); the flag exists so that packs can be built and
measured. The default command and every exact route are unchanged when the flag is absent.

## What it is

Every decode cycle reads the dense Q8_0 projections once (GDN input and output, indexed-attention
query / key / value / output, shared-expert gate and up): 8.5 bits per weight. The pack holds
re-quantised copies of those matrices as standard GGUF `Q4_K` (4.5 bits), `Q5_K` (5.5 bits)
or `Q6_K` (6.5625 bits) tensors under their own names. With the pack loaded, the admitted dense A8
decode windows (1 and 2 rows by default, up to 8 by switch) read
the copies through the typed dense A8 kernel (`project_rows_dense_a8_gsq_kernel`,
`src/hip_q8_rows_gemv.hpp`); prefill (16 rows and more), the hyperconnection projections, the
shared-expert down projection (640 wide), the LM head and the routed experts keep their own tensors.

| | Q8_0 | Q4_K pack | Q5_K pack | Q6_K pack |
|---|---|---|---|---|
| bytes of the replaced tensors (252 matrices) | 3.008 GB | 1.593 GB | 1.946 GB | 2.322 GB |
| weight error (relative RMS against the Q8_0 values) | - | 7.5-8.5 % | 3.7-4.2 % | 2.0-2.3 % |

## Build a pack

```sh
python3 tools/dense_fast_quantize.py MODEL.gguf PACK.gguf                      # all Q4_K
python3 tools/dense_fast_quantize.py MODEL.gguf PACK.gguf --default q5_k       # all Q5_K (or q6_k)
python3 tools/dense_fast_quantize.py MODEL.gguf PACK.gguf --spec 'attn_qkv=q5_k' --spec 'blk\.(0|47)\.=q8'
```

`--spec REGEX=FMT` (first match wins, `FMT` = `q4_k`, `q5_k`, `q6_k` or `q8` = leave on Q8_0) gives mixed
precision per tensor or layer; `--imatrix F` weights the error by column importance. The scale rule is
the K-quant weighted least-squares search per block (32 weights with a scale and a minimum for Q4_K
and Q5_K, followed by a one-step refinement of the 6-bit scale and minimum codes against the same
weighted error; 16 weights with a signed scale for Q6_K). The pack header pins the model
file by SHA-256 and content identity. `python3 -m unittest tools/test_dense_fast_quantize.py` checks
the block layout against a scalar decoder.

A pack is a derived model artifact: build it locally from your own copy of the model.

## Use it

```sh
./build/strix-hip-decode-token MODEL.gguf $(cat prompt.ids) --generate 256 --dense-fast-pack PACK.gguf
```

- The flag does not change the selected route: the automatic prompt route, MTP and the decode profile
  stay as they are; only the weight bytes of the admitted dense windows change. MTP calibration
  goldens are keyed by the pack, so the first run with a new pack calibrates once.
- One-shot C1 only: it is refused with the HTTP server, batches and cohorts, and with the Q6P32 and
  trellis packs. Session and prefix-state identities include the pack file.
- `STRIX_DENSE_FAST=0` keeps the Q8_0 tensors with the pack loaded (A/B in one command line).
  `STRIX_DENSE_FAST_ROWS=MASK` admits the pack per window size (bit r = windows of r rows). The
  default `0x6` admits 1 and 2 rows, the sizes where the typed kernel measured faster; `0x1fe`
  admits every size up to 8 rows. With a mask that admits only some sizes, a row is computed with
  the copies in the admitted windows and with the Q8_0 values in the others.
- `STRIX_DENSE_FAST_Q8_TYPED=1` (diagnostic): the tensors that have a copy run the typed kernel on
  their own Q8_0 bits. The logits must then equal the default bitwise; it separates the kernel route
  from the values of a pack.
- Receipt: `dense_fast_pack_sha256`, `dense_fast_spec`, `dense_fast_tensors`,
  `dense_fast_packed_bytes`, `dense_fast_replaced_q8_bytes`, and per run `dense_fast_launches`,
  `dense_fast_launch_bytes` (bytes the typed launches cover) against `dense_fast_launch_q8_bytes`.

## Quality gate and speed

One Strix Halo node, production UD-Q4_K_XL checkpoint, default MTP command. The gate is
`tools/strix-gate/fast_logit_gate.py` (pooled KL <= 0.02, top-1 >= 93 %, dNLL <= 0.01) on the
teacher-forced logits of 256 generated tokens for each of a chat, a code and a JSON continuation
(`--teacher-forced-score` on the serial decode rows with the decode numerical profile, reference =
the same command without the pack).

| pack | chat KL | code KL | JSON KL | pooled KL | top-1 | dNLL | gate |
|---|---|---|---|---|---|---|---|
| Q4_K, all | 0.0405 | 0.0172 | 0.1770 | 0.0782 | 95.6 % | +0.0296 | fail |
| Q5_K, all | 0.0110 | 0.0077 | 0.1021 | 0.0403 | 96.6 % | -0.0004 | fail |
| Q6_K, all | 0.0055 | - | 0.1278 | - | - | - | (JSON alone fails) |
| Q5_K, GDN matrices only | - | - | 0.0675 | - | - | - | - |
| Q5_K, indexed-attention matrices only | - | - | 0.0317 | - | - | - | - |

Implementation review of the failure (same node, same commands):

- The route is exact. `STRIX_DENSE_FAST_Q8_TYPED=1` (diagnostic) runs the typed kernel and its
  separate activation quantiser on the Q8_0 bits of every tensor that has a copy: the JSON logits are
  bitwise equal to the reference. So are a second reference run and a run with the pack loaded and
  `STRIX_DENSE_FAST=0`, and a Q6_K run repeats bitwise. Every difference in the table comes from the
  values of the copies, not from the loader, the type dispatch, the block layout or the kernel.
- The quantiser is at the precision of its formats. Against the Q8_0 values the copies differ by
  7.5-8.5 % (Q4_K), 3.7-4.2 % (Q5_K) and 2.0-2.3 % (Q6_K) relative RMS; in a sample of the tensors, per output row at most
  5.5 % and 3.9 % for Q5_K and Q6_K, with a gain of 0.9992-0.9996 (no systematic shrink). Q6_K is
  not near-lossless here: 64 levels per 16 weights give about 2 %, against about 0.5 % for Q8_0.
  The only blocks stored as zero are blocks whose source values are below 2e-4 of the tensor RMS.
  The copies are made from the Q8_0 values because those are what the reference arm computes with.
- The JSON number is dominated by a few rows that flip under any perturbation. The dense A8
  activation path of the default profile, compared with the profile before it
  (`hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1`), gives JSON KL 0.148 on the same sequence:
  five rows carry 88 % of it, among them rows 63 and 2, which also lead the Q6_K run. The same three-prompt gate would therefore also reject a step that is already the default.
- Without those rows the packs are still above the step that is default: the JSON mean without the
  ten largest rows is 0.008 for the A8 step, 0.031 for Q6_K, 0.034 for Q5_K and 0.056 for Q4_K, and
  the median row KL is 0.0002 against 0.0005, 0.0007 and 0.0012.

So the failure is precision, not a defect: no K-quant depth that the typed kernel reads is within
the limit on this content. Greedy text diverges from the reference early (token 4 at 1K, token 23
at 8K).

Speed, Q4_K pack with every size admitted (`STRIX_DENSE_FAST_ROWS=0x1fe`), cycle cost per verify
window size (controller EMA, ms; default / pack):

| context | 1 row | 2 rows | 4 rows | 8 rows | tok/s | MTP acceptance |
|---|---|---|---|---|---|---|
| 1K | 43.7 / 39.2 | 44.0 / 41.6 | 54.0 / 55.9 | 74.0 / 72.9 | 71.4 / 57.1 | 0.746 / 0.618 |
| 8K | 45.1 / 42.7 | 45.1 / 43.1 | 58.0 / 58.6 | 78.9 / 81.0 | 61.0 / 61.4 | 0.702 / 0.688 |

Plain decode (`--no-mtp`, one row per step) at 1K: 41.96 -> 40.77 ms per step. The pack reads
1.42 GB fewer bytes per cycle, but the typed path launches the activation quantiser separately
and decodes more per weight (read from the code, not traced), and the measured gain is 1-4 ms at 1 and 2 rows and nothing at 4 and 8.
End-to-end tok/s follows the acceptance of the changed text, not the kernel. The row-mask default
(`0x6`) follows this table and has not been timed on its own.

Q6_K, the depth closest to the limit, is not faster. Serial decode at 1K (one row per step, the
draft bundle not admitted in that run, default row mask), A B B A: 21.70 and 21.75 tok/s without
the pack, 21.26 and 21.50 tok/s with it. The Q6_K copies are 23 % smaller than the Q8_0 tensors,
and the typed decode costs more than that saves. Not measured with the draft bundle.

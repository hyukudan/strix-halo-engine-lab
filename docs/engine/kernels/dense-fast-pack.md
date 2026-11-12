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

The error is spread, not localised: the logit error does not grow along the sequence, the KL of a
row follows the entropy of the reference distribution, and top-1 changes sit at rows whose reference
margin is small. On prose-like text the KL falls with the weight error as expected (chat: 0.041,
0.011, 0.006). On JSON it does not (0.18, 0.10, 0.13 at 8 %, 4 % and 2 % weight error): a few rows
carry 40-55 % of the total. That is consistent with a discrete decision (such as the block selection
of indexed attention) flipping under any perturbation of the hidden state rather than with bit
depth; the selection itself was not dumped, so this is a hypothesis. Greedy
text diverges from the reference early (token 4 at 1K, token 23 at 8K).

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
